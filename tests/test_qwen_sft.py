# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Offline Mobius Qwen3 model, gradients, masks and SFT acceptance."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional integration dependencies.

import numpy as np
import pytest

pytest.importorskip("mobius")
pytest.importorskip("trl")
torch = pytest.importorskip("torch")

from examples.huggingface.sft import prepare, run, token_loss, tokenizer
from examples.huggingface.validation import add_reference_adapters, audit


@pytest.fixture(scope="module")
def workload():
    torch.set_num_threads(1)
    model, reference = prepare()
    return model, add_reference_adapters(reference, model)


def batch():
    ids = torch.tensor([[1, 4, 5, 6, 2], [1, 7, 2, 0, 0]])
    mask = ids.ne(0).long()
    labels = ids.clone()
    labels[mask == 0] = -100
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "position_ids": (mask.cumsum(-1) - 1).clamp(min=0),
        "labels": labels,
    }


def test_qwen_full_parameter_gradients_and_nonzero_b(workload):
    model, reference = workload
    audit(model, reference, batch())
    with torch.no_grad():
        for name, p in zip(model.onnx.names, model.onnx.adapters, strict=True):
            if name.endswith(".B"):
                p.add_(0.01)
    audit(model, reference, batch())


def test_causal_padding_positions_and_dynamic_sequence(workload):
    model, _ = workload
    b = batch()
    a = model(**b).logits.detach()
    changed = {k: v.clone() for k, v in b.items()}
    changed["input_ids"][0, -1] = 17
    changed["input_ids"][1, -2:] = 19
    z = model(**changed).logits.detach()
    torch.testing.assert_close(a[0, :-1], z[0, :-1], rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(a[1, :3], z[1, :3], rtol=1e-4, atol=1e-6)
    short = {k: v[:1, :3] for k, v in b.items()}
    torch.testing.assert_close(a[:1, :3], model(**short).logits, rtol=1e-4, atol=1e-6)


def test_token_normalization_and_all_ignored(workload):
    model, _ = workload
    b = batch()
    model.zero_grad()
    model(**b).loss.backward()
    full = [p.grad.clone() for p in model.onnx.adapters]
    model.zero_grad()
    count = b["labels"][:, 1:].ne(-100).sum()
    for i in range(2):
        model(
            **{k: v[i : i + 1] for k, v in b.items()}, num_items_in_batch=count
        ).loss.backward()
    for p, expected in zip(model.onnx.adapters, full, strict=True):
        torch.testing.assert_close(p.grad, expected, rtol=1e-4, atol=1e-6)
    logits = torch.randn(2, 5, 256, requires_grad=True)
    loss = token_loss(logits, torch.full((2, 5), -100))
    loss.backward()
    assert loss.item() == 0 and logits.grad.count_nonzero() == 0


@pytest.mark.parametrize("mode,nf4", [("full", False), ("joint", True)])
def test_tiny_sft_complete_training(tmp_path, mode, nf4):
    result = run(str(tmp_path), mode=mode, nf4=nf4)
    assert result["steps"] == 100
    assert result["validation_loss"][1] < result["validation_loss"][0]
    assert tokenizer().padding_side == "right"


def test_independent_multi_step_updates():
    model, reference = prepare()
    reference = add_reference_adapters(reference, model)
    actual_optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0)
    expected_optimizer = torch.optim.AdamW(
        [p for p in reference.parameters() if p.requires_grad], lr=0.003, weight_decay=0
    )
    for _ in range(3):
        audit(model, reference, batch(), sync=False)
        actual_optimizer.step()
        expected_optimizer.step()
        for row in model.mapping:
            layer = reference.get_submodule(row["module"])
            for key in ("A", "B"):
                value = model.onnx.arrays()[row[key]]
                np.testing.assert_allclose(
                    value, getattr(layer, key).detach(), rtol=1e-4, atol=1e-6
                )


@pytest.mark.parametrize(
    "mode,nf4,precision,provider",
    [
        ("full", False, "fp32", "cpu"),
        ("joint", True, "fp32", "cpu"),
        ("joint", False, "bf16", "reference"),
        ("joint", True, "bf16", "reference"),
    ],
)
def test_sft_fresh_process_checkpoint_resume(tmp_path, mode, nf4, precision, provider):
    import os
    import subprocess
    import sys

    result = run(
        str(tmp_path),
        mode=mode,
        nf4=nf4,
        projection_precision=precision,
        provider=provider,
        diagnose_rounding=precision == "bf16",
    )
    assert result["frozen_buffers_unchanged"]
    assert result["validation_loss"][1] < result["validation_loss"][0]
    resumed = tmp_path / "resumed"
    with np.load(tmp_path / "export" / "adapter.npz") as data:
        expected = {n: data[n].copy() for n in data.files}
    subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.huggingface.sft",
            "--output-dir",
            str(resumed),
            "--resume",
            os.fspath(tmp_path / "checkpoint-50"),
            "--mode",
            mode,
            "--projection-precision",
            precision,
            "--provider",
            provider,
            *(["--diagnose-rounding"] if precision == "bf16" else []),
            *(["--nf4"] if nf4 else []),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with np.load(resumed / "export" / "adapter.npz") as data:
        for n, value in expected.items():
            np.testing.assert_array_equal(data[n], value)

    from examples.huggingface.replay import verify

    replay = verify(str(tmp_path), str(resumed), step=100, provider=provider)
    assert replay["optimizer_bitwise_equal"]
    assert replay["export_load_outputs_bitwise_equal"]
    if mode == "joint":
        assert replay["export_load_loss_gradients_bitwise_equal"]
        assert replay["exported_gradient_tensors"] == len(expected)


def test_noncontiguous_position_ids(workload):
    model, reference = workload
    inputs = batch()
    inputs["position_ids"] = 2 * inputs["position_ids"] + 3
    audit(model, reference, inputs)


def test_fused_mobius_and_decomposed_graphs():
    import onnx_ir as ir

    from examples.huggingface.qwen import build_base, decompose, reference_model
    from examples.lora.train import session

    source = build_base(reference_model())
    lowered = decompose(source)
    feeds = {n: v.numpy() for n, v in batch().items() if n != "labels"}
    expected = session(ir.to_proto(source).SerializeToString()).run(None, feeds)
    actual = session(ir.to_proto(lowered).SerializeToString()).run(None, feeds)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)


def test_swish_decomposition_retains_negative_tail():
    import math

    import onnx_ir as ir

    import onnax
    from examples.huggingface.qwen import decompose
    from examples.lora.train import session

    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[5])
    node = ir.node("Swish", [x])
    y = node.outputs[0]
    y.name, y.dtype, y.shape = "y", x.dtype, x.shape
    graph = ir.Model(
        ir.Graph([x], [y], nodes=[node], opset_imports={"": 24}, name="swish_tail"),
        ir_version=12,
    )
    graph = decompose(graph)
    derivative = onnax.vjp(graph)
    values = np.array([-40, -18, 0, 18, 40], np.float32)
    expected = np.array(
        [
            math.exp(-abs(float(v))) / (1 + math.exp(-abs(float(v))))
            if v < 0
            else 1 / (1 + math.exp(-float(v)))
            for v in values
        ]
    )
    actual, gradient = session(ir.to_proto(derivative).SerializeToString()).run(
        None, {"x": values, derivative.graph.inputs[-1].name: np.ones_like(values)}
    )
    np.testing.assert_allclose(actual, values * expected, rtol=2e-5, atol=0)
    np.testing.assert_allclose(
        gradient, expected + values * expected * (1 - expected), rtol=2e-5, atol=0
    )


def test_same_input_layer_audit():
    from examples.huggingface.layer_audit import run

    report = run()
    assert len(report["input_ids"][0]) == 64
    assert {row["op"] for row in report["operators"]} == {
        "MatMul",
        "RMSNormalization",
        "Swish",
    }
    dots = [row["worst_scaled_dot"] for row in report["operators"] if row["op"] == "MatMul"]
    assert dots and all(np.isfinite(dot["scalar_fsum_oracle"]) for dot in dots)
    assert report["logits"]["relative_l2"] < 1e-4


def test_completed_training_report_survives_optional_audit_failure(tmp_path, monkeypatch):
    import json

    from examples.huggingface import peft_adapter

    def fail(*args, **kwargs):
        raise RuntimeError("optional audit failed")

    monkeypatch.setattr(peft_adapter, "roundtrip", fail)
    with pytest.raises(RuntimeError, match="optional audit failed"):
        run(str(tmp_path), steps=1, peft_roundtrip=True)
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["steps"] == 1
    assert report["requested_audits"]["peft"]
    assert (tmp_path / "export" / "forward.onnx").is_file()
    assert (tmp_path / "adapter" / "adapter_model.safetensors").is_file()


def test_replay_verification_checks_state_bytes():
    from examples.huggingface.replay import _equal_array, _equal_tree

    state = {"optimizer": [torch.tensor([0.0, 1.0]), {"step": 3}]}
    _equal_tree(state, {"optimizer": [torch.tensor([0.0, 1.0]), {"step": 3}]})
    with pytest.raises(AssertionError):
        _equal_array(np.array([0.0], np.float32), np.array([-0.0], np.float32))
    with pytest.raises(AssertionError):
        _equal_tree(state, {"optimizer": [torch.tensor([-0.0, 1.0]), {"step": 3}]})
    with pytest.raises(AssertionError):
        _equal_tree(state, {"optimizer": [torch.tensor([0.0, 1.0]), {"step": 4}]})


def test_reloaded_output_digests_preserve_bytes_and_reject_nonfinite():
    from examples.huggingface.replay import _output_digest

    value = np.arange(24, dtype=np.float32).reshape(4, 6).T
    assert _output_digest(value) == _output_digest(value.copy())
    assert _output_digest(np.array([0.0], np.float32)) != _output_digest(
        np.array([-0.0], np.float32)
    )
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        _output_digest(np.array([np.nan], np.float32))


@pytest.mark.parametrize("side", ["onnx", "reference"])
def test_diagnostic_gradient_audit_rejects_nonfinite_values(side):
    from examples.huggingface.validation import add_reference_adapters, audit

    model, reference = prepare()
    reference = add_reference_adapters(reference, model)
    parameter = (
        model.onnx.adapters[0]
        if side == "onnx"
        else reference.get_submodule(model.mapping[0]["module"]).A
    )
    parameter.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
    ids = torch.tensor([[1, 4, 5, 2]])
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids}
    with pytest.raises(AssertionError, match="Nonfinite gradient"):
        audit(model, reference, batch, diagnose_rounding=True)


def test_sft_stable_reference_includes_peft_control(tmp_path):
    from examples.huggingface.sft import run

    report = run(
        str(tmp_path),
        steps=2,
        mode="chunked",
        nf4=True,
        stable_reference=True,
        peft_roundtrip=True,
    )
    assert report["frozen_buffers_unchanged"]
    assert report["peft"]["format_roundtrip_bitwise"]
    control = report["peft"]["stable_reference_audit"]
    assert control["loss_check_passed"]
    assert control["gradient_check_passed"]
    assert control["update_check_passed"]


@pytest.mark.parametrize("max_length", [32, 64])
def test_custom_text_training_fresh_process_resume(tmp_path, monkeypatch, max_length):
    import json
    import subprocess
    import sys

    from transformers import AutoTokenizer

    from examples.huggingface import sft

    train = tmp_path / "train.txt"
    validation = tmp_path / "validation.txt"
    lines = ["Amber birds " * 40 + "glide."] + [
        f"Amber birds glide over blue water {index}." for index in range(8)
    ]
    train.write_text("\n".join(lines) + "\n", encoding="utf-8")
    validation.write_text("Unseenword birds glide.\nBlue water is quiet.\n", encoding="utf-8")
    original = sft.make_trainer

    def checkpoint_each_step(*args, **kwargs):
        trainer = original(*args, **kwargs)
        assert trainer.args.max_length == max_length
        assert max(map(len, trainer.train_dataset["input_ids"])) <= max_length
        trainer.args.save_steps = 1
        return trainer

    monkeypatch.setattr(sft, "make_trainer", checkpoint_each_step)
    directory = tmp_path / "continuous"
    report = sft.run(
        str(directory),
        steps=4,
        mode="joint",
        nf4=True,
        train_file=train,
        validation_file=validation,
        max_length=max_length,
    )
    assert report["data"]["max_length"] == max_length
    assert report["data"]["train_lines"] == 9
    assert report["data"]["validation_lines"] == 2
    assert report["frozen_buffers_unchanged"]
    tok = AutoTokenizer.from_pretrained(directory / "export")
    assert "Unseenword" not in tok.get_vocab()
    assert sft._tokenizer_digest(tok) == report["data"]["tokenizer_sha256"]
    resumed = tmp_path / "resumed"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.huggingface.sft",
            "--output-dir",
            str(resumed),
            "--mode",
            "joint",
            "--nf4",
            "--steps",
            "4",
            "--max-length",
            str(max_length),
            "--resume",
            str(directory / "checkpoint-1"),
            "--train-file",
            str(train),
            "--validation-file",
            str(validation),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    resumed_report = json.loads((resumed / "report.json").read_text())
    assert resumed_report["data"] == report["data"]
    with (
        np.load(directory / "export" / "adapter.npz") as a,
        np.load(resumed / "export" / "adapter.npz") as b,
    ):
        assert a.files == b.files
        for name in a.files:
            np.testing.assert_array_equal(a[name], b[name])
    assert (resumed / "export" / "forward.onnx").is_file()
    assert (resumed / "export" / "loss_gradient.onnx").is_file()
    import functools

    import onnx

    from examples.huggingface.replay import _equal_array, _training_outputs
    from examples.lora.train import session

    encoded = tok(lines[0], truncation=True, max_length=max_length, return_tensors="np")
    feeds = {name: encoded[name] for name in ("input_ids", "attention_mask")}
    feeds["position_ids"] = (np.cumsum(feeds["attention_mask"], axis=1) - 1).clip(min=0)
    factory = functools.partial(session, memory_reuse=False)
    forward, training = [], []
    for root in (directory, resumed):
        for name in ("forward.onnx", "loss_gradient.onnx"):
            onnx.checker.check_model(root / "export" / name, full_check=True)
        forward.append(factory(root / "export" / "forward.onnx").run(None, feeds))
        interface = json.loads((root / "export" / "interface.json").read_text())
        training.append(_training_outputs(root, interface, feeds, factory))
    for outputs in (forward, training):
        for a, b in zip(*outputs, strict=True):
            _equal_array(a, b)
    with pytest.raises(ValueError, match="No checkpoint found"):
        sft.run(str(tmp_path / "missing-checkpoint"), resume=True)
    saved_identity = json.loads((directory / "checkpoint-1" / "onnax_data.json").read_text())
    assert saved_identity == report["data"]
    with monkeypatch.context() as context:

        def unexpected_model(*args, **kwargs):
            pytest.fail("A changed context must be rejected before model construction")

        context.setattr(sft, "prepare", unexpected_model)
        with pytest.raises(ValueError, match="context length differs"):
            sft.run(
                str(tmp_path / "changed-context"),
                resume=directory / "checkpoint-1",
                train_file=train,
                validation_file=validation,
                max_length=64 if max_length == 32 else 32,
            )
    train.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
    reordered = sft.tokenizer(training_texts=list(reversed(lines)))
    assert sft._tokenizer_digest(reordered) == report["data"]["tokenizer_sha256"]
    with pytest.raises(
        ValueError, match="Checkpoint training data, tokenizer or context length differs"
    ):
        sft.run(
            str(tmp_path / "changed"),
            resume=directory / "checkpoint-1",
            train_file=train,
            validation_file=validation,
            max_length=max_length,
        )
    train.write_text("\n".join(lines) + "\n", encoding="utf-8")
    validation.write_text("A different held-out sequence.\n", encoding="utf-8")
    with pytest.raises(
        ValueError, match="Checkpoint training data, tokenizer or context length differs"
    ):
        sft.run(
            str(tmp_path / "changed-validation"),
            resume=directory / "checkpoint-1",
            train_file=train,
            validation_file=validation,
            max_length=max_length,
        )
    (directory / "checkpoint-1" / "onnax_data.json").unlink()
    with pytest.raises(ValueError, match="requires checkpoint data identity"):
        sft.run(
            str(tmp_path / "missing-identity"),
            resume=directory / "checkpoint-1",
            train_file=train,
            validation_file=validation,
            max_length=max_length,
        )

    with pytest.raises(ValueError, match="both training and validation"):
        sft.run(str(tmp_path / "invalid"), train_file=train)
    validation.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="both be nonempty"):
        sft.run(str(tmp_path / "invalid"), train_file=train, validation_file=validation)


@pytest.mark.parametrize("max_length", [1, True, 65, 2.5])
def test_sft_rejects_invalid_context_before_model_construction(
    tmp_path, monkeypatch, max_length
):
    from examples.huggingface import sft

    def unexpected_model(*args, **kwargs):
        pytest.fail("Invalid context must be rejected before model construction")

    monkeypatch.setattr(sft, "prepare", unexpected_model)
    with pytest.raises(ValueError, match="max_length"):
        sft.run(str(tmp_path), max_length=max_length)


def test_requested_sft_reference_comparisons_are_diagnostics_by_default(tmp_path, monkeypatch):
    from examples.huggingface import validation

    original = validation.audit

    def shifted_reference(model, reference, inputs, **kwargs):
        # A uniform logit shift preserves the probability distribution and loss.
        hook = reference.lm_head.register_forward_hook(
            lambda module, args, output: output + 0.01
        )
        try:
            return original(model, reference, inputs, **kwargs)
        finally:
            hook.remove()

    monkeypatch.setattr(validation, "audit", shifted_reference)
    report = run(str(tmp_path / "diagnostic"), steps=1, mode="joint", reference_audit=True)
    assert not report["requested_audits"]["strict_reference"]
    assert len(report["audits"]) == 2
    assert all(not row["strict_logit_check_passed"] for row in report["audits"])
    assert report["frozen_buffers_unchanged"]
    with pytest.raises(AssertionError):
        run(str(tmp_path / "strict"), steps=1, mode="joint", diagnose_rounding=False)


@pytest.mark.parametrize("failure", ["logits", "loss", "shape"])
def test_diagnostic_audit_rejects_invalid_forward_values(failure):
    model, reference = prepare()
    reference = add_reference_adapters(reference, model)

    def corrupt_forward(module, args, output):
        if failure == "logits":
            output.logits = torch.full_like(output.logits, float("nan"))
        elif failure == "loss":
            output.loss = output.loss * float("nan")
        else:
            output.logits = output.logits[..., :-1]
        return output

    hook = model.register_forward_hook(corrupt_forward)
    try:
        with pytest.raises(AssertionError, match=r"Nonfinite|shape mismatch"):
            audit(model, reference, batch(), diagnose_rounding=True)
    finally:
        hook.remove()


def test_default_sft_releases_reference_and_preserves_training(tmp_path, monkeypatch):
    import weakref

    from examples.huggingface import sft

    original_prepare, original_trainer = sft.prepare, sft.make_trainer
    references, heads = [], []
    expect_reference = False

    def tracked_prepare(*args, **kwargs):
        model, reference = original_prepare(*args, **kwargs)
        references.append(weakref.ref(reference))
        return model, reference

    def checked_trainer(model, *args, **kwargs):
        assert (references[-1]() is not None) == expect_reference
        heads.append(model.head)
        return original_trainer(model, *args, **kwargs)

    monkeypatch.setattr(sft, "prepare", tracked_prepare)
    monkeypatch.setattr(sft, "make_trainer", checked_trainer)
    ordinary = sft.run(str(tmp_path / "ordinary"), mode="chunked", nf4=True, steps=2)
    assert ordinary["audits"] == []
    assert not ordinary["requested_audits"]["reference"]
    assert not ordinary["reference_retained_during_training"]
    assert heads[0].forward_executor is None
    expect_reference = True
    audited = sft.run(
        str(tmp_path / "audited"), mode="chunked", nf4=True, steps=2, reference_audit=True
    )
    assert len(audited["audits"]) == 2
    assert audited["requested_audits"]["reference"]
    assert heads[1].forward_executor is not None
    assert ordinary["validation_loss"] == audited["validation_loss"]
    with (
        np.load(tmp_path / "ordinary/export/adapter.npz") as a,
        np.load(tmp_path / "audited/export/adapter.npz") as b,
    ):
        assert a.files == b.files
        for name in a.files:
            np.testing.assert_array_equal(a[name], b[name])


def test_default_sft_rejects_nonfinite_update_before_export(tmp_path, monkeypatch):
    from examples.huggingface import sft

    original = sft.prepare

    def invalid_gradient(*args, **kwargs):
        model, reference = original(*args, **kwargs)
        model.onnx.adapters[0].register_hook(
            lambda gradient: torch.full_like(gradient, float("nan"))
        )
        return model, reference

    monkeypatch.setattr(sft, "prepare", invalid_gradient)
    with pytest.raises(FloatingPointError, match="Nonfinite adapter parameter"):
        sft.run(str(tmp_path), mode="joint", steps=1)
    assert not (tmp_path / "export" / "forward.onnx").exists()


def test_jsonl_preserves_multiline_training_examples(tmp_path):
    import json

    from examples.huggingface.sft import texts

    path = tmp_path / "train.jsonl"
    expected = ["User: line one\nline two\nAssistant: answer", " next sample "]
    path.write_text("\n".join(json.dumps({"text": text}) for text in expected))
    assert texts(path) == expected
    path.write_text('{"text": ""}\n')
    with pytest.raises(ValueError, match="text"):
        texts(path)


def test_multiline_data_identity_preserves_example_boundaries():
    import hashlib

    from examples.huggingface.sft import _data_identity

    tok = tokenizer()
    plain = _data_identity(["a", "b"], ["c"], tok)
    assert plain["version"] == 1
    assert plain["train_sha256"] == hashlib.sha256(b"a\nb").hexdigest()
    left = _data_identity(["a\nb", "c"], ["heldout"], tok)
    right = _data_identity(["a", "b\nc"], ["heldout"], tok)
    assert left["version"] == right["version"] == 2
    assert left["train_sha256"] != right["train_sha256"]
