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


@pytest.mark.parametrize("mode,nf4", [("full", False), ("joint", True)])
def test_sft_fresh_process_checkpoint_resume(tmp_path, mode, nf4):
    import os
    import subprocess
    import sys

    run(str(tmp_path), mode=mode, nf4=nf4)
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

    replay = verify(str(tmp_path), str(resumed), step=100)
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
