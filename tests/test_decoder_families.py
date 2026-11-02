# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Mobius architecture adapters through shared execution, TRL and PEFT."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional model dependencies.

import json
import os
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("mobius")
pytest.importorskip("trl")
torch = pytest.importorskip("torch")

from examples.huggingface.decoder_families import prepare_graph
from examples.huggingface.decoder_sft import OnnxDecoder, audit, batch, prepare, run


@pytest.mark.parametrize("family", ["phi3", "qwen3", "qwen35"])
@pytest.mark.parametrize("fused", [False, True])
def test_source_architecture_gradients_updates_and_peft(tmp_path, family, fused):
    torch.set_num_threads(1)
    report = audit(family, str(tmp_path), fused_ops=fused)
    assert report["frozen_source_weights_unchanged"] and report["reload"]
    assert report["peft"]["format_roundtrip_bitwise"]
    assert report["peft"]["gradient_check_passed"] and report["peft"]["update_check_passed"]
    assert report["peft"]["logit_violations"] == 0
    assert report["adapter_tensors"] == {"phi3": 4, "qwen3": 8, "qwen35": 10}[family]


def test_phi3_source_mapping_and_shared_low_rank_product():
    graph, arrays, mapping, reference = prepare_graph("phi3")
    expected = (
        reference.config.hidden_size
        + 2 * reference.config.num_key_value_heads * reference.config.head_dim
    )
    for row in mapping:
        assert row["module"].endswith("self_attn.qkv_proj")
        assert row["source_weight"] in reference.state_dict()
        assert arrays[row["B"]].shape == (4, expected)
        assert [piece["module"].rsplit(".", 1)[-1] for piece in row["pieces"]] == [
            "q_proj",
            "k_proj",
            "v_proj",
        ]
        consumers = [
            node
            for node in graph.graph
            if node.op_type == "MatMul" and node.inputs[1].name == row["A"]
        ]
        assert len(consumers) == 1
        assert consumers[0].inputs[0].shape[-1] == reference.config.hidden_size
        assert not any(
            node.op_type == "MatMul"
            and node.inputs[0].name == row["A"]
            and node.inputs[1].name == row["B"]
            for node in graph.graph
        )


@pytest.mark.parametrize("family", ["phi3", "qwen3", "qwen35"])
def test_dynamic_padding_positions_ignore_labels_and_accumulation(family):
    model, _ = prepare(family)
    try:
        data = batch()
        target = {name: value[:1, :4] for name, value in data.items()}
        model(**data).loss.backward()
        first = {
            name: value.grad.clone() for name, value in model.onnx.parameters_by_name().items()
        }
        model(**data).loss.backward()
        for name, value in model.onnx.parameters_by_name().items():
            torch.testing.assert_close(value.grad, 2 * first[name], rtol=1e-4, atol=1e-6)
        model.zero_grad()
        # Sum unequal microbatches using their total effective shifted-token count.
        total = sum(int(values["labels"][:, 1:].ne(-100).sum()) for values in (data, target))
        for values in (data, target):
            model(**values, num_items_in_batch=total).loss.backward()
        accumulated = {
            name: value.grad.clone() for name, value in model.onnx.parameters_by_name().items()
        }
        model.zero_grad()
        loss = (
            sum(
                model(**values).loss * values["labels"][:, 1:].ne(-100).sum()
                for values in (data, target)
            )
            / total
        )
        loss.backward()
        for name, value in model.onnx.parameters_by_name().items():
            torch.testing.assert_close(value.grad, accumulated[name], rtol=1e-4, atol=1e-6)
        model.zero_grad()
        target["labels"] = torch.full_like(target["labels"], -100)
        ignored = model(**target).loss
        assert ignored.item() == 0
        ignored.backward()
        assert all(value.grad.count_nonzero() == 0 for value in model.parameters())
    finally:
        model.close()


@pytest.mark.parametrize("family", ["phi3", "qwen35"])
def test_load_reuses_derivatives(tmp_path, monkeypatch, family):
    import onnax

    model, _ = prepare(family)
    try:
        expected = model(**batch()).loss
        expected.backward()
        gradients = [p.grad.clone() for p in model.parameters()]
        model.export(str(tmp_path))
    finally:
        model.close()

    def fail(*args, **kwargs):
        raise AssertionError("Reuse the saved VJP modules")

    monkeypatch.setattr(onnax, "vjp", fail)
    restored = OnnxDecoder.load(str(tmp_path))
    try:
        actual = restored(**batch()).loss
        actual.backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for parameter, gradient in zip(restored.parameters(), gradients, strict=True):
            torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
    finally:
        restored.close()


@pytest.mark.parametrize("family", ["phi3", "qwen35"])
@pytest.mark.parametrize("head_chunk_size", [None, 3])
def test_trl_training_and_fresh_process_resume(tmp_path, family, head_chunk_size):
    torch.set_num_threads(1)
    full, partial, resumed = (str(tmp_path / name) for name in ("full", "partial", "resumed"))
    report = run(full, family=family, steps=4, head_chunk_size=head_chunk_size)
    assert report["validation_loss"][1] < report["validation_loss"][0]
    run(partial, family=family, steps=4, stop=2, head_chunk_size=head_chunk_size)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.huggingface.decoder_sft",
            "--family",
            family,
            "--steps",
            "4",
            "--resume",
            os.path.join(partial, "checkpoint-2"),
            "--output-dir",
            resumed,
            *(["--head-chunk-size", str(head_chunk_size)] if head_chunk_size else []),
        ],
        check=True,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
    )
    models = [
        OnnxDecoder.load(os.path.join(directory, "export")) for directory in (full, resumed)
    ]
    try:
        left, right = (model.onnx.arrays() for model in models)
        for name in left:
            assert (
                left[name].dtype == right[name].dtype and left[name].shape == right[name].shape
            )
            assert left[name].tobytes() == right[name].tobytes()
        torch.testing.assert_close(
            models[0](**batch()).loss, models[1](**batch()).loss, rtol=0, atol=0
        )
    finally:
        for model in models:
            model.close()
    a = torch.load(os.path.join(full, "checkpoint-4/optimizer.pt"), weights_only=True)
    b = torch.load(os.path.join(resumed, "checkpoint-4/optimizer.pt"), weights_only=True)
    for key, state in a["state"].items():
        for name, value in state.items():
            other = b["state"][key][name]
            assert value.dtype == other.dtype and value.shape == other.shape
            assert torch.equal(
                value.contiguous().reshape(-1).view(torch.uint8),
                other.contiguous().reshape(-1).view(torch.uint8),
            )
    assert torch.load(
        os.path.join(full, "checkpoint-4/scheduler.pt"), weights_only=True
    ) == torch.load(os.path.join(resumed, "checkpoint-4/scheduler.pt"), weights_only=True)


@pytest.mark.parametrize("family", ["phi3", "qwen3", "qwen35"])
def test_native_family_gradients_and_peft(tmp_path, family):
    if os.environ.get("ONNAX_TEST_MLX_EP") != "1":
        pytest.skip("Native MLX opt-in")
    from examples.lora.mlx_runtime import register_library, session

    if path := os.environ.get("ONNAX_MLX_EP_LIBRARY"):
        register_library(path)
    report = audit(family, str(tmp_path), session_factory=session, fused_ops=True)
    assert report["peft"]["gradient_check_passed"] and report["peft"]["update_check_passed"]


@pytest.mark.parametrize("family", ["phi3", "qwen3", "qwen35", "legacy_qwen3"])
def test_trainer_counts_shifted_tokens_for_unequal_microbatches(tmp_path, family):
    from examples.huggingface.sft import make_trainer, tokenizer

    if family == "legacy_qwen3":
        from examples.huggingface.sft import prepare as legacy_prepare

        model, _ = legacy_prepare()
    else:
        model, _ = prepare(family)
    try:
        trainer = make_trainer(model, tokenizer(), str(tmp_path), steps=2)
        values = batch()
        smaller = {name: tensor[:1, :3] for name, tensor in values.items()}
        actual = trainer._get_num_items_in_batch([values, smaller], torch.device("cpu"))
        expected = sum(value["labels"][:, 1:].ne(-100).sum() for value in (values, smaller))
        assert actual == expected
    finally:
        if hasattr(model, "close"):
            model.close()


@pytest.mark.parametrize("change", ["family", "parameters", "ranges"])
def test_exported_task_rejects_inconsistent_source_mapping(tmp_path, change):
    model, _ = prepare("phi3")
    try:
        model.export(str(tmp_path))
    finally:
        model.close()
    path = tmp_path / "task.json"
    record = json.loads(path.read_text())
    if change == "family":
        record["family"] = "qwen3"
    elif change == "parameters":
        record["mapping"][0]["A"] = "missing"
    else:
        record["mapping"][0]["pieces"][1]["start"] += 1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        OnnxDecoder.load(str(tmp_path))


@pytest.mark.parametrize("checkpoint_layers", [1, 2, 4])
def test_hybrid_decoder_grouping_preserves_source_layout_and_gradients(checkpoint_layers):
    from examples.huggingface.validation import add_reference_parameters

    torch.set_num_threads(1)
    model, reference = prepare("qwen35", checkpoint_layers=checkpoint_layers)
    try:
        assert len(model.onnx.blocks) == 2 + (4 + checkpoint_layers - 1) // checkpoint_layers
        assert (
            sum(row["module"].endswith(".linear_attn.in_proj_qkv") for row in model.mapping)
            == 3
        )
        rng = np.random.default_rng(71)
        with torch.no_grad():
            for name, parameter in model.onnx.parameters_by_name().items():
                if name.endswith(".B"):
                    parameter.copy_(
                        torch.from_numpy(
                            rng.normal(0, 0.02, tuple(parameter.shape)).astype(np.float32)
                        )
                    )
        reference = add_reference_parameters(reference, model.onnx.arrays(), model.mapping)
        ids = torch.from_numpy(rng.integers(3, 128, (2, 37), dtype=np.int64))
        mask = torch.ones_like(ids)
        mask[0, -5:] = 0
        ids[mask == 0] = 0
        values = {
            "input_ids": ids,
            "attention_mask": mask,
            "position_ids": (mask.cumsum(-1) - 1).clamp(min=0),
        }
        labels = ids.masked_fill(mask == 0, -100)
        actual = model(**values, labels=labels)
        expected = reference(**values, labels=labels, use_cache=False)
        torch.testing.assert_close(
            actual.logits[mask.bool()], expected.logits[mask.bool()], rtol=1e-4, atol=1e-6
        )
        torch.testing.assert_close(actual.loss, expected.loss, rtol=1e-4, atol=1e-6)
        actual.loss.backward()
        expected.loss.backward()
        for row in model.mapping:
            target = reference.get_submodule(row["module"])
            assert (
                model.onnx.parameters_by_name()[row["B"]].shape[1]
                == target.base.weight.shape[0]
            )
            for key in ("A", "B"):
                torch.testing.assert_close(
                    model.onnx.parameters_by_name()[row[key]].grad,
                    getattr(target, key).grad,
                    rtol=1e-4,
                    atol=1e-6,
                )
    finally:
        model.close()
