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


@pytest.mark.parametrize("family", ["phi3", "qwen3"])
@pytest.mark.parametrize("fused", [False, True])
def test_source_architecture_gradients_updates_and_peft(tmp_path, family, fused):
    torch.set_num_threads(1)
    report = audit(family, str(tmp_path), fused_ops=fused)
    assert report["frozen_source_weights_unchanged"] and report["reload"]
    assert report["peft"]["format_roundtrip_bitwise"]
    assert report["peft"]["gradient_check_passed"] and report["peft"]["update_check_passed"]
    assert report["peft"]["logit_violations"] == 0
    assert report["adapter_tensors"] == (4 if family == "phi3" else 8)


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


@pytest.mark.parametrize("family", ["phi3", "qwen3"])
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


def test_load_reuses_derivatives(tmp_path, monkeypatch):
    import onnax

    model, _ = prepare("phi3")
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


def test_phi3_trl_training_and_fresh_process_resume(tmp_path):
    torch.set_num_threads(1)
    full, partial, resumed = (str(tmp_path / name) for name in ("full", "partial", "resumed"))
    report = run(full, family="phi3", steps=4)
    assert report["validation_loss"][1] < report["validation_loss"][0]
    run(partial, family="phi3", steps=4, stop=2)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.huggingface.decoder_sft",
            "--family",
            "phi3",
            "--steps",
            "4",
            "--resume",
            os.path.join(partial, "checkpoint-2"),
            "--output-dir",
            resumed,
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
            np.testing.assert_array_equal(left[name], right[name])
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
            torch.testing.assert_close(value, b["state"][key][name], rtol=0, atol=0)


@pytest.mark.parametrize("family", ["phi3", "qwen3"])
def test_native_family_gradients_and_peft(tmp_path, family):
    if os.environ.get("ONNAX_TEST_MLX_EP") != "1":
        pytest.skip("Native MLX opt-in")
    from examples.lora.mlx_runtime import register_library, session

    if path := os.environ.get("ONNAX_MLX_EP_LIBRARY"):
        register_library(path)
    report = audit(family, str(tmp_path), session_factory=session, fused_ops=True)
    assert report["peft"]["gradient_check_passed"] and report["peft"]["update_check_passed"]


@pytest.mark.parametrize("family", ["phi3", "qwen3", "legacy_qwen3"])
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
