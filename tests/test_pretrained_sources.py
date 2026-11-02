# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Pinned text-backbone loading and source-name provenance without downloads."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional model dependencies.

import dataclasses
import json

import pytest

pytest.importorskip("mobius")
torch = pytest.importorskip("torch")
from safetensors.torch import load_file, save_file
from transformers import Qwen3_5Config

from examples.huggingface.decoder_families import FAMILIES, prepare_graph, reference_model
from examples.huggingface.pretrained import PretrainedSource, load_tokenizer
from examples.huggingface.qwen35 import reference_model as tiny_reference
from examples.huggingface.sft import tokenizer


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    reference = tiny_reference()
    config = Qwen3_5Config(text_config=reference.config.to_dict())
    config.save_pretrained(tmp_path)
    weights = {
        "model.language_model." + name.removeprefix("model."): value.to(torch.bfloat16).clone()
        for name, value in reference.state_dict().items()
        if name != "lm_head.weight"
    }
    weights["model.visual.fixture.weight"] = torch.ones(2, 2, dtype=torch.bfloat16)
    weights["mtp.fixture.weight"] = torch.zeros(2, 2, dtype=torch.bfloat16)
    save_file(weights, str(tmp_path / "model.safetensors"), metadata={"format": "pt"})
    tokenizer().save_pretrained(tmp_path)
    source = PretrainedSource(str(tmp_path), "1" * 40, (("model.language_model.", "model."),))
    monkeypatch.setitem(
        FAMILIES, "qwen35", dataclasses.replace(FAMILIES["qwen35"], pretrained=source)
    )
    return reference, source


def test_load_complete_text_backbone_fp32_tied_and_mapped(checkpoint):
    original, source = checkpoint
    loaded = reference_model("qwen35", real=True)
    assert loaded.model.embed_tokens.weight.data_ptr() == loaded.lm_head.weight.data_ptr()
    assert all(p.dtype == torch.float32 for p in loaded.parameters())
    for name, actual in loaded.state_dict().items():
        expected = original.state_dict()[name].to(torch.bfloat16).to(torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    record = loaded.config.onnax_pretrained
    assert record["revision"] == source.revision
    assert record["checkpoint_to_reference_prefix"] == {"model.language_model.": "model."}
    assert not any(record["loading"].values())
    assert load_tokenizer(source).padding_side == "right"


@pytest.mark.parametrize("change", ["missing", "unexpected", "shape"])
def test_loading_mismatches_stop_before_training(checkpoint, change):
    _, source = checkpoint
    filename = source.model_id + "/model.safetensors"
    weights = load_file(filename)
    name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    if change == "missing":
        del weights[name]
    elif change == "shape":
        weights[name] = weights[name][:-1].clone()
    else:
        weights["model.language_model.unexpected.weight"] = torch.ones(2, 2)
    save_file(weights, filename, metadata={"format": "pt"})
    with pytest.raises((ValueError, RuntimeError)):
        reference_model("qwen35", real=True)


def test_graph_retains_original_checkpoint_parameter_names(checkpoint):
    _, source = checkpoint
    graph, arrays, mapping, _ = prepare_graph("qwen35", real=True)
    assert len(arrays) == 10
    for row in mapping:
        assert row["checkpoint_weight"] == source.checkpoint_name(row["source_weight"])
        assert row["checkpoint_weight"].startswith("model.language_model.layers.")
    assert "onnax.example.pretrained_source" in graph.metadata_props


def test_unpinned_revision_and_missing_source_are_rejected():
    with pytest.raises(ValueError, match="commit SHA"):
        PretrainedSource("example/model", "main")
    with pytest.raises(ValueError, match="No pinned pretrained"):
        reference_model("phi3", real=True)


def test_pretrained_sft_uses_pinned_tokenizer_and_training_boundary(checkpoint, tmp_path):
    import subprocess
    import sys

    from examples.huggingface.decoder_sft import OnnxDecoder, run, text_batch
    from examples.huggingface.peft_adapter import export_adapter

    _, source = checkpoint
    result = run(str(tmp_path / "training"), family="qwen35", real=True, steps=2)
    assert result["pretrained_source"]["revision"] == source.revision
    assert result["microbatch_size"] == 1
    assert result["gradient_accumulation_steps"] == 4
    training = [row for row in result["batches"] if row["training"]]
    assert len(training) == 8
    assert all(row["batch"] == 1 and row["sequence"] == 64 for row in training)
    assert all(row["valid_shifted_targets"] > 0 for row in training)
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import dataclasses, sys; "
                "from examples.huggingface.decoder_families import FAMILIES; "
                "from examples.huggingface.pretrained import PretrainedSource; "
                "from examples.huggingface.decoder_sft import run; "
                "FAMILIES['qwen35'] = dataclasses.replace(FAMILIES['qwen35'], "
                "pretrained=PretrainedSource(sys.argv[1], sys.argv[2], "
                "(('model.language_model.', 'model.'),))); "
                "run(sys.argv[3], family='qwen35', real=True, steps=2, resume=sys.argv[4])"
            ),
            source.model_id,
            source.revision,
            str(tmp_path / "resumed"),
            str(tmp_path / "training" / "checkpoint-1"),
        ],
        check=True,
    )
    models = [
        OnnxDecoder.load(str(tmp_path / name / "export")) for name in ("training", "resumed")
    ]
    try:
        assert models[0].config.onnax_pretrained == models[1].config.onnax_pretrained
        assert models[0].config._commit_hash == source.revision
        export_adapter(models[0], str(tmp_path / "reexported-adapter"))
        adapter_config = json.loads(
            (tmp_path / "reexported-adapter" / "adapter_config.json").read_text()
        )
        assert adapter_config["revision"] == source.revision
        for name, value in models[0].onnx.arrays().items():
            assert value.tobytes() == models[1].onnx.arrays()[name].tobytes()
        data = text_batch(load_tokenizer(source))
        torch.testing.assert_close(
            models[0](**data).logits, models[1](**data).logits, rtol=0, atol=0
        )
    finally:
        for model in models:
            model.close()


@pytest.mark.parametrize("real", [False, True])
def test_checkpoint_source_checked_before_model_construction(checkpoint, tmp_path, real):
    from examples.huggingface.decoder_sft import run

    path = tmp_path / "resume"
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_text",
                "onnax_pretrained": {"model_id": "other", "revision": "2" * 40},
            }
        )
    )
    with pytest.raises(ValueError, match="pretrained"):
        run(str(tmp_path / "unused"), family="qwen35", real=real, resume=str(path))
    assert not (tmp_path / "unused").exists()


def test_stable_peft_control_accepts_checkpointed_parameter_interface(checkpoint, tmp_path):
    from examples.huggingface.decoder_sft import batch, prepare
    from examples.huggingface.peft_adapter import roundtrip

    model, reference = prepare("qwen35", real=True)
    try:
        report = roundtrip(
            model,
            reference,
            batch(),
            str(tmp_path / "adapter"),
            diagnose_rounding=True,
            stable_reference=True,
        )
        control = report["stable_reference_audit"]
        assert control["adapter_tensors"] == 10
        assert control["loss_check_passed"]
        assert control["gradient_check_passed"]
    finally:
        model.close()
