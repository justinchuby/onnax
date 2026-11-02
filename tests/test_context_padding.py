# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Padding policy, shifted-token counts and checkpoint input identity."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional workload dependencies.

import json

import pytest

pytest.importorskip("mobius")
pytest.importorskip("trl")
torch = pytest.importorskip("torch")

from examples.huggingface import sft


@pytest.mark.parametrize("multiple", [0, -1, True, 8.0, 3, 128])
def test_invalid_padding_multiple(multiple):
    with pytest.raises(ValueError, match="dividing max_length"):
        sft._validate_padding(64, multiple)


def test_padding_multiple_masks_only_added_tokens_and_preserves_counts(tmp_path):
    torch.set_num_threads(1)
    model, _ = sft.prepare()
    tok = sft.tokenizer()
    trainer = sft.make_trainer(model, tok, str(tmp_path), steps=2, pad_to_multiple_of=8)
    batch = trainer.data_collator([{"input_ids": [1, 4, 2]}, {"input_ids": [1, 5, 6, 7, 2]}])
    assert batch["input_ids"].shape == (2, 8)
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == -100)
    assert (batch["labels"] == tok.eos_token_id).sum() == 2
    assert trainer._get_num_items_in_batch([batch], torch.device("cpu")) == 6
    outputs = model(**batch)
    outputs.loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_resume_rejects_changed_padding_before_model_construction(tmp_path, monkeypatch):
    train, validation = sft._training_texts()
    record = sft._data_identity(
        train, validation, sft.tokenizer(), max_length=64, pad_to_multiple_of=8
    )
    assert record["version"] == 3
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "onnax_data.json").write_text(json.dumps(record))

    def no_construction(*args, **kwargs):
        raise AssertionError("Validate input policy before constructing a model")

    monkeypatch.setattr(sft, "prepare", no_construction)
    with pytest.raises(ValueError, match="Checkpoint training data"):
        sft.run(
            str(tmp_path / "resumed"), steps=2, resume=str(checkpoint), pad_to_multiple_of=16
        )


def test_capacity_phase_observer_keeps_training_values():
    from examples.huggingface.benchmark import measure

    events = []
    options = dict(samples=1, warmup=0, batch_size=1, sequence_length=8, checkpoint_layers=1)
    baseline = measure("chunked", **options)
    traced = measure(
        "chunked",
        **options,
        phase_observer=lambda phase, **details: events.append((phase, details)),
        validate_updates=True,
    )
    assert traced["losses"] == baseline["losses"]
    assert traced["validated_updates"] == 1 and traced["parameter_tensors"] == 8
    names = [name for name, _ in events]
    assert names.index("source_load_complete") < names.index("execution_begin")
    assert names.index("construction_complete") < names.index("step_begin")
    assert names[-1] == "step_complete"
    assert names.count("session_run_begin") == names.count("session_run_complete")


def test_pytorch_benchmark_executes_requested_half_precision():
    from examples.huggingface.benchmark import measure

    options = dict(nf4=True, samples=1, warmup=0, batch_size=1, sequence_length=8)
    bf16 = measure("pytorch", **options, projection_precision="bf16")
    fp16 = measure("pytorch", **options, projection_precision="fp16")
    assert fp16["losses"] != bf16["losses"]
    assert fp16["gradient_and_optimizer_dtype"] == "float32"
