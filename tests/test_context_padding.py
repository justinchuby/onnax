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
