# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Train actual text contexts and reject incompatible checkpoint data early."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional example dependencies.

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("mobius")
pytest.importorskip("trl")
torch = pytest.importorskip("torch")

from examples.huggingface.decoder_sft import run, task_tokenizer
from examples.huggingface.replay import verify
from examples.huggingface.sft import _data_identity


@pytest.fixture
def corpus(tmp_path):
    train = [
        ("ONNX models learn from examples. " * 12).strip(),
        ("Adapters update a small model. " * 12).strip(),
    ]
    validation = [("ONNX models learn from examples. " * 10).strip()]
    paths = [tmp_path / "train.jsonl", tmp_path / "validation.jsonl"]
    for path, rows in zip(paths, (train, validation), strict=True):
        path.write_text("\n".join(json.dumps({"text": row}) for row in rows) + "\n")
    return paths, train, validation


@pytest.mark.parametrize("family", ["phi3", "qwen3", "qwen35"])
def test_custom_context_training_and_checkpoint_recovery(tmp_path, corpus, family):
    torch.set_num_threads(1)
    paths, _, _ = corpus
    options = dict(
        family=family,
        steps=4,
        head_chunk_size=3,
        max_length=32,
        pad_to_multiple_of=8,
        train_file=paths[0],
        validation_file=paths[1],
    )
    full = tmp_path / "full"
    report = run(full, **options)
    batches = [row for row in report["batches"] if row["training"]]
    assert batches and all(row["sequence"] == 32 for row in batches)
    assert all(row["valid_shifted_targets"] == row["batch"] * 31 for row in batches)
    assert report["data"] == json.loads((full / "checkpoint-4/onnax_data.json").read_text())
    if family != "phi3":
        return
    partial, resumed = tmp_path / "partial", tmp_path / "resumed"
    run(partial, stop=2, **options)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.huggingface.decoder_sft",
            "--family",
            family,
            "--steps",
            "4",
            "--head-chunk-size",
            "3",
            "--max-length",
            "32",
            "--pad-to-multiple-of",
            "8",
            "--train-file",
            str(paths[0]),
            "--validation-file",
            str(paths[1]),
            "--resume",
            str(partial / "checkpoint-2"),
            "--output-dir",
            str(resumed),
        ],
        check=True,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
    )
    replay = verify(full, resumed, step=4)
    assert replay["checkpoint_bitwise_equal"] and replay["optimizer_bitwise_equal"]
    assert replay["standalone_outputs_bitwise_equal"]


@pytest.mark.parametrize(
    "change", ["train", "validation", "tokenizer", "context", "padding", "missing"]
)
def test_resume_identity_rejected_before_graph_build(tmp_path, corpus, monkeypatch, change):
    paths, train, validation = corpus
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "model_type": "phi3",
                "onnax_head": {"identity": "unused"},
            }
        )
    )
    tok = task_tokenizer("phi3", training_texts=train)
    identity = _data_identity(train, validation, tok, max_length=32, pad_to_multiple_of=8)
    if change == "tokenizer":
        identity["tokenizer_sha256"] = "different"
    if change != "missing":
        (checkpoint / "onnax_data.json").write_text(json.dumps(identity))
    if change in ("train", "validation"):
        path = paths[0 if change == "train" else 1]
        path.write_text(json.dumps({"text": "Changed training example."}) + "\n")

    def unexpected_build(*args, **kwargs):
        pytest.fail("Incompatible data reached graph construction")

    monkeypatch.setattr("examples.huggingface.decoder_sft.prepare", unexpected_build)
    with pytest.raises(ValueError, match=r"differs|identity metadata"):
        run(
            tmp_path / "output",
            family="phi3",
            steps=2,
            resume=checkpoint,
            head_chunk_size=3,
            max_length=16 if change == "context" else 32,
            pad_to_multiple_of=4 if change == "padding" else 8,
            train_file=paths[0],
            validation_file=paths[1],
        )


@pytest.mark.parametrize(
    "options",
    [
        {"max_length": 1},
        {"max_length": True},
        {"max_length": 65},
        {"max_length": 32, "pad_to_multiple_of": 3},
        {"train_file": "missing.jsonl"},
    ],
)
def test_invalid_context_rejected_before_graph_build(tmp_path, monkeypatch, options):
    def unexpected_build(*args, **kwargs):
        pytest.fail("Invalid context reached graph construction")

    monkeypatch.setattr("examples.huggingface.decoder_sft.prepare", unexpected_build)
    with pytest.raises(ValueError):
        run(tmp_path / "output", **options)
