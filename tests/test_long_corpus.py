# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Offline long-text selection and document-isolation checks."""

import pytest

from examples.huggingface.long_corpus import QUOTAS, _collect, _render, _select


class Tokenizer:
    eos_token = "<eos>"

    def __call__(self, texts, **kwargs):
        return {"input_ids": [text.split() for text in texts]}


def test_complete_text_and_transitive_document_groups():
    rows = [
        {
            "instruction": "Context\nThe material ends. Question one?",
            "output": "Answer",
            "file": "a",
        },
        {
            "instruction": "Context\nThe material ends. Question two?",
            "output": "Answer",
            "file": "b",
        },
        {"instruction": "Other question", "output": "Answer", "file": "b"},
        {"instruction": "Unrelated", "output": "Answer", "file": "c"},
    ]
    candidates, counts = _collect([*rows, rows[0], {"instruction": ""}], Tokenizer())
    assert counts["invalid"] == 1
    assert counts["duplicate_text"] == 1
    assert len({r["document_sha256"] for r in candidates[:3]}) == 1
    assert candidates[3]["document_sha256"] != candidates[0]["document_sha256"]
    assert candidates[0]["text"] == _render(rows[0]) + "<eos>"
    assert "Context\nThe material" in candidates[0]["text"]


def test_selection_is_deterministic_and_document_disjoint():
    candidates = [
        {"document_sha256": f"doc-{i}", "text_sha256": f"text-{bucket}-{i}", "bucket": bucket}
        for bucket in ("short", "long")
        for i in range(160)
    ]
    result = _select(candidates)
    assert result == _select(list(reversed(candidates)))
    for split, quotas in QUOTAS.items():
        for bucket, count in quotas.items():
            assert sum(row["bucket"] == bucket for row in result[split]) == count
    assert not (
        {r["document_sha256"] for r in result["train"]}
        & {r["document_sha256"] for r in result["validation"]}
    )
    with pytest.raises(ValueError, match="document-separated"):
        _select(candidates[:2])


def test_8k_selection_keeps_complete_long_examples():
    rows = [
        {"instruction": "word " * count, "output": "Answer"}
        for count in (10, 3500, 7000, 8500)
    ]
    old, _ = _collect(rows, Tokenizer())
    extended, _ = _collect(rows, Tokenizer(), max_length=8192)
    assert [row["source_index"] for row in old] == [0, 1]
    assert [row["source_index"] for row in extended] == [0, 2]
    assert extended[1]["bucket"] == "long"
    assert extended[1]["tokens"] == 7003
    assert extended[1]["text"] == _render(rows[2]) + "<eos>"
    assert old[0] == extended[0]


@pytest.mark.parametrize("length", [-1, True, 4096.0, 2048, 16384])
def test_unsupported_corpus_context(length):
    with pytest.raises(ValueError, match="max_length"):
        _collect([], Tokenizer(), max_length=length)


def test_qualification_rejects_shorter_corpus_for_8k():
    from examples.huggingface.long_context import _validate_corpus_context

    _validate_corpus_context({"max_length": 8192}, 8192)
    with pytest.raises(ValueError, match="corpus manifest"):
        _validate_corpus_context({"max_length": 4096}, 8192)
