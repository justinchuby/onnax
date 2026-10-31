# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Offline checks for the external training-corpus selection policy."""

import pytest

pytest.importorskip("mobius")
pytest.importorskip("trl")

from examples.huggingface.quality_corpus import _select


def test_selection_preserves_pairs_and_disjoint_samples():
    def row(identity, user, answer="The answer is four."):
        return {
            "prompt_id": identity,
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": answer},
            ],
        }

    def tok(text):
        return {"input_ids": text.split()}

    records = [
        row("long", "word " * 40),
        row("a", "What\n is two plus two?"),
        row("a", "Repeated ID"),
        row("b", "Name a color.", "Amber."),
    ]
    selected = _select(records, 2, tok)
    assert [r["id"] for r in selected] == ["a", "b"]
    assert selected[0]["text"] == "User: What is two plus two? Assistant: The answer is four."
    assert "\n" not in selected[0]["text"]
    test = [
        row("a", "Repeated ID"),
        row("c", "Name a color.", "Amber."),
        row("d", "Name a tree.", "Oak."),
    ]
    assert _select(test, 1, tok, excluded=selected)[0]["id"] == "d"
    with pytest.raises(ValueError, match="eligible unique sequences"):
        _select(records, 3, tok)
