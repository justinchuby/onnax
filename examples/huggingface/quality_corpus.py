# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Prepare a pinned No Robots corpus for matched Qwen training experiments."""

import argparse
import hashlib
import importlib.metadata
import json
import os

from datasets import load_dataset

from examples.huggingface.qwen import MODEL_ID, REVISION
from examples.huggingface.sft import tokenizer
from examples.huggingface.training_impact import _tokenizer_digest

DATASET = "HuggingFaceH4/no_robots"
DATASET_REVISION = "e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b"
SEED = 20261029


def _select(records, count, tok, *, excluded=()):
    seen_ids = {row["id"] for row in excluded}
    seen_texts = {row["text"] for row in excluded}
    selected = []
    for row in records:
        messages = row["messages"]
        if [message["role"] for message in messages] != ["user", "assistant"]:
            continue
        user, answer = [" ".join(message["content"].split()) for message in messages]
        if not user or not answer:
            continue
        prefix = f"User: {user} Assistant:"
        prefix_tokens = len(tok(prefix)["input_ids"])
        if prefix_tokens > 32:
            continue
        text = f"{prefix} {answer}"
        identity = row["prompt_id"]
        if identity in seen_ids or text in seen_texts:
            continue
        tokens = min(64, len(tok(text)["input_ids"]))
        selected.append(
            {
                "id": identity,
                "text": text,
                "prefix_tokens": prefix_tokens,
                "retained_tokens": tokens,
            }
        )
        seen_ids.add(identity)
        seen_texts.add(text)
        if len(selected) == count:
            return selected
    raise ValueError(f"Only {len(selected)} eligible unique sequences; requested {count}")


def prepare(output, manifest, *, train_count=128, heldout_count=32):
    if os.path.abspath(output) == os.path.abspath(manifest):
        raise ValueError("Corpus and manifest paths must differ")
    if min(train_count, heldout_count) < 1:
        raise ValueError("Select positive training and held-out counts")
    if os.path.exists(output) or os.path.exists(manifest):
        raise FileExistsError("Select new corpus and manifest paths")
    data = load_dataset(DATASET, revision=DATASET_REVISION)
    tok = tokenizer(real=True)
    train = _select(data["train"].shuffle(seed=SEED), train_count, tok)
    heldout = _select(data["test"].shuffle(seed=SEED), heldout_count, tok, excluded=train)
    rows = train + heldout
    text = "\n".join(row["text"] for row in rows)
    report = {
        "version": 1,
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "dataset_url": f"https://huggingface.co/datasets/{DATASET}/tree/{DATASET_REVISION}",
        "license": "cc-by-nc-4.0",
        "attribution": "No Robots, HuggingFaceH4, 2023",
        "tokenizer": MODEL_ID,
        "tokenizer_revision": REVISION,
        "tokenizer_sha256": _tokenizer_digest(tok),
        "shuffle_seed": SEED,
        "train_lines": train_count,
        "heldout_lines": heldout_count,
        "source_splits": ["train", "test"],
        "selection": "Unique single-turn user/assistant pairs; whitespace normalized; prefix <=32 tokens",
        "training_sequence_limit": 64,
        "loss": "All shifted nonpadding tokens, including user and assistant text",
        "corpus_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "samples": [
            {
                **{key: value for key, value in row.items() if key != "text"},
                "split": "train" if index < train_count else "test",
            }
            for index, row in enumerate(rows)
        ],
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("datasets", "transformers", "tokenizers", "huggingface-hub")
        },
    }
    with open(output, "w", encoding="utf-8") as stream:
        stream.write(text + "\n")
    with open(manifest, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-count", type=int, default=128)
    parser.add_argument("--heldout-count", type=int, default=32)
    args = parser.parse_args()
    report = prepare(
        args.output,
        args.manifest,
        train_count=args.train_count,
        heldout_count=args.heldout_count,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "dataset_revision",
                    "corpus_sha256",
                    "train_lines",
                    "heldout_lines",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
