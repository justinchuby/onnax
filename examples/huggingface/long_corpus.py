# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Prepare complete short and 4k LongAlpaca QA with document-group holdout."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path  # ruff: ignore[banned-api] - CLI artifact management.

DATASET = "Yukang/LongAlpaca-12k"
REVISION = "46dce924ed8786979556018e191c0f557d8f4aa2"
SEED = 20261029
QUOTAS = {"train": {"short": 96, "long": 32}, "validation": {"short": 24, "long": 8}}


def _digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _render(row):
    if not isinstance(row, dict):
        return None
    instruction, answer = row.get("instruction"), row.get("output")
    if not isinstance(instruction, str) or not isinstance(answer, str):
        return None
    if not instruction.strip() or not answer.strip():
        return None
    extra = row.get("input")
    if extra is not None and not isinstance(extra, str):
        return None
    prompt = instruction + ("\n" + extra if extra else "")
    return f"User: {prompt}\nAssistant: {answer}"


def _document_keys(row):
    prompt = row["instruction"] + ("\n" + row["input"] if row.get("input") else "")
    ends = list(
        re.finditer(r"\b(?:now\s+)?the\s+(?:paper|material|book)\s+ends\.", prompt, re.I)
    )
    context = prompt[: ends[-1].start()] if ends else prompt
    keys = ["context:" + _digest(" ".join(context.split()).casefold())]
    if isinstance(row.get("file"), str) and row["file"].strip():
        keys.append("file:" + _digest(row["file"].strip().replace("\\", "/")))
    return keys


def _collect(rows, tokenizer):
    parents, candidates, seen = {}, [], set()
    counts = {
        "source_rows": len(rows),
        "invalid": 0,
        "duplicate_text": 0,
        "outside_buckets": 0,
    }

    def root(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    for start in range(0, len(rows), 128):
        batch = []
        for index, row in enumerate(rows[start : start + 128], start):
            text = _render(row)
            if text is None:
                counts["invalid"] += 1
                continue
            if not text.endswith(tokenizer.eos_token):
                text += tokenizer.eos_token
            keys = _document_keys(row)
            groups = sorted({root(key) for key in keys})
            for group in groups:
                parents[group] = groups[0]
            identity = _digest(text)
            if identity in seen:
                counts["duplicate_text"] += 1
                continue
            seen.add(identity)
            batch.append((index, text, identity, keys[0]))
        if not batch:
            continue
        encoded = tokenizer(
            [item[1] for item in batch],
            add_special_tokens=True,
            return_attention_mask=False,
            verbose=False,
        )["input_ids"]
        for (index, text, identity, key), ids in zip(batch, encoded, strict=True):
            length = len(ids)
            bucket = (
                "short" if 2 <= length <= 1024 else "long" if 3072 <= length <= 4096 else None
            )
            if bucket is None:
                counts["outside_buckets"] += 1
                continue
            candidates.append(
                {
                    "source_index": index,
                    "text": text,
                    "text_sha256": identity,
                    "tokens": length,
                    "bucket": bucket,
                    "group_key": key,
                }
            )
    for row in candidates:
        row["document_sha256"] = _digest(root(row.pop("group_key")))
    counts["eligible_short"] = sum(row["bucket"] == "short" for row in candidates)
    counts["eligible_long"] = sum(row["bucket"] == "long" for row in candidates)
    return candidates, counts


def _select(candidates):
    pools = {
        bucket: [row for row in candidates if row["bucket"] == bucket]
        for bucket in ("short", "long")
    }
    documents = {row["document_sha256"] for row in candidates}
    counts = {
        bucket: Counter(row["document_sha256"] for row in pool)
        for bucket, pool in pools.items()
    }
    remaining = {bucket: len(pool) for bucket, pool in pools.items()}
    validation_counts = dict.fromkeys(pools, 0)
    heldout = set()

    def priority(key):
        return _digest(f"{SEED}:{key}")

    for bucket in ("long", "short"):
        groups = sorted(
            (key for key in documents - heldout if counts[bucket][key]),
            key=lambda key: (counts[bucket][key], priority(key)),
        )
        for group in groups:
            if validation_counts[bucket] >= QUOTAS["validation"][bucket]:
                break
            if all(
                remaining[kind] - counts[kind][group] >= QUOTAS["train"][kind]
                for kind in pools
            ):
                heldout.add(group)
                for kind in pools:
                    remaining[kind] -= counts[kind][group]
                    validation_counts[kind] += counts[kind][group]
    result = {}
    for split in ("train", "validation"):
        selected = []
        for bucket in ("short", "long"):
            pool = sorted(
                (
                    row
                    for row in pools[bucket]
                    if (row["document_sha256"] in heldout) == (split == "validation")
                ),
                key=lambda row: priority(row["text_sha256"]),
            )
            count = QUOTAS[split][bucket]
            if len(pool) < count:
                raise ValueError(
                    f"Only {len(pool)} document-separated {split}/{bucket} samples; need {count}"
                )
            selected.extend(pool[:count])
        result[split] = sorted(selected, key=lambda row: priority(row["text_sha256"]))
    if {row["document_sha256"] for row in result["train"]} & {
        row["document_sha256"] for row in result["validation"]
    }:
        raise AssertionError("Training and validation document groups overlap")
    return result


def prepare(directory):
    from huggingface_hub import hf_hub_download

    from examples.huggingface.qwen import MODEL_ID
    from examples.huggingface.qwen import REVISION as MODEL_REVISION
    from examples.huggingface.sft import _tokenizer_digest, tokenizer

    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("Select a new corpus directory")
    source = hf_hub_download(
        DATASET, "LongAlpaca-12k.json", repo_type="dataset", revision=REVISION
    )
    tok = tokenizer(real=True)
    with open(source, encoding="utf-8") as stream:
        rows = json.load(stream)
    candidates, counts = _collect(rows, tok)
    selected = _select(candidates)
    with open(source, "rb") as stream:
        source_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    directory.mkdir(parents=True)
    files = {}
    for split, samples in selected.items():
        path = directory / (split + ".jsonl")
        path.write_text(
            "".join(
                json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n" for row in samples
            ),
            encoding="utf-8",
        )
        files[split] = {
            "filename": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(samples),
            "minimum_tokens": min(r["tokens"] for r in samples),
            "maximum_tokens": max(r["tokens"] for r in samples),
            "valid_shifted_tokens": sum(r["tokens"] - 1 for r in samples),
        }
    benchmark = None
    for index, row in enumerate(rows):
        text = _render(row)
        if text is None:
            continue
        if not text.endswith(tok.eos_token):
            text += tok.eos_token
        length = len(tok(text, verbose=False)["input_ids"])
        if length >= 8192:
            benchmark = {"source_index": index, "text": text, "tokens": length}
            break
    if benchmark is None:
        raise ValueError("No natural example reaches 8192 benchmark tokens")
    path = directory / "benchmark.jsonl"
    path.write_text(
        json.dumps({"text": benchmark["text"]}, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    benchmark_metadata = {
        "filename": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_index": benchmark["source_index"],
        "natural_tokens": benchmark["tokens"],
        "usage": "Fixed-length performance/capacity prefixes; quality training uses complete train/validation examples",
    }
    manifest = {
        "version": 1,
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "dataset_url": f"https://huggingface.co/datasets/{DATASET}/tree/{REVISION}",
        "source_sha256": source_sha256,
        "license": "cc-by-nc-4.0",
        "attribution": "LongAlpaca, Yukang Chen et al., 2023",
        "tokenizer": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "tokenizer_sha256": _tokenizer_digest(tok),
        "seed": SEED,
        "quotas": QUOTAS,
        "selection": "Complete QA; short 2-1024 tokens, long 3072-4096; shared context/source-file groups held out together; smallest holdout groups first with seeded ties",
        "format": "jsonl-text",
        "packing": False,
        "max_length": 4096,
        "eos_token": tok.eos_token,
        "loss": "All shifted nonpadding tokens",
        "source_counts": counts,
        "files": files,
        "benchmark": benchmark_metadata,
        "samples": {
            split: [{k: v for k, v in row.items() if k != "text"} for row in samples]
            for split, samples in selected.items()
        },
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = prepare(args.output_dir)
    print(
        json.dumps(
            {"dataset_revision": report["dataset_revision"], "files": report["files"]},
            indent=2,
        )
    )
