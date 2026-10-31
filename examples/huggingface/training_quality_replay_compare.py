# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Compare losses and gradients from shared-adapter replay processes."""

import argparse
import json

import numpy as np

from examples.huggingface.training_quality_compare import _metrics, _read, _sha


def compare_replays(reports, captures):
    if len(reports) != 2 or len(captures) != 2:
        raise ValueError("Provide two replay report/capture pairs")
    left, right = [_read(path) for path in reports]
    if left["engine"] == right["engine"]:
        raise ValueError("Select distinct replay engines")
    for field in ("settings", "source_captures", "gradient_scope"):
        if left[field] != right[field]:
            raise ValueError(f"Replay experiments differ: {field}")
    tables = []
    for report, capture in zip((left, right), captures, strict=True):
        if not report["frozen_unchanged"] or report["capture"]["sha256"] != _sha(capture):
            raise ValueError("Replay frozen-base or capture checksum mismatch")
        table = {(row["source_engine"], row["seed"]): row for row in report["states"]}
        if not table or len(table) != len(report["states"]):
            raise ValueError("Expected unique nonempty replay states")
        tables.append(table)
    if tables[0].keys() != tables[1].keys():
        raise ValueError("Replay states differ")
    if left["capture"]["parameter_order"] != right["capture"]["parameter_order"]:
        raise ValueError("Replay parameter order differs")
    comparisons = []
    with (
        np.load(captures[0], allow_pickle=False) as a,
        np.load(captures[1], allow_pickle=False) as b,
    ):
        for key in sorted(tables[0]):
            x, y = tables[0][key], tables[1][key]
            if x["parameter_sha256"] != y["parameter_sha256"]:
                raise ValueError("Replays evaluated different parameter states")
            values = [
                row[field]
                for row in (x, y)
                for field in ("validation_loss", "gradient_batch_loss")
            ]
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite replay loss")
            comparisons.append(
                {
                    "source_engine": key[0],
                    "seed": key[1],
                    "parameter_sha256": x["parameter_sha256"],
                    "validation_loss_gap": x["validation_loss"] - y["validation_loss"],
                    "gradient_batch_loss_gap": x["gradient_batch_loss"]
                    - y["gradient_batch_loss"],
                    "gradient": _metrics(a[x["gradient_key"]], b[y["gradient_key"]]),
                }
            )
    return {
        "engines": [left["engine"], right["engine"]],
        "comparisons": comparisons,
        "components": [left, right],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs=2, required=True)
    parser.add_argument("--captures", nargs=2, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare_replays(args.reports, args.captures)
    with open(args.output, "w") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
