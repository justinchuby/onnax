# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Compare quality reports and FP32 snapshots from isolated runtime processes."""

import argparse
import hashlib
import itertools
import json

import numpy as np

from examples.huggingface.training_impact import _vector_metrics

MATCHED = (
    "model",
    "revision",
    "nf4",
    "projection_precision",
    "base_seed",
    "steps",
    "microbatch",
    "accumulation",
    "evaluation_interval",
    "optimizer",
    "precision_policy",
    "reference_head",
    "source_base_sha256",
    "corpus_sha256",
    "train_lines",
    "heldout_lines",
    "gradient_scope",
)


def _read(path):
    with open(path) as stream:
        return json.load(stream)


def _sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _metrics(a, b):
    if a.dtype != np.float32 or b.dtype != np.float32 or a.ndim != 1 or a.shape != b.shape:
        raise ValueError("Snapshots must be matching FP32 vectors")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite snapshot")
    return _vector_metrics(a, b)


def compare(reports, captures, *, executions=None):
    if len(reports) < 2 or len(reports) != len(captures):
        raise ValueError("Provide at least two report/capture pairs")
    components = [_read(path) for path in reports]
    baseline = components[0]
    seeds = sorted(baseline["seeds"])
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate trial seeds")
    engines, trials = [], {}
    for component, capture in zip(components, captures, strict=True):
        if len(component["engines"]) != 1:
            raise ValueError("Each isolated report must contain one engine")
        engine = component["engines"][0]
        if engine in engines:
            raise ValueError("Duplicate engine report")
        engines.append(engine)
        for field in MATCHED:
            if field not in component or field not in baseline:
                raise ValueError(f"Missing experiment setting: {field}")
            if component[field] != baseline[field]:
                raise ValueError(f"Experiment settings differ: {field}")
        if sorted(component["seeds"]) != seeds or not component["frozen_unchanged"]:
            raise ValueError("Trial seeds or frozen-base checks differ")
        metadata = component.get("capture")
        if (
            metadata is None
            or metadata.get("version") != 1
            or metadata["sha256"] != _sha(capture)
        ):
            raise ValueError("Capture checksum or format mismatch")
        if metadata["parameter_order"] != baseline["capture"]["parameter_order"]:
            raise ValueError("Snapshot parameter order differs")
        rows = component["trials"]
        if (
            len(rows) != len(seeds)
            or {row["seed"] for row in rows} != set(seeds)
            or any(row["engine"] != engine for row in rows)
        ):
            raise ValueError("Expected one trial per engine and seed")
        trials.update({(engine, row["seed"]): row for row in rows})
    comparisons = []
    for i, j in itertools.combinations(range(len(engines)), 2):
        left, right = engines[i], engines[j]
        with (
            np.load(captures[i], allow_pickle=False) as ca,
            np.load(captures[j], allow_pickle=False) as cb,
        ):
            for seed in seeds:
                a, b = trials[left, seed], trials[right, seed]
                for field in ("initial_adapter_sha256", "schedule_sha256"):
                    if a[field] != b[field]:
                        raise ValueError(f"Paired trials differ: {field}")
                if [row["step"] for row in a["validation_curve"]] != [
                    row["step"] for row in b["validation_curve"]
                ]:
                    raise ValueError("Validation checkpoints differ")
                pa, pb = f"{left}__{seed}__", f"{right}__{seed}__"
                sa = sorted(
                    int(name.removeprefix(pa + "gradient_"))
                    for name in ca.files
                    if name.startswith(pa + "gradient_")
                )
                sb = sorted(
                    int(name.removeprefix(pb + "gradient_"))
                    for name in cb.files
                    if name.startswith(pb + "gradient_")
                )
                if not sa or sa != sb:
                    raise ValueError("Gradient checkpoints differ")
                comparisons.append(
                    {
                        "left": left,
                        "right": right,
                        "seed": seed,
                        "final_validation_loss_gap": a["final_validation_loss"]
                        - b["final_validation_loss"],
                        "validation_curve": [
                            {
                                "step": x["step"],
                                "left": x["loss"],
                                "right": y["loss"],
                                "gap": x["loss"] - y["loss"],
                            }
                            for x, y in zip(
                                a["validation_curve"], b["validation_curve"], strict=True
                            )
                        ],
                        "parameters": _metrics(ca[pa + "parameters"], cb[pb + "parameters"]),
                        "gradient_snapshots": [
                            {
                                "step": step,
                                **_metrics(
                                    ca[pa + f"gradient_{step}"], cb[pb + f"gradient_{step}"]
                                ),
                            }
                            for step in sa
                        ],
                    }
                )
    summaries = {}
    for engine in engines:
        values = [trials[engine, seed]["final_validation_loss"] for seed in seeds]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite validation loss")
        summaries[engine] = {
            "values": values,
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        }
    journals = None
    if executions is not None:
        if len(executions) != len(components):
            raise ValueError("Provide one execution journal per report")
        journals = [_read(path) for path in executions]
        if any(
            row.get("returncode") != 0 or row.get("guard_triggered", False) for row in journals
        ):
            raise ValueError("A component run did not finish successfully")
        if any(
            row.get("engine") != engine for row, engine in zip(journals, engines, strict=True)
        ):
            raise ValueError("Execution journals refer to different engines")
        if len({row["source_commit"] for row in journals}) != 1:
            raise ValueError("Component source commits differ")
    return {
        "settings": {field: baseline.get(field) for field in MATCHED},
        "seeds": seeds,
        "engines": engines,
        "cuda_executed": "torch_cuda" in engines,
        "validation_summary": summaries,
        "comparisons": comparisons,
        "components": components,
        "executions": journals,
        "execution_scope": "Quality runs including construction, training, evaluation and host snapshots",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--captures", nargs="+", required=True)
    parser.add_argument("--executions", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare(args.reports, args.captures, executions=args.executions)
    with open(args.output, "w") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
