# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Summarize completed long-context training and isolated measurement records."""

import argparse
import hashlib
import json
import math
import os


def _read(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def _sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _training(directory, precision, *, resumed=False):
    report = _read(os.path.join(directory, "report.json"))
    journal = _read(os.path.join(directory, "qualification.json"))
    _require(journal["status"] == "complete", "Training worker has not completed")
    _require(
        report["model"] == "Qwen/Qwen3-0.6B"
        and report["revision"] == "c1899de289a04d12100db370d81485cdf75e47ca",
        "Training base or revision differs",
    )
    _require(
        journal["physical_footprint"]["lifetime_peak_bytes"] <= 24 * 1024**3,
        "Training exceeded its physical-memory budget",
    )
    _require(report["steps"] == 30, "Expected 30 completed optimizer updates")
    _require(report["projection_precision"] == precision, "Training precision differs")
    _require(report["data"]["max_length"] == 4096, "Training context differs")
    _require(report["frozen_buffers_unchanged"], "Frozen weight check failed")
    _require(report["checkpoint_layers"] == 1, "Expected one decoder layer per group")
    _require(report["mode"] == "chunked", "Expected chunked output-head execution")
    _require(report["ort_graph_optimization"] == "disabled", "ORT optimizer was enabled")
    _require(report["mlx_compile_disabled"], "MLX compilation was enabled")
    steps = [row["step"] for row in journal["updates"]]
    _require(steps == list(range(16 if resumed else 1, 31)), "Update journal is incomplete")
    train, validation = report["fixed_training_loss"], report["validation_loss"]
    _require(
        len(train) == len(validation) == 2
        and all(math.isfinite(value) for value in [*train, *validation]),
        "Loss endpoints must be finite",
    )
    _require(train[1] < train[0], "Fixed training loss did not decrease")
    return {
        "training_loss": train,
        "validation_loss": validation,
        "steps": report["steps"],
        "data": report["data"],
        "model": report["model"],
        "revision": report["revision"],
        "source_commit": journal["source_commit"],
        "library_sha256": journal["library_sha256"],
        "versions": journal["versions"],
        "physical_footprint": journal["physical_footprint"],
        "frozen_buffers_unchanged": True,
        "report_sha256": _sha(os.path.join(directory, "report.json")),
        "journal_sha256": _sha(os.path.join(directory, "qualification.json")),
    }


def collect(suite):
    queue = _read(os.path.join(suite, "queue.json"))
    _require(queue["status"] == "complete", "Qualification queue is still incomplete")
    cases = {row["name"]: row for row in queue["cases"]}
    _require(len(cases) == len(queue["cases"]), "Duplicate qualification cases")
    for name in (
        "fp32-resumed",
        "fp32-recovery",
        "fp32-peft",
        "bf16-training",
        "bf16-resumed",
        "bf16-recovery",
        "bf16-peft",
    ):
        _require(
            name in cases and cases[name]["status"] == "complete",
            f"Required qualification case is incomplete: {name}",
        )
    first = queue["existing_fp32_training"]["directory"]
    training = {}
    for precision in ("fp32", "bf16"):
        directory = first if precision == "fp32" else os.path.join(suite, "bf16-training")
        baseline = _training(directory, precision)
        resumed = _training(
            os.path.join(suite, precision + "-resumed"), precision, resumed=True
        )
        _require(baseline["data"] == resumed["data"], "Recovery data identity differs")
        _require(
            baseline["library_sha256"] == resumed["library_sha256"], "Recovery library differs"
        )
        recovery = _read(os.path.join(suite, precision + "-recovery.json"))
        for field in (
            "adapter_bitwise_equal",
            "checkpoint_bitwise_equal",
            "optimizer_bitwise_equal",
            "scheduler_equal",
            "frozen_hashes_equal",
            "export_load_outputs_bitwise_equal",
            "export_load_loss_gradients_bitwise_equal",
        ):
            _require(recovery[field], f"Recovery failed: {field}")
        _require(
            recovery["input_shape"] == [1, 4096] and recovery["padding_tokens"] == 7,
            "Expected a padded 4096-token replay",
        )
        _require(recovery["exported_gradient_tensors"] == 112, "Missing adapter gradients")
        peft = _read(os.path.join(suite, precision + "-peft", "report.json"))
        for field in (
            "format_roundtrip_bitwise",
            "trained_checkpoint_restored_bitwise",
            "restored_loss_and_gradients_bitwise_equal",
        ):
            _require(peft[field], f"PEFT exchange failed: {field}")
        _require(peft["adapter_tensors"] == 112, "PEFT adapter mapping is incomplete")
        training[precision] = {
            "baseline": baseline,
            "resumed": resumed,
            "recovery": recovery,
            "peft": peft,
        }
    _require(
        training["fp32"]["baseline"]["data"] == training["bf16"]["baseline"]["data"],
        "Precision runs used different training data",
    )
    measurements, capacity = [], []
    tokens = {}
    for case in queue["cases"]:
        if "-capacity" in case["name"] or case["status"] != "complete":
            _require(
                case["status"] == "complete" or case.get("memory_capacity_failure"),
                "Unresolved execution failure",
            )
            capacity.append(case)
            continue
        if not any(f"-{length}-" in case["name"] for length in (512, 1024, 4096, 8192)):
            continue
        path = os.path.join(suite, case["name"] + ".json")
        report = _read(path)
        _require(
            report["warmup"] == 10 and report["samples"] == 100,
            "Expected 10 warmups and 100 measured samples",
        )
        _require(
            len(report["losses"]) == 110 and all(map(math.isfinite, report["losses"])),
            "Incomplete or nonfinite benchmark updates",
        )
        _require(
            report["batch"] == 1 and report["head_chunk_size"] == 128,
            "Benchmark batch or head chunk differs",
        )
        _require(
            report["ort_optimizer"] == "disabled" and report["mlx_compile_disabled"],
            "Benchmark runtime policy differs",
        )
        _require(
            report["cpu_fallback_disabled"]
            and report["mlx_memory_limit_bytes"] == 20 * 1024**3
            and report["base_storage"] == "packed_nf4_decoder"
            and report["gradient_and_optimizer_dtype"] == "float32",
            "Benchmark memory, provider or parameter policy differs",
        )
        _require(
            report["library_sha256"] == queue["library_sha256"], "Measurement library differs"
        )
        _require(
            report["physical_footprint"]["lifetime_peak_bytes"] <= 24 * 1024**3,
            "Measurement exceeded its physical-memory budget",
        )
        for values in report["timings_ms"].values():
            _require(
                0 <= values["median"] <= values["p95"] and math.isfinite(values["p95"]),
                "Invalid timing statistics",
            )
        length = report["sequence"]
        _require(report["valid_shifted_tokens"] == length - 1, "Benchmark token count differs")
        key = length
        identity = report["input_ids_sha256"]
        _require(tokens.setdefault(key, identity) == identity, "Benchmark input tokens differ")
        measurements.append(
            {
                "case": case["name"],
                "sequence": length,
                "precision": report["projection_precision"],
                "checkpoint_layers": report["checkpoint_layers"],
                "save_activations": report["save_activations"],
                "warmup": report["warmup"],
                "samples": report["samples"],
                "losses": report["losses"],
                "microbatch": report["batch"],
                "gradient_accumulation": 1,
                "valid_shifted_tokens": report["valid_shifted_tokens"],
                "input_ids_sha256": identity,
                "timings_ms": report["timings_ms"],
                "tokens_per_second": report["effective_tokens_per_second"],
                "peak_rss_bytes": report["peak_rss_bytes"],
                "physical_footprint": report["physical_footprint"],
                "guard_peak_physical_bytes": case["physical_footprint_peak_bytes"],
                "mlx_allocator": report["mlx_allocator"],
                "logical_saved_activation_bytes": report["logical_saved_activation_bytes"],
                "saved_activation_count": report["saved_activation_count"],
                "initialization_seconds": report["initialization_seconds"],
                "initialization_scope": report["initialization_scope"],
                "native_construction": report["native_construction"],
                "input_view_median_us": report["input_view_median_us"],
                "input_view_scope": report["input_view_scope"],
                "exchange_scope": report["exchange_scope"],
                "nodes": report["nodes"],
                "parameter_bytes": report["parameter_bytes"],
                "parameter_bytes_scope": "FP32 trainable LoRA tensors",
                "initializer_bytes_by_session": report["initializer_bytes_by_session"],
                "initializer_scope": report["initializer_scope"],
                "decoder_storage_bytes": report["decoder_storage_bytes"],
                "base_storage": report["base_storage"],
                "gradient_and_optimizer_dtype": report["gradient_and_optimizer_dtype"],
                "head_chunk_size": report["head_chunk_size"],
                "head_forward_session_created": report["head_forward_session_created"],
                "full_logits_bytes": report["full_logits_bytes"],
                "chunk_logits_bytes": report["chunk_logits_bytes"],
                "threads": report["threads"],
                "thread_environment": report["thread_environment"],
                "cpu_fallback_disabled": report["cpu_fallback_disabled"],
                "mlx_memory_limit_bytes": report["mlx_memory_limit_bytes"],
                "hardware": report["hardware"],
                "library_sha256": report["library_sha256"],
                "versions": report["versions"],
                "report_sha256": _sha(path),
            }
        )
    names = {row["case"] for row in measurements}
    for precision in ("fp32", "bf16"):
        for length in (512, 1024, 4096):
            _require(
                f"{precision}-{length}-group1" in names,
                "Required one-layer-group measurement is missing",
            )
    for precision in ("fp32", "bf16"):
        for length in (512, 1024, 4096):
            for strategy in ("recompute", "saved", "group1", "group2", "group4"):
                name = f"{precision}-{length}-{strategy}"
                if name not in names:
                    limited = cases.get(name + "-capacity", cases.get(name, {}))
                    _require(
                        limited.get("status") == "failed"
                        and limited.get("memory_capacity_failure"),
                        f"Missing strategy measurement or capacity finding: {name}",
                    )
        name = precision + "-8192-group1"
        probe = cases.get(name + "-capacity", {})
        _require(
            probe.get("status") == "complete" or probe.get("memory_capacity_failure"),
            f"Missing 8192-token capacity probe: {precision}",
        )
        if probe.get("status") == "complete":
            _require(name in names, "Successful 8192-token probe requires measurement")
    comparisons = []
    for precision in ("fp32", "bf16"):
        for length in (512, 1024, 4096, 8192):
            group = [
                row
                for row in measurements
                if row["precision"] == precision and row["sequence"] == length
            ]
            if not group:
                continue
            reference = next(
                (row for row in group if row["case"].endswith("-recompute")), group[0]
            )
            for row in group:
                if row is reference:
                    continue
                pairs = list(zip(reference["losses"], row["losses"], strict=True))
                comparisons.append(
                    {
                        "reference_case": reference["case"],
                        "compared_case": row["case"],
                        "updates": len(pairs),
                        "loss_values_equal": all(a == b for a, b in pairs),
                        "max_absolute_loss_gap": max(abs(a - b) for a, b in pairs),
                        "rtol": 1e-4,
                        "atol": 1e-6,
                        "within_tolerance": all(
                            abs(a - b) <= 1e-6 + 1e-4 * abs(a) for a, b in pairs
                        ),
                    }
                )
    return {
        "version": 1,
        "training_policy": {
            "microbatch": 1,
            "gradient_accumulation": 4,
            "steps": 30,
            "learning_rate": 0.003,
            "head_chunk_tokens": 128,
        },
        "training_loss_reduction": "Valid-token mean across accumulated microbatches",
        "evaluation_loss_reduction": "Mean of per-example valid-token means",
        "benchmark_step": "One microbatch forward/backward and one AdamW update",
        "source_commit": queue["source_commit"],
        "queue_sha256": _sha(os.path.join(suite, "queue.json")),
        "training": training,
        "measurements": measurements,
        "strategy_loss_comparisons": comparisons,
        "capacity": capacity,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = collect(args.suite)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
