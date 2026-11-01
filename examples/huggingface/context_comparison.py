# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Compare isolated PyTorch 4k measurements with the qualified ONNX strategy."""

import argparse
import hashlib
import json
import math
import os


def _read(path):
    with open(path) as stream:
        return json.load(stream)


def collect(suite, onnx_report):
    queue = _read(os.path.join(suite, "queue.json"))
    if queue["status"] != "complete":
        raise ValueError("Wait for the complete measurement suite")
    onnx = _read(onnx_report)
    rows, findings, omitted = [], [], []
    cases = {case["name"]: case for case in queue["cases"]}
    requested = [
        f"{device}-{precision}-4096-group4"
        for device in ("cpu", "mps")
        for precision in ("fp32", "bf16")
    ]
    requested.extend(
        sorted({name.removesuffix("-capacity") for name in cases if "-sdpa" in name})
    )
    for name in requested:
        if name in queue.get("omitted_measurements", {}):
            item = {"case": name, "reason": queue["omitted_measurements"][name]}
            probe = cases.get(name + "-capacity")
            if probe is not None:
                item["capacity_case"] = probe
                if probe["status"] == "complete":
                    item["capacity_report"] = _read(
                        os.path.join(suite, name + "-capacity.json")
                    )
            if name in cases:
                item["interrupted_measurement"] = cases[name]
            omitted.append(item)
            continue
        device, precision = name.split("-")[:2]
        case = cases.get(name, cases.get(name + "-capacity"))
        if case is None:
            raise ValueError(f"Missing measurement: {name}")
        guard = _read(os.path.join(suite, case["name"] + ".guard", "execution.json"))
        if case["status"] != "complete":
            findings.append({"case": case, "guard": guard})
            continue
        report_path = os.path.join(suite, name + ".json")
        measured = _read(report_path)
        expected = next(
            row for row in onnx["measurements"] if row["case"] == precision + "-4096-group4"
        )
        for key in (
            "input_ids_sha256",
            "projection_precision",
            "batch",
            "sequence",
            "parameter_bytes",
        ):
            other = {"projection_precision": "precision", "batch": "microbatch"}.get(key, key)
            if measured[key] != expected[other]:
                raise ValueError(f"Unmatched {key}: {name}")
        for key, value in {
            "provider": "pytorch_" + device,
            "real": True,
            "weight_values": "nf4_reconstructed_fp32",
            "base_storage": "fp32",
            "checkpoint_layers": 4,
            "head_chunk_size": 128,
            "valid_shifted_tokens": 4095,
            "warmup": 10,
            "samples": 100,
            "gradient_and_optimizer_dtype": "float32",
            "torch_chunked_head": True,
            "mps_fallback_enabled": False,
        }.items():
            if measured[key] != value:
                raise ValueError(f"Unexpected {key}: {name}")
        if len(measured["losses"]) != 110 or not all(map(math.isfinite, measured["losses"])):
            raise ValueError(f"Incomplete or nonfinite updates: {name}")
        if measured["threads"]["torch"] != expected["threads"]["torch"]:
            raise ValueError(f"Different CPU thread count: {name}")
        attention = "sdpa" if name.endswith("-sdpa") else "eager"
        if measured["attention_implementation"] != attention:
            raise ValueError(f"Different attention path: {name}")
        if measured["hardware"] != expected["hardware"]:
            raise ValueError(f"Different measurement hardware: {name}")
        if measured["versions"] != expected["versions"]:
            raise ValueError(f"Different dependency versions: {name}")
        with open(report_path, "rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        rows.append(
            {
                "case": case,
                "pytorch": measured,
                "guard_peak_physical_bytes": case["physical_footprint_peak_bytes"],
                "report_sha256": digest,
                "onnx_case": expected["case"],
                "onnx_step_median_ms": expected["timings_ms"]["step"]["median"],
                "onnx_peak_physical_bytes": expected["physical_footprint"][
                    "lifetime_peak_bytes"
                ],
                "onnx_speedup_over_pytorch": measured["timings_ms"]["step"]["median"]
                / expected["timings_ms"]["step"]["median"],
            }
        )
    with open(onnx_report, "rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "version": 1,
        "source_commit": queue["source_commit"],
        "supplemental_source_commit": queue.get("supplemental_source_commit"),
        "model": "Qwen/Qwen3-0.6B",
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "onnx_report": os.path.basename(onnx_report),
        "onnx_report_sha256": digest,
        "scope": "M1 Max Qwen3-0.6B; batch 1, accumulation 1, rank-4 q/v, frozen NF4 decoder values; FP32 adapters/gradients/AdamW; four-layer decoder checkpoints and 128-row loss chunks",
        "storage": "PyTorch stores reconstructed FP32 base weights; ONNX uses packed NF4 decoder weights",
        "parameter_devices": "PyTorch parameters, gradients and AdamW state reside on the selected CPU/MPS device; ONNX uses CPU-owned adapters and optimizer state with MLX execution",
        "inputs": "Fixed benchmark tokens are prepared once before timing; ONNX execution times include runtime transfers",
        "loss": "PyTorch native cross entropy and ONNX target-excluded stable CE use the same shifted valid-token mean",
        "attention": "PyTorch eager and SDPA cases are labeled separately; ONNX uses decomposed attention. No PyTorch compilation, ONNX Runtime optimization or native MLX compilation",
        "timing": "10 warmups and 100 updates, phase synchronization on MPS; initialization reported separately",
        "memory": "Whole-process physical footprint includes construction, device allocations and runtime; RSS is recorded separately",
        "measurements": rows,
        "findings": findings,
        "omitted_measurements": omitted,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--onnx-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.output, "w") as stream:
        json.dump(collect(args.suite, args.onnx_report), stream, indent=2, allow_nan=False)
        stream.write("\n")
