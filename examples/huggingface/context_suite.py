# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Run isolated 4k PyTorch CPU/MPS baselines under the Mac process guard."""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys

from examples.huggingface.guard import run


def source_identity():
    names = (
        "benchmark.py",
        "torch_context.py",
        "qwen.py",
        "validation.py",
        "sft.py",
        "mixed_precision.py",
        "nf4.py",
        "context_suite.py",
        "../lora/nf4.py",
    )
    hashes = {}
    for name in names:
        with open(os.path.join(os.path.dirname(__file__), name), "rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "commit": result.stdout.strip() if result.returncode == 0 else None,
        "files": hashes,
    }


def run_suite(directory, train_file, *, sdpa=True, memory_bytes=24 * 1024**3):
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError("Disable PyTorch MPS CPU fallback for the native comparison")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    directory = os.path.abspath(directory)
    train_file = os.path.abspath(train_file)
    with open(train_file, "rb") as stream:
        corpus = hashlib.file_digest(stream, "sha256").hexdigest()
    identity = source_identity()
    policy = {
        "source": identity,
        "corpus_sha256": corpus,
        "sdpa": sdpa,
        "memory_bytes": memory_bytes,
    }
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "queue.json")
    if os.path.exists(path):
        with open(path) as stream:
            queue = json.load(stream)
        if queue["policy"] != policy:
            raise ValueError(
                "Source, data or measurement policy changed since the suite started"
            )
    else:
        queue = {
            "version": 1,
            "source_commit": identity["commit"],
            "policy": policy,
            "cases": [],
        }
    queue["status"] = "running"

    def persist():
        with open(path + ".tmp", "w") as stream:
            json.dump(queue, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(path + ".tmp", path)

    configurations = [
        (device, precision, "eager")
        for device in ("mps", "cpu")
        for precision in ("fp32", "bf16")
    ]
    if sdpa:
        configurations.extend(("mps", precision, "sdpa") for precision in ("fp32", "bf16"))
    for device, precision, attention in configurations:
        base = f"{device}-{precision}-4096-group4" + ("-sdpa" if attention == "sdpa" else "")
        for probe in (True, False):
            name = base + ("-capacity" if probe else "")
            previous = next((row for row in queue["cases"] if row["name"] == name), None)
            if previous is not None:
                if previous["status"] == "complete":
                    with open(os.path.join(directory, name + ".json"), "rb") as stream:
                        if (
                            hashlib.file_digest(stream, "sha256").hexdigest()
                            != previous["report_sha256"]
                        ):
                            raise ValueError(f"Stored measurement changed: {name}")
                    continue
                if previous.get("guard_reason") == "Physical-footprint budget exceeded":
                    break
                raise ValueError(f"Inspect the existing failed or interrupted worker: {name}")
            command = [
                sys.executable,
                "-m",
                "examples.huggingface.benchmark",
                "--mode",
                "pytorch",
                "--torch-device",
                device,
                "--torch-attention",
                attention,
                "--torch-chunked",
                "--checkpoint-layers",
                "4",
                "--real",
                "--nf4",
                "--projection-precision",
                precision,
                "--sequence-length",
                "4096",
                "--batch-size",
                "1",
                "--chunk-size",
                "128",
                "--train-file",
                train_file,
                "--warmup",
                "0" if probe else "10",
                "--samples",
                "1" if probe else "100",
            ]
            row = {
                "name": name,
                "command": command,
                "source_commit": identity["commit"],
                "status": "running",
                "started": datetime.datetime.now(datetime.UTC).isoformat(),
            }
            queue["cases"].append(row)
            queue["current"] = name
            persist()
            guard_dir = os.path.join(directory, name + ".guard")
            guarded = run(command, guard_dir, memory_bytes=memory_bytes)
            row.update(
                status=guarded["status"],
                finished=guarded["finished"],
                returncode=guarded["returncode"],
                guard_reason=guarded.get("guard_reason"),
                physical_footprint_peak_bytes=max(
                    (
                        max(item["current_bytes"], item["peak_bytes"])
                        for item in guarded["observations"]
                    ),
                    default=None,
                ),
            )
            if source_identity() != identity:
                row["status"] = "failed"
                persist()
                raise ValueError("Measurement source changed while the worker ran")
            if row["status"] == "complete":
                with open(os.path.join(guard_dir, "worker.log")) as stream:
                    report = json.loads(
                        next(
                            line
                            for line in reversed(stream.readlines())
                            if line.startswith("{")
                        )
                    )
                report_path = os.path.join(directory, name + ".json")
                with open(report_path, "w") as stream:
                    json.dump(report, stream, indent=2, allow_nan=False)
                    stream.write("\n")
                with open(report_path, "rb") as stream:
                    row["report_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            persist()
            if row["status"] != "complete":
                if row["guard_reason"] == "Physical-footprint budget exceeded":
                    break
                raise RuntimeError(f"Worker failed; inspect {guard_dir}")
    queue.update(status="complete", current=None)
    persist()
    return queue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--no-sdpa", action="store_true")
    args = parser.parse_args()
    run_suite(args.output_dir, args.train_file, sdpa=not args.no_sdpa)
