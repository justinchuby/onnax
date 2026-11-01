# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Measurement provenance, resumption and failure classification."""

import json
import os

import pytest

from examples.huggingface import context_suite


def setup(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    monkeypatch.setattr(
        context_suite, "source_identity", lambda: {"commit": "fixed", "files": {}}
    )
    corpus = tmp_path / "data.jsonl"
    corpus.write_text('{"text":"example"}\n')
    return tmp_path / "suite", corpus


def worker(command, directory, *, memory_bytes):
    os.makedirs(directory)
    with open(os.path.join(directory, "worker.log"), "w") as stream:
        stream.write('a loader message\n{"losses": [1.0]}\n')
    return {
        "status": "complete",
        "finished": "now",
        "returncode": 0,
        "observations": [{"current_bytes": 10, "peak_bytes": 20}],
    }


def test_resume_checks_reports_and_data_before_execution(monkeypatch, tmp_path):
    directory, corpus = setup(monkeypatch, tmp_path)
    calls = []

    def execute(*args, **kwargs):
        calls.append(args[0])
        return worker(*args, **kwargs)

    monkeypatch.setattr(context_suite, "run", execute)
    first = context_suite.run_suite(directory, corpus)
    assert first["status"] == "complete" and first["current"] is None
    assert len(calls) == 12  # Four eager and two SDPA configurations, probe + timing.
    assert len({row["name"] for row in first["cases"]}) == 12
    for command in calls:
        assert command[command.index("--samples") + 1] in ("1", "100")
    context_suite.run_suite(directory, corpus)
    assert len(calls) == 12
    corpus.write_text('{"text":"different"}\n')
    with pytest.raises(ValueError, match="policy changed"):
        context_suite.run_suite(directory, corpus)
    assert len(calls) == 12
    corpus.write_text('{"text":"example"}\n')
    (directory / "mps-fp32-4096-group4.json").write_text("{}")
    with pytest.raises(ValueError, match="measurement changed"):
        context_suite.run_suite(directory, corpus)
    assert len(calls) == 12


def test_memory_budget_finding_skips_timing_but_keeps_other_cases(monkeypatch, tmp_path):
    directory, corpus = setup(monkeypatch, tmp_path)

    def limited(command, guard_directory, **kwargs):
        report = worker(command, guard_directory, **kwargs)
        if guard_directory.endswith("mps-fp32-4096-group4-capacity.guard"):
            report.update(
                status="failed",
                returncode=-15,
                guard_reason="Physical-footprint budget exceeded",
            )
        return report

    monkeypatch.setattr(context_suite, "run", limited)
    report = context_suite.run_suite(directory, corpus)
    names = {row["name"] for row in report["cases"]}
    assert "mps-fp32-4096-group4" not in names
    assert "mps-bf16-4096-group4" in names and "cpu-fp32-4096-group4" in names
    assert context_suite.run_suite(directory, corpus) == report


def test_execution_failure_is_not_a_memory_finding(monkeypatch, tmp_path):
    directory, corpus = setup(monkeypatch, tmp_path)

    def broken(command, guard_directory, **kwargs):
        report = worker(command, guard_directory, **kwargs)
        report.update(status="failed", returncode=1)
        return report

    monkeypatch.setattr(context_suite, "run", broken)
    with pytest.raises(RuntimeError, match="Worker failed"):
        context_suite.run_suite(directory, corpus)
    stored = json.loads((directory / "queue.json").read_text())
    assert len(stored["cases"]) == 1
    assert stored["cases"][0]["guard_reason"] is None
    with pytest.raises(ValueError, match="failed or interrupted"):
        context_suite.run_suite(directory, corpus)


def test_comparison_rejects_unmatched_inputs_and_nonfinite_updates(tmp_path):
    from examples.huggingface.context_comparison import collect

    cases, references = [], []
    for precision in ("fp32", "bf16"):
        references.append(
            {
                "case": precision + "-4096-group4",
                "precision": precision,
                "input_ids_sha256": "same tokens",
                "sequence": 4096,
                "microbatch": 1,
                "threads": {"torch": 1},
                "hardware": {"cpu": "same machine"},
                "versions": {},
                "timings_ms": {"step": {"median": 2000}},
                "physical_footprint": {"lifetime_peak_bytes": 100},
            }
        )
        for device in ("cpu", "mps"):
            name = f"{device}-{precision}-4096-group4"
            cases.append(
                {"name": name, "status": "complete", "physical_footprint_peak_bytes": 50}
            )
            report = {
                "input_ids_sha256": "same tokens",
                "projection_precision": precision,
                "batch": 1,
                "sequence": 4096,
                "provider": "pytorch_" + device,
                "checkpoint_layers": 4,
                "head_chunk_size": 128,
                "valid_shifted_tokens": 4095,
                "warmup": 10,
                "samples": 100,
                "gradient_and_optimizer_dtype": "float32",
                "torch_chunked_head": True,
                "mps_fallback_enabled": False,
                "losses": [1.0] * 110,
                "attention_implementation": "eager",
                "threads": {"torch": 1},
                "hardware": {"cpu": "same machine"},
                "versions": {},
                "timings_ms": {"step": {"median": 1000}},
            }
            (tmp_path / (name + ".json")).write_text(json.dumps(report))
            guard = tmp_path / (name + ".guard")
            guard.mkdir()
            (guard / "execution.json").write_text('{"status":"complete"}')
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps({"status": "complete", "source_commit": "fixed", "cases": cases})
    )
    onnx = tmp_path / "onnx.json"
    onnx.write_text(json.dumps({"measurements": references}))
    comparison = collect(tmp_path, onnx)
    assert len(comparison["measurements"]) == 4
    assert all(
        row["onnx_speedup_over_pytorch"] == pytest.approx(0.5)
        for row in comparison["measurements"]
    )
    path = tmp_path / "cpu-fp32-4096-group4.json"
    data = json.loads(path.read_text())
    data["input_ids_sha256"] = "different tokens"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Unmatched input_ids"):
        collect(tmp_path, onnx)
    data["input_ids_sha256"] = "same tokens"
    data["losses"][-1] = float("nan")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="nonfinite"):
        collect(tmp_path, onnx)
