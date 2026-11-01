# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Guard budget enforcement and recovery from transient counter errors."""

import sys
import time
from types import SimpleNamespace

from examples.huggingface import guard


def test_transient_observation_errors_keep_worker_running(tmp_path, monkeypatch):
    calls = 0
    release = tmp_path / "release-worker"

    def memory(pid):
        nonlocal calls
        calls += 1
        assert pid > 0
        if calls <= 3:
            raise OSError("Temporary counter failure")
        if calls == 5:
            release.touch()
        return {"available": True, "current_bytes": 100, "lifetime_peak_bytes": 100}

    monkeypatch.setattr(guard, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(guard, "physical_footprint", memory)
    monkeypatch.setattr(guard, "time", SimpleNamespace(sleep=lambda _: time.sleep(0.01)))
    directory = tmp_path / "worker"
    report = guard.run(
        [
            sys.executable,
            "-c",
            (
                "import pathlib,sys,time\n"
                "flag=pathlib.Path(sys.argv[1])\n"
                "while not flag.exists(): time.sleep(0.001)\n"
                "print('finished')"
            ),
            str(release),
        ],
        directory,
        reserve_bytes=0,
    )
    assert report["status"] == "complete"
    assert report["returncode"] == 0
    assert report["observation_errors"] == 3
    assert report["observations"]
    assert "guard_reason" not in report
    assert (directory / "worker.log").read_text().strip() == "finished"


def test_observed_memory_budget_stops_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(
        guard,
        "physical_footprint",
        lambda pid: {"available": True, "current_bytes": 101, "lifetime_peak_bytes": 101},
    )
    report = guard.run(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        tmp_path / "worker",
        memory_bytes=100,
        reserve_bytes=0,
    )
    assert report["status"] == "failed" and report["returncode"] != 0
    assert report["guard_reason"] == "Physical-footprint budget exceeded"
    assert report["observations"][0]["peak_bytes"] == 101
