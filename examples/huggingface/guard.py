# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Run a Mac qualification worker with physical-memory and disk budgets."""

import argparse
import datetime
import json
import os
import shutil
import signal
import subprocess
import sys
import time

from examples.process_memory import physical_footprint


def run(command, directory, *, memory_bytes=24 * 1024**3, reserve_bytes=5 * 1024**3):
    if sys.platform != "darwin":
        raise ValueError("The physical-footprint guard requires macOS kernel counters")
    if shutil.disk_usage(os.getcwd()).free < reserve_bytes:
        raise OSError("Free disk is below the qualification reserve")
    os.makedirs(directory, exist_ok=False)
    report = {
        "command": command,
        "cwd": os.getcwd(),
        "status": "running",
        "physical_memory_budget_bytes": memory_bytes,
        "minimum_free_disk_bytes": reserve_bytes,
        "observations": [],
        "memory_source": "proc_pid_rusage/RUSAGE_INFO_V4",
        "observation_errors": 0,
    }
    path = os.path.join(directory, "execution.json")

    def persist():
        with open(path + ".tmp", "w") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        os.replace(path + ".tmp", path)

    def stop(child):
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()

    def now():
        return datetime.datetime.now(datetime.UTC).isoformat()

    with open(os.path.join(directory, "worker.log"), "w") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        report.update(pid=child.pid, started=now())
        persist()
        try:
            while child.poll() is None:
                usage = None
                try:
                    counters = physical_footprint(child.pid)
                    if counters["available"]:
                        usage = (counters["current_bytes"], counters["lifetime_peak_bytes"])
                    else:
                        report["observation_errors"] += 1
                        report["last_observation_error"] = counters["reason"]
                except OSError as error:
                    report["observation_errors"] += 1
                    report["last_observation_error"] = str(error)
                if child.poll() is not None:
                    break
                free = shutil.disk_usage(os.getcwd()).free
                if usage:
                    report["observations"].append(
                        {
                            "utc": now(),
                            "current_bytes": usage[0],
                            "peak_bytes": max(usage),
                            "free_disk_bytes": free,
                        }
                    )
                if usage and max(usage) > memory_bytes:
                    report["guard_reason"] = "Physical-footprint budget exceeded"
                elif free < reserve_bytes:
                    report["guard_reason"] = "Free disk below reserve"
                if "guard_reason" in report:
                    stop(child)
                    break
                persist()
                time.sleep(1)
        finally:
            if child.poll() is None:
                stop(child)
            report.update(returncode=child.wait(), finished=now())
            report["status"] = (
                "complete"
                if child.returncode == 0 and "guard_reason" not in report
                else "failed"
            )
            persist()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Provide a worker command after --")

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    result = run(command, args.output_dir)
    print(json.dumps({key: value for key, value in result.items() if key != "observations"}))
    sys.exit(0 if result["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
