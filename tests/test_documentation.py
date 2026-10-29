# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Keep executable documentation under the repository's construction policy."""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _lint(source: str, filename: str, *options: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--config",
            str(_ROOT / "pyproject.toml"),
            "--stdin-filename",
            str(_ROOT / filename),
            *options,
            "-",
        ],
        input=source,
        capture_output=True,
        text=True,
        check=False,
    )


def test_readme_python_lint():
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    snippets = [block.split("```", 1)[0] for block in readme.split("```python\n")[1:]]
    assert snippets, "README must contain an executable Python example."
    for source in snippets:
        result = _lint(source, "README.py")
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "filename", ["README.py", "examples/construction.py", "tools/check_distribution.py"]
)
@pytest.mark.parametrize("module", ["onnx.helper", "onnx.numpy_helper"])
def test_documentation_disallows_protobuf_builders(filename, module):
    result = _lint(f"import {module}\n", filename, "--select", "TID251")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "banned-api" in result.stdout, result.stdout + result.stderr


def test_documentation_allows_onnx_boundaries():
    result = _lint(
        "import onnx\nfrom onnx.reference import ReferenceEvaluator\n",
        "examples/evaluate.py",
        "--select",
        "TID251",
    )
    assert result.returncode == 0, result.stdout + result.stderr
