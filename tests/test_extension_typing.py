# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Public registration preserves concrete callable types and validates contracts."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("mode", ["jvp", "vjp", "batching"])
def test_registration_preserves_callable_type_and_rejects_invalid_signature(tmp_path, mode):
    signatures = {
        "jvp": "ctx: onnax.AutodiffContext, node: ir.Node, tangents: Mapping[int, ir.Value]",
        "vjp": "ctx: onnax.AutodiffContext, node: ir.Node, cotangents: Sequence[ir.Value], indices: Sequence[int]",
        "batching": "ctx: onnax.BatchingContext, node: ir.Node",
    }
    returns = {
        "jvp": ("list[ir.Value]", "[]"),
        "vjp": ("dict[int, ir.Value]", "{}"),
        "batching": ("None", "None"),
    }
    annotation, result = returns[mode]
    source = f"""
from collections.abc import Mapping, Sequence
from typing import assert_type
import onnx_ir as ir
import onnax

class Rule:
    label: str = "plugin"
    def __call__(self, {signatures[mode]}) -> {annotation}:
        return {result}

registered = onnax.register_{mode}("example", "Op", since_version=1, until_version=1)(Rule())
assert_type(registered, Rule)
assert_type(registered.label, str)
"""
    script = tmp_path / "plugin.py"
    script.write_text(source)
    command = [sys.executable, "-m", "mypy", "--follow-imports=silent", str(script)]
    checked = subprocess.run(command, capture_output=True, text=True, check=False)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    script.write_text(
        source
        + f"""
@onnax.register_{mode}("example", "Bad", since_version=1, until_version=1)
def invalid(value: int) -> str:
    return "bad"
"""
    )
    checked = subprocess.run(command, capture_output=True, text=True, check=False)
    assert checked.returncode != 0
    assert "Callable[[int], str]" in checked.stdout and "type-var" in checked.stdout, (
        checked.stdout
    )
