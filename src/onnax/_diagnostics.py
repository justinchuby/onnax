# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Small metadata-only descriptions for transformation diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import islice

import onnx_ir as ir


def _argument_context(value: object, *, _depth: int = 0) -> str:
    """Describe malformed arguments without invoking arbitrary object reprs."""
    if isinstance(value, ir.Value):
        return _value_context(value)
    if type(value) in (str, int, float, bool, type(None)):
        return repr(value)
    if (
        _depth < 3
        and isinstance(value, (list, tuple, dict))
        and type(value) in (list, tuple, dict)
    ):
        if isinstance(value, dict):
            entries = [
                f"{_argument_context(k, _depth=_depth + 1)}: "
                f"{_argument_context(v, _depth=_depth + 1)}"
                for k, v in islice(value.items(), 8)
            ]
            left, right = "{", "}"
        else:
            entries = [_argument_context(v, _depth=_depth + 1) for v in value[:8]]
            left, right = ("[", "]") if isinstance(value, list) else ("(", ")")
        if len(value) > 8:
            entries.append("...")
        suffix = "," if isinstance(value, tuple) and len(value) == 1 else ""
        return left + ", ".join(entries) + suffix + right
    return f"<{type(value).__name__}>"


def _node_context(node: ir.Node, imports: Mapping[str, int], mode: str) -> str:
    version = node.version
    if version is None:
        version = imports.get(node.domain)
        if version is None and node.domain in ("", "ai.onnx"):
            version = imports.get("" if node.domain == "ai.onnx" else "ai.onnx")
    location = (
        f"node {node.name!r}"
        if node.name
        else (f"outputs {[value.name for value in node.outputs]!r}")
    )
    if node.graph is not None:
        location += f", graph {node.graph.name!r}"
    return (
        f"{mode}: {node.domain or 'ai.onnx'}::{node.op_type} "
        f"at opset {_argument_context(version) if version is not None else 'missing'} "
        f"({location})"
    )


def _value_context(value: ir.Value) -> str:
    # Never format the Value itself: initializers can include tensor contents.
    shape = list(value.shape) if value.shape is not None else None
    return f"{value.name!r} (type={value.type}, shape={shape})"
