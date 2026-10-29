# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Small metadata-only descriptions for transformation diagnostics."""

from __future__ import annotations

from collections.abc import Mapping

import onnx_ir as ir


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
        f"at opset {version if version is not None else 'missing'} ({location})"
    )


def _value_context(value: ir.Value) -> str:
    # Never format the Value itself: initializers can include tensor contents.
    shape = list(value.shape) if value.shape is not None else None
    return f"{value.name!r} (type={value.type}, shape={shape})"
