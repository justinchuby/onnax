# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Static return-type regression checks for functional container overloads."""

from typing import assert_type

import onnx
import onnx_ir as ir

import onnax


def check(m: ir.Model, g: ir.Graph, view: ir.GraphView, p: onnx.ModelProto) -> None:
    assert_type(onnax.vmap(m, [0]), ir.Model)
    assert_type(onnax.vmap(g, [0]), ir.Graph)
    assert_type(onnax.vmap(view, [0]), ir.Graph)
    assert_type(onnax.vmap(p, [0]), onnx.ModelProto)
    assert_type(onnax.jvp(m), ir.Model)
    assert_type(onnax.jvp(g), ir.Graph)
    assert_type(onnax.jvp(view), ir.Graph)
    assert_type(onnax.jvp(p), onnx.ModelProto)
    assert_type(onnax.vjp(m), ir.Model)
    assert_type(onnax.vjp(g), ir.Graph)
    assert_type(onnax.vjp(view), ir.Graph)
    assert_type(onnax.vjp(p), onnx.ModelProto)
    assert_type(onnax.grad(m), ir.Model)
    assert_type(onnax.grad(g), ir.Graph)
    assert_type(onnax.grad(view), ir.Graph)
    assert_type(onnax.grad(p), onnx.ModelProto)
    assert_type(onnax.jvp(onnax.grad(g)), ir.Graph)
    assert_type(onnax.jvp(onnax.grad(p)), onnx.ModelProto)


def check_union(value: ir.Model | ir.Graph | ir.GraphView | onnx.ModelProto) -> None:
    assert_type(onnax.grad(value), ir.Model | ir.Graph | onnx.ModelProto)
