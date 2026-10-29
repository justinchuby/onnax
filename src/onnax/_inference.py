# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Functional adapter for the in-place symbolic inference engine."""

from __future__ import annotations

import copy

import onnx_ir as ir
from onnx_shape_inference import SYM_DATA_KEY, infer_symbolic_shapes


def _infer_shapes(model: ir.Model) -> ir.Model:
    """Infer a clone without sharing mutable tensor names with the source.

    IR cloning shares tensor payload objects. Inference exposes Constant tensors
    as value.const_value; subsequent serialization can name those tensors.
    Shallow copies isolate tensor metadata without duplicating array storage.
    The engine's refine policy preserves declared shapes and propagates errors.
    """
    result = model.clone(deep_copy=True)
    defaults: list[tuple[ir.Value, ir.TensorProtocol]] = []
    for graph in (result.graph, *(f.graph for f in result.functions.values())):
        for node in graph.all_nodes():
            for output in node.outputs:
                output.const_value = None
                output.meta.pop(SYM_DATA_KEY, None)
            for name, attr in tuple(node.attributes.items()):
                if not attr.is_ref() and attr.type == ir.AttributeType.TENSOR:
                    node.attributes[name] = ir.AttrTensor(
                        name, copy.copy(attr.as_tensor()), doc_string=attr.doc_string
                    )
        for subgraph in (graph, *graph.subgraphs()):
            for value in (*subgraph.inputs, *subgraph.initializers.values()):
                if value.const_value is not None:
                    value.const_value = copy.copy(value.const_value)
            for value in subgraph.inputs:
                value.meta.pop(SYM_DATA_KEY, None)
                if value.const_value is not None:
                    defaults.append((value, value.const_value))
                    value.const_value = None
    try:
        return infer_symbolic_shapes(result, policy="refine")
    finally:
        # A graph-input initializer is an overridable default, not a constant
        # for inference. Restore its stored payload without specializing users.
        for value, tensor in defaults:
            value.const_value = tensor
