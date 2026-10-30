# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Functional adapter for the in-place symbolic inference engine."""

from __future__ import annotations

import copy

import onnx_ir as ir
from onnx_shape_inference import (
    SYM_DATA_KEY,
    OpUsageError,
    ShapeInferenceError,
    infer_symbolic_shapes,
)

from onnax._diagnostics import _argument_context


def _copy_tensor_metadata(tensor: ir.TensorProtocol) -> ir.TensorProtocol:
    # TensorProtoTensor stores its name in the backing protobuf. A shallow copy
    # still shares that mutable name; inference can name anonymous Constants.
    # Named initializer payloads keep the existing storage-sharing behavior.
    if isinstance(tensor, ir.serde.TensorProtoTensor) and not tensor.name:
        result = ir.serde.TensorProtoTensor(ir.serde.serialize_tensor(tensor))
        result.meta.update(tensor.meta)
        return result
    return copy.copy(tensor)


def _infer_shapes(model: ir.Model, *, context: str = "shape inference") -> ir.Model:
    """Infer a clone without sharing mutable tensor names with the source.

    IR cloning shares tensor payload objects. Inference exposes Constant tensors
    as value.const_value; subsequent serialization can name those tensors.
    Shallow copies isolate tensor metadata while sharing array storage. Anonymous
    protobuf-backed tensors receive independent protos before inference names them.
    The engine's refine policy preserves declared shapes and propagates errors.
    Known engine failures retain their type/cause and receive a graph-specific
    exception note naming the calling transform and the corrective action.
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
                        name,
                        _copy_tensor_metadata(attr.as_tensor()),
                        doc_string=attr.doc_string,
                    )
        for subgraph in (graph, *graph.subgraphs()):
            for value in (*subgraph.inputs, *subgraph.initializers.values()):
                if value.const_value is not None:
                    value.const_value = _copy_tensor_metadata(value.const_value)
            for value in subgraph.inputs:
                value.meta.pop(SYM_DATA_KEY, None)
                if value.const_value is not None:
                    defaults.append((value, value.const_value))
                    value.const_value = None
    try:
        return infer_symbolic_shapes(result, policy="refine")
    except (OpUsageError, ShapeInferenceError) as error:
        error.add_note(
            f"{context}: onnx-shape-inference failed for graph {model.graph.name!r} "
            f"with imports {_argument_context(dict(model.graph.opset_imports))}, policy='refine'. "
            "Correct the reported operator inputs/attributes or incompatible shape "
            "annotations in the source model, then retry. The source is unchanged."
        )
        raise
    finally:
        # A graph-input initializer is an overridable default, not a constant
        # for inference. Restore its stored payload without specializing users.
        for value, tensor in defaults:
            value.const_value = tensor
