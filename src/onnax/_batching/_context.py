# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Batch mapping state and graph construction primitives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import onnx
import onnx_ir as ir

from onnax._batching._registry import _MULTI_OUTPUT
from onnax._diagnostics import _argument_context, _value_context
from onnax._rule_registry import _effective_opset


def _axis(axis: int, rank: int, *, what: str = "Axis") -> int:
    if isinstance(axis, bool) or not isinstance(axis, int) or not -rank <= axis < rank:
        remedy = (
            f"Use an integer in [{-rank}, {rank - 1}] (not bool)."
            if rank > 0
            else "A scalar has no valid axis; use an operator/input with rank >= 1."
        )
        raise ir.passes.PreconditionError(
            f"Axis {_argument_context(axis)} is invalid for rank {rank} ({what}). {remedy}"
        )
    return axis % rank


class _Context:
    def __init__(self, graph: ir.Graph, axis_size: int | None) -> None:
        self.graph = graph
        self.axis_size = axis_size
        self.ranks: dict[ir.Value, int] = {}
        self.mapped: dict[ir.Value, bool] = {}
        self.values: dict[ir.Value, ir.Value] = {}
        self.before: ir.Node | None = None
        self.batch_source: ir.Value | None = None
        self.batch_shape: ir.Value | None = None

    def _describe(self, value: ir.Value) -> str:
        if value not in self.ranks:
            return _value_context(value)
        return (
            f"{value.name!r} (example shape={list(self._example_shape(value))}, "
            f"rank={self.ranks[value]}, dtype={value.dtype}, mapped={self.mapped[value]})"
        )

    def _input_error(self, node: ir.Node) -> ir.passes.PreconditionError:
        version = _effective_opset(node, self.graph.opset_imports)
        schema = onnx.defs.get_schema(node.op_type, version, "")
        return ir.passes.PreconditionError(
            f"VmapPass has no batching rule for {node.op_type!r} with "
            f"{sum(value is not None for value in node.inputs)} present inputs "
            f"in {len(node.inputs)} slots. Expected schema inputs "
            f"{[value.name for value in schema.inputs]!r}, "
            f"count {schema.min_input}..{schema.max_input}. "
            "Supply required inputs in schema order and retain empty slots for "
            "omitted optional inputs."
        )

    def _emit(
        self,
        op_type: str,
        inputs: Sequence[ir.Value | None],
        attributes: Mapping[
            str, ir.Attr | int | float | str | Sequence[int] | ir.TensorProtocol
        ]
        | None = None,
    ) -> ir.Value:
        node = ir.node(op_type, inputs, attributes)
        if self.before is None:
            self.graph.append(node)
        else:
            self.graph.insert_before(self.before, node)
        return node.outputs[0]

    def _ints(self, values: Sequence[int]) -> ir.Value:
        return self._emit(
            "Constant", [], {"value": ir.tensor(values, dtype=ir.DataType.INT64)}
        )

    def _unsqueeze(self, value: ir.Value, axes: Sequence[int]) -> ir.Value:
        if not axes:
            return value
        result = self._emit("Unsqueeze", [value, self._ints(axes)])
        result.type = value.type
        if value.shape is not None:
            dims = list(value.shape)
            for axis in sorted(axes):
                dims.insert(axis, 1)
            result.shape = ir.Shape(dims)
        return result

    def _transpose(self, value: ir.Value, perm: Sequence[int]) -> ir.Value:
        if list(perm) == list(range(len(perm))):
            return value
        result = self._emit("Transpose", [value], {"perm": perm})
        result.type = value.type
        if value.shape is not None:
            result.shape = ir.Shape(value.shape[i] for i in perm)
        return result

    def _get_batch_shape(self) -> ir.Value:
        if self.batch_shape is None:
            if self.axis_size is not None:
                self.batch_shape = self._ints([self.axis_size])
            else:
                assert self.batch_source is not None
                batch_source_shape = self._emit("Shape", [self.batch_source])
                self.batch_shape = self._emit(
                    "Gather", [batch_source_shape, self._ints([0])], {"axis": 0}
                )
        return self.batch_shape

    def _repeat(self, value: ir.Value) -> ir.Value:
        batch_shape = self._get_batch_shape()
        shape = self._emit("Shape", [value])
        target = self._emit("Concat", [batch_shape, shape], {"axis": 0})
        result = self._emit("Expand", [self._unsqueeze(value, [0]), target])
        result.type = value.type
        if value.shape is not None:
            dims: list[int | ir.SymbolicDim | None] = [self.axis_size, *value.shape]
            result.shape = ir.Shape(dims)
        return result

    def _align(self, value: ir.Value, rank: int, mapped: bool, target: int) -> ir.Value:
        # Padding belongs *after* a mapped batch axis, never before it.
        start = 1 if mapped else 0
        return self._unsqueeze(value, list(range(start, target - rank + 1)))

    def _example_shape(self, value: ir.Value) -> ir.Shape:
        shape = self.values[value].shape
        assert shape is not None
        return ir.Shape(list(shape)[int(self.mapped[value]) :])

    def _register(self, value: ir.Value, rank: int, mapped: bool) -> None:
        if value.type is not None and not isinstance(value.type, ir.TensorType):
            raise ir.passes.PreconditionError(
                f"Expected a tensor value, got {_value_context(value)}. "
                "Use tensor-valued inputs/outputs; sequence and optional batching is unsupported."
            )
        if value.shape is not None and len(value.shape) != rank:
            raise ir.passes.PreconditionError(
                f"Inconsistent rank annotation for {_value_context(value)}: "
                f"batching rule expects example rank {rank}. "
                "Correct the source output shape annotation to match the operator's result."
            )
        self.ranks[value] = rank
        self.mapped[value] = mapped
        self.values[value] = value
        if mapped:
            dims: list[int | ir.SymbolicDim | None] = (
                list(value.shape) if value.shape is not None else [None] * rank
            )
            value.shape = ir.Shape([self.axis_size, *dims])
            value.const_value = None
        elif value.shape is None:
            value.shape = ir.Shape([None] * rank)

    def _constant_tensor(self, value: ir.Value) -> ir.TensorProtocol | None:
        tensor = None
        if not value.is_graph_input() and not self.mapped.get(value, False):
            if value.is_initializer():
                tensor = value.const_value
            else:
                producer = value.producer()
                if producer is not None and producer.op_type == "Constant":
                    attr = producer.attributes.get("value")
                    if attr is not None and not attr.is_ref():
                        tensor = attr.as_tensor()
        return tensor

    def _constant_ints(self, value: ir.Value, purpose: str) -> list[int]:
        tensor = self._constant_tensor(value)
        if (
            tensor is None
            or tensor.dtype != ir.DataType.INT64
            or len(tensor.shape) != 1
            or self.mapped.get(value, False)
        ):
            raise ir.passes.PreconditionError(
                f"{purpose} must be a shared constant 1-D INT64 tensor, not a runtime input; "
                f"got {self._describe(value)}. Supply a non-input INT64 initializer/Constant, "
                "not an overridable graph input. If this control must vary "
                "between examples, execute the examples independently instead."
            )
        return [int(axis) for axis in tensor.numpy().tolist()]

    def _require_shared(self, values: Sequence[ir.Value], op: str) -> None:
        mapped = [value for value in values if self.mapped[value]]
        if mapped:
            raise ir.passes.PreconditionError(
                f"{op} requires shared control/parameter inputs; mapped controls: "
                f"{[self._describe(value) for value in mapped]}. "
                "Keep these controls independent of mapped values; use in_axes=None "
                "for graph inputs only if they are truly shared; "
                "for per-example controls/parameters, execute examples independently."
            )

    def _constant_axis(self, value: ir.Value, rank: int) -> int:
        tensor = self._constant_tensor(value)
        if (
            tensor is None
            or tensor.dtype not in (ir.DataType.INT32, ir.DataType.INT64)
            or len(tensor.shape) != 0
        ):
            raise ir.passes.PreconditionError(
                "Axis must be a shared constant integer scalar (INT32 or INT64); "
                f"got {self._describe(value)}. Supply a rank-zero integer non-input initializer/Constant "
                "and keep it shared; runtime or per-example CumSum axes are unsupported."
            )
        return _axis(int(tensor.numpy().item()), rank)

    def _scalar_int(self, value: int) -> ir.Value:
        return self._emit("Constant", [], {"value": ir.tensor(value, dtype=ir.DataType.INT64)})

    def _cast_int64(self, value: ir.Value) -> ir.Value:
        return self._emit("Cast", [value], {"to": int(ir.DataType.INT64)})

    def _shape_vector_length(self, value: ir.Value) -> int:
        shape = self.values[value].shape
        if (
            self.mapped[value]
            or self.ranks[value] != 1
            or shape is None
            or not isinstance(shape[0], int)
            or value.dtype not in (None, ir.DataType.INT64)
        ):
            raise ir.passes.PreconditionError(
                "Shape targets must be shared 1-D INT64 tensors with statically known lengths; "
                f"got {self._describe(value)}. Annotate the target as shape=[N], dtype=INT64 "
                "where N is the target rank, and keep it shared. Values may remain dynamic; "
                "per-example target shapes require independent execution."
            )
        return shape[0]

    def _prepare_node(self, node: ir.Node) -> tuple[list[ir.Value], bool]:
        inputs = [value for value in node.inputs if value is not None]
        mapped = any(self.mapped[value] for value in inputs)
        if len(node.outputs) != 1 and node.op_type not in _MULTI_OUTPUT:
            raise ir.passes.PreconditionError(
                f"VmapPass requires single-output operators for {node.op_type}; "
                f"got {len(node.outputs)} outputs {[value.name for value in node.outputs]!r}. "
                "Use a single primary output only if secondary outputs are not needed; "
                "otherwise execute examples independently. Multiple outputs are supported "
                "only for the documented multi-output rules."
            )
        for i, value in enumerate(node.inputs):
            if value is not None:
                node.replace_input_with(i, self.values[value])
        return inputs, mapped
