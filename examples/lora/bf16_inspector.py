# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Observe mixed-precision matrix inputs/results for accumulation-bound checks."""

import onnx
import onnx_ir as ir

from examples.lora.bf16_bounds import product_bounds
from examples.lora.fp32_bounds import product_bounds as fp32_product_bounds


def instrument(model, *, include_fp32=False):
    """Expose selected MatMul operands and results through FP32 cast outputs."""
    graph = model.clone()
    products = [
        node
        for node in graph.graph
        if node.domain == ""
        and node.op_type == "MatMul"
        and (
            node.outputs[0].dtype == ir.DataType.BFLOAT16
            or (include_fp32 and node.outputs[0].dtype == ir.DataType.FLOAT)
        )
    ]
    if not products:
        raise ValueError("Expected selected MatMul nodes")
    original_count = len(graph.graph.outputs)
    tape = ir.tape.Tape(graph.graph)
    taps = []
    for node in products:
        indices = []
        for value in (*node.inputs, node.outputs[0]):
            output = tape.op("Cast", [value], {"to": int(ir.DataType.FLOAT)})
            output.dtype = ir.DataType.FLOAT
            output.shape = ir.Shape(value.shape)
            indices.append(len(graph.graph.outputs))
            graph.graph.outputs.append(output)
        taps.append((node.outputs[0].name, indices))
    onnx.checker.check_model(ir.to_proto(graph), full_check=True)
    return graph, original_count, taps


class ProductInspector:
    """Execute an instrumented copy using the caller's runtime factory."""

    def __init__(self, model, *, session_factory, include_fp32=False):
        graph, self.original_count, self.taps = instrument(model, include_fp32=include_fp32)
        self.dtypes = {
            node.outputs[0].name: node.outputs[0].dtype
            for node in model.graph
            if node.op_type == "MatMul" and node.domain == ""
        }
        self.executor = session_factory(graph)

    def check(self, feeds, original):
        values = self.executor.run(None, feeds)
        unchanged = all(
            a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
            for a, b in zip(original, values[: self.original_count], strict=True)
        )
        products = {}
        for name, (a, b, output) in self.taps:
            dtype = self.dtypes[name]
            factory = product_bounds if dtype == ir.DataType.BFLOAT16 else fp32_product_bounds
            bounds = factory(values[a], values[b])
            products[name] = {
                "dtype": str(dtype),
                "inner_dimension": bounds.inner_dimension,
                **bounds.check(values[output]),
            }
        return {"original_outputs_equal": unchanged, "products": products}
