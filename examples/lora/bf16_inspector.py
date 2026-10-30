# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Observe BF16 matrix inputs/results for local accumulation-bound checks."""

import onnx
import onnx_ir as ir

from examples.lora.bf16_bounds import product_bounds


def instrument(model):
    """Expose each BF16 MatMul operand and result through FP32 cast outputs."""
    graph = model.clone()
    products = [
        node
        for node in graph.graph
        if node.domain == ""
        and node.op_type == "MatMul"
        and node.outputs[0].dtype == ir.DataType.BFLOAT16
    ]
    if not products:
        raise ValueError("Expected BF16 MatMul nodes")
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

    def __init__(self, model, *, session_factory):
        graph, self.original_count, self.taps = instrument(model)
        self.executor = session_factory(graph)

    def check(self, feeds, original):
        values = self.executor.run(None, feeds)
        unchanged = all(
            a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
            for a, b in zip(original, values[: self.original_count], strict=True)
        )
        products = {}
        for name, (a, b, output) in self.taps:
            bounds = product_bounds(values[a], values[b])
            products[name] = {
                "inner_dimension": bounds.inner_dimension,
                **bounds.check(values[output]),
            }
        return {"original_outputs_equal": unchanged, "products": products}
