# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Differentiate ONNXScript functions: python -m examples.onnxscript_autodiff."""

import onnx_ir as ir
from onnx_ir.passes.common import InlinePass
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op

import onnax
from examples.authoring import run_cli


@script(default_opset=op)
def mean_squared_error(residual: FLOAT["batch"]) -> FLOAT[()]:
    return op.ReduceMean(residual * residual, keepdims=0)


@script(default_opset=op)
def linear_regression(
    features: FLOAT["batch", 2],
    weight: FLOAT[2],
    bias: FLOAT[()],
    target: FLOAT["batch"],
) -> tuple[FLOAT["batch"], FLOAT[()]]:
    prediction = op.MatMul(features, weight) + bias
    loss = mean_squared_error(prediction - target)
    return prediction, loss


def differentiate():
    proto = linear_regression.to_model_proto()
    graph = ir.from_proto(proto)
    graph = InlinePass()(graph).model
    gradient = onnax.grad(graph, inputs=["weight", "bias"], output="loss")
    return proto, graph, gradient


if __name__ == "__main__":
    run_cli(differentiate, "onnxscript_models")
