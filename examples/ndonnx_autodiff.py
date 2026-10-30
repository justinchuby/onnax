# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Differentiate an ndonnx array program: python -m examples.ndonnx_autodiff."""

from __future__ import annotations

import ndonnx as ndx
import onnx_ir as ir

import onnax
from examples.authoring import run_cli


def model():
    features = ndx.argument(shape=("batch", 2), dtype=ndx.float32)
    weight = ndx.argument(shape=(2,), dtype=ndx.float32)
    bias = ndx.argument(shape=(), dtype=ndx.float32)
    target = ndx.argument(shape=("batch",), dtype=ndx.float32)
    prediction = ndx.sum(features * weight, axis=1) + bias
    residual = prediction - target
    loss = ndx.mean(residual * residual)
    return ndx.build(
        {"features": features, "weight": weight, "bias": bias, "target": target},
        {"prediction": prediction, "loss": loss},
    )


def differentiate():
    proto = model()
    graph = ir.from_proto(proto)
    gradient = onnax.grad(graph, inputs=["weight", "bias"], output="loss")
    return proto, graph, gradient


if __name__ == "__main__":
    run_cli(differentiate, "ndonnx_models")
