# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Register a plugin once, then use ordinary onnax transformation calls."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence

import numpy as np
import onnx
import onnx_ir as ir
from onnx.reference import ReferenceEvaluator

import onnax


@onnax.register_jvp("example", "Square", since_version=1, until_version=1)
def square_jvp(
    ctx: onnax.AutodiffContext, node: ir.Node, tangents: Mapping[int, ir.Value]
) -> Sequence[ir.Value]:
    x = ctx.input(node)
    return [ctx.op("Mul", ctx.op("Mul", ctx.scalar(x, 2), x), tangents[0])]


@onnax.register_vjp("example", "Square", since_version=1, until_version=1)
def square_vjp(
    ctx: onnax.AutodiffContext,
    node: ir.Node,
    cotangents: Sequence[ir.Value],
    input_indices: Sequence[int],
) -> Mapping[int, ir.Value]:
    x = ctx.input(node)
    return {
        i: ctx.op("Mul", ctx.op("Mul", ctx.scalar(x, 2), x), cotangents[0])
        for i in input_indices
    }


@onnax.register_batching("example", "Square", since_version=1, until_version=1)
def square_batching(ctx: onnax.BatchingContext, node: ir.Node) -> None:
    ctx.elementwise(node)


def model() -> ir.Model:
    # A local ONNX function supplies executable primal semantics. Registration
    # supplies transformation rules, not runtime kernels or inference schemas.
    formal = ir.val("arg")
    body = ir.tape.Tape()
    squared = body.op("Mul", [formal, formal])
    function = ir.Function(
        "example",
        "Square",
        graph=ir.Graph(
            [formal], [squared], nodes=body.nodes, opset_imports={"": 18}, name="square_body"
        ),
        attributes=[],
    )
    x = ir.val("x", dtype=ir.DataType.DOUBLE, shape=[2])
    tape = ir.tape.Tape()
    prediction = tape.op("Square", [x], domain="example")
    prediction.name, prediction.dtype, prediction.shape = (
        "prediction",
        ir.DataType.DOUBLE,
        ir.Shape([2]),
    )
    loss = tape.op("ReduceSum", [prediction], {"keepdims": 0})
    loss.name, loss.dtype, loss.shape = "loss", ir.DataType.DOUBLE, ir.Shape([])
    return ir.Model(
        ir.Graph(
            [x],
            [prediction, loss],
            nodes=tape.nodes,
            opset_imports={"": 18, "example": 1},
            name="custom_square",
        ),
        ir_version=10,
        functions=[function],
    )


def main(output_dir: str) -> None:
    source = model()
    gradient = onnax.grad(source, output="loss")
    forward, reverse = onnax.jvp(source), onnax.vjp(source)
    product = onnax.jvp(gradient)
    batched = onnax.vmap(gradient, in_axes=(0,))
    os.makedirs(output_dir, exist_ok=True)
    for name, graph in (
        ("primal", source),
        ("gradient", gradient),
        ("jvp", forward),
        ("vjp", reverse),
        ("hvp", product),
        ("batched", batched),
    ):
        path = os.path.join(output_dir, f"{name}.onnx")
        ir.save(graph, path)
        onnx.checker.check_model(path, full_check=True)

    def run(graph, feeds):
        return ReferenceEvaluator(ir.to_proto(graph)).run(None, feeds)

    x, direction = np.array([2.0, -3.0]), np.array([0.5, -0.25])
    np.testing.assert_allclose(run(gradient, {"x": x})[-1], 2 * x)
    feeds = {"x": x, forward.graph.inputs[-1].name: direction}
    np.testing.assert_allclose(run(forward, feeds)[-1], np.sum(2 * x * direction))
    feeds = {
        "x": x,
        reverse.graph.inputs[-2].name: direction,
        reverse.graph.inputs[-1].name: np.array(0.75),
    }
    np.testing.assert_allclose(run(reverse, feeds)[-1], 2 * x * (direction + 0.75))
    feeds = {"x": x, product.graph.inputs[-1].name: direction}
    np.testing.assert_allclose(run(product, feeds)[-1], 2 * direction)
    batch = np.stack([x, -x])
    np.testing.assert_allclose(run(batched, {"x": batch})[-1], 2 * batch)
    print(
        f"Global extension rules: JVP, VJP, grad, HVP and vmap validated; models: {output_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="custom_rule_models")
    main(parser.parse_args().output_dir)
