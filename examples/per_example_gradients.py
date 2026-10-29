# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Compute per-example parameter-gradient norms for a small multi-output regressor.

Run ``python examples/per_example_gradients.py --output-dir per_example_models``.
This prints and saves before/after ONNX and readable graphs, then reloads and
executes them. Only onnax and its ordinary dependencies are required.

The source predicts two responses from three features and has a scalar
single-example mean-squared loss. grad differentiates its shared weight and
bias initializers without promoting them to inputs. vmap then maps samples
and targets, retaining a distinct example axis in each parameter gradient.
The same exported graph accepts batches of 1, 3, and 7.

Each norm combines that example's weight and bias gradients. These can be used
to inspect unusually influential examples or as the input to per-example
gradient clipping. No clipping, privacy guarantee, or optimizer is implemented.
The gradient of the mean batch loss is the *mean of the gradient tensors*;
its norm is generally not the mean of their individual norms.
"""

from __future__ import annotations

import argparse
from pathlib import Path  # ruff: ignore[banned-api] - CLI artifact paths, not library code.

import numpy as np
import onnx
import onnx_ir as ir
from onnx.reference import ReferenceEvaluator
from onnx_ir.passes.common import CheckerPass

from onnax import grad, vmap
from onnax._inference import _infer_shapes


def regressor(weight: np.ndarray, bias: np.ndarray) -> ir.Model:
    """Build a single-example two-response linear regression loss."""
    features = ir.val("features", dtype=ir.DataType.DOUBLE, shape=[3])
    target = ir.val("target", dtype=ir.DataType.DOUBLE, shape=[2])
    tape = ir.tape.Tape()
    w = tape.initializer(ir.tensor(weight, name="weight"))
    b = tape.initializer(ir.tensor(bias, name="bias"))
    prediction = tape.op("Add", [tape.op("MatMul", [features, w]), b])
    prediction.name = "prediction"
    residual = tape.op("Sub", [prediction, target])
    loss = tape.op("ReduceMean", [tape.op("Mul", [residual, residual])], {"keepdims": 0})
    loss.name = "loss"
    return ir.Model(
        ir.Graph(
            [features, target],
            [prediction, loss],
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": 18},
            name="single_example_regressor",
        ),
        ir_version=10,
    )


def save_graph(model: ir.Model, name: str, output_dir: Path) -> ir.Model:
    """Print, save, check, and reload an ordinary executable ONNX artifact."""
    inferred = _infer_shapes(model)
    CheckerPass(full_check=True)(inferred)
    proto = ir.to_proto(inferred)
    path = output_dir / f"{name}.onnx"
    onnx.save(proto, path)
    text = onnx.printer.to_text(proto)
    path.with_suffix(".txt").write_text(text, encoding="utf-8")
    print(f"\n=== {name} ===\n{text}")
    return ir.load(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("per_example_models"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weight = np.array([[0.2, -0.3], [0.1, 0.4], [-0.2, 0.1]])
    bias = np.array([0.3, -0.1])
    original = regressor(weight, bias)
    before = ir.to_proto(original).SerializeToString()
    save_graph(original, "single_example", args.output_dir)
    gradient = grad(original, inputs=["weight", "bias"], output="loss")
    gradient = save_graph(gradient, "single_gradient", args.output_dir)
    mapped = vmap(gradient, in_axes=[0, 0])
    mapped = save_graph(mapped, "per_example_gradient", args.output_dir)
    actual_inputs = [v.name for v in mapped.graph.inputs]
    assert actual_inputs == ["features", "target"], (
        f"Per-example gradient example expected runtime inputs ['features', 'target'], "
        f"got {actual_inputs!r}. Check that weight/bias remain stored initializers "
        "and grad adds no cotangent seed inputs."
    )
    evaluator = ReferenceEvaluator(ir.to_proto(mapped))
    independent = ReferenceEvaluator(ir.to_proto(gradient))

    batch_loss = vmap(original, in_axes=[0, 0])
    tape = ir.tape.Tape(batch_loss.graph)
    mean_loss = tape.op("ReduceMean", [batch_loss.graph.outputs[1]], {"keepdims": 0})
    mean_loss.name = "mean_loss"
    batch_loss.graph.outputs.clear()
    batch_loss.graph.outputs.append(mean_loss)
    batch_gradient = grad(batch_loss, inputs=["weight", "bias"])
    batch_gradient = save_graph(batch_gradient, "mean_loss_gradient", args.output_dir)
    mean_evaluator = ReferenceEvaluator(ir.to_proto(batch_gradient))

    rng = np.random.default_rng(42)
    for batch in (1, 3, 7):
        features = rng.normal(size=(batch, 3))
        target = features @ np.array([[1.0, -0.5], [0.5, 0.8], [-0.3, 0.7]]) + 0.4
        feeds = {"features": features, "target": target}
        prediction, loss, gw, gb = evaluator.run(None, feeds)
        residual = prediction - target
        np.testing.assert_allclose(loss, np.mean(residual**2, axis=1))
        np.testing.assert_allclose(gw, features[:, :, None] * residual[:, None, :])
        np.testing.assert_allclose(gb, residual)
        for i in range(batch):
            expected = independent.run(None, {"features": features[i], "target": target[i]})
            for value, reference in zip((prediction, loss, gw, gb), expected, strict=True):
                np.testing.assert_allclose(value[i], reference)
        norms = np.sqrt(np.sum(gw**2, axis=(1, 2)) + np.sum(gb**2, axis=1))
        _, mean_gw, mean_gb = mean_evaluator.run(None, feeds)
        np.testing.assert_allclose(mean_gw, gw.mean(axis=0))
        np.testing.assert_allclose(mean_gb, gb.mean(axis=0))
        mean_gradient_norm = np.sqrt(np.sum(mean_gw**2) + np.sum(mean_gb**2))
        print(f"Batch {batch}: per-example parameter-gradient norms: {norms}")
        print(
            f"Norm of mean-loss gradient: {mean_gradient_norm:.6f}; "
            f"mean of individual norms: {norms.mean():.6f}"
        )
    assert ir.to_proto(original).SerializeToString() == before, (
        "Per-example gradient example detected source-model mutation after grad/vmap/export. "
        "Check clone/tensor-metadata isolation before reusing the original model."
    )


if __name__ == "__main__":
    main()
