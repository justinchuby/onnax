# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Fit a tiny linear regressor and inspect its sensitivities using ONNX autodiff.

Run ``python examples/autodiff.py --output-dir autodiff_models`` after
installing onnax. Only onnax and its ordinary dependencies are needed.

The original model computes predictions and mean-squared error for a symbolic
batch size. This example prints the before/after IR graphs, checks parameter
gradients against NumPy, takes one loss-reducing optimization step, and computes
both directional prediction sensitivities and a reverse-mode attribution.
It then applies JvpPass to the reloaded gradient graph to compute an exact
parameter Hessian-vector product and directional curvature without materializing
the Hessian. Initializer values stay stored; only their direction seeds are inputs.

All derivative models are ordinary, serializable ONNX models. For example,
``ir.save(gradient_model, "linear_regression_grad.onnx")`` exports the executable
gradient graph, not a Python callback or a training-domain operator.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx  # ruff: ignore[banned-api]
import onnx_ir as ir
from onnx.reference import ReferenceEvaluator  # ruff: ignore[banned-api]

from onnax import GradPass, JvpPass, VjpPass


def linear_regression(weight: np.ndarray, bias: np.ndarray) -> ir.Model:
    """Create prediction = features @ weight + bias and its scalar MSE loss."""
    features = ir.val("features", dtype=ir.DataType.DOUBLE, shape=["batch", 2])
    target = ir.val("target", dtype=ir.DataType.DOUBLE, shape=["batch", 1])
    tape = ir.tape.Tape()
    w = tape.initializer(ir.tensor(weight, name="weight"))
    b = tape.initializer(ir.tensor(bias, name="bias"))
    prediction = tape.op("Add", [tape.op("MatMul", [features, w]), b])
    prediction.name = "prediction"
    residual = tape.op("Sub", [prediction, target])
    loss = tape.op("ReduceMean", [tape.op("Mul", [residual, residual])], {"keepdims": 0})
    loss.name = "mse"
    return ir.Model(
        ir.Graph(
            [features, target],
            [prediction, loss],
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": 18},
            name="linear_regression",
        ),
        ir_version=9,
    )


def evaluate(model: ir.Model, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Check and execute a generated graph without an autodiff runtime."""
    proto = onnx.shape_inference.infer_shapes(ir.to_proto(model), strict_mode=True)
    onnx.checker.check_model(proto, full_check=True)
    return ReferenceEvaluator(proto).run(None, feeds)


def save_graph(model: ir.Model, name: str, output_dir: Path) -> ir.Model:
    """Save ONNX and printable graphs, then reload the executable artifact."""
    path = output_dir / f"{name}.onnx"
    proto = onnx.shape_inference.infer_shapes(ir.to_proto(model), strict_mode=True)
    onnx.checker.check_model(proto, full_check=True)
    onnx.save(proto, path)
    text = onnx.printer.to_text(proto)
    path.with_suffix(".txt").write_text(text, encoding="utf-8")
    print(f"\n=== {name} ===\n{text}")
    return ir.load(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("autodiff_models"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weight = np.array([[0.2], [-0.1]])
    bias = np.array([0.3])
    features = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]])
    target = features @ np.array([[1.5], [-0.5]]) + 0.7
    feeds = {"features": features, "target": target}
    original = linear_regression(weight, bias)
    source_bytes = ir.to_proto(original).SerializeToString()
    print("BEFORE: predictions and scalar MSE")
    save_graph(original, "original", args.output_dir)

    gradient_model = GradPass(inputs=["weight", "bias"], output="mse")(original).model
    print("\nAFTER GradPass: predictions, MSE, weight gradient, bias gradient")
    gradient_model = save_graph(gradient_model, "grad", args.output_dir)
    prediction, loss, grad_weight, grad_bias = evaluate(gradient_model, feeds)
    residual = prediction - target
    expected_weight = 2 * features.T @ residual / residual.size
    expected_bias = 2 * residual.mean(axis=0)
    np.testing.assert_allclose(grad_weight, expected_weight)
    np.testing.assert_allclose(grad_bias, expected_bias)
    updated = linear_regression(weight - 0.1 * grad_weight, bias - 0.1 * grad_bias)
    _, updated_loss = evaluate(updated, feeds)
    assert updated_loss < loss
    print(f"MSE before/after one gradient step: {loss:.6f} -> {updated_loss:.6f}")
    print("d(MSE)/d(weight):", grad_weight.ravel())
    print("d(MSE)/d(bias):", grad_bias)

    # Default outputs on a reverse-mode graph are its generated gradients.
    hvp_model = JvpPass(inputs=["weight", "bias"])(gradient_model).model
    dw, db = np.array([[0.4], [-0.2]]), np.array([0.1])
    hvp_feeds = dict(feeds)
    for seed, direction in zip(
        hvp_model.graph.inputs[len(gradient_model.graph.inputs) :], (dw, db)
    ):
        hvp_feeds[seed.name] = direction
    print("\nAFTER JvpPass(GradPass): parameter Hessian-vector product")
    hvp_model = save_graph(hvp_model, "hvp", args.output_dir)
    _, _, _, _, h_weight, h_bias = evaluate(hvp_model, hvp_feeds)
    response = features @ dw + db
    np.testing.assert_allclose(h_weight, 2 * features.T @ response / response.size)
    np.testing.assert_allclose(h_bias, 2 * response.mean(axis=0))
    curvature = np.vdot(dw, h_weight) + np.vdot(db, h_bias)
    np.testing.assert_allclose(curvature, 2 * np.mean(response**2))
    print("Hessian-vector product (weight, bias):", h_weight.ravel(), h_bias)
    print(f"Directional curvature v.T @ H @ v: {curvature:.6f}")

    jvp_model = JvpPass(inputs=["features"], outputs=["prediction"])(original).model
    direction = np.broadcast_to(np.array([1.0, 0.0]), features.shape).copy()
    jvp_feeds = {**feeds, jvp_model.graph.inputs[-1].name: direction}
    print("\nAFTER JvpPass: sensitivity to increasing the first feature")
    jvp_model = save_graph(jvp_model, "jvp", args.output_dir)
    _, _, prediction_tangent = evaluate(jvp_model, jvp_feeds)
    np.testing.assert_allclose(prediction_tangent, direction @ weight)
    print("Directional prediction derivative:", prediction_tangent.ravel())

    vjp_model = VjpPass(inputs=["features"], outputs=["prediction"])(original).model
    cotangent = np.array([[1.0], [0.0], [0.0], [0.0]])
    vjp_feeds = {**feeds, vjp_model.graph.inputs[-1].name: cotangent}
    print("\nAFTER VjpPass: attribute the first prediction to its input features")
    vjp_model = save_graph(vjp_model, "vjp", args.output_dir)
    _, _, feature_cotangent = evaluate(vjp_model, vjp_feeds)
    np.testing.assert_allclose(feature_cotangent, cotangent @ weight.T)
    print("Feature cotangent:\n", feature_cotangent)
    np.testing.assert_allclose(
        np.sum(prediction_tangent * cotangent),
        np.sum(direction * feature_cotangent),
    )
    assert ir.to_proto(original).SerializeToString() == source_bytes


if __name__ == "__main__":
    main()
