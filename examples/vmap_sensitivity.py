# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Batch input-sensitivity directions: python -m examples.vmap_sensitivity."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort

import onnax

WEIGHT = np.array([[0.7, -0.2], [0.3, 0.8], [-0.5, 0.4]], dtype=np.float32)
BIAS = np.array([0.1, -0.3], dtype=np.float32)


def predictor() -> ir.Model:
    """Predict two responses from three input features."""
    x = ir.val("features", dtype=ir.DataType.FLOAT, shape=[3])
    tape = ir.tape.Tape()
    weight = tape.initializer(ir.tensor(WEIGHT, name="weight"))
    bias = tape.initializer(ir.tensor(BIAS, name="bias"))
    output = tape.op("Tanh", [tape.op("Add", [tape.op("MatMul", [x, weight]), bias])])
    output.name = "prediction"
    output.dtype = ir.DataType.FLOAT
    output.shape = ir.Shape([2])
    return ir.Model(
        ir.Graph(
            [x],
            [output],
            nodes=tape.nodes,
            initializers=[weight, bias],
            opset_imports={"": 24},
            name="sensitivity",
        ),
        ir_version=10,
    )


def _session(path: str) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])


def run(output_dir: str) -> dict:
    """Export, reload and verify batched JVPs at one shared input point."""
    os.makedirs(output_dir, exist_ok=True)
    model = predictor()
    before = ir.to_proto(model).SerializeToString()
    directional = onnax.jvp(model, inputs=["features"], outputs=["prediction"])
    # The feature vector is shared; only the tangent has a direction axis.
    batched = onnax.vmap(directional, in_axes=(None, 0))
    sessions = {}
    for name, graph in (("forward", model), ("jvp", directional), ("batched_jvp", batched)):
        path = os.path.join(output_dir, f"{name}.onnx")
        ir.save(graph, path)
        onnx.checker.check_model(path, full_check=True)
        sessions[name] = _session(path)

    seed_name = directional.graph.inputs[-1].name
    point = np.array([0.4, -0.6, 0.2], dtype=np.float32)
    prediction = np.tanh(point @ WEIGHT + BIAS)
    rng = np.random.default_rng(42)
    cases = []
    for count in (1, 3, 7):
        directions = rng.normal(size=(count, 3)).astype(np.float32)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        feeds = {"features": point, seed_name: directions}
        primal, products = sessions["batched_jvp"].run(None, feeds)
        # Analytic directional derivative, without constructing a Jacobian.
        expected = (directions @ WEIGHT) * (1 - prediction**2)
        np.testing.assert_allclose(products, expected, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(
            primal, np.broadcast_to(prediction, (count, 2)), rtol=1e-4, atol=1e-6
        )
        sequential = [
            sessions["jvp"].run(None, {"features": point, seed_name: direction})
            for direction in directions
        ]
        for index, actual in enumerate((primal, products)):
            np.testing.assert_allclose(
                actual, np.stack([row[index] for row in sequential]), rtol=1e-4, atol=1e-6
            )
        cases.append(
            {
                "directions": count,
                "product_shape": list(products.shape),
                "sensitivity_norms": np.linalg.norm(products, axis=1).tolist(),
                "maximum_absolute_error": float(np.max(np.abs(products - expected))),
            }
        )
    np.testing.assert_allclose(
        sessions["forward"].run(None, {"features": point})[0], prediction, rtol=1e-4, atol=1e-6
    )
    assert ir.to_proto(model).SerializeToString() == before
    np.savez(
        os.path.join(output_dir, "directions.npz"),
        **feeds,
        prediction=primal,
        products=products,
    )
    report = {
        "version": 1,
        "dtype": "float32",
        "provider": "CPUExecutionProvider",
        "ort_graph_optimizer": False,
        "input": "features",
        "direction_input": seed_name,
        "outputs": [value.name for value in batched.graph.outputs],
        "cases": cases,
    }
    with open(os.path.join(output_dir, "report.json"), "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="vmap_sensitivity_models")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
