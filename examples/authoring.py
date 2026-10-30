# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Shared execution and numerical checks for the two graph-authoring tutorials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path  # ruff: ignore[banned-api] - Example artifact paths.

import numpy as np
import onnx
import onnx_ir as ir
from onnx.reference import ReferenceEvaluator
from onnxscript import optimizer


def reference(features, target, weight, bias):
    prediction = features @ weight + bias
    residual = prediction - target
    return (
        prediction,
        np.mean(residual * residual),
        np.float32(2 / len(features)) * (features.T @ residual),
        np.asarray(2 * residual.mean(), np.float32),
    )


def run(models, output_dir):
    proto, graph, gradient = models
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    original = proto.SerializeToString()
    onnx.save(proto, output_dir / "source.onnx")
    onnx.checker.check_model(proto, full_check=True)
    before = ir.to_proto(graph).SerializeToString()
    artifacts = {
        "prepared": graph,
        "gradient": gradient,
        "gradient.optimized": optimizer.optimize(gradient.clone()),
    }
    for name, artifact in artifacts.items():
        path = output_dir / f"{name}.onnx"
        ir.save(artifact, path)
        onnx.checker.check_model(str(path), full_check=True)
    names = [value.name for value in gradient.graph.outputs]
    interface = {
        "version": 1,
        "prediction": names[0],
        "loss": names[1],
        "parameters": [
            {"input": "weight", "gradient": names[2], "shape": [2], "dtype": "float32"},
            {"input": "bias", "gradient": names[3], "shape": [], "dtype": "float32"},
        ],
    }
    (output_dir / "interface.json").write_text(json.dumps(interface, indent=2) + "\n")
    execute = ReferenceEvaluator(str(output_dir / "gradient.optimized.onnx"))
    features = np.array(
        [[1, 0], [0, 1], [1, 1], [2, -1], [-1, 2], [0.5, 0.5], [-0.5, -1]], np.float32
    )
    target = features @ np.array([1.5, -0.5], np.float32) + np.float32(0.7)
    weight, bias = np.array([0.2, -0.1], np.float32), np.array(0.3, np.float32)
    ref_weight, ref_bias = weight.copy(), bias.copy()
    history = []
    for _ in range(20):
        actual = execute.run(
            names, {"features": features, "target": target, "weight": weight, "bias": bias}
        )
        expected = reference(features, target, weight, bias)
        for av, ev in zip(actual, expected, strict=True):
            np.testing.assert_allclose(av, ev, rtol=1e-4, atol=1e-6)
        independent = reference(features, target, ref_weight, ref_bias)
        np.testing.assert_allclose(actual[1], independent[1], rtol=1e-4, atol=1e-6)
        weight = weight - np.float32(0.1) * actual[2]
        bias = bias - np.float32(0.1) * actual[3]
        ref_weight = ref_weight - np.float32(0.1) * independent[2]
        ref_bias = ref_bias - np.float32(0.1) * independent[3]
        np.testing.assert_allclose(weight, ref_weight, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(bias, ref_bias, rtol=1e-4, atol=1e-6)
        history.append(float(actual[1]))
    for batch in (1, 3, 7):
        feeds = {
            "features": features[:batch],
            "target": target[:batch],
            "weight": weight,
            "bias": bias,
        }
        expected = reference(features[:batch], target[:batch], weight, bias)
        for name in ("gradient", "gradient.optimized"):
            loaded = ReferenceEvaluator(str(output_dir / f"{name}.onnx"))
            for av, ev in zip(loaded.run(names, feeds), expected, strict=True):
                np.testing.assert_allclose(av, ev, rtol=1e-4, atol=1e-6)
    np.savez(output_dir / "parameters.npz", weight=weight, bias=bias)
    with np.load(output_dir / "parameters.npz") as saved:
        reloaded = ReferenceEvaluator(str(output_dir / "prepared.onnx"))
        prediction, loss = reloaded.run(
            None, {"features": features, "target": target, **dict(saved)}
        )
        np.testing.assert_allclose(
            prediction, reference(features, target, weight, bias)[0], rtol=1e-4, atol=1e-6
        )
    assert float(loss) < history[0] * 0.5
    assert original == proto.SerializeToString()
    assert before == ir.to_proto(graph).SerializeToString()
    report = {
        "steps": 20,
        "initial_loss": history[0],
        "final_loss": float(loss),
        "weight": weight.tolist(),
        "bias": float(bias),
        "loss_history": history,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run_cli(differentiate, default_directory):
    parser = argparse.ArgumentParser(
        description="Export and execute parameter gradients of an authored ONNX model."
    )
    parser.add_argument("--output-dir", type=Path, default=Path(default_directory))
    args = parser.parse_args()
    result = run(differentiate(), args.output_dir)
    print(json.dumps({k: v for k, v in result.items() if k != "loss_history"}, indent=2))
