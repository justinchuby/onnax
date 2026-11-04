# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Learn a regularization strength through a matrix-free implicit derivative."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import onnx
import onnx_ir as ir
from onnxscript import optimizer

import onnax
from examples.fitting import cg
from examples.lora.train import session


def objective(*, regularized):
    """Build a linear-regression loss with optional a scalar ridge penalty."""
    theta = ir.val("parameters", dtype=ir.DataType.FLOAT, shape=[8])
    x = ir.val("features", dtype=ir.DataType.FLOAT, shape=["samples", 8])
    y = ir.val("targets", dtype=ir.DataType.FLOAT, shape=["samples"])
    tape = ir.tape.Tape()
    residual = tape.op("Sub", [tape.op("MatMul", [x, theta]), y])
    half = tape.initializer(ir.tensor(np.array(0.5, np.float32), name="half"))
    loss = tape.op(
        "Mul",
        [half, tape.op("ReduceMean", [tape.op("Mul", [residual, residual])], {"keepdims": 0})],
    )
    inputs = [theta, x, y]
    if regularized:
        rho = ir.val("log_penalty", dtype=ir.DataType.FLOAT, shape=[])
        inputs.append(rho)
        penalty = tape.op("Mul", [tape.op("Exp", [rho]), tape.op("Mul", [theta, theta])])
        loss = tape.op(
            "Add",
            [loss, tape.op("Mul", [half, tape.op("ReduceSum", [penalty], {"keepdims": 0})])],
        )
    loss.name, loss.dtype, loss.shape = "loss", ir.DataType.FLOAT, ir.Shape([])
    return ir.Model(
        ir.Graph(
            inputs,
            [loss],
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": 18},
            name="ridge" if regularized else "validation",
        ),
        ir_version=10,
    )


def run(output_dir, *, steps=30):
    """Export derivative products, solve inner problems and tune their penalties."""
    os.makedirs(output_dir, exist_ok=True)
    train = objective(regularized=True)
    before = ir.to_proto(train).SerializeToString()
    gradient = onnax.grad(train, inputs=["parameters"], output="loss")
    gradient_name = gradient.graph.outputs[-1].name
    graphs = {
        "gradient": gradient,
        "hvp": onnax.jvp(gradient, inputs=["parameters"], outputs=[gradient_name]),
        "mixed_jvp": onnax.jvp(gradient, inputs=["log_penalty"], outputs=[gradient_name]),
        "validation": onnax.grad(
            objective(regularized=False), inputs=["parameters"], output="loss"
        ),
    }
    executors, interface = {}, {}
    for name, graph in graphs.items():
        deployed = optimizer.optimize(graph.clone())
        path = os.path.join(output_dir, name + ".onnx")
        ir.save(deployed, path)
        onnx.checker.check_model(path, full_check=True)
        executors[name] = session(path)
        interface[name] = {
            "file": name + ".onnx",
            "inputs": [v.name for v in deployed.graph.inputs],
            "outputs": [v.name for v in deployed.graph.outputs],
        }
    assert ir.to_proto(train).SerializeToString() == before
    rng = np.random.default_rng(20261104)
    truth = np.array([1.8, -1.2, 0.7, 0, 0, 0, 0, 0], np.float32)
    x = rng.normal(size=(64, 8)).astype(np.float32)
    xv = rng.normal(size=(96, 8)).astype(np.float32)
    y = x @ truth + rng.normal(scale=0.1, size=64).astype(np.float32)
    yv = xv @ truth + rng.normal(scale=0.1, size=96).astype(np.float32)
    zero = np.zeros(8, np.float32)
    rho = np.array(np.log(1.5), np.float32)
    history, checks = [], []

    def evaluate(name, parameters, penalties, *, direction=None, validation=False):
        feeds = {
            "parameters": parameters,
            "features": xv if validation else x,
            "targets": yv if validation else y,
            "log_penalty": np.asarray(penalties, dtype=np.float32),
        }
        if direction is not None:
            feeds[graphs[name].graph.inputs[-1].name] = direction
        return executors[name].run(
            None, {v.name: feeds[v.name] for v in executors[name].get_inputs()}
        )

    def inner(penalties):
        rhs = -evaluate("gradient", zero, penalties)[-1]

        def product(v):
            return evaluate("hvp", zero, penalties, direction=v)[-1]

        theta, iterations = cg(product, rhs, tolerance=2e-6, max_iterations=32)
        return theta, iterations

    for step in range(steps + 1):
        theta, inner_iterations = inner(rho)
        loss, validation_gradient = evaluate("validation", theta, rho, validation=True)
        u, adjoint_iterations = cg(
            lambda v, theta=theta, rho=rho: evaluate("hvp", theta, rho, direction=v)[-1],
            validation_gradient,
            tolerance=2e-6,
            max_iterations=32,
        )
        mixed = evaluate("mixed_jvp", theta, rho, direction=np.array(1, np.float32))[-1]
        hypergradient = -(u @ mixed)
        # Independent dense formulas validate the products and both linear solves.
        hessian = x.T @ x / np.float32(len(x)) + np.exp(rho) * np.eye(8, dtype=np.float32)
        reference_theta = np.linalg.solve(hessian, x.T @ y / np.float32(len(x)))
        reference_u = np.linalg.solve(
            hessian, xv.T @ (xv @ reference_theta - yv) / np.float32(len(xv))
        )
        reference_hypergradient = -np.exp(rho) * (reference_theta @ reference_u)
        np.testing.assert_allclose(mixed, np.exp(rho) * theta, rtol=1e-4, atol=1e-6)
        probe = np.arange(1, 9, dtype=np.float32) / 8
        np.testing.assert_allclose(
            evaluate("hvp", theta, rho, direction=probe)[-1],
            hessian @ probe,
            rtol=1e-4,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            loss, 0.5 * np.mean((xv @ theta - yv) ** 2), rtol=1e-4, atol=1e-6
        )
        train_loss, train_gradient = evaluate("gradient", theta, rho)
        np.testing.assert_allclose(
            train_loss,
            0.5 * (np.mean((x @ theta - y) ** 2) + np.exp(rho) * (theta @ theta)),
            rtol=1e-4,
            atol=1e-6,
        )
        assert np.linalg.norm(train_gradient) < 5e-6

        np.testing.assert_allclose(theta, reference_theta, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(u, reference_u, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(
            hypergradient, reference_hypergradient, rtol=1e-4, atol=1e-6
        )
        history.append(float(loss))
        checks.append(
            {
                "inner_cg_iterations": inner_iterations,
                "adjoint_cg_iterations": adjoint_iterations,
                "hypergradient_max_abs_error": float(
                    np.max(np.abs(hypergradient - reference_hypergradient))
                ),
            }
        )
        if step == 0:
            epsilon = np.float32(0.01)
            plus, _ = inner(rho + epsilon)
            minus, _ = inner(rho - epsilon)
            lp = evaluate("validation", plus, rho + epsilon, validation=True)[0]
            lm = evaluate("validation", minus, rho - epsilon, validation=True)[0]
            finite_difference = (lp - lm) / (2 * epsilon)
            np.testing.assert_allclose(hypergradient, finite_difference, rtol=1e-3, atol=1e-5)
            finite_difference_error = float(np.max(np.abs(hypergradient - finite_difference)))
        if step < steps:
            rho = rho - np.float32(1.0) * hypergradient
    assert history[-1] < history[0] * 0.5
    report = {
        "version": 1,
        "seed": 20261104,
        "training_samples": len(x),
        "validation_samples": len(xv),
        "dtype": "float32",
        "steps": steps,
        "validation_loss": history,
        "penalty": np.exp(rho).tolist(),
        "checks": checks,
        "finite_difference_max_abs_error": finite_difference_error,
        "runtime": "CPUExecutionProvider",
        "ort_graph_optimizer": False,
        "derivative_optimizer": "ONNXScript",
        "interface": interface,
    }
    with open(os.path.join(output_dir, "report.json"), "w") as stream:
        json.dump(report, stream, indent=2)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="implicit_tuning_models")
    args = parser.parse_args()
    result = run(args.output_dir)
    print(
        json.dumps(
            {
                "initial_validation_loss": result["validation_loss"][0],
                "final_validation_loss": result["validation_loss"][-1],
                "penalty": result["penalty"],
            },
            indent=2,
        )
    )
