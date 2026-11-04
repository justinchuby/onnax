# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Inspect LoRA curvature using exported FP32 Hessian and Gauss-Newton products."""

import argparse
import json
from pathlib import Path  # ruff: ignore[banned-api] - Example artifact paths.

import numpy as np
import onnx
import onnx_ir as ir
from onnxscript import optimizer

import onnax
from examples.lora.model import PARAMETERS, adapt, base_model, with_loss
from examples.lora.train import problem, session


def interface(model):
    """Read collision-resolved derivative names from the exported graph."""
    return json.loads(model.graph.metadata_props["onnax.autodiff.interface"])


def dot(left, right):
    """Pair named parameter blocks without constructing a dense matrix."""
    return sum(float(np.vdot(a, b)) for a, b in zip(left, right, strict=True))


def reference_products(x, target, params, weights, head, direction):
    """Use independent PyTorch FP32 differentiation for both products."""
    import torch

    a, b = (torch.tensor(params[name], requires_grad=True) for name in PARAMETERS)
    xx, yy, ww, cc = map(torch.from_numpy, (x, target, weights, head))
    vectors = tuple(torch.from_numpy(v) for v in direction)

    def predict(a, b):
        return torch.tanh(xx @ ww + 0.25 * ((xx @ a) @ b)) @ cc

    prediction = predict(a, b)
    loss = (prediction - yy).square().mean()
    gradient = torch.autograd.grad(loss, (a, b), create_graph=True)
    hessian = torch.autograd.grad(
        sum((g * v).sum() for g, v in zip(gradient, vectors, strict=True)), (a, b)
    )
    _, tangent = torch.autograd.functional.jvp(predict, (a, b), vectors)
    gauss_newton = torch.autograd.grad(
        predict(a, b), (a, b), grad_outputs=tangent * (2 / target.size)
    )
    return tuple(
        [value.detach().numpy() for value in values] for values in (hessian, gauss_newton)
    )


def run(output_dir):
    """Export products, validate zero/nonzero adapters, and find initial negative curvature."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    w, c, x, target, _, _, initial = problem()
    source = with_loss(adapt(base_model(w, c)))
    original = ir.to_proto(source).SerializeToString()
    gradient = onnax.grad(source, inputs=list(PARAMETERS), output="loss")
    graphs = {
        "hvp": onnax.jvp(gradient, inputs=list(PARAMETERS)),
        "jvp": onnax.jvp(source, inputs=list(PARAMETERS), outputs=["prediction"]),
        "vjp": onnax.vjp(source, inputs=list(PARAMETERS), outputs=["prediction"]),
    }
    executors = {}
    for name, graph in graphs.items():
        for suffix, value in (("", graph), (".optimized", optimizer.optimize(graph.clone()))):
            path = output_dir / f"{name}{suffix}.onnx"
            ir.save(value, path)
            onnx.checker.check_model(path, full_check=True)
            executors[name + suffix] = session(path)

    def execute(name, feeds, direction):
        record = interface(graphs[name.split(".")[0]])
        values = direction if name.startswith(("hvp", "jvp")) else [direction]
        seeds = graphs[name.split(".")[0]].graph.inputs[-len(values) :]
        return executors[name].run(
            record["derivative_outputs"],
            {**feeds, **{seed.name: v for seed, v in zip(seeds, values, strict=True)}},
        )

    def products(feeds, direction, suffix=""):
        exact = execute("hvp" + suffix, feeds, direction)
        (tangent,) = execute("jvp" + suffix, feeds, direction)
        cotangent = tangent * np.float32(2 / feeds["target"].size)
        approximate = execute("vjp" + suffix, feeds, cotangent)
        return exact, approximate

    rng = np.random.default_rng(42)
    states = [initial, {name: value.copy() for name, value in initial.items()}]
    states[1][PARAMETERS[1]] = rng.normal(0, 0.02, initial[PARAMETERS[1]].shape).astype(
        np.float32
    )
    checks = 0
    for params in states:
        for batch in (1, 7, 32):
            feeds = {"features": x[:batch], "target": target[:batch], **params}
            direction = [
                rng.normal(size=params[n].shape).astype(np.float32) for n in PARAMETERS
            ]
            reference = reference_products(x[:batch], target[:batch], params, w, c, direction)
            for suffix in ("", ".optimized"):
                actual = products(feeds, direction, suffix)
                for got, wanted in zip(actual, reference, strict=True):
                    for a, b in zip(got, wanted, strict=True):
                        if not np.isfinite(a).all():
                            raise AssertionError("Nonfinite curvature product")
                        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)
                checks += 1

    feeds = {"features": x[:32], "target": target[:32], **initial}
    vb = rng.normal(size=initial[PARAMETERS[1]].shape).astype(np.float32)
    vb /= np.linalg.norm(vb)
    ha, hb = execute("hvp", feeds, [np.zeros_like(initial[PARAMETERS[0]]), vb])
    cross_squared = dot([ha], [ha])
    if cross_squared <= 0:
        raise AssertionError("Expected nonzero initial A/B cross curvature")
    scale = (abs(dot([vb], [hb])) + 0.001) / (2 * cross_squared)
    direction = [-np.float32(scale) * ha, vb]
    norm = np.sqrt(dot(direction, direction))
    direction = [v / np.float32(norm) for v in direction]
    exact, approximate = products(feeds, direction)
    curvature, ggn_curvature = dot(direction, exact), dot(direction, approximate)
    if not curvature < 0 or not ggn_curvature >= 0:
        raise AssertionError("Expected an indefinite Hessian and nonnegative GGN product")
    zero_seeds = {
        value.name: np.zeros_like(initial[name])
        for value, name in zip(graphs["hvp"].graph.inputs[-2:], PARAMETERS, strict=True)
    }
    (a_gradient,) = executors["hvp"].run(
        interface(gradient)["derivative_outputs"][:1], {**feeds, **zero_seeds}
    )
    np.testing.assert_array_equal(a_gradient, np.zeros_like(initial[PARAMETERS[0]]))
    if ir.to_proto(source).SerializeToString() != original:
        raise AssertionError("Source graph or frozen weights changed")
    report = dict(
        dtype="float32",
        parameter_count=sum(value.size for value in initial.values()),
        reference_cases=checks,
        full_checker_graphs=len(executors),
        initial_a_gradient_zero=True,
        cross_block_norm=float(np.sqrt(cross_squared)),
        negative_direction_curvature=curvature,
        gauss_newton_direction_curvature=ggn_curvature,
        reference="Independent PyTorch FP32 HVP and JVP/VJP",
        rtol=1e-4,
        atol=1e-6,
        ort_graph_optimizer=False,
    )
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    """Run the offline curvature diagnostic and write ordinary ONNX products."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
