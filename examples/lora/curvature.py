# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Inspect and fit LoRA with exported FP32 Hessian and Gauss-Newton products."""

import argparse
import json
from pathlib import Path  # ruff: ignore[banned-api] - Example artifact paths.

import numpy as np
import onnx
import onnx_ir as ir
from onnxscript import optimizer

import onnax
from examples.fitting import cg
from examples.lora.model import PARAMETERS, adapt, base_model, with_loss
from examples.lora.train import problem, reference, session


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


def reference_gauss_newton(x, params, weights, head, direction):
    """Apply the prediction Jacobian and its transpose using NumPy formulas."""
    a, b = (params[name] for name in PARAMETERS)
    da, db = direction
    hidden = np.tanh(x @ weights + np.float32(0.25) * ((x @ a) @ b))
    slope = 1 - hidden * hidden
    dz = np.float32(0.25) * ((x @ da) @ b + (x @ a) @ db)
    tangent = (slope * dz) @ head
    cotangent = ((np.float32(2 / tangent.size) * tangent) @ head.T) * slope
    return [
        np.float32(0.25) * (x.T @ cotangent @ b.T),
        np.float32(0.25) * ((x @ a).T @ cotangent),
    ]


def fit(initial, evaluate, product, *, steps, damping=1e-4):
    """Update adapters with external damped CG and Armijo backtracking."""
    shapes = [initial[name].shape for name in PARAMETERS]
    split = initial[PARAMETERS[0]].size

    def unpack(vector):
        return {
            name: value.reshape(shape)
            for name, value, shape in zip(
                PARAMETERS, np.split(vector, [split]), shapes, strict=True
            )
        }

    vector = np.concatenate([initial[name].ravel() for name in PARAMETERS])
    history = []
    for _ in range(steps):
        params = unpack(vector)
        loss, gradients = evaluate(params)
        gradient = np.concatenate([g.ravel() for g in gradients])
        if np.linalg.norm(gradient) < 1e-7:
            break

        def action(direction, params=params):
            blocks = product(params, list(unpack(direction).values()))
            flat = np.concatenate([v.ravel() for v in blocks])
            return flat + np.float32(damping) * direction

        tolerance = max(1e-7, float(np.linalg.norm(gradient)) * 1e-3)
        delta, iterations = cg(
            action, -gradient, tolerance=tolerance, max_iterations=len(vector)
        )
        residual = float(np.linalg.norm(action(delta) + gradient))
        if residual > 2 * tolerance:
            raise AssertionError("Damped Gauss-Newton linear residual exceeds tolerance")
        for backtrack in range(20):
            scale = np.float32(0.5**backtrack)
            candidate = vector + scale * delta
            trial, _ = evaluate(unpack(candidate))
            if trial <= loss + 1e-4 * float(scale * (gradient @ delta)):
                break
        else:
            raise RuntimeError("Gauss-Newton backtracking did not find a descent step")
        if candidate.dtype != np.float32:
            raise AssertionError("Expected FP32 adapter updates")
        history.append(
            dict(
                loss=float(loss),
                next_loss=float(trial),
                cg_iterations=iterations,
                linear_residual=residual,
                linear_tolerance=tolerance,
                step_size=float(scale),
            )
        )
        vector = candidate
    return unpack(vector), history


def run(output_dir, *, fit_steps=12):
    """Inspect adapter curvature and fit the model with exported Gauss-Newton products."""
    if isinstance(fit_steps, bool) or not isinstance(fit_steps, int) or fit_steps < 0:
        raise ValueError("fit_steps must be a nonnegative integer")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    w, c, x, target, vx, vy, initial = problem()
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

    # Reuse the reloaded products in an external solver; parameters stay explicit inputs.
    gradient_names = interface(gradient)["derivative_outputs"]
    base_feeds = {"features": x, "target": target}
    solve_checks = []

    def evaluate(params, *, features=x, observed=target):
        prediction, loss, *gradients = executors["hvp.optimized"].run(
            ["prediction", "loss", *gradient_names],
            {"features": features, "target": observed, **params, **zero_seeds},
        )
        expected_prediction, expected_loss, expected_gradients = reference(
            features, observed, params, w, c
        )
        for actual, expected in zip(
            [prediction, loss, *gradients],
            [expected_prediction, expected_loss, *expected_gradients.values()],
            strict=True,
        ):
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)
        return float(loss), gradients

    def solve_product(params, direction):
        feeds = {**base_feeds, **params}
        (tangent,) = execute("jvp.optimized", feeds, direction)
        result = execute("vjp.optimized", feeds, tangent * np.float32(2 / target.size))
        expected = reference_gauss_newton(x, params, w, c, direction)
        for actual, wanted in zip(result, expected, strict=True):
            np.testing.assert_allclose(actual, wanted, rtol=1e-4, atol=1e-6)
        solve_checks.append(
            max(float(np.max(np.abs(a - b))) for a, b in zip(result, expected, strict=True))
        )
        return result

    fitted, history = fit(initial, evaluate, solve_product, steps=fit_steps)
    for row in history:
        if not row["next_loss"] < row["loss"]:
            raise AssertionError("Expected a descending Gauss-Newton update")
    initial_loss, _ = evaluate(initial)
    final_loss, _ = evaluate(fitted)
    validation = [evaluate(params, features=vx, observed=vy)[0] for params in (initial, fitted)]
    np.savez(output_dir / "adapter.npz", **fitted)
    with np.load(output_dir / "adapter.npz", allow_pickle=False) as archive:
        restored = {name: archive[name] for name in PARAMETERS}
    restored_loss, _ = evaluate(restored)
    if restored_loss != final_loss:
        raise AssertionError("Reloaded adapters changed loss")
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
        fit=dict(
            damping=1e-4,
            steps=history,
            train_loss=[initial_loss, final_loss],
            validation_loss=validation,
            adapter="adapter.npz",
            parameter_inputs=list(PARAMETERS),
            gradient_outputs=gradient_names,
            reference_products=len(solve_checks),
            product_max_abs_error=max(solve_checks, default=0.0),
        ),
    )
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    """Run FP32 curvature checks and external Gauss-Newton adapter updates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fit-steps", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, fit_steps=args.fit_steps), indent=2))


if __name__ == "__main__":
    main()
