# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Independent numerical checks of the checked-in browser artifacts."""

import json
from pathlib import Path  # ruff: ignore[banned-api] - demo artifact tests.

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnx.reference import ReferenceEvaluator
from onnx_ir.passes.common import CheckerPass

from examples.jelly.export_models import (
    EPSILON,
    OUTPUT,
    energy_difference_model,
    energy_model,
    graph_summary,
    mesh,
)
from onnax import grad, jvp


@pytest.fixture(scope="module")
def sessions():
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return {
        name: ort.InferenceSession(
            str(OUTPUT / f"{name}.onnx"), options, providers=["CPUExecutionProvider"]
        )
        for name in ("energy", "gradient", "hvp")
    }


def feeds(seed=5):
    rng = np.random.default_rng(seed)
    positions, _, _ = mesh()
    n = len(positions)
    free = np.ones_like(positions)
    free[[0, 5]] = 0
    return {
        "q": (positions + rng.normal(0, 0.04, positions.shape)).astype(np.float32),
        "free": free,
        "anchor": positions * (1 - free),
        "diagonal": np.full((n, 2), 25, dtype=np.float32),
        "target": positions.copy(),
        "load": np.tile(np.array([0, 0.04 * 9.81], dtype=np.float32), (n, 1)),
        "stiffness": np.array(130, dtype=np.float32),
    }


def analytic(data, direction):
    """Analytic spring derivatives for tests only, independent of graph rules."""
    _, _, incidence = mesh()
    b = incidence.astype(np.float64)
    reference = mesh()[0].astype(np.float64)
    x = data["free"] * data["q"] + data["anchor"]
    delta = b @ x
    rest = np.sqrt(np.sum((b @ reference) ** 2, axis=1, keepdims=True) + EPSILON**2)
    length = np.sqrt(np.sum(delta**2, axis=1, keepdims=True) + EPSILON**2)
    k = float(data["stiffness"])
    stretch = length - rest
    offset = x - data["target"]
    energy = (
        0.5 * k * np.sum(stretch**2)
        + 0.5 * np.sum(data["diagonal"] * offset**2)
        - np.sum(data["load"] * x)
    )
    gradient = data["free"] * (
        b.T @ (k * (1 - rest / length) * delta) + data["diagonal"] * offset - data["load"]
    )
    d = b @ (direction * data["free"])
    response = k * (
        (1 - rest / length) * d
        + rest / length**3 * delta * np.sum(delta * d, axis=1, keepdims=True)
    )
    hvp = data["free"] * (b.T @ response + data["diagonal"] * direction * data["free"])
    return energy, gradient, hvp


@pytest.mark.parametrize("case", ["deformed", "rest", "compressed", "collapsed", "stiff"])
def test_force_hvp_and_symmetry(sessions, case):
    data = feeds()
    if case == "rest":
        data["q"] = mesh()[0]
    elif case == "compressed":
        data["q"] *= 0.3
    elif case == "collapsed":
        data["q"][:] = 0
    elif case == "stiff":
        data["stiffness"] = np.array(8000, dtype=np.float32)
    rng = np.random.default_rng(42)
    u = rng.normal(size=data["q"].shape).astype(np.float32)
    v = rng.normal(size=data["q"].shape).astype(np.float32)
    result = sessions["hvp"].run(None, {**data, "q_tangent": v})
    expected_energy, expected_gradient, expected_hvp = analytic(data, v)
    np.testing.assert_allclose(result[1], expected_energy, rtol=2e-5, atol=3e-4)
    np.testing.assert_allclose(result[2], expected_gradient, rtol=5e-4, atol=0.003)
    np.testing.assert_allclose(result[3], expected_hvp, rtol=5e-4, atol=0.02)
    assert np.isfinite(result[3]).all()
    np.testing.assert_array_equal(result[2][[0, 5]], 0)
    np.testing.assert_array_equal(result[3][[0, 5]], 0)
    hu = sessions["hvp"].run(None, {**data, "q_tangent": u})[-1]
    np.testing.assert_allclose(np.vdot(u, result[3]), np.vdot(v, hu), rtol=2e-5, atol=0.005)


def test_finite_differences(sessions):
    data = feeds()
    direction = np.random.default_rng(8).normal(size=data["q"].shape).astype(np.float32)
    step = 2e-4
    plus = {**data, "q": data["q"] + step * direction}
    minus = {**data, "q": data["q"] - step * direction}
    p = sessions["gradient"].run(None, plus)
    m = sessions["gradient"].run(None, minus)
    base = sessions["hvp"].run(None, {**data, "q_tangent": direction})
    np.testing.assert_allclose(
        (p[1] - m[1]) / (2 * step), np.vdot(base[2], direction), rtol=0.002, atol=0.002
    )
    np.testing.assert_allclose((p[2] - m[2]) / (2 * step), base[3], rtol=0.003, atol=0.08)


def test_lowered_artifacts_match_current_source(sessions):
    source = energy_model()
    model = jvp(grad(source, inputs=["q"], output="energy"), inputs=["q"])
    data = feeds()
    data["q_tangent"] = np.full_like(data["q"], 0.3)
    expected = ReferenceEvaluator(ir.to_proto(model)).run(None, data)
    actual = sessions["hvp"].run(None, data)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=1e-4)


def test_serialized_interfaces():
    manifest = json.loads((OUTPUT / "mesh.json").read_text())
    assert manifest["grid"] == 6
    assert len(manifest["edges"]) == 110
    assert manifest["opset"] == 22
    for name in ("energy", "gradient", "hvp"):
        path = OUTPUT / f"{name}.onnx"
        onnx.checker.check_model(str(path), full_check=True)
        model = ir.load(path)
        assert model.graph.opset_imports[""] == 22
        assert manifest["models"][name]["graph"] == graph_summary(model)
        CheckerPass(full_check=True)(model)
        assert [v.name for v in model.graph.inputs] == manifest["models"][name]["inputs"]
        assert [v.name for v in model.graph.outputs] == manifest["models"][name]["outputs"]
        assert all(not node.domain and node.op_type != "NonZero" for node in model.graph)
    assert Path(OUTPUT).is_dir()
    for name, info in manifest["runtime"].items():
        path = OUTPUT / "runtime" / f"{name}.onnx"
        model = ir.load(path)
        assert model.graph.opset_imports[""] == 22
        assert info["graph"] == graph_summary(model)
        assert info["nodes"] == len(model.graph)
        onnx.checker.check_model(str(path), full_check=True)


@pytest.mark.parametrize("case", ["deformed", "rest", "compressed", "collapsed", "stiff"])
def test_optimized_runtime_equivalence(sessions, case):
    data = feeds()
    if case == "rest":
        data["q"] = mesh()[0]
    elif case == "compressed":
        data["q"] *= 0.3
    elif case == "collapsed":
        data["q"][:] = 0
    elif case == "stiff":
        data["stiffness"] = np.array(8000, dtype=np.float32)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    for kind in ("energy", "gradient", "hvp"):
        path = OUTPUT / "runtime" / f"{kind}.onnx"
        model = ir.load(path)
        CheckerPass(full_check=True)(model)
        onnx.checker.check_model(str(path), full_check=True)
        runtime = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        inputs = dict(data)
        if kind == "hvp":
            inputs["q_tangent"] = (
                np.random.default_rng(42).normal(size=data["q"].shape).astype(np.float32)
            )
        expected = sessions[kind].run(None, inputs)
        if kind == "gradient":
            physical = {
                **inputs,
                "diagonal": inputs["diagonal"] * 0.1,
                "target": inputs["target"] + 0.03,
            }
            force = sessions[kind].run(None, physical)[-1]
            expected = np.concatenate([a.flatten() for a in [*expected, force]])
            inputs.update(
                physical_diagonal=physical["diagonal"], physical_target=physical["target"]
            )
        else:
            expected = expected[-1]
        actual = runtime.run(None, inputs)[0]
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=0.005)
        if kind != "energy":
            assert len(model.graph) <= (55 if kind == "gradient" else 45)
            assert "onnax.jelly.source_autodiff_interface" in model.graph.metadata_props


@pytest.mark.parametrize("case", ["deformed", "compressed", "collapsed", "zero"])
def test_gpu_cg_graph_matches_independent_matrix_free_cg(sessions, case):
    data = feeds()
    if case in ("compressed", "collapsed"):
        data["q"] *= 0.1 if case == "compressed" else 0
        data["diagonal"][:] = 0
    gradient = np.random.default_rng(3).normal(size=data["q"].shape).astype(np.float32)
    gradient *= data["free"]
    if case == "zero":
        gradient[:] = 0
    regularization = np.float32(0.001)
    r = -gradient.copy()
    d = r.copy()
    p = np.zeros_like(r)
    rr = np.vdot(r, r)
    threshold = max(1e-12, rr * 0.02**2)
    negative, iterations = False, 0
    for i in range(8):
        if rr <= threshold:
            break
        hd = sessions["hvp"].run(None, {**data, "q_tangent": d})[-1] + regularization * d
        curvature = np.vdot(d, hd)
        iterations += 1
        if curvature <= 1e-8 * np.vdot(d, d):
            negative = True
            if i == 0:
                p = -gradient
            break
        alpha = rr / curvature
        p += alpha * d
        r -= alpha * hd
        next_rr = np.vdot(r, r)
        d = r + next_rr / rr * d
        rr = next_rr
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    path = OUTPUT / "runtime" / "cg.onnx"
    CheckerPass(full_check=True)(ir.load(path))
    runtime = ReferenceEvaluator(str(path))
    result = runtime.run(
        None, {**data, "gradient": gradient, "regularization": np.array(regularization)}
    )[0]
    np.testing.assert_allclose(result[:-2].reshape(p.shape), p, rtol=0.003, atol=1e-4)
    assert result[-2] == iterations
    assert bool(result[-1]) == negative


@pytest.mark.parametrize("case", ["deformed", "collapsed", "stiff"])
def test_batched_proposals_equal_serial_generated_graphs(sessions, case):
    data = feeds()
    if case == "collapsed":
        data["q"][:] = 0
        data["free"][:] = 1
        data["anchor"][:] = 0
        data["diagonal"][:] = 0
    elif case == "stiff":
        data["stiffness"] = np.array(8000, dtype=np.float32)
    data.update(
        physical_diagonal=np.zeros_like(data["q"]), physical_target=np.zeros_like(data["q"])
    )
    data["regularization"] = np.array(0.001, dtype=np.float32)
    models = {
        kind: ir.load(OUTPUT / "runtime" / f"{kind}.onnx")
        for kind in ("gradient", "cg", "step", "batch")
    }
    for model in models.values():
        CheckerPass(full_check=True)(model)
        onnx.checker.check_model(ir.to_proto(model), full_check=True)
    runners = {kind: ReferenceEvaluator(ir.to_proto(model)) for kind, model in models.items()}

    def run(kind, values):
        names = {v.name for v in models[kind].graph.inputs}
        return runners[kind].run(None, {k: v for k, v in values.items() if k in names})[0]

    initial = run("gradient", data)
    n = data["q"].size
    stages = []
    current = dict(data)
    evaluation = initial
    for _ in range(2):
        current["gradient"] = evaluation[2 : n + 2].reshape(data["q"].shape)
        cg = run("cg", current)
        p = cg[:-2].reshape(data["q"].shape)
        if np.vdot(current["gradient"], p) >= 0:
            p = -current["gradient"]
        length = np.linalg.norm(p)
        if length > 0.45:
            p = p * np.float32(0.45 / length)
        candidate = current["q"] + p
        evaluation = run("gradient", {**current, "q": candidate})
        as_double = {k: v.astype(np.float64) for k, v in current.items() if k in feeds()}
        zero = np.zeros_like(candidate)
        change = (
            analytic({**as_double, "q": candidate.astype(np.float64)}, zero)[0]
            - analytic(as_double, zero)[0]
        )
        expected = np.concatenate((candidate.flatten(), evaluation, [change], cg[-2:]))
        actual = run("step", current)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=0.003)
        stages.append(actual)
        current["q"] = actual[:n].reshape(data["q"].shape)
        if cg[-1]:
            current["regularization"] = current["regularization"] * 10
    actual = run("batch", data)
    np.testing.assert_allclose(
        actual, np.concatenate((initial, *stages)), rtol=2e-5, atol=0.003
    )


@pytest.mark.parametrize("scale", [0.01, 1e-5, 1e-7])
def test_stable_energy_difference_at_float32_resolution(scale):
    model = energy_difference_model()
    CheckerPass(full_check=True)(model)
    data = feeds()
    old = data["q"].copy()
    data["q"] = old + np.random.default_rng(42).normal(0, scale, old.shape).astype(np.float32)
    actual = ReferenceEvaluator(ir.to_proto(model)).run(None, {**data, "reference_q": old})[0]
    double = {k: v.astype(np.float64) for k, v in data.items()}
    zero = np.zeros_like(old)
    expected = (
        analytic(double, zero)[0] - analytic({**double, "q": old.astype(np.float64)}, zero)[0]
    )
    np.testing.assert_allclose(actual, expected, rtol=0.001, atol=2e-8)
