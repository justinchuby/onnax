# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Matrix-free nonlinear fitting, derivative oracles and solver failure contracts."""

import json

import numpy as np
import pytest

from examples.fitting import cg, reference, run, solve


def test_complete_matrix_free_fitting(tmp_path):
    report = run(tmp_path)
    history = report["loss_history"]
    assert history[-1] < history[0] * 0.01
    assert np.all(np.diff(history) <= 0)
    np.testing.assert_allclose(report["parameters"], report["truth"], atol=0.01)
    assert report["measurement"]["peak_process_rss_bytes"] > 0
    for metric in report["measurement"]["timings_ms"].values():
        assert 0 < metric["median"] <= metric["p95"]
    interface = json.loads((tmp_path / "interface.json").read_text())
    assert interface["version"] == 1
    assert set(interface["graphs"]) == {"residual", "jvp", "vjp", "hvp", "batched_jvp"}


def test_analytic_reference_against_finite_differences():
    p = np.array([1.4, 0.8, 0.1])
    t = np.linspace(0, 3, 17)
    y = 2 * np.exp(-1.3 * t) + 0.4
    r, jacobian, hessian = reference(p, t, y)
    for direction in np.eye(3):
        plus, jp, _ = reference(p + 1e-5 * direction, t, y)
        minus, jm, _ = reference(p - 1e-5 * direction, t, y)
        np.testing.assert_allclose(
            (plus - minus) / 2e-5, jacobian @ direction, rtol=1e-8, atol=1e-9
        )
        np.testing.assert_allclose(
            (jp.T @ plus - jm.T @ minus) / 2e-5, hessian @ direction, rtol=1e-8, atol=1e-9
        )
    assert np.linalg.norm(hessian - jacobian.T @ jacobian) > 1
    assert np.linalg.norm(jacobian.T @ r) > 1


def test_cg_matches_dense_solution_and_handles_zero_rhs():
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    rhs = np.array([1.0, 2.0])
    result, iterations = cg(lambda v: matrix @ v, rhs)
    np.testing.assert_allclose(result, np.linalg.solve(matrix, rhs), atol=1e-12)
    assert iterations == 2
    result, iterations = cg(lambda v: matrix @ v, np.zeros(2))
    np.testing.assert_array_equal(result, np.zeros(2))
    assert iterations == 0
    with pytest.raises(ValueError, match="positive curvature"):
        cg(lambda v: -v, rhs)
    with pytest.raises(ValueError, match="positive curvature"):
        cg(lambda v: v * np.nan, rhs)
    with pytest.raises(RuntimeError, match="residual tolerance"):
        cg(lambda v: matrix @ v, rhs, max_iterations=1)


def test_line_search_failure_is_explicit():
    with pytest.raises(RuntimeError, match="line search"):
        solve(lambda p: np.ones(1), lambda p, v: np.ones(1), lambda p, v: v, np.zeros(1))
