# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Implicit hypergradients agree with dense solves and outer finite differences."""

import json

import numpy as np

from examples.implicit_tuning import run


def test_implicit_regularization_tuning(tmp_path):
    report = run(tmp_path)
    history = report["validation_loss"]
    assert len(history) == 31
    assert np.all(np.diff(history) < 0)
    assert history[-1] < history[0] * 0.05
    assert 0 < report["penalty"] < 0.1
    assert report["finite_difference_max_abs_error"] < 2e-5
    assert max(row["hypergradient_max_abs_error"] for row in report["checks"]) < 1e-6
    assert report["dtype"] == "float32"
    assert not report["ort_graph_optimizer"]
    assert set(report["interface"]) == {"gradient", "hvp", "mixed_jvp", "validation"}
    assert json.loads((tmp_path / "report.json").read_text()) == report
    for row in report["interface"].values():
        assert (tmp_path / row["file"]).is_file()
