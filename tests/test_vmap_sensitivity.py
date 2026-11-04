# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Exercise the runnable FP32 batched-direction tutorial after ONNX reload."""

import json

import numpy as np
import onnxruntime as ort

from examples.vmap_sensitivity import run


def test_exported_sensitivity_directions(tmp_path):
    report = run(str(tmp_path))
    assert [case["product_shape"] for case in report["cases"]] == [[1, 2], [3, 2], [7, 2]]
    assert report == json.loads((tmp_path / "report.json").read_text())
    with np.load(tmp_path / "directions.npz") as saved:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(tmp_path / "batched_jvp.onnx"), options, providers=["CPUExecutionProvider"]
        )
        # Replay a different direction count from the saved seven-direction input.
        prediction, product = session.run(
            None,
            {
                report["input"]: saved[report["input"]],
                report["direction_input"]: saved[report["direction_input"]][:2],
            },
        )
        np.testing.assert_array_equal(prediction, saved["prediction"][:2])
        np.testing.assert_allclose(product, saved["products"][:2], rtol=1e-4, atol=1e-6)
