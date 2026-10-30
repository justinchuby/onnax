# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Streaming materialization and native classifier microbatch training."""

import os

import numpy as np
import pytest

from examples.lora.accumulation import accumulate_gradients
from examples.mlp_microbatch import run, slices


def test_materialize_each_chunk_before_requesting_next():
    observations = []

    def chunks():
        for i, count in enumerate([3, 2, 1]):
            assert len(observations) == i
            yield count, np.array(i + 1, np.float32), {"p": np.full(2, i + 1, np.float32)}

    def evaluate(loss_sum, gradient_sum):
        observations.append((loss_sum.copy(), gradient_sum.copy()))

    loss, gradients = accumulate_gradients(chunks(), xp=np, evaluate=evaluate)
    assert len(observations) == 3
    np.testing.assert_allclose(loss, 10 / 6)
    np.testing.assert_allclose(gradients["p"], [10 / 6, 10 / 6])
    np.testing.assert_array_equal(observations[-1][1], [10, 10])


def test_tail_batch_partition():
    parts = slices(1024, 300)
    assert [(p.start, p.stop) for p in parts] == [
        (0, 300),
        (300, 600),
        (600, 900),
        (900, 1024),
    ]
    assert slices(3, 10) == [slice(0, 3)]
    for size in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            slices(5, size)


@pytest.mark.skipif(os.environ.get("ONNAX_TEST_MLX_EP") != "1", reason="Native MLX opt-in")
def test_medium_metal_accumulation_training_and_reload(tmp_path):
    report = run(tmp_path, measure=False)
    assert report["batch_sizes"] == [300, 300, 300, 124]
    assert report["same_parameter_acceptance"]
    assert report["checkpoint_replay"] and report["masters_moments_on_metal"]
    assert report["steps"] == 150
    assert report["final_validation_loss"] < report["initial_validation_loss"] * 0.5
