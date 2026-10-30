# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Deterministic LoRA workloads for parameter-size and runtime comparisons."""

import numpy as np

from examples.lora.model import PARAMETERS
from examples.lora.train import problem, reference

# input, hidden, output, rank, training rows, validation rows
PROFILES = {
    "tiny": (16, 24, 8, 4, 256, 128),
    "medium": (256, 512, 64, 16, 512, 128),
    "large": (1024, 2048, 128, 128, 2048, 512),
}


def problem_for(profile):
    if profile not in PROFILES:
        raise ValueError(f"Expected a workload profile in {sorted(PROFILES)}")
    if profile == "tiny":
        return problem()
    width, hidden, outputs, rank, train_rows, validation_rows = PROFILES[profile]
    rng = np.random.default_rng(20261030)
    w = rng.normal(0, 0.4 / np.sqrt(width), (width, hidden)).astype(np.float32)
    c = rng.normal(0, 0.6 / np.sqrt(hidden), (hidden, outputs)).astype(np.float32)
    teacher = {
        PARAMETERS[0]: rng.normal(0, 0.15 / np.sqrt(width), (width, rank)).astype(np.float32),
        PARAMETERS[1]: rng.normal(0, 0.2, (rank, hidden)).astype(np.float32),
    }
    x = rng.normal(size=(train_rows + validation_rows, width)).astype(np.float32)
    y = reference(x, np.zeros((len(x), outputs), np.float32), teacher, w, c)[0]
    params = {
        PARAMETERS[0]: rng.normal(0, 0.1 / np.sqrt(width), (width, rank)).astype(np.float32),
        PARAMETERS[1]: np.zeros((rank, hidden), np.float32),
    }
    return w, c, x[:train_rows], y[:train_rows], x[train_rows:], y[train_rows:], params


def optimizer_epsilon(profile):
    if profile not in PROFILES:
        raise ValueError(f"Expected a workload profile in {sorted(PROFILES)}")
    # Mean losses over more output coordinates produce smaller gradients.
    # epsilon bounds the first-step sensitivity by lr / epsilon.
    return 1e-8 if profile == "tiny" else 1e-6
