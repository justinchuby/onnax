# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Offline Mobius DiT construction and example-local attention adapter injection."""

import numpy as np
import onnx_ir as ir
from mobius.models.dit import DiTConfig, DiTTransformer2DModel
from mobius.tasks import DenoisingTask
from onnx_ir.passes.common import ShapeInferencePass


def build():
    """Build a two-layer conditional latent denoiser without tracing or downloads."""
    config = DiTConfig(
        in_channels=4,
        out_channels=4,
        patch_size=2,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        cross_attention_dim=32,
        caption_channels=32,
        sample_size=8,
        act_fn="gelu",
    )
    model = DenoisingTask().build(DiTTransformer2DModel(config), config)["model"]
    rng = np.random.default_rng(20261102)
    weights = {}
    for name, value in model.graph.initializers.items():
        if value.const_value is not None:
            continue
        shape = tuple(value.shape)
        if name.endswith(".bias"):
            array = np.zeros(shape, np.float32)
        elif ".norm." in name and name.endswith(".weight"):
            array = np.ones(shape, np.float32)
        else:
            array = rng.normal(0, 0.06, shape).astype(np.float32)
        value.const_value = ir.tensor(array, name=name)
        weights[name] = array.copy()
    ShapeInferencePass()(model)
    return config, model, weights


def inject(model, *, rank=4, alpha=4.0):
    """Adapt the eight q/v projections in the offline two-layer DiT."""
    from examples.diffusion.adapters import inject as inject_attention

    result, params, mapping = inject_attention(model, rank=rank, alpha=alpha)
    if len(mapping) != 8:
        raise ValueError("Expected q/v adapters in both attention blocks of both layers")
    return result, params, mapping
