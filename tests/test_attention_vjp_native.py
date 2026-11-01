# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Opt-in native attention deployment qualification on a complete decoder."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional native workload dependencies.

import os

import numpy as np
import pytest

LIBRARY = os.environ.get("ONNAX_TEST_MLX_LIBRARY")
pytestmark = pytest.mark.skipif(
    not LIBRARY, reason="Set ONNAX_TEST_MLX_LIBRARY for native qualification"
)
pytest.importorskip("mobius")
pytest.importorskip("trl")
torch = pytest.importorskip("torch")

from examples.huggingface.sft import prepare
from examples.huggingface.validation import add_reference_adapters, audit
from examples.lora.mlx_runtime import register_library, session


@pytest.mark.parametrize("mode", ["full", "chunked", "joint"])
@pytest.mark.parametrize("dense_limit", [0, 64 * 1024**2])
def test_independent_native_updates(mode, dense_limit, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setenv("ONNAX_TILED_ATTENTION", "1")
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "1")
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_ATTENTION_VJP_DENSE_LIMIT_BYTES", str(dense_limit))
    register_library(LIBRARY)
    model, reference = prepare(
        mode=mode,
        nf4=True,
        fused_ops=True,
        checkpoint_layers=1 if mode == "chunked" else 0,
        session_factory=session,
    )
    reference = add_reference_adapters(reference, model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0)
    target_optimizer = torch.optim.AdamW(
        [p for p in reference.parameters() if p.requires_grad], lr=0.003, weight_decay=0
    )
    ids = torch.tensor([[1, 4, 5, 6, 2], [1, 7, 2, 0, 0]])
    mask = ids.ne(0).long()
    batch = {
        "input_ids": ids,
        "attention_mask": mask,
        "position_ids": (mask.cumsum(-1) - 1).clamp(min=0),
        "labels": ids.masked_fill(mask == 0, -100),
    }
    for _ in range(3):
        audit(model, reference, batch, sync=False)
        optimizer.step()
        target_optimizer.step()
        for row in model.mapping:
            layer = reference.get_submodule(row["module"])
            for key in ("A", "B"):
                np.testing.assert_allclose(
                    model.onnx.arrays()[row[key]],
                    getattr(layer, key).detach().numpy(),
                    rtol=1e-4,
                    atol=1e-6,
                )
