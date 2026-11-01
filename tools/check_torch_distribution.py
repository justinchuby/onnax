# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Run outside the checkout against an installed wheel."""

import tempfile
from pathlib import Path  # ruff: ignore[banned-api] - Installed distribution smoke script.

import numpy as np
import onnx_ir as ir
import torch

import onnax
from onnax.torch import OnnxModule

assert "site-packages" in str(Path(onnax.__file__).resolve())
x = ir.val("x", dtype=ir.DataType.FLOAT, shape=["batch", 2])
w = ir.val("weight", dtype=ir.DataType.FLOAT, shape=[2, 1])
tape = ir.tape.Tape()
y = tape.op("MatMul", [x, w])
y.name = "prediction"
source = ir.Model(
    ir.Graph([x, w], [y], nodes=tape.nodes, opset_imports={"": 18}, name="installed"),
    ir_version=10,
)
with tempfile.TemporaryDirectory() as directory:
    with OnnxModule(source, {"weight": np.ones((2, 1), np.float32)}) as model:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        values = torch.ones(3, 2)
        initial = model(x=values).square().mean().item()
        for _ in range(10):
            optimizer.zero_grad()
            model(x=values).square().mean().backward()
            optimizer.step()
        assert model(x=values).square().mean().item() < initial * 0.01
        expected = model(x=values).detach()
        model.export(Path(directory) / "bundle")
    with OnnxModule.load(Path(directory) / "bundle") as restored:
        torch.testing.assert_close(restored(x=values), expected, rtol=0, atol=0)
print("Installed Torch module: training, export and reload passed")
