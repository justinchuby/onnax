# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Run outside the checkout against an installed wheel."""

import tempfile
from pathlib import Path  # ruff: ignore[banned-api] - Installed distribution smoke script.

import numpy as np
import onnx_ir as ir
import torch

import onnax
from onnax.torch import CheckpointedModule, OnnxModule

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

with tempfile.TemporaryDirectory() as directory:
    with CheckpointedModule(
        source,
        {"weight": np.ones((2, 1), np.float32)},
        boundaries={"projection": "prediction"},
    ) as model:
        value = model(x=torch.ones(3, 2))
        value.square().mean().backward()
        assert model.trainable[0].grad is not None
        model.export(Path(directory) / "checkpoints")
    with CheckpointedModule.load(Path(directory) / "checkpoints") as restored:
        torch.testing.assert_close(restored(x=torch.ones(3, 2)), value, rtol=0, atol=0)
        restored(x=torch.ones(3, 2)).sum().backward()
print("Installed checkpoint module: backward, export and reload passed")

saved_source = source.clone()
projection = saved_source.graph.outputs[0]
activation = ir.val("activation", dtype=ir.DataType.FLOAT, shape=["batch", 1])
saved_source.graph.append(ir.node("Tanh", [projection], outputs=[activation]))
saved_source.graph.outputs.clear()
saved_source.graph.outputs.append(activation)
with tempfile.TemporaryDirectory() as directory:
    with OnnxModule(
        saved_source, {"weight": np.full((2, 1), 0.1, np.float32)}, save_activations=True
    ) as model:
        values = torch.ones(3, 2, requires_grad=True)
        actual = model(x=values)
        actual.square().mean().backward()
        weight = model.trainable[0].detach().clone().requires_grad_()
        reference = torch.tanh(values.detach() @ weight)
        reference.square().mean().backward()
        torch.testing.assert_close(model.trainable[0].grad, weight.grad)
        model.export(Path(directory) / "saved")
    with OnnxModule.load(Path(directory) / "saved") as restored:
        restored(x=values.detach()).square().mean().backward()
        torch.testing.assert_close(restored.trainable[0].grad, weight.grad)
print("Installed saved-activation module: gradients, export and reload passed")
