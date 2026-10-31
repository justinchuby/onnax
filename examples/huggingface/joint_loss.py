# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Execute a causal loss and selected parameter gradients in one ONNX session."""

import tempfile

import numpy as np
import onnx_ir as ir
import torch
from torch.autograd.function import once_differentiable

import onnax
from examples.huggingface.bridge import graph_session
from examples.huggingface.optimization import optimize_training_graph
from examples.huggingface.stable_losses import hard_cross_entropy, install
from examples.lora.train import session


def causal_loss_graph(forward):
    """Append summed, shifted hard-label CE to an FP32 logits model."""
    model = forward.clone()
    graph = model.graph
    values = ir.convenience.create_value_mapping(graph, include_subgraphs=False)
    if "labels" in values:
        raise ValueError("The forward graph already contains the name 'labels'")
    logits = next(v for v in graph.outputs if v.name == "logits")
    if logits.dtype != ir.DataType.FLOAT or logits.shape is None or len(logits.shape) != 3:
        raise ValueError("Expected FP32 logits with shape [batch, sequence, vocabulary]")
    vocab = logits.shape[-1]
    if not isinstance(vocab, int) or vocab < 1:
        raise ValueError("Expected a fixed positive vocabulary size")
    labels = ir.val("labels", dtype=ir.DataType.INT64, shape=list(logits.shape[:2]))
    graph.inputs.append(labels)
    ctx = onnax.AutodiffContext(graph)
    zero, one = ctx.constant(0), ctx.constant(1)
    sequence = ctx.op("Gather", ctx.op("Shape", logits), one, axis=0)
    length = ctx.op("Max", ctx.op("Sub", sequence, one), zero)
    indices = ctx.op("Range", zero, length, one)
    rows = ctx.op("Gather", logits, indices, axis=1)
    rows = ctx.op("Reshape", rows, ctx.constant([-1, vocab]))
    rows.type, rows.shape = ir.TensorType(ir.DataType.FLOAT), ir.Shape(["tokens", vocab])
    targets = ctx.op("Gather", labels, ctx.op("Add", indices, one), axis=1)
    targets = ctx.op("Reshape", targets, ctx.constant([-1]))
    targets.type, targets.shape = ir.TensorType(ir.DataType.INT64), ir.Shape(["tokens"])
    loss = ctx.op("ReduceSum", hard_cross_entropy(ctx, rows, targets), keepdims=0)
    loss.name, loss.type, loss.shape = ctx.name("loss_sum"), rows.type, ir.Shape([])
    graph.outputs.clear()
    graph.outputs.append(loss)
    install(model)
    return model


class _JointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, runtime, feeds, *parameters):
        values = runtime.executor.run(
            runtime.outputs,
            {
                **feeds,
                **{
                    n: p.detach().numpy()
                    for n, p in zip(runtime.names, parameters, strict=True)
                },
            },
        )
        gradients = [torch.from_numpy(v) for v in values[1:]]
        ctx.count = len(parameters)
        ctx.save_for_backward(*parameters, *gradients)
        return torch.from_numpy(values[0])

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        # Reading saved parameters also validates their mutation versions.
        gradients = ctx.saved_tensors[ctx.count :]
        return None, None, *(gradient * upstream for gradient in gradients)


class JointLoss(torch.nn.Module):
    """Own explicit CPU FP32 parameters; retain only returned gradients for backward."""

    def __init__(self, model, params, *, session_factory=session):
        super().__init__()
        self.model = model
        self.names = tuple(params)
        inputs = {v.name: v for v in model.graph.inputs}
        for name, array in params.items():
            if name not in inputs or name in model.graph.initializers:
                raise ValueError(f"Expected an explicit parameter input: {name}")
            if array.dtype != np.float32 or list(array.shape) != list(inputs[name].shape):
                raise ValueError(f"FP32 parameter shape mismatch: {name}")
        if len(model.graph.outputs) != 1 or model.graph.outputs[0].shape != ir.Shape([]):
            raise ValueError("Expected one scalar loss output")
        self.adapters = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.from_numpy(value.copy())) for value in params.values()]
        )
        self.gradient = optimize_training_graph(
            onnax.grad(model, inputs=list(self.names), output=model.graph.outputs[0].name)
        )
        self.outputs = [v.name for v in self.gradient.graph.outputs]
        if len(self.outputs) != 1 + len(self.names):
            raise ValueError("Expected one returned derivative per parameter")
        self.directory = tempfile.TemporaryDirectory(prefix="onnax-joint-loss-")
        self.executor = graph_session(
            self.gradient,
            self.directory.name,
            "loss_gradient",
            session_factory=session_factory,
        )

    def arrays(self):
        return {
            name: value.detach().numpy().copy()
            for name, value in zip(self.names, self.adapters, strict=True)
        }

    def forward(self, **feeds):
        if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in self.adapters):
            raise ValueError("Expected CPU FP32 adapter parameters")
        arrays = {}
        for name, value in feeds.items():
            if name in self.names:
                raise ValueError(f"Parameter {name} is owned by the module")
            if isinstance(value, torch.Tensor):
                if value.device.type != "cpu" or value.requires_grad:
                    raise ValueError("Feed tensors must be nondifferentiable CPU tensors")
                value = value.detach().contiguous().numpy()
            arrays[name] = value
        return _JointFunction.apply(self, arrays, *self.adapters)
