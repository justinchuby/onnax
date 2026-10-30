# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Bounded-token ONNX LM-head loss and hidden-state derivatives."""

import tempfile

import numpy as np
import onnx_ir as ir
import torch
from onnx_ir.passes.common import ShapeInferencePass
from torch.autograd.function import once_differentiable

import onnax
from examples.huggingface.bridge import graph_session
from examples.huggingface.optimization import optimize_training_graph
from examples.huggingface.stable_losses import hard_cross_entropy


def head_graph(weight):
    width, vocab = weight.shape
    hidden = ir.val("hidden", dtype=ir.DataType.FLOAT, shape=["tokens", width])
    labels = ir.val("labels", dtype=ir.DataType.INT64, shape=["tokens"])
    tape = ir.tape.Tape()

    w = tape.initializer(ir.tensor(weight, name="head.weight"))
    logits = tape.op("MatMul", [hidden, w])
    logits.dtype, logits.shape = hidden.dtype, ir.Shape(["tokens", vocab])
    graph = ir.Graph(
        [hidden, labels],
        [],
        nodes=tape.nodes,
        initializers=tape.initializers,
        opset_imports={"": 24},
        name="head_loss",
    )
    ctx = onnax.AutodiffContext(graph)
    losses = hard_cross_entropy(ctx, logits, labels)
    total = ctx.op("ReduceSum", losses, keepdims=0)
    total.name, total.dtype, total.shape = "loss_sum", ir.DataType.FLOAT, ir.Shape([])
    graph.outputs.append(total)
    model = ir.Model(graph, ir_version=12)
    ShapeInferencePass()(model)
    return model


class _ChunkLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, runtime, hidden, labels):
        array = hidden.detach().contiguous().numpy()
        # Only positions predicting a following token participate in the loss.
        rows = array[:, :-1].reshape(-1, array.shape[-1])
        targets = labels[:, 1:].contiguous().numpy().reshape(-1)
        gradient = np.zeros_like(rows)
        total = np.float32(0)
        for start in range(0, len(rows), runtime.chunk_size):
            stop = min(start + runtime.chunk_size, len(rows))
            feeds = {"hidden": rows[start:stop], "labels": targets[start:stop]}
            value, derivative = runtime.executor.run(None, feeds)
            total += np.float32(value)
            gradient[start:stop] = derivative
        full = np.zeros_like(array)
        full[:, :-1] = gradient.reshape(array.shape[0], array.shape[1] - 1, array.shape[2])
        ctx.save_for_backward(torch.from_numpy(full))
        return torch.tensor(total, dtype=torch.float32)

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        (gradient,) = ctx.saved_tensors
        return None, gradient * upstream, None


class ChunkedHead:
    def __init__(self, weight, *, chunk_size=128):
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        self.chunk_size = chunk_size
        self.vocab_size = weight.shape[1]
        self.model = head_graph(weight)
        self.gradient = optimize_training_graph(
            onnax.grad(self.model, inputs=["hidden"], output="loss_sum")
        )
        self.gradient.graph.sort()
        self.directory = tempfile.TemporaryDirectory(prefix="onnax-head-")
        self.executor = graph_session(self.gradient, self.directory.name, "gradient")
        forward = self.model.clone()
        logits = next(n.outputs[0] for n in forward.graph if n.op_type == "MatMul")
        logits.name = "logits"
        forward.graph.outputs.clear()
        forward.graph.outputs.append(logits)
        forward.graph.inputs.pop()  # labels are unnecessary for prediction
        from onnxscript import optimizer

        optimizer.remove_unused_nodes(forward)
        self.forward_executor = graph_session(forward, self.directory.name, "forward")

    def forward(self, hidden):
        array = hidden.detach().contiguous().numpy()
        logits = self.forward_executor.run(
            None, {"hidden": array.reshape(-1, array.shape[-1])}
        )[0]
        return torch.from_numpy(logits.reshape(*array.shape[:-1], logits.shape[-1]))

    def __call__(self, hidden, labels, num_items=None):
        if hidden.device.type != "cpu" or hidden.dtype != torch.float32 or hidden.ndim != 3:
            raise ValueError("Expected CPU FP32 hidden states [batch,sequence,width]")
        if (
            labels.device.type != "cpu"
            or labels.dtype != torch.int64
            or labels.shape != hidden.shape[:2]
        ):
            raise ValueError("Expected CPU INT64 labels matching batch and sequence")
        if torch.any(((labels < 0) & (labels != -100)) | (labels >= self.vocab_size)):
            raise ValueError("Labels must be vocabulary IDs or -100")
        total = _ChunkLoss.apply(self, hidden, labels)
        count = (
            labels[:, 1:].ne(-100).sum() if num_items is None else torch.as_tensor(num_items)
        )
        return total / count.clamp(min=1)
