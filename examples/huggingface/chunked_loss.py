# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Bounded-token ONNX LM-head loss and hidden-state derivatives."""

import json
import os
import tempfile

import numpy as np
import onnx_ir as ir
import torch
from onnx_ir.passes.common import ShapeInferencePass
from torch.autograd.function import once_differentiable

import onnax
from examples.huggingface.optimization import optimize_training_graph
from examples.huggingface.serialization import SharedWeights
from examples.huggingface.stable_losses import hard_cross_entropy, install
from examples.lora.train import session


def head_graph(weight, *, projection_precision="fp32"):
    if projection_precision not in ("fp32", "bf16"):
        raise ValueError("Expected fp32 or bf16 head projection precision")
    width, vocab = weight.shape
    hidden = ir.val("hidden", dtype=ir.DataType.FLOAT, shape=["tokens", width])
    labels = ir.val("labels", dtype=ir.DataType.INT64, shape=["tokens"])
    tape = ir.tape.Tape()

    w = tape.initializer(ir.tensor(weight, name="head.weight"))
    logits = tape.op("MatMul", [hidden, w])
    logits.name = "logits"
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
    install(model)
    ShapeInferencePass()(model)
    if projection_precision == "bf16":
        from examples.huggingface.mixed_precision import bf16_projections

        model = bf16_projections(model)
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
    def __init__(
        self, weight, *, chunk_size=128, projection_precision="fp32", session_factory=session
    ):
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        self.chunk_size = chunk_size
        self.projection_precision = projection_precision
        self.vocab_size = weight.shape[1]
        self.model = head_graph(weight, projection_precision=projection_precision)
        self.gradient = optimize_training_graph(
            onnax.grad(self.model, inputs=["hidden"], output="loss_sum")
        )
        self.gradient.graph.sort()
        self.directory = tempfile.TemporaryDirectory(prefix="onnax-head-")
        forward = self.model.clone()
        logits = ir.convenience.create_value_mapping(forward.graph)["logits"]
        forward.graph.outputs.clear()
        forward.graph.outputs.append(logits)
        forward.graph.inputs.pop()  # labels are unnecessary for prediction
        from onnxscript import optimizer

        optimizer.remove_unused_nodes(forward)
        # All three modules snapshot the same immutable weight payload. Keep the
        # prediction session lazy while retaining portable training artifacts.
        with SharedWeights(self.directory.name) as weights:
            sources = {
                "loss": weights.save(self.model, "loss"),
                "gradient": weights.save(self.gradient, "gradient"),
                "forward": weights.save(forward, "forward"),
            }
        self._load_sources(sources, session_factory)

    def _load_sources(self, sources, session_factory):
        self.model = ir.load(sources["loss"])
        self.gradient = ir.load(sources["gradient"])
        self.forward_model = ir.load(sources["forward"])
        weight = self.model.graph.initializers["head.weight"]
        if list(weight.shape)[1] != self.vocab_size:
            raise ValueError("Exported head vocabulary differs from its interface")
        policy = json.loads(self.model.metadata_props.get("onnax.example.precision", "{}"))
        actual_precision = "bf16" if policy.get("projection_compute") == "bfloat16" else "fp32"
        if actual_precision != self.projection_precision:
            raise ValueError("Output-head projection precision differs from its interface")
        self.executor = session_factory(sources["gradient"])
        self._forward_source = sources["forward"]
        self._session_factory = session_factory
        self.forward_executor = None

    def export_to(self, weights):
        """Store the head modules alongside the checkpointed body's shared weights."""
        sources = {
            "loss": weights.save(self.model, "head_loss"),
            "gradient": weights.save(self.gradient, "head_gradient"),
            "forward": weights.save(self.forward_model, "head_forward"),
        }
        return {
            "version": 1,
            "chunk_size": self.chunk_size,
            "projection_precision": self.projection_precision,
            "vocab_size": self.vocab_size,
            "reduction": "valid_shifted_token_mean",
            **{name: os.path.basename(path) for name, path in sources.items()},
        }

    @classmethod
    def load(cls, directory, *, session_factory=session):
        """Load a saved output head without constructing new derivative graphs."""
        with open(os.path.join(directory, "checkpoint_plan.json")) as stream:
            plan = json.load(stream)
        if plan.get("version") != 1 or "head" not in plan:
            raise ValueError("Expected a checkpoint execution plan with an output head")
        record = plan["head"]
        if record.get("version") != 1 or record.get("reduction") != "valid_shifted_token_mean":
            raise ValueError("Unsupported output-head interface")
        if (
            isinstance(record["chunk_size"], bool)
            or not isinstance(record["chunk_size"], int)
            or record["chunk_size"] < 1
            or record["projection_precision"] not in ("fp32", "bf16")
        ):
            raise ValueError("Invalid output-head execution options")
        result = cls.__new__(cls)
        result.chunk_size = record["chunk_size"]
        result.vocab_size = record["vocab_size"]
        result.projection_precision = record["projection_precision"]
        result._load_sources(
            {
                name: os.path.join(directory, record[name])
                for name in ("loss", "gradient", "forward")
            },
            session_factory,
        )
        return result

    def forward(self, hidden):
        array = hidden.detach().contiguous().numpy()
        if self.forward_executor is None:
            self.forward_executor = self._session_factory(self._forward_source)
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
