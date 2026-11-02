# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Bounded-token ONNX LM-head loss and hidden-state derivatives."""

import json
import os
import tempfile
import time

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


def head_graph(weight, *, projection_precision="fp32", weight_layout="hidden_vocab"):
    if projection_precision not in ("fp32", "bf16", "fp16"):
        raise ValueError("Expected fp32, bf16 or fp16 head projection precision")
    if weight_layout not in ("hidden_vocab", "vocab_hidden"):
        raise ValueError("Expected hidden_vocab or vocab_hidden weight layout")
    if weight_layout == "vocab_hidden" and projection_precision != "fp32":
        raise ValueError("The source-layout head currently uses FP32 projection compute")
    width, vocab = weight.shape if weight_layout == "hidden_vocab" else weight.shape[::-1]
    hidden = ir.val("hidden", dtype=ir.DataType.FLOAT, shape=["tokens", width])
    labels = ir.val("labels", dtype=ir.DataType.INT64, shape=["tokens"])
    tape = ir.tape.Tape()

    w = tape.initializer(ir.tensor(weight, name="head.weight"))
    logits = (
        tape.op("Gemm", [hidden, w], {"transB": 1})
        if weight_layout == "vocab_hidden"
        else tape.op("MatMul", [hidden, w])
    )
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
    if weight_layout != "hidden_vocab":
        model.metadata_props["onnax.example.head.weight_layout"] = weight_layout
    install(model)
    ShapeInferencePass()(model)
    if projection_precision != "fp32":
        from examples.huggingface.mixed_precision import precision_projections

        model = precision_projections(model, precision=projection_precision)
    return model


class _ChunkLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, runtime, hidden, labels):
        before = time.perf_counter()
        array = hidden.detach().numpy()
        # Shift before gathering: hidden[b, i] predicts labels[b, i + 1].
        targets = labels[:, 1:].numpy()
        boundary_seconds = time.perf_counter() - before
        before = time.perf_counter()
        batches, positions = np.nonzero(targets != -100)
        rows = array[batches, positions]
        selected = targets[batches, positions]
        gather_seconds = time.perf_counter() - before
        before = time.perf_counter()
        full = np.zeros_like(array)
        total = np.float32(0)
        scatter_seconds = time.perf_counter() - before
        execute_seconds, input_bytes, output_bytes, chunks = 0.0, 0, 0, 0
        for start in range(0, len(rows), runtime.chunk_size):
            stop = min(start + runtime.chunk_size, len(rows))
            feeds = {"hidden": rows[start:stop], "labels": selected[start:stop]}
            before = time.perf_counter()
            value, derivative = runtime.executor.run(None, feeds)
            execute_seconds += time.perf_counter() - before
            input_bytes += sum(value.nbytes for value in feeds.values())
            output_bytes += np.asarray(value).nbytes + derivative.nbytes
            chunks += 1
            before = time.perf_counter()
            total += np.float32(value)
            full[batches[start:stop], positions[start:stop]] = derivative
            scatter_seconds += time.perf_counter() - before
        before = time.perf_counter()
        ctx.save_for_backward(torch.from_numpy(full))
        result = torch.tensor(total, dtype=torch.float32, device=hidden.device)
        boundary_seconds += time.perf_counter() - before
        runtime.last_profile = {
            "causal_rows": targets.size,
            "valid_rows": len(rows),
            "chunks": chunks,
            "boundary_seconds": boundary_seconds,
            "gather_seconds": gather_seconds,
            "execute_seconds": execute_seconds,
            "scatter_seconds": scatter_seconds,
            "runtime_input_bytes": input_bytes,
            "runtime_output_bytes": output_bytes,
        }
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        (gradient,) = ctx.saved_tensors
        return None, gradient * upstream, None


class ChunkedHead:
    def __init__(
        self,
        weight,
        *,
        chunk_size=128,
        projection_precision="fp32",
        weight_layout="hidden_vocab",
        session_factory=session,
    ):
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        self._closed = False
        self.chunk_size = chunk_size
        self.projection_precision = projection_precision
        self.weight_layout = weight_layout
        self.model = head_graph(
            weight, projection_precision=projection_precision, weight_layout=weight_layout
        )
        self.vocab_size = weight.shape[0 if weight_layout == "vocab_hidden" else 1]
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
        self.last_profile = None
        self.model = ir.load(sources["loss"])
        self.gradient = ir.load(sources["gradient"])
        self.forward_model = ir.load(sources["forward"])
        weight = self.model.graph.initializers["head.weight"]
        layout = self.model.metadata_props.get(
            "onnax.example.head.weight_layout", "hidden_vocab"
        )
        if layout != self.weight_layout:
            raise ValueError("Exported head weight layout differs from its interface")
        if list(weight.shape)[0 if layout == "vocab_hidden" else 1] != self.vocab_size:
            raise ValueError("Exported head vocabulary differs from its interface")
        policy = json.loads(self.model.metadata_props.get("onnax.example.precision", "{}"))
        actual_precision = {"bfloat16": "bf16", "float16": "fp16"}.get(
            policy.get("projection_compute"), "fp32"
        )
        if actual_precision != self.projection_precision:
            raise ValueError("Output-head projection precision differs from its interface")
        self.executor = session_factory(sources["gradient"])
        self._forward_source = sources["forward"]
        self._session_factory = session_factory
        self.forward_executor = None

    def export_to(self, weights):
        """Store the head modules alongside the checkpointed body's shared weights."""
        self._ensure_open()
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
            **(
                {"weight_layout": self.weight_layout}
                if self.weight_layout != "hidden_vocab"
                else {}
            ),
            **{name: os.path.basename(path) for name, path in sources.items()},
        }

    @classmethod
    def load(cls, directory, *, session_factory=session):
        """Load a saved output head without constructing new derivative graphs."""
        with open(os.path.join(directory, "checkpoint_plan.json")) as stream:
            plan = json.load(stream)
        if plan.get("version") != 1 or "head" not in plan:
            raise ValueError("Expected a checkpoint execution plan with an output head")
        return cls.from_record(directory, plan["head"], session_factory=session_factory)

    @classmethod
    def from_record(cls, directory, record, *, session_factory=session):
        """Restore reusable head modules from their serialized task record."""
        if record.get("version") != 1 or record.get("reduction") != "valid_shifted_token_mean":
            raise ValueError("Unsupported output-head interface")
        if (
            isinstance(record["chunk_size"], bool)
            or not isinstance(record["chunk_size"], int)
            or record["chunk_size"] < 1
            or record["projection_precision"] not in ("fp32", "bf16", "fp16")
        ):
            raise ValueError("Invalid output-head execution options")
        result = cls.__new__(cls)
        result._closed = False
        result.weight_layout = record.get("weight_layout", "hidden_vocab")
        if result.weight_layout not in ("hidden_vocab", "vocab_hidden"):
            raise ValueError("Invalid output-head weight layout")
        for name in ("loss", "gradient", "forward"):
            if os.path.basename(record[name]) != record[name]:
                raise ValueError("Output-head files must be inside the task directory")
        result.chunk_size = record["chunk_size"]
        result.vocab_size = record["vocab_size"]
        result.projection_precision = record["projection_precision"]
        if result.weight_layout == "vocab_hidden" and result.projection_precision != "fp32":
            raise ValueError("The source-layout head currently uses FP32 projection compute")
        result._load_sources(
            {
                name: os.path.join(directory, record[name])
                for name in ("loss", "gradient", "forward")
            },
            session_factory,
        )
        return result

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("The output head is closed")

    def close(self):
        self._closed = True
        self.executor = self.forward_executor = None
        if hasattr(self, "directory"):
            self.directory.cleanup()

    def forward(self, hidden):
        self._ensure_open()
        array = hidden.detach().contiguous().numpy()
        if self.forward_executor is None:
            self.forward_executor = self._session_factory(self._forward_source)
        logits = self.forward_executor.run(
            None, {"hidden": array.reshape(-1, array.shape[-1])}
        )[0]
        return torch.from_numpy(logits.reshape(*array.shape[:-1], logits.shape[-1]))

    def __call__(self, hidden, labels, num_items=None):
        self._ensure_open()
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
            labels[:, 1:].ne(-100).sum()
            if num_items is None
            else torch.as_tensor(num_items, device=hidden.device)
        )
        return total / count.clamp(min=1)
