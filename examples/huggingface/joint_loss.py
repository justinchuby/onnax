# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Execute a causal loss and selected parameter gradients in one ONNX session."""

import hashlib
import json
import os
import tempfile
import time

import numpy as np
import onnx
import onnx_ir as ir
import torch
from onnx_ir.passes import common
from torch.autograd.function import once_differentiable

import onnax
from examples.huggingface.bridge import frozen_hashes, graph_session
from examples.huggingface.optimization import optimize_training_graph
from examples.huggingface.serialization import save_graph
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
    # An ignored row keeps reductions nonempty even when sequence length is one.
    dummy = ctx.op(
        "ConstantOfShape",
        ctx.constant([1, vocab]),
        value=ir.tensor([0], dtype=ir.DataType.FLOAT),
    )
    rows = ctx.op("Concat", rows, dummy, axis=0)
    rows.type, rows.shape = ir.TensorType(ir.DataType.FLOAT), ir.Shape(["loss_rows", vocab])
    targets = ctx.op("Concat", targets, ctx.constant([-100]), axis=0)
    targets.type, targets.shape = ir.TensorType(ir.DataType.INT64), ir.Shape(["loss_rows"])
    loss = ctx.op("ReduceSum", hard_cross_entropy(ctx, rows, targets), keepdims=0)
    loss.name, loss.type, loss.shape = ctx.name("loss_sum"), rows.type, ir.Shape([])
    graph.outputs.clear()
    graph.outputs.append(loss)
    install(model)
    model.metadata_props["onnax.example.causal_loss"] = json.dumps(
        {"version": 1, "labels": "labels", "vocabulary": vocab, "reduction": "sum"}
    )
    return model


class _JointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, runtime, feeds, preparation_seconds, *parameters):
        start = time.perf_counter()
        parameter_feeds = {
            name: value.detach().numpy()
            for name, value in zip(runtime.names, parameters, strict=True)
        }
        prepared = time.perf_counter()
        inputs = {**feeds, **parameter_feeds}
        values = runtime.executor.run(runtime.outputs, inputs)
        runtime.timings["exchange"].append(preparation_seconds + prepared - start)
        runtime.last_input_bytes = sum(value.nbytes for value in inputs.values())
        runtime.last_output_bytes = sum(value.nbytes for value in values)
        runtime.timings["loss_gradient"].append(time.perf_counter() - prepared)
        gradients = [torch.from_numpy(v) for v in values[1:]]
        runtime.last_saved_gradient_bytes = sum(value.nbytes for value in values[1:])
        ctx.count = len(parameters)
        ctx.save_for_backward(*parameters, *gradients)
        return torch.from_numpy(values[0])

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        # Reading saved parameters also validates their mutation versions.
        gradients = ctx.saved_tensors[ctx.count :]
        return None, None, None, *(gradient * upstream for gradient in gradients)


class JointLoss(torch.nn.Module):
    """Own explicit CPU FP32 parameters; retain only returned gradients for backward."""

    def __init__(self, model, params, *, optimize=True, session_factory=session):
        super().__init__()
        self.model = model
        self.names = tuple(params)
        inputs = {v.name: v for v in model.graph.inputs}
        self.feed_names = set(inputs) - set(self.names)
        self.required_feeds = {
            name for name in self.feed_names if inputs[name].const_value is None
        }
        metadata = model.metadata_props.get("onnax.example.causal_loss")
        self.causal_loss = json.loads(metadata) if metadata is not None else None
        if self.causal_loss is not None and self.causal_loss["version"] != 1:
            raise ValueError("Unsupported causal-loss interface version")
        for name, array in params.items():
            if name not in inputs or name in model.graph.initializers:
                raise ValueError(f"Expected an explicit parameter input: {name}")
            if (
                inputs[name].dtype != ir.DataType.FLOAT
                or inputs[name].shape is None
                or array.dtype != np.float32
                or list(array.shape) != list(inputs[name].shape)
            ):
                raise ValueError(f"FP32 parameter shape mismatch: {name}")
        if len(model.graph.outputs) != 1 or model.graph.outputs[0].shape != ir.Shape([]):
            raise ValueError("Expected one scalar loss output")
        self.adapters = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.from_numpy(value.copy())) for value in params.values()]
        )
        self.frozen = frozen_hashes(model)
        identity = {"frozen": self.frozen, "names": self.names}
        if precision := model.metadata_props.get("onnax.example.precision"):
            identity["precision"] = json.loads(precision)
        identity = json.dumps(identity, sort_keys=True)
        self.register_buffer(
            "base_identity",
            torch.tensor(list(hashlib.sha256(identity.encode()).digest()), dtype=torch.uint8),
        )
        self.output = model.graph.outputs[0].name
        self.last_saved_gradient_bytes = 0
        self.last_input_bytes = self.last_output_bytes = 0
        self.timings = {"exchange": [], "loss_gradient": []}
        self.evaluation_timings = {"session_creation": [], "exchange": [], "loss": []}
        self.loss_model = self.loss_executor = None
        self._session_factory = session_factory
        self._closed = False
        self.gradient = onnax.grad(model, inputs=list(self.names), output=self.output)
        if optimize:
            self.gradient = optimize_training_graph(self.gradient)
        self.outputs = [v.name for v in self.gradient.graph.outputs]
        self.gradients = self.outputs[1:]
        if len(self.outputs) != 1 + len(self.names):
            raise ValueError("Expected one returned derivative per parameter")
        self.directory = tempfile.TemporaryDirectory(prefix="onnax-joint-loss-")
        self.executor = graph_session(
            self.gradient,
            self.directory.name,
            "loss_gradient",
            session_factory=session_factory,
        )

    def close(self):
        """Release training/evaluation sessions and their temporary model files."""
        self.executor = self.loss_executor = None
        self.directory.cleanup()
        self._closed = True

    def _evaluate(self, feeds, preparation_seconds):
        if self.loss_executor is None:
            start = time.perf_counter()
            # Keep the optimized primal exactly as used by the training graph.
            self.loss_model = self.gradient.clone()
            del self.loss_model.graph.outputs[1:]
            common.RemoveUnusedNodesPass()(self.loss_model)
            self.loss_executor = graph_session(
                self.loss_model,
                self.directory.name,
                "loss",
                session_factory=self._session_factory,
            )
            self._loss_inputs = {value.name for value in self.loss_model.graph.inputs}
            self.evaluation_timings["session_creation"].append(time.perf_counter() - start)
        start = time.perf_counter()
        parameters = {
            name: value.detach().numpy()
            for name, value in zip(self.names, self.adapters, strict=True)
        }
        inputs = {
            name: value
            for name, value in {**feeds, **parameters}.items()
            if name in self._loss_inputs
        }
        prepared = time.perf_counter()
        values = self.loss_executor.run([self.loss_model.graph.outputs[0].name], inputs)
        self.evaluation_timings["exchange"].append(preparation_seconds + prepared - start)
        self.evaluation_timings["loss"].append(time.perf_counter() - prepared)
        self.last_input_bytes = sum(value.nbytes for value in inputs.values())
        self.last_output_bytes = values[0].nbytes
        self.last_saved_gradient_bytes = 0
        return torch.from_numpy(values[0])

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        identity = state_dict.get(prefix + "base_identity")
        if identity is None or not torch.equal(identity.cpu(), self.base_identity.cpu()):
            raise ValueError(
                "Checkpoint base model, parameter mapping or precision policy mismatch"
            )
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def export(self, directory, *, forward_model=None):
        """Save a portable loss/gradient module, adapters and an optional bound forward."""
        if frozen_hashes(self.model) != self.frozen:
            raise ValueError("Frozen ONNX weights changed")
        if forward_model is not None and frozen_hashes(forward_model) != self.frozen:
            raise ValueError("Forward model has different frozen weights")
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "loss_gradient.onnx")
        save_graph(self.gradient, path, data_name="loss_gradient.data")
        onnx.checker.check_model(path, full_check=True)
        arrays = self.arrays()
        forward_outputs = None
        if forward_model is not None:
            forward = forward_model.clone()
            inputs = {value.name: value for value in forward.graph.inputs}
            for name, array in arrays.items():
                value = inputs[name]
                forward.graph.inputs.remove(value)
                value.const_value = ir.tensor(array, name=name)
                forward.graph.register_initializer(value)
            path = os.path.join(directory, "forward.onnx")
            save_graph(forward, path, data_name="forward.data")
            onnx.checker.check_model(path, full_check=True)
            forward_outputs = [value.name for value in forward.graph.outputs]
        np.savez(os.path.join(directory, "adapter.npz"), **arrays)
        interface = {
            "version": 1,
            "kind": "loss_gradient",
            "model": "loss_gradient.onnx",
            "output": self.output,
            "frozen": self.frozen,
            "causal_loss": self.causal_loss,
            "forward_outputs": forward_outputs,
            "parameters": [
                {
                    "name": name,
                    "gradient": gradient,
                    "shape": list(parameter.shape),
                    "dtype": "float32",
                }
                for name, gradient, parameter in zip(
                    self.names, self.gradients, self.adapters, strict=True
                )
            ],
        }
        if policy := self.model.metadata_props.get("onnax.example.precision"):
            interface["precision"] = json.loads(policy)
        with open(os.path.join(directory, "interface.json"), "w") as stream:
            json.dump(interface, stream, indent=2)
        return interface

    def arrays(self):
        return {
            name: value.detach().numpy().copy()
            for name, value in zip(self.names, self.adapters, strict=True)
        }

    def forward(self, **feeds):
        if self._closed:
            raise RuntimeError("JointLoss sessions are closed")
        start = time.perf_counter()
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
            if not isinstance(value, np.ndarray):
                raise TypeError("Feeds must be NumPy arrays or CPU tensors")
            arrays[name] = value
        missing, extra = self.required_feeds - arrays.keys(), arrays.keys() - self.feed_names
        if missing or extra:
            raise ValueError(
                f"Invalid feeds: missing={sorted(missing)}, unexpected={sorted(extra)}"
            )
        if self.causal_loss is not None:
            labels = arrays[self.causal_loss["labels"]]
            if (
                not isinstance(labels, np.ndarray)
                or labels.dtype != np.int64
                or labels.ndim != 2
            ):
                raise ValueError("Expected INT64 labels [batch, sequence]")
            if "input_ids" in arrays and labels.shape != arrays["input_ids"].shape:
                raise ValueError("Labels must match input_ids batch and sequence")
            if np.any(
                ((labels < 0) & (labels != -100)) | (labels >= self.causal_loss["vocabulary"])
            ):
                raise ValueError("Labels must be vocabulary IDs or -100")
        preparation_seconds = time.perf_counter() - start
        if not torch.is_grad_enabled():
            return self._evaluate(arrays, preparation_seconds)
        return _JointFunction.apply(self, arrays, preparation_seconds, *self.adapters)
