# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""CPU PyTorch autograd boundary for ordinary ONNX forward/VJP modules."""

import hashlib
import json
import os
import tempfile
import time

import numpy as np
import onnx
import onnx_ir as ir
import torch
from torch.autograd.function import once_differentiable

import onnax
from examples.huggingface.optimization import optimize_training_graph
from examples.lora.train import session


def frozen_hashes(model):
    return {
        name: hashlib.sha256(v.const_value.numpy().tobytes()).hexdigest()
        for name, v in model.graph.initializers.items()
    }


def save_graph(model, path, *, data_name):
    """Serialize lazy transpose views through contiguous buffers once per export."""
    result = model.clone()
    for name, value in result.graph.initializers.items():
        tensor = value.const_value
        if tensor.nbytes >= 64 * 1024:
            value.const_value = ir.tensor(np.ascontiguousarray(tensor.numpy()), name=name)
    ir.save(result, path, external_data=data_name)


def graph_session(model, directory, name):
    """Use external data for large graphs and in-memory protobufs for small ones."""
    size = sum(v.const_value.nbytes for v in model.graph.initializers.values())
    if size < 64 * 1024**2:
        return session(ir.to_proto(model).SerializeToString())
    path = os.path.join(directory, name + ".onnx")
    save_graph(model, path, data_name=name + ".data")
    return session(path)


class _OnnxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, runtime, feeds, *params):
        ctx.runtime = runtime
        ctx.feeds = {k: v.copy() for k, v in feeds.items()}
        ctx.save_for_backward(*params)
        result = runtime.forward_session.run(
            [runtime.output], {**feeds, **runtime.parameter_feeds(params)}
        )[0]
        return torch.from_numpy(result)

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        runtime = ctx.runtime
        # Saved tensors also detect optimizer updates between forward and backward.
        feeds = {**ctx.feeds, **runtime.parameter_feeds(ctx.saved_tensors)}
        feeds[runtime.seed] = upstream.detach().contiguous().numpy()
        start = time.perf_counter()
        values = runtime.backward_session.run(runtime.gradients, feeds)
        runtime.timings["backward"].append(time.perf_counter() - start)
        return (None, None, *(torch.from_numpy(v) for v in values))


class OnnxModule(torch.nn.Module):
    """Own trainable CPU FP32 parameters while ONNX computes values and VJPs."""

    def __init__(self, model, params, *, output=None):
        super().__init__()
        self.model = model
        self.names = tuple(params)
        self.output = output or model.graph.outputs[0].name
        inputs = {v.name: v for v in model.graph.inputs}
        for name, array in params.items():
            if name not in inputs or name in model.graph.initializers:
                raise ValueError(f"Expected explicit trainable input: {name}")
            if array.dtype != np.float32 or list(array.shape) != list(inputs[name].shape):
                raise ValueError(f"FP32 parameter shape mismatch: {name}")
        self.adapters = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.from_numpy(p.copy())) for p in params.values()]
        )
        self.data_names = tuple(v.name for v in model.graph.inputs if v.name not in params)
        self.frozen = frozen_hashes(model)
        identity = json.dumps({"frozen": self.frozen, "names": self.names}, sort_keys=True)
        self.register_buffer(
            "base_identity",
            torch.tensor(list(hashlib.sha256(identity.encode()).digest()), dtype=torch.uint8),
        )
        vjp = onnax.vjp(model, inputs=list(params), outputs=[self.output])
        self.seed = vjp.graph.inputs[-1].name
        self.gradients = [v.name for v in vjp.graph.outputs[len(model.graph.outputs) :]]
        self.vjp_model = vjp
        self.optimized_vjp = optimize_training_graph(vjp)
        self.runtime_directory = tempfile.TemporaryDirectory(prefix="onnax-runtime-")

        self.forward_session = graph_session(model, self.runtime_directory.name, "forward")
        self.backward_session = graph_session(
            self.optimized_vjp, self.runtime_directory.name, "backward"
        )
        self.timings = {n: [] for n in ("forward", "backward", "exchange")}

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        identity = state_dict.get(prefix + "base_identity")
        if identity is None or not torch.equal(identity.cpu(), self.base_identity.cpu()):
            raise ValueError("Checkpoint base model or parameter mapping mismatch")
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def parameter_feeds(self, params):
        if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in params):
            raise ValueError("This bridge executes CPU FP32 parameters")
        return {
            n: p.detach().contiguous().numpy() for n, p in zip(self.names, params, strict=True)
        }

    def forward(self, **inputs):
        if set(inputs) != set(self.data_names):
            raise ValueError(
                f"Expected ONNX data inputs {self.data_names}, got {tuple(inputs)}"
            )
        start = time.perf_counter()
        if any(v.device.type != "cpu" for v in inputs.values()):
            raise ValueError("This bridge executes CPU input tensors")
        feeds = {n: v.detach().contiguous().numpy() for n, v in inputs.items()}
        exchanged = time.perf_counter()
        value = _OnnxFunction.apply(self, feeds, *self.adapters)
        self.timings["exchange"].append(exchanged - start)
        self.timings["forward"].append(time.perf_counter() - exchanged)
        return value

    def arrays(self):
        return {
            n: p.detach().numpy().copy()
            for n, p in zip(self.names, self.adapters, strict=True)
        }

    def export(self, directory):
        os.makedirs(directory, exist_ok=True)
        if frozen_hashes(self.model) != self.frozen:
            raise ValueError("Frozen ONNX weights changed")
        result = self.model.clone()
        values = {v.name: v for v in result.graph.inputs}
        for name, array in self.arrays().items():
            value = values[name]
            result.graph.inputs.remove(value)
            value.const_value = ir.tensor(array, name=name)
            result.graph.register_initializer(value)
        path = os.path.join(directory, "forward.onnx")
        save_graph(result, path, data_name="forward.data")
        onnx.checker.check_model(path, full_check=True)
        np.savez(os.path.join(directory, "adapter.npz"), **self.arrays())
        with open(os.path.join(directory, "interface.json"), "w") as stream:
            json.dump(
                {
                    "version": 1,
                    "output": self.output,
                    "frozen": self.frozen,
                    "parameters": [
                        {"name": n, "gradient": g, "shape": list(p.shape), "dtype": "float32"}
                        for n, g, p in zip(
                            self.names, self.gradients, self.adapters, strict=True
                        )
                    ],
                },
                stream,
                indent=2,
            )
        return result
