# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Autograd execution shared by package and task-specific checkpoint adapters."""

import time
from dataclasses import dataclass

import numpy as np
import onnx_ir as ir
import torch
from torch.autograd.function import once_differentiable

from onnax._checkpoint import Block


@dataclass(frozen=True)
class _Execution:
    block: Block
    forward: object
    backward: object
    seeds: tuple[str, ...]
    gradients: tuple[str, ...]
    derivative: ir.Model | None
    forward_inputs: tuple[str, ...] | None = None


def _forward_arrays(execution, feeds):
    names = (
        execution.block.inputs
        if execution.forward_inputs is None
        else execution.forward_inputs
    )
    arrays = list(
        execution.forward.run(list(execution.block.outputs), {n: feeds[n] for n in names})
    )
    if len(arrays) != len(execution.block.outputs) or any(
        not isinstance(a, np.ndarray) for a in arrays
    ):
        raise TypeError("Session.run must return one NumPy array per checkpoint output")
    return [
        array.copy()
        if not array.flags.writeable
        or any(np.shares_memory(array, feed) for feed in feeds.values())
        or any(i != j and np.shares_memory(array, other) for j, other in enumerate(arrays))
        else array
        for i, array in enumerate(arrays)
    ]


class _GroupFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, runtime, execution, invocation, *inputs):
        ctx.runtime, ctx.execution, ctx.invocation = runtime, execution, invocation
        needed = {value.name for value in execution.derivative.graph.inputs}
        ctx.saved_names = tuple(name for name in execution.block.inputs if name in needed)
        ctx.save_for_backward(
            *(
                value
                for name, value in zip(execution.block.inputs, inputs, strict=True)
                if name in needed
            )
        )
        feeds = {
            name: tensor.detach().contiguous().numpy()
            for name, tensor in zip(execution.block.inputs, inputs, strict=True)
        }
        arrays = _forward_arrays(execution, feeds)
        outputs = [torch.from_numpy(array) for array in arrays]
        ctx.specs = [(value.shape, value.dtype) for value in arrays]
        ctx.mark_non_differentiable(
            *(
                value
                for name, value in zip(execution.block.outputs, outputs, strict=True)
                if name not in execution.block.active_outputs
            )
        )
        ctx.set_materialize_grads(False)
        return tuple(outputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, *upstreams):
        runtime, execution = ctx.runtime, ctx.execution
        block = execution.block
        if hasattr(runtime, "_ensure_open"):
            runtime._ensure_open()
        feeds = {
            name: value.detach().contiguous().numpy()
            for name, value in zip(ctx.saved_names, ctx.saved_tensors, strict=True)
        }
        gradients = dict(zip(block.outputs, upstreams, strict=True))
        specs = dict(zip(block.outputs, ctx.specs, strict=True))
        for name, seed in zip(block.active_outputs, execution.seeds, strict=True):
            value = gradients[name]
            feeds[seed] = (
                np.zeros(*specs[name])
                if value is None
                else value.detach().contiguous().numpy()
            )
        start = time.perf_counter()
        values = execution.backward.run(
            list(execution.gradients),
            {v.name: feeds[v.name] for v in execution.derivative.graph.inputs},
        )
        elapsed = time.perf_counter() - start
        derivatives = dict(zip(block.active_inputs, values, strict=True))
        if ctx.invocation["remaining"] == 0:
            ctx.invocation.update(remaining=ctx.invocation["count"], elapsed=0.0)
        ctx.invocation["elapsed"] += elapsed
        ctx.invocation["remaining"] -= 1
        if ctx.invocation["remaining"] == 0:
            runtime.timings["backward"].append(ctx.invocation["elapsed"])
        return (
            None,
            None,
            None,
            *(
                torch.from_numpy(derivatives[name]) if name in derivatives else None
                for name in block.inputs
            ),
        )


def _execute_block(runtime, execution, inputs, *, save, invocation, checkpoint_bytes):
    """Scope exchange arrays and input references to a single block invocation."""
    block = execution.block
    if (
        save
        and block.active_inputs
        and block.active_outputs
        and any(value.requires_grad for value in inputs)
    ):
        invocation["remaining"] += 1
        invocation["count"] += 1
        result = _GroupFunction.apply(runtime, execution, invocation, *inputs)
        needed = {value.name for value in execution.derivative.graph.inputs}
        for name, value in zip(block.inputs, inputs, strict=True):
            if (
                name in needed
                and name not in runtime.parameter_names
                and name not in runtime.data_names
            ):
                checkpoint_bytes[name] = value.numel() * value.element_size()
    else:
        arguments = {
            name: value.detach().contiguous().numpy()
            for name, value in zip(block.inputs, inputs, strict=True)
        }
        arrays = _forward_arrays(execution, arguments)
        result = tuple(torch.from_numpy(value) for value in arrays)
    return result


def execute(self, feeds, *, save):
    values = {
        **getattr(self, "_constant_outputs", {}),
        **feeds,
        **dict(zip(self.parameter_names, self.trainable, strict=True)),
    }
    remaining = self.forward_uses.copy()
    checkpoint_bytes: dict[str, int] = {}
    invocation = {"elapsed": 0.0, "remaining": 0, "count": 0}
    for execution in self.executions:
        block = execution.block
        inputs = tuple(values[name] for name in block.inputs)
        result = _execute_block(
            self,
            execution,
            inputs,
            save=save,
            invocation=invocation,
            checkpoint_bytes=checkpoint_bytes,
        )
        values.update(zip(block.outputs, result, strict=True))
        del inputs, result
        for name in block.inputs:
            remaining[name] -= 1
            if remaining[name] == 0 and name not in self.output_names:
                values.pop(name, None)
    self.last_residual_bytes = sum(checkpoint_bytes.values())
    return tuple(values[name] for name in self.output_names)
