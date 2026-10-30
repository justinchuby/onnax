# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""JAX CPU / ONNX Runtime DLPack exchange with fresh, independently owned outputs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import onnxruntime as ort

from examples.lora.binding import gradient_shapes
from examples.lora.model import PARAMETERS


def synchronize(params, state=None):
    """Wait at the runtime boundary, including optimizer-state computations."""
    arrays = list(params.values())
    if state is not None:
        arrays.extend(state.first.values())
        arrays.extend(state.second.values())
    jax.block_until_ready(arrays)


class JaxCpuDLPack:
    """Share FP32 buffers on CPU; each run gives JAX fresh output ownership."""

    def __init__(self, executor, interface):
        if executor.get_providers() != ["CPUExecutionProvider"]:
            raise ValueError("This adapter requires a CPUExecutionProvider session.")
        self.shapes, self.output_shapes = gradient_shapes(executor, interface)
        self.executor = executor
        self.binding = executor.io_binding()
        self.loss_name = interface["loss"]
        self.gradient_names = {r["input"]: r["gradient"] for r in interface["parameters"]}
        self.input_arrays = {}
        self.input_values = {}
        self.output_values = None
        self.ready = False

    def bind(self, features, target, params):
        self.ready = False
        self.output_values = None
        if params.keys() != set(PARAMETERS):
            raise ValueError("Expected exactly lora.A and lora.B parameters.")
        feeds = {"features": features, "target": target, **params}
        for name, array in feeds.items():
            shape = self.shapes[name]
            if (
                not isinstance(array, jax.Array)
                or array.dtype != jnp.float32
                or array.ndim != len(shape)
                or any(
                    wanted is not None and array.shape[i] != wanted
                    for i, wanted in enumerate(shape)
                )
            ):
                raise ValueError(f"Expected an FP32 JAX array with shape {shape} for {name}.")
            if (
                not array.is_fully_addressable
                or len(array.devices()) != 1
                or array.__dlpack_device__() != (1, 0)
            ):
                raise ValueError(f"Expected one locally addressable CPU device for {name}.")
        if len(features) == 0 or len(features) != len(target):
            raise ValueError("Features and target need matching nonempty batches.")
        synchronize(feeds)
        for name, array in feeds.items():
            if self.input_arrays.get(name) is array:
                continue
            value = ort.OrtValue.from_dlpack(array)
            if value.data_ptr() != array.unsafe_buffer_pointer():
                raise RuntimeError(f"DLPack input import copied storage for {name}.")
            self.binding.bind_ortvalue_input(name, value)
            self.input_arrays[name] = array
            self.input_values[name] = value
        self.ready = True

    def execute(self):
        """Allocate fresh output OrtValues and run synchronously on CPU."""
        if not self.ready:
            raise RuntimeError("Bind inputs before executing the gradient graph.")
        self.output_values = None
        self.binding.clear_binding_outputs()
        for name in self.output_shapes:
            self.binding.bind_output(name, "cpu")
        self.executor.run_with_iobinding(self.binding)
        self.output_values = dict(
            zip(self.output_shapes, self.binding.get_outputs(), strict=True)
        )

    def import_outputs(self):
        """Import runtime-owned storage with copy=False and verify pointer identity."""
        if self.output_values is None:
            raise RuntimeError("Execute the gradient graph before importing outputs.")
        arrays = {}
        for name, value in self.output_values.items():
            array = jax.dlpack.from_dlpack(value, copy=False)
            if array.unsafe_buffer_pointer() != value.data_ptr():
                raise RuntimeError(f"DLPack output import copied storage for {name}.")
            arrays[name] = array
        return arrays[self.loss_name], {
            n: arrays[output] for n, output in self.gradient_names.items()
        }

    def run(self):
        self.execute()
        return self.import_outputs()
