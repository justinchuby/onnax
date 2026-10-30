# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Example-owned CPU bindings for the LoRA loss and selected gradients."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

from examples.lora.model import PARAMETERS


def gradient_shapes(executor, interface):
    """Validate the LoRA manifest against runtime metadata and return shape maps."""
    rows = interface["parameters"]
    if (
        interface.get("version") != 1
        or [r["input"] for r in rows] != list(PARAMETERS)
        or any(r["dtype"] != "float32" for r in rows)
    ):
        raise ValueError("Expected the version-one LoRA parameter interface.")
    shapes = {
        "features": (None, 16),
        "target": (None, 8),
        **{r["input"]: tuple(r["shape"]) for r in rows},
    }
    inputs = {v.name: v for v in executor.get_inputs()}
    if inputs.keys() != shapes.keys():
        raise ValueError("Model inputs do not match the LoRA interface.")
    outputs = {v.name: v for v in executor.get_outputs()}
    loss_name = interface["loss"]
    output_shapes = {
        loss_name: (),
        **{r["gradient"]: tuple(r["shape"]) for r in rows},
    }
    if len(output_shapes) != len(rows) + 1:
        raise ValueError("Loss and gradient output names must be distinct.")
    for name, shape in shapes.items():
        actual = inputs[name]
        if (
            actual.type != "tensor(float)"
            or len(actual.shape) != len(shape)
            or any(
                wanted is not None and actual.shape[i] != wanted
                for i, wanted in enumerate(shape)
            )
        ):
            raise ValueError(f"Model input shape/dtype mismatch for {name}.")
    for name, shape in output_shapes.items():
        if (
            name not in outputs
            or outputs[name].type != "tensor(float)"
            or tuple(outputs[name].shape) != shape
        ):
            raise ValueError(f"Model output shape/dtype mismatch for {name}.")
    return shapes, output_shapes


class CpuGradientBinding:
    """Reuse owned output arrays; results are borrowed until the next run.

    Bind contiguous FP32 NumPy inputs, keeping their owners alive. External
    functional updates allocate new parameter arrays, which must be bound again.
    This adapter accepts the LoRA example's fixed parameter shapes/dynamic batch.
    """

    def __init__(self, executor, interface):
        if executor.get_providers() != ["CPUExecutionProvider"]:
            raise ValueError("This adapter requires a CPUExecutionProvider session.")
        self.executor = executor
        self.binding = executor.io_binding()
        self.shapes, output_shapes = gradient_shapes(executor, interface)
        self.loss_name = interface["loss"]
        self.gradient_names = {r["input"]: r["gradient"] for r in interface["parameters"]}
        self.output_arrays = {}
        self.output_values = {}
        for name, shape in output_shapes.items():
            array = np.empty(shape, np.float32)
            value = ort.OrtValue.ortvalue_from_numpy(array)
            if value.data_ptr() != array.ctypes.data:
                raise RuntimeError("CPU output OrtValue must share its NumPy buffer.")
            self.output_arrays[name] = array
            self.output_values[name] = value
            self.binding.bind_ortvalue_output(name, value)
        self.ready = False
        self.input_arrays = {}
        self.input_values = {}

    def bind(self, features, target, params):
        self.ready = False
        if params.keys() != set(PARAMETERS):
            raise ValueError("Expected exactly lora.A and lora.B parameters.")
        feeds = {"features": features, "target": target, **params}
        for name, array in feeds.items():
            shape = self.shapes[name]
            if (
                not isinstance(array, np.ndarray)
                or array.dtype != np.float32
                or array.ndim != len(shape)
                or any(
                    wanted is not None and array.shape[i] != wanted
                    for i, wanted in enumerate(shape)
                )
            ):
                raise ValueError(
                    f"Expected an FP32 NumPy array with shape {shape} for {name}."
                )
            if not array.flags.c_contiguous:
                raise ValueError(f"CPU binding requires contiguous input {name}.")
            if any(np.shares_memory(array, output) for output in self.output_arrays.values()):
                raise ValueError(f"Input {name} overlaps a bound output buffer.")
        if len(features) == 0 or len(features) != len(target):
            raise ValueError("Features and target need matching nonempty batches.")
        for name, array in feeds.items():
            if self.input_arrays.get(name) is array:
                continue
            value = ort.OrtValue.ortvalue_from_numpy(array)
            if value.data_ptr() != array.ctypes.data:
                raise RuntimeError(f"CPU input OrtValue must share its NumPy buffer: {name}.")
            self.binding.bind_ortvalue_input(name, value)
            self.input_arrays[name] = array
            self.input_values[name] = value
        self.ready = True

    def run(self):
        """Execute synchronously and return loss/gradient views into reusable storage."""
        if not self.ready:
            raise RuntimeError("Bind inputs before executing the gradient graph.")
        self.executor.run_with_iobinding(self.binding)
        return self.output_arrays[self.loss_name], {
            name: self.output_arrays[output] for name, output in self.gradient_names.items()
        }
