# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Shared unified-memory boundaries between MLX arrays and the MLX ORT provider."""

import numpy as np
import onnxruntime as ort

from examples.lora.binding import gradient_shapes
from examples.lora.model import PARAMETERS


class CpuView:
    """Request the producer's real CPU view of shared Metal storage."""

    def __init__(self, array):
        self.array = array

    def __dlpack_device__(self):
        return (1, 0)

    def __dlpack__(self, **kwargs):
        if kwargs.get("dl_device") not in (None, (1, 0)) or kwargs.get("copy") is True:
            raise BufferError("This adapter exports a shared CPU view without copying")
        kwargs.update(dl_device=(1, 0), copy=False)
        return self.array.__dlpack__(**kwargs)


def synchronize(params, state=None):
    """Finish MLX GPU work before crossing a runtime boundary or reading a timer."""
    import mlx.core as mx

    arrays = list(params.values())
    if state is not None:
        arrays.extend(state.first.values())
        arrays.extend(state.second.values())
    mx.eval(*arrays)


class MlxDLPack:
    """Share input storage and import fresh output ownership on Apple Silicon.

    Both DLPack directions enforce pointer identity. The plugin's internal
    wrapping and output copies are measured separately from this Python boundary.
    Caller arrays must remain unchanged while a synchronous run is in progress.
    """

    def __init__(self, executor, interface):
        if "MLXExecutionProvider" not in executor.get_providers():
            raise ValueError("Expected an MLXExecutionProvider session")
        options = executor.get_session_options()
        try:
            disabled = options.get_session_config_entry("session.disable_cpu_ep_fallback")
        except RuntimeError as error:
            raise ValueError("Disable CPU EP fallback for this native adapter") from error
        if (
            disabled != "1"
            or options.graph_optimization_level != ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        ):
            raise ValueError("Expected MLX EP with CPU fallback and ORT optimization disabled")
        import mlx.core as mx

        self.xp = mx
        self.shapes, self.output_shapes = gradient_shapes(executor, interface)
        inputs = {value.name: value for value in executor.get_inputs()}
        self.batch = inputs["features"].shape[0]
        if not isinstance(self.batch, int) or self.batch < 1:
            raise ValueError("This native workload requires a fixed positive batch")
        if inputs["target"].shape[0] != self.batch:
            raise ValueError("Features and target metadata need the same fixed batch")
        outputs = {value.name: value for value in executor.get_outputs()}
        self.prediction_name = interface.get("prediction", "prediction")
        if self.prediction_name is not None:
            prediction = outputs.get(self.prediction_name)
            if (
                prediction is None
                or prediction.type != "tensor(float)"
                or prediction.shape != [self.batch, 8]
            ):
                raise ValueError("Expected FP32 prediction output with the fixed batch")
            self.output_shapes = {self.prediction_name: (self.batch, 8), **self.output_shapes}
        self.loss_name = interface["loss"]
        self.gradient_names = {
            row["input"]: row["gradient"] for row in interface["parameters"]
        }
        self.executor = executor
        self.binding = executor.io_binding()
        self.input_arrays = {}
        self.input_values = {}
        self.output_values = None
        self.ready = False

    def bind(self, features, target, params):
        self.ready = False
        self.output_values = None
        if params.keys() != set(PARAMETERS):
            raise ValueError("Expected exactly lora.A and lora.B parameters")
        feeds = {"features": features, "target": target, **params}
        for name, array in feeds.items():
            shape = self.shapes[name]
            if (
                not isinstance(array, self.xp.array)
                or array.dtype != self.xp.float32
                or array.ndim != len(shape)
                or any(
                    wanted is not None and array.shape[i] != wanted
                    for i, wanted in enumerate(shape)
                )
                or array.__dlpack_device__() != (8, 0)
            ):
                raise ValueError(
                    f"Expected an FP32 MLX Metal array with shape {shape} for {name}"
                )
        if len(features) != self.batch or len(target) != self.batch:
            raise ValueError(f"Expected batch {self.batch} for features and target")
        synchronize(feeds)
        arrays, values = {}, {}
        for name, array in feeds.items():
            host = np.asarray(array, copy=False)
            if not host.flags.c_contiguous:
                raise ValueError(f"Expected contiguous MLX input {name}")
            owner = CpuView(array)
            value = ort.OrtValue.from_dlpack(owner)
            if value.data_ptr() != host.ctypes.data:
                raise RuntimeError(f"DLPack import copied input storage for {name}")
            arrays[name], values[name] = owner, value
        # Rebind even the same Python object: MLX indexed assignment can replace
        # its storage while preserving object identity.
        for name, value in values.items():
            self.binding.bind_ortvalue_input(name, value)
        self.input_arrays, self.input_values = arrays, values
        self.ready = True

    def execute(self):
        if not self.ready:
            raise RuntimeError("Bind inputs before executing")
        self.output_values = None
        self.binding.clear_binding_outputs()
        for name in self.output_shapes:
            self.binding.bind_output(name, "cpu")
        self.executor.run_with_iobinding(self.binding)
        self.output_values = dict(
            zip(self.output_shapes, self.binding.get_outputs(), strict=True)
        )

    def import_outputs(self):
        if self.output_values is None:
            raise RuntimeError("Execute before importing outputs")
        arrays = {}
        for name, value in self.output_values.items():
            array = self.xp.from_dlpack(value, copy=False)
            if np.asarray(array, copy=False).ctypes.data != value.data_ptr():
                raise RuntimeError(f"DLPack import copied output storage for {name}")
            arrays[name] = array
        return (
            arrays.get(self.prediction_name),
            arrays[self.loss_name],
            {name: arrays[output] for name, output in self.gradient_names.items()},
        )

    def run(self):
        self.execute()
        return self.import_outputs()
