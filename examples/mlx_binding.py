# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Shared unified-memory bindings for static FP32 ONNX graph boundaries."""

import numpy as np
import onnxruntime as ort


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


def static_shapes(metadata):
    shapes = {}
    for value in metadata:
        if (
            value.shape is None
            or value.type != "tensor(float)"
            or any(isinstance(d, bool) or not isinstance(d, int) or d < 1 for d in value.shape)
        ):
            raise ValueError(f"Expected a static positive FP32 shape for {value.name}")
        shapes[value.name] = tuple(value.shape)
    return shapes


class MlxBinding:
    """Bind named Metal arrays and share fresh FP32 output owners.

    Keep caller inputs unchanged until synchronous execution returns. Imported
    arrays retain each execution's output storage, including after rebinding.
    """

    def __init__(self, executor, *, outputs=None):
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
        self.shapes = static_shapes(executor.get_inputs())
        available = {value.name: value for value in executor.get_outputs()}
        outputs = list(available) if outputs is None else list(outputs)
        if (
            not outputs
            or len(set(outputs)) != len(outputs)
            or not set(outputs).issubset(available)
        ):
            raise ValueError("Select distinct existing output names")
        self.output_shapes = static_shapes([available[name] for name in outputs])
        import mlx.core as mx

        self.xp = mx
        self.executor = executor
        self.binding = executor.io_binding()
        self.input_arrays = {}
        self.input_values = {}
        self.output_values = None
        self.ready = False

    def bind(self, feeds):
        self.ready = False
        self.output_values = None
        if feeds.keys() != self.shapes.keys():
            raise ValueError("Feed names must match all graph inputs")
        for name, array in feeds.items():
            shape = self.shapes[name]
            if (
                not isinstance(array, self.xp.array)
                or array.dtype != self.xp.float32
                or tuple(array.shape) != shape
                or array.__dlpack_device__() != (8, 0)
            ):
                raise ValueError(
                    f"Expected an FP32 MLX Metal array with shape {shape} for {name}"
                )
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
        return arrays

    def run(self):
        self.execute()
        return self.import_outputs()
