# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Optional native MLX execution-provider sessions for workload examples."""

import os

import onnx_ir as ir
import onnxruntime as ort

_development_library = None


def register_library(path):
    """Register one development binary and reject conflicting selections."""
    global _development_library
    path = os.path.realpath(path)
    if _development_library is not None:
        if _development_library != path:
            raise ValueError("A different MLX EP library is already registered")
        return
    ort.register_execution_provider_library("MLXExecutionProvider", path)
    _development_library = path


def session(graph, *, profile_prefix=None, target_nodes=None):
    """Load the optional plugin and require complete MLX provider assignment."""
    if target_nodes is not None:
        import onnx

        from examples.huggingface.segmented import SegmentedSession

        if profile_prefix is not None:
            raise ValueError("Select per-segment profiling explicitly through session_factory")
        if not isinstance(graph, ir.Model):
            graph = (
                ir.from_proto(onnx.load_model_from_string(graph))
                if isinstance(graph, bytes)
                else ir.load(graph)
            )
        return SegmentedSession(graph, target_nodes=target_nodes, session_factory=session)
    import onnxruntime_ep_mlx

    if not any(device.ep_name == "MLXExecutionProvider" for device in ort.get_ep_devices()):
        onnxruntime_ep_mlx.register_execution_provider_library()
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if profile_prefix is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    source = ir.to_proto(graph).SerializeToString() if isinstance(graph, ir.Model) else graph
    executor = ort.InferenceSession(source, options, providers=["MLXExecutionProvider"])
    executor.disable_fallback()
    return executor
