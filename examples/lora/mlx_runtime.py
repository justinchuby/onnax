# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Optional native MLX execution-provider sessions for workload examples."""

import onnx_ir as ir
import onnxruntime as ort


def session(graph, *, profile_prefix=None):
    """Load the optional plugin and require complete MLX provider assignment."""
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
    executor = ort.InferenceSession(
        ir.to_proto(graph).SerializeToString(), options, providers=["MLXExecutionProvider"]
    )
    executor.disable_fallback()
    return executor
