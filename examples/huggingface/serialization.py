# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""External-data serialization for workload execution graphs."""

import numpy as np
import onnx_ir as ir


def reference_session(source):
    """Execute portable graphs with ONNX's reference kernels, including BF16."""
    import onnx
    from onnx.reference import ReferenceEvaluator

    return ReferenceEvaluator(
        onnx.load_model_from_string(source) if isinstance(source, bytes) else str(source)
    )


def save_graph(model, path, *, data_name):
    """Serialize lazy transpose views through contiguous buffers once per export."""
    result = model.clone()
    for name, value in result.graph.initializers.items():
        tensor = value.const_value
        if tensor.nbytes >= 64 * 1024:
            value.const_value = ir.tensor(np.ascontiguousarray(tensor.numpy()), name=name)
    ir.save(result, path, external_data=data_name)
