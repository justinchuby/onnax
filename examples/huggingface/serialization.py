# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""External-data serialization for workload execution graphs."""

import numpy as np
import onnx_ir as ir


def save_graph(model, path, *, data_name):
    """Serialize lazy transpose views through contiguous buffers once per export."""
    result = model.clone()
    for name, value in result.graph.initializers.items():
        tensor = value.const_value
        if tensor.nbytes >= 64 * 1024:
            value.const_value = ir.tensor(np.ascontiguousarray(tensor.numpy()), name=name)
    ir.save(result, path, external_data=data_name)
