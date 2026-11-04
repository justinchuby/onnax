# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Optimize large derivative graphs with ONNXScript folding and IR cleanup."""

import hashlib
import json
import os

import onnx_ir as ir
from onnx_ir.passes import common
from onnxscript import optimizer


def freeze_constants(model):
    """Evaluate frozen expressions once and give derived tensors stable names."""
    result = model.clone()
    original = set(result.graph.initializers)
    public_outputs = {value.name for value in result.graph.outputs}
    optimizer.fold_constants(result)
    common.LiftConstantsToInitializersPass(lift_all_constants=True, size_limit=0)(result)
    used = set(ir.convenience.create_value_mapping(result.graph))
    canonical = {}
    for old, value in list(result.graph.initializers.items()):
        if old in original or old in public_outputs:
            continue
        tensor = value.const_value
        header = json.dumps([str(tensor.dtype), list(tensor.shape)]).encode()
        digest = hashlib.sha256(header + tensor.tobytes()).hexdigest()
        del result.graph.initializers[old]
        if digest in canonical:
            value.replace_all_uses_with(canonical[digest])
            continue
        name = "pkg.onnax.frozen." + digest
        base, index = name, 0
        while name in used:
            index += 1
            name = f"{base}_{index}"
        value.name = name
        used.add(name)
        result.graph.register_initializer(value)
        canonical[digest] = value
    return result


def optimize_training_graph(model):
    """Return a clone with folded shapes and shared derivative expressions.

    The default pattern matcher repeatedly scans large graphs. This pipeline
    uses constant propagation and CSE directly, preserving local function calls.
    """
    result = model.clone()
    if os.environ.get("ONNAX_TILED_ATTENTION") == "1":
        from onnax import FuseAttentionVJPPass

        result = FuseAttentionVJPPass()(result).model
    for _ in range(2):
        optimizer.fold_constants(result)
        common.RemoveUnusedNodesPass()(result)
        common.LiftConstantsToInitializersPass(lift_all_constants=True, size_limit=0)(result)
        common.DeduplicateInitializersPass()(result)
        common.CommonSubexpressionEliminationPass()(result)
    common.OutputFixPass()(result)
    common.NameFixPass()(result)
    result.graph.sort()
    return result
