# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Optimize large derivative graphs with ONNXScript folding and IR cleanup."""

from onnx_ir.passes import common
from onnxscript import optimizer


def optimize_training_graph(model):
    """Return a clone with folded shapes and shared derivative expressions.

    The default pattern matcher repeatedly scans large graphs. This pipeline
    uses constant propagation and CSE directly, preserving local function calls.
    """
    result = model.clone()
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
