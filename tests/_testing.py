# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""IR-first validation and bounded-thread execution at real runtime boundaries."""

from pathlib import Path

import onnx_ir as ir
import onnxruntime as ort
from onnx.reference import ReferenceEvaluator
from onnx_ir.passes.common import CheckerPass

from onnax._inference import _infer_shapes


def evaluator(model: ir.Model) -> ReferenceEvaluator:
    inferred = _infer_shapes(model)
    CheckerPass(full_check=True)(inferred)
    return ReferenceEvaluator(ir.to_proto(inferred))


def runtime_session(model: ir.Model | str | Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    # Validate the actual returned graph, without runtime optimizer cleanup.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    artifact = (
        ir.to_proto(model).SerializeToString() if isinstance(model, ir.Model) else str(model)
    )
    return ort.InferenceSession(
        artifact, sess_options=options, providers=["CPUExecutionProvider"]
    )
