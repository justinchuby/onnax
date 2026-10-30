# SPDX-License-Identifier: MIT
"""Experimental, source-preserving ONNX graph transformations.

The functional transforms ``vmap``, ``jvp``, ``vjp`` and ``grad`` return a
transformed model or graph in the input container type without modifying the
source. ``onnx_ir.GraphView`` inputs return independent ``onnx_ir.Graph`` objects. Compatible pass
entry points return an ``onnx_ir.passes.PassResult`` instead. Generated graphs
use ordinary ONNX operators. Save IR models with ``onnx_ir.save`` and protobuf
models with ``onnx.save``. See the functions and passes for selection and support contracts.
Global registration decorators and public context types support extensions.
Implementation modules remain private.
"""

from onnax._autodiff import GradPass, JvpPass, VjpPass
from onnax._autodiff._context import AutodiffContext
from onnax._batching import VmapPass
from onnax._batching._context import BatchingContext
from onnax._functional import grad, jvp, vjp, vmap
from onnax._linear import HoistLinearCoefficientsPass
from onnax._rules import (
    BatchingRule,
    JvpRule,
    VjpRule,
    register_batching,
    register_jvp,
    register_vjp,
)

__all__ = [
    "GradPass",
    "JvpPass",
    "VjpPass",
    "VmapPass",
    "HoistLinearCoefficientsPass",
    "grad",
    "jvp",
    "vjp",
    "vmap",
    "register_jvp",
    "register_vjp",
    "register_batching",
    "JvpRule",
    "VjpRule",
    "BatchingRule",
    "AutodiffContext",
    "BatchingContext",
]
