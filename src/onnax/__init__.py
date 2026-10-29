# SPDX-License-Identifier: MIT
"""Experimental, source-preserving ONNX graph transformations.

The model-first functions ``vmap``, ``jvp``, ``vjp`` and ``grad`` return a
transformed ``onnx_ir.Model`` without modifying the source. Compatible pass
entry points return an ``onnx_ir.passes.PassResult`` instead. Generated graphs
use ordinary ONNX operators, can be saved with ``onnx_ir.save``, and run without
Python rules. See the functions and passes for selection and support contracts.
Implementation modules and rule registries are private, not a plugin API.
"""

from onnax._autodiff import GradPass, JvpPass, VjpPass
from onnax._batching import VmapPass
from onnax._functional import grad, jvp, vjp, vmap

__all__ = ["GradPass", "JvpPass", "VjpPass", "VmapPass", "grad", "jvp", "vjp", "vmap"]
