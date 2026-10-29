# SPDX-License-Identifier: MIT
"""Experimental, source-preserving ONNX graph transformations.

The four functional passes return an ``onnx_ir.passes.PassResult`` whose
``model`` is a transformed clone. Their generated graphs use ordinary ONNX
operators and can be saved with ``onnx_ir.save`` and run without Python rules.
See each pass's docstring for selection, shape, dtype, and operator limitations.
Implementation modules and rule registries are private, not a plugin API.
"""

from onnax._autodiff import GradPass, JvpPass, VjpPass
from onnax._batching import VmapPass

__all__ = ["GradPass", "JvpPass", "VjpPass", "VmapPass"]
