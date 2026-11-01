# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Optional PyTorch autograd integration for ordinary ONNX modules."""

from onnax.torch._module import OnnxModule, Session, SessionFactory

__all__ = ["OnnxModule", "Session", "SessionFactory"]
