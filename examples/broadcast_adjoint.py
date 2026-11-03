# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Example compatibility entry point for the public broadcast-adjoint pass."""

from onnax import FuseBroadcastAdjointsPass


def fuse_broadcast_adjoints(model):
    """Return the portable deployment copy produced by the public pass."""
    return FuseBroadcastAdjointsPass()(model).model
