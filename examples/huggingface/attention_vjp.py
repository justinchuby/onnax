# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Example compatibility entry point for the public attention VJP pass."""

from onnax import FuseAttentionVJPPass


def fuse_attention_vjp(model, *, recompute=False):
    """Return the portable deployment copy produced by the public pass."""
    return FuseAttentionVJPPass(recompute=recompute)(model).model
