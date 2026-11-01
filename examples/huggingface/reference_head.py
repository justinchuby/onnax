# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""FP32 pairwise accumulation for the independent frozen-head reference."""

from contextlib import contextmanager
from types import MethodType

import torch
from torch.autograd.function import once_differentiable


def pairwise_matmul(left, right, *, block_size=128):
    """Reduce short K-block products with a binary carry tree."""
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("Expected a positive integer block size")
    if left.ndim == 0 or right.ndim != 2 or left.shape[-1] != right.shape[0]:
        raise ValueError("Expected (..., K) and (K, N) matrices")
    if right.shape[0] == 0:
        return left @ right
    slots = []
    for offset in range(0, right.shape[0], block_size):
        partial = (
            left[..., offset : offset + block_size].contiguous()
            @ right[offset : offset + block_size]
        )
        level = 0
        while level < len(slots) and slots[level] is not None:
            partial = slots[level] + partial
            slots[level] = None
            level += 1
        if level == len(slots):
            slots.append(partial)
        else:
            slots[level] = partial
    result = None
    for partial in reversed(slots):
        if partial is not None:
            result = partial if result is None else result + partial
    return result


class _FrozenHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, block_size):
        if weight.requires_grad:
            raise ValueError("The reference head weight must be frozen")
        ctx.save_for_backward(weight)
        ctx.block_size = block_size
        return torch.nn.functional.linear(hidden, weight)

    @staticmethod
    @once_differentiable
    def backward(ctx, cotangent):
        return (
            pairwise_matmul(cotangent, ctx.saved_tensors[0], block_size=ctx.block_size),
            None,
            None,
        )


class _BF16FrozenHead(_FrozenHead):
    @staticmethod
    def forward(ctx, hidden, weight, block_size):
        if weight.requires_grad:
            raise ValueError("The reference head weight must be frozen")
        rounded = weight.to(torch.bfloat16)
        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(rounded.float())
        ctx.block_size = block_size
        return (hidden.to(torch.bfloat16) @ rounded.T).float()


class _FP16FrozenHead(_FrozenHead):
    @staticmethod
    def forward(ctx, hidden, weight, block_size):
        if weight.requires_grad:
            raise ValueError("The reference head weight must be frozen")
        rounded = weight.to(torch.float16)
        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(rounded.float())
        ctx.block_size = block_size
        return (hidden.to(torch.float16) @ rounded.T).float()


@contextmanager
def pairwise_head(reference, *, block_size=128, projection_precision="fp32"):
    """Preserve Linear's forward and state dict; change frozen-head accumulation."""
    if projection_precision not in ("fp32", "bf16", "fp16"):
        raise ValueError("Expected fp32, bf16 or fp16 projection precision")
    product = {"bf16": _BF16FrozenHead, "fp16": _FP16FrozenHead, "fp32": _FrozenHead}[
        projection_precision
    ]
    head = reference.lm_head
    if head.weight.requires_grad or head.bias is not None:
        raise ValueError("Expected a frozen bias-free lm_head")
    if head.weight.dtype != torch.float32:
        raise ValueError("The head reference uses FP32 weights")
    original = head.__dict__.get("forward")

    def forward(module, hidden):
        return product.apply(hidden, module.weight, block_size)

    head.forward = MethodType(forward, head)
    try:
        yield reference
    finally:
        if original is None:
            del head.forward
        else:
            head.forward = original
