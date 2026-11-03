# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Shared half-forward, FP32-adjoint products for independent Torch references."""

import torch


def _save_needed_operands(ctx, a, b):
    if any(ctx.needs_input_grad):
        ctx.save_for_backward(
            *[
                value.float()
                for value, needed in (
                    (a, ctx.needs_input_grad[1]),
                    (b, ctx.needs_input_grad[0]),
                )
                if needed
            ]
        )


class BFloat16Product(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        qa, qb = a.to(torch.bfloat16), b.to(torch.bfloat16)
        _save_needed_operands(ctx, qa, qb)
        return (qa @ qb).float()

    @staticmethod
    def backward(ctx, gradient):
        saved = iter(ctx.saved_tensors)
        a = next(saved) if ctx.needs_input_grad[1] else None
        b = next(saved) if ctx.needs_input_grad[0] else None
        da = gradient @ b.T if ctx.needs_input_grad[0] else None
        db = (
            a.reshape(-1, a.shape[-1]).T @ gradient.reshape(-1, gradient.shape[-1])
            if ctx.needs_input_grad[1]
            else None
        )
        return da, db


class Float16Product(BFloat16Product):
    @staticmethod
    def forward(ctx, a, b):
        qa, qb = a.to(torch.float16), b.to(torch.float16)
        _save_needed_operands(ctx, qa, qb)
        return (qa @ qb).float()


def projection_product(precision):
    """Select the declared BF16/FP16 compute policy with FP32 boundary tensors."""
    if precision == "bf16":
        return BFloat16Product
    if precision == "fp16":
        return Float16Product
    raise ValueError("Expected bf16 or fp16 projection precision")
