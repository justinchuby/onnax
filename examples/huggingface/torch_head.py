# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Native Torch chunked cross entropy with eager hidden-state gradients."""

import torch
from torch.autograd.function import once_differentiable


class _EagerHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, labels, head, chunk_size, compute_gradient):
        rows = hidden[:, :-1].reshape(-1, hidden.shape[-1])
        targets = labels[:, 1:].reshape(-1)
        selected = torch.nonzero(targets != -100, as_tuple=True)[0]
        total = hidden.new_zeros((), dtype=torch.float32)
        gradient = torch.zeros_like(hidden) if compute_gradient else None
        for start in range(0, selected.numel(), chunk_size):
            indices = selected[start : start + chunk_size]
            values = rows.index_select(0, indices).detach()
            with torch.set_grad_enabled(compute_gradient):
                values.requires_grad_(compute_gradient)
                loss = torch.nn.functional.cross_entropy(
                    head(values).float(), targets.index_select(0, indices), reduction="sum"
                )
                if compute_gradient:
                    (partial,) = torch.autograd.grad(loss, values)
            total = total + loss.detach()
            if compute_gradient:
                batches = indices // (hidden.shape[1] - 1)
                positions = indices % (hidden.shape[1] - 1)
                gradient[batches, positions] = partial
            del loss, values
        ctx.compute_gradient = compute_gradient
        if compute_gradient:
            # Retain version checks for the frozen parameters, without copying them.
            ctx.save_for_backward(gradient, *tuple(head.parameters()))
        return total

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        if not ctx.compute_gradient:
            return (None,) * 5
        gradient = ctx.saved_tensors[0]
        return gradient * upstream, None, None, None, None


def eager_head_sum(head, hidden, labels, chunk_size):
    """Sum valid shifted-token losses and save their hidden gradient per chunk."""
    if any(parameter.requires_grad for parameter in head.parameters()):
        raise ValueError("Eager head requires frozen head parameters")
    return _EagerHead.apply(
        hidden, labels, head, chunk_size, torch.is_grad_enabled() and hidden.requires_grad
    )
