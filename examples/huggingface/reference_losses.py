# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Independent PyTorch CE reference with stable target-class gradients."""

import torch


def stable_token_loss(logits, labels, num_items=None):
    """Causal hard-label CE, accumulating half-precision logits in FP32.

    Compute softplus(logsumexp(other classes) - target) so the target
    derivative avoids subtracting one from a probability near one.
    """
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or labels.dtype != torch.int64:
        raise ValueError("Expected floating logits [batch,sequence,vocab] and INT64 labels")
    if not logits.is_floating_point() or logits.shape[-1] < 1:
        raise ValueError("Expected floating logits with at least one class")
    if labels.device != logits.device:
        raise ValueError("Logits and labels must use the same device")
    working = logits if logits.dtype == torch.float64 else logits.float()
    targets = labels[:, 1:].reshape(-1)
    if torch.any(((targets < 0) & (targets != -100)) | (targets >= logits.shape[-1])):
        raise ValueError("Labels must be vocabulary IDs or -100")
    if logits.shape[-1] == 1:
        return working[:, :-1, :0].sum()
    rows = working[:, :-1].reshape(-1, logits.shape[-1])
    valid = targets.ne(-100)
    rows = torch.where(valid[:, None], rows, 0)
    index = targets.clamp(min=0)
    selected = rows.gather(-1, index[:, None]).squeeze(-1)
    other = torch.arange(rows.shape[-1], device=rows.device)[None, :] != index[:, None]
    # The shift cancels algebraically; differentiating it adds a cancelling
    # reduction and can leave rounding error at its selected maximum.
    maximum = torch.where(other, rows, -torch.inf).max(-1).values.detach()
    shifted = torch.where(other, rows - maximum[:, None], 0)
    mass = torch.where(other, shifted.exp(), 0).sum(-1)
    losses = torch.nn.functional.softplus((maximum - selected) + mass.log())
    count = (
        valid.sum() if num_items is None else torch.as_tensor(num_items, device=rows.device)
    )
    return torch.where(valid, losses, 0).sum() / count.clamp(min=1)
