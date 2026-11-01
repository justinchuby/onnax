# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""PyTorch Qwen checkpoint and bounded output-head benchmark paths."""

import torch
from torch.utils.checkpoint import checkpoint


class _DecoderGroup(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states, **kwargs):
        def execute(hidden):
            for layer in self.layers:
                hidden = layer(hidden, **kwargs)
            return hidden

        if torch.is_grad_enabled():
            return checkpoint(execute, hidden_states, use_reentrant=False)
        return execute(hidden_states)


def checkpoint_qwen(model, group_size):
    """Group the pinned Qwen decoder while preserving its Transformers forward."""
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1:
        raise ValueError("Expected a positive decoder group size")
    if any(kind != "full_attention" for kind in model.config.layer_types):
        raise ValueError("The Qwen benchmark groups require full-attention decoder layers")
    layers = list(model.model.layers)
    if any(isinstance(layer, _DecoderGroup) for layer in layers):
        raise ValueError("Decoder groups have already been installed")
    model.model.layers = torch.nn.ModuleList(
        _DecoderGroup(layers[start : start + group_size])
        for start in range(0, len(layers), group_size)
    )
    return model


def chunked_token_loss(model, batch, labels, chunk_size):
    """Checkpoint each frozen head/CE chunk and normalize over shifted targets.

    Transformers computes the decoder hidden states. Recomputing each head chunk
    during backward bounds live logits to chunk_size rows. Native PyTorch CE
    provides the independent loss implementation.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("Expected a positive output-head chunk size")
    hidden = model.model(**batch, use_cache=False).last_hidden_state
    rows = hidden[:, :-1].reshape(-1, hidden.shape[-1])
    targets = labels[:, 1:].reshape(-1)

    def loss_sum(values, indices):
        return torch.nn.functional.cross_entropy(
            model.lm_head(values).float(), indices, ignore_index=-100, reduction="sum"
        )

    total = hidden.new_zeros(())
    for start in range(0, len(rows), chunk_size):
        total = total + checkpoint(
            loss_sum,
            rows[start : start + chunk_size],
            targets[start : start + chunk_size],
            use_reentrant=False,
        )
    return total / targets.ne(-100).sum().clamp(min=1)


def synchronize(device):
    """Complete queued device work at benchmark timing boundaries."""
    if torch.device(device).type == "mps":
        torch.mps.synchronize()
