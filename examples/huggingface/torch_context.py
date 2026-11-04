# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""PyTorch decoder checkpoint and bounded output-head benchmark paths."""

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


def checkpoint_decoder(model, group_size):
    """Checkpoint decoder layers while preserving their Transformers mask dispatch."""
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1:
        raise ValueError("Expected a positive decoder group size")
    layers = list(model.model.layers)
    kinds = getattr(model.config, "layer_types", ())
    if group_size > 1 and (
        len(kinds) != len(layers) or any(kind != "full_attention" for kind in kinds)
    ):
        raise ValueError("Use one-layer checkpoints for hybrid or unspecified decoder layouts")
    if any(isinstance(layer, _DecoderGroup) for layer in layers):
        raise ValueError("Decoder groups have already been installed")
    model.model.layers = torch.nn.ModuleList(
        _DecoderGroup(layers[start : start + group_size])
        for start in range(0, len(layers), group_size)
    )
    return model


def chunked_token_loss(model, batch, labels, chunk_size, *, head_policy="checkpoint"):
    """Checkpoint each frozen head/CE chunk and normalize over shifted targets.

    Transformers computes the decoder hidden states. Recomputing each head chunk
    during backward bounds live logits to chunk_size rows. Native PyTorch CE
    provides the independent loss implementation.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("Expected a positive output-head chunk size")
    if head_policy not in ("checkpoint", "eager"):
        raise ValueError("Expected checkpoint or eager head policy")
    hidden = model.model(**batch, use_cache=False).last_hidden_state
    if head_policy == "eager":
        from examples.huggingface.torch_head import eager_head_sum

        total = eager_head_sum(model.lm_head, hidden, labels, chunk_size)
        return total / labels[:, 1:].ne(-100).sum().clamp(min=1)
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
