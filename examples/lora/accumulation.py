# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""External streaming FP32 accumulation of sample-mean losses and gradients."""

from __future__ import annotations


def accumulate_gradients(chunks, *, xp, evaluate=None):
    """Combine (sample_count, mean_loss, mean_gradients) without host array reads.

    Gradients must be independent sample means with the same output reduction.
    All chunks must use the same parameter point, names, shapes, dtype and device.
    Only running sums survive each iteration; input arrays are never mutated.
    Supply a backend evaluation function to materialize lazy sums per chunk.
    """
    total = 0
    loss_sum = None
    sums = None
    for count, loss, grads in chunks:
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("Each microbatch needs a positive integer sample count.")
        if loss.shape != () or loss.dtype != xp.float32 or not grads:
            raise ValueError("Expected scalar FP32 mean loss and nonempty FP32 gradients.")
        if sums is None:
            sums = {name: xp.zeros_like(value) for name, value in grads.items()}
            loss_sum = xp.zeros_like(loss)
        if sums.keys() != grads.keys():
            raise ValueError("Microbatch gradient names must match exactly.")
        for name, value in grads.items():
            if value.dtype != xp.float32 or value.shape != sums[name].shape:
                raise ValueError(f"Microbatch gradient shape/dtype mismatch for {name}.")
            device = getattr(value, "device", None)
            if device != getattr(sums[name], "device", None) or device != getattr(
                loss, "device", None
            ):
                raise ValueError(f"Microbatch gradient/loss device mismatch for {name}.")
            sums[name] = sums[name] + count * value
        loss_sum = loss_sum + count * loss
        if evaluate is not None:
            evaluate(loss_sum, *sums.values())
        total += count
    if sums is None:
        raise ValueError("At least one microbatch is required.")
    return loss_sum / total, {name: value / total for name, value in sums.items()}
