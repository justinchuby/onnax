# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""External, functional Adam using only Array API operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AdamState:
    step: int
    first: dict[str, Any]
    second: dict[str, Any]


def _validated_state(params, grads, state, *, xp):
    """Initialize and validate state before entering array-only arithmetic."""
    if params.keys() != grads.keys():
        raise ValueError("Parameter and gradient names must match exactly.")
    if state is None:
        state = AdamState(
            0,
            {k: xp.zeros_like(p) for k, p in params.items()},
            {k: xp.zeros_like(p) for k, p in params.items()},
        )
    if state.first.keys() != params.keys() or state.second.keys() != params.keys():
        raise ValueError("Adam state names must match parameters.")
    for name, param in params.items():
        gradient = grads[name]
        if param.dtype != xp.float32:
            raise ValueError(f"This example requires FP32 parameters: {name}.")
        if getattr(gradient, "device", None) != getattr(param, "device", None):
            raise ValueError(f"Gradient device mismatch for {name}.")
        for moment in (state.first[name], state.second[name]):
            if moment.shape != param.shape or moment.dtype != param.dtype:
                raise ValueError(f"Adam state shape/dtype mismatch for {name}.")
            if getattr(moment, "device", None) != getattr(param, "device", None):
                raise ValueError(f"Adam state device mismatch for {name}.")
        if gradient.shape != param.shape or gradient.dtype != param.dtype:
            raise ValueError(f"Gradient shape/dtype mismatch for {name}.")
    return state


def _adam_arrays(
    params, grads, first, second, corrections, *, xp, lr=0.03, beta1=0.9, beta2=0.999, eps=1e-8
):
    """Pure Array API arithmetic with explicit bias-correction inputs."""
    updated, next_first, next_second = {}, {}, {}
    for name, param in params.items():
        gradient = grads[name]
        next_first[name] = beta1 * first[name] + (1 - beta1) * gradient
        next_second[name] = beta2 * second[name] + (1 - beta2) * gradient * gradient
        mean = next_first[name] / corrections[0]
        variance = next_second[name] / corrections[1]
        updated[name] = param - lr * mean / (xp.sqrt(variance) + eps)
    return updated, next_first, next_second


def adam_step(params, grads, state=None, *, xp, lr=0.03, beta1=0.9, beta2=0.999, eps=1e-8):
    """Return new parameters/state; never mutate arrays or cross a device boundary."""
    state = _validated_state(params, grads, state, xp=xp)
    step = state.step + 1
    updated, first, second = _adam_arrays(
        params,
        grads,
        state.first,
        state.second,
        (1 - beta1**step, 1 - beta2**step),
        xp=xp,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
    )
    return updated, AdamState(step, first, second)
