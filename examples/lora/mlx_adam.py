# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Compile the example's external Array API Adam arithmetic for MLX."""

from dataclasses import dataclass, field
from typing import Any

from examples.lora.adam import AdamState, _adam_arrays, _validated_state


@dataclass(frozen=True)
class CompiledAdam:
    """A stateless compiled update; parameters and optimizer state stay explicit."""

    lr: float = 0.03
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    _kernel: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        import mlx.core as mx

        lr, beta1, beta2, eps = self.lr, self.beta1, self.beta2, self.eps

        def kernel(params, grads, first, second, corrections):
            return _adam_arrays(
                params,
                grads,
                first,
                second,
                corrections,
                xp=mx,
                lr=lr,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
            )

        object.__setattr__(self, "_kernel", mx.compile(kernel))

    def __call__(self, params, grads, state=None):
        import mlx.core as mx

        state = _validated_state(params, grads, state, xp=mx)
        if isinstance(state.step, bool) or not isinstance(state.step, int) or state.step < 0:
            raise ValueError("Compiled Adam requires a nonnegative Python integer step")
        step = state.step + 1
        # Preserve Python's bias-correction calculation and pass values as data,
        # so successive steps reuse the compiled graph instead of retracing.
        corrections = mx.array([1 - self.beta1**step, 1 - self.beta2**step], dtype=mx.float32)
        updated, first, second = self._kernel(
            params, grads, state.first, state.second, corrections
        )
        return updated, AdamState(step, first, second)
