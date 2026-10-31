# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Compare stable causal CE with analytic values and derivatives."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional reference backend.

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from examples.huggingface.reference_losses import stable_token_loss


def test_long_vocabulary_dominant_class_matches_scalar_oracle():
    vocab = 151936
    logits = torch.full((1, 2, vocab), -4.0, requires_grad=True)
    with torch.no_grad():
        logits[0, 0, 0] = 10
    loss = stable_token_loss(logits, torch.zeros((1, 2), dtype=torch.int64))
    loss.backward()
    mass = (vocab - 1) * math.exp(-14)
    assert loss.dtype == logits.grad.dtype == torch.float32
    np.testing.assert_allclose(loss.detach(), math.log1p(mass), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(logits.grad[0, 0, 0], -mass / (1 + mass), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(
        logits.grad[0, 0, 1:], math.exp(-14) / (1 + mass), rtol=1e-6, atol=1e-12
    )


def test_confident_target_keeps_fp32_tail_gradient():
    logits = torch.tensor([[[0.0, 40.0], [0.0, 0.0]]], requires_grad=True)
    loss = stable_token_loss(logits, torch.ones((1, 2), dtype=torch.int64))
    loss.backward()
    mass = math.exp(-40)
    np.testing.assert_allclose(loss.detach(), math.log1p(mass), rtol=1e-5, atol=1e-25)
    np.testing.assert_allclose(
        logits.grad[0, 0],
        [mass / (1 + mass), -mass / (1 + mass)],
        rtol=1e-5,
        atol=1e-25,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mask_normalization_and_dtype(dtype):
    torch.manual_seed(37)
    logits = torch.randn(2, 5, 7).to(dtype).requires_grad_()
    labels = torch.tensor([[0, 1, 2, -100, 4], [1, 2, -100, 0, 3]])
    expected_logits = logits.detach().double().requires_grad_()
    expected = (
        torch.nn.functional.cross_entropy(
            expected_logits[:, :-1].reshape(-1, 7),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        / 11
    )
    expected.backward()
    actual = stable_token_loss(logits, labels, num_items=11)
    actual.backward()
    assert actual.dtype == torch.float32
    assert logits.grad.dtype == dtype
    torch.testing.assert_close(actual.double(), expected.detach(), rtol=1e-5, atol=1e-7)
    # Compare low-precision input gradients to their representable oracle.
    torch.testing.assert_close(
        logits.grad,
        expected_logits.grad.to(dtype),
        rtol=1e-5 if dtype == torch.float32 else torch.finfo(dtype).eps,
        atol=1e-7,
    )
    assert torch.count_nonzero(logits.grad[:, -1]) == 0
    assert torch.count_nonzero(logits.grad[:, :-1][labels[:, 1:] == -100]) == 0


@pytest.mark.parametrize("shape", [(2, 5, 1), (2, 1, 7), (0, 5, 7), (2, 0, 7)])
def test_single_class_and_empty_token_loss(shape):
    logits = torch.zeros(shape, requires_grad=True)
    labels = torch.zeros(shape[:2], dtype=torch.int64)
    loss = stable_token_loss(logits, labels)
    loss.backward()
    assert loss.item() == 0
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_all_ignored_rows_have_zero_loss_and_gradient():
    logits = torch.full((2, 3, 7), float("nan"), requires_grad=True)
    loss = stable_token_loss(logits, torch.full((2, 3), -100, dtype=torch.int64))
    loss.backward()
    assert loss.item() == 0
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_higher_order_reference_away_from_max_ties():
    logits = torch.randn(1, 3, 5, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[0, 3, 1]])

    def function(x):
        return stable_token_loss(x, labels)

    assert torch.autograd.gradcheck(function, (logits,))
    assert torch.autograd.gradgradcheck(function, (logits,))


@pytest.mark.parametrize("label", [-2, 7])
def test_invalid_target_ids_raise(label):
    with pytest.raises(ValueError, match="vocabulary"):
        stable_token_loss(torch.zeros(1, 2, 7), torch.tensor([[0, label]]))
