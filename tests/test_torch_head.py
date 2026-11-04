# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Independent eager-head loss, ownership and backward contracts."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional Torch dependency.

import copy
import math

import pytest

torch = pytest.importorskip("torch")

from examples.huggingface.torch_head import eager_head_sum


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("chunk_size", [1, 3, 8])
def test_eager_head_matches_native_loss_and_upstream(precision, chunk_size):
    torch.manual_seed(7)
    head = torch.nn.Linear(4, 7).requires_grad_(False)
    if precision == "bf16":
        from examples.huggingface.mixed_precision import reference_projections

        reference_projections(head)
    reference = copy.deepcopy(head)
    calls = []
    head.register_forward_hook(lambda *args: calls.append(1))
    hidden = torch.randn(2, 5, 4, requires_grad=True)
    expected_hidden = hidden.detach().clone().requires_grad_()
    labels = torch.tensor([[1, 2, -100, 3, 4], [2, -100, 5, 6, -100]])
    actual = eager_head_sum(head, hidden, labels, chunk_size)
    expected = torch.nn.functional.cross_entropy(
        reference(expected_hidden[:, :-1]).float().reshape(-1, 7),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
    count = math.ceil(int(labels[:, 1:].ne(-100).sum()) / chunk_size)
    assert len(calls) == count
    actual.backward(torch.tensor(0.37), retain_graph=True)
    expected.backward(torch.tensor(0.37))
    torch.testing.assert_close(hidden.grad, expected_hidden.grad, rtol=1e-4, atol=1e-6)
    actual.backward(torch.tensor(0.37))
    torch.testing.assert_close(hidden.grad, 2 * expected_hidden.grad, rtol=1e-4, atol=1e-6)
    assert len(calls) == count
    assert all(p.grad is None for p in head.parameters())
    assert torch.count_nonzero(hidden.grad[:, -1]) == 0


def test_eager_head_all_ignored_skips_projection():
    head = torch.nn.Linear(4, 7).requires_grad_(False)
    calls = []
    head.register_forward_hook(lambda *args: calls.append(1))
    hidden = torch.randn(2, 5, 4, requires_grad=True)
    labels = torch.full((2, 5), -100)
    loss = eager_head_sum(head, hidden, labels, 3)
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(hidden.grad) == 0
    assert calls == []
    with torch.no_grad():
        evaluation = eager_head_sum(head, hidden, labels, 3)
    assert not evaluation.requires_grad and evaluation.item() == 0


def test_eager_head_frozen_weight_version_is_checked():
    head = torch.nn.Linear(4, 7).requires_grad_(False)
    hidden = torch.randn(1, 3, 4, requires_grad=True)
    loss = eager_head_sum(head, hidden, torch.ones(1, 3, dtype=torch.int64), 2)
    with torch.no_grad():
        head.weight.add_(0.1)
    with pytest.raises(RuntimeError, match="inplace"):
        loss.backward()


def test_eager_head_requires_frozen_parameters():
    with pytest.raises(ValueError, match="frozen"):
        eager_head_sum(torch.nn.Linear(4, 7), torch.ones(1, 3, 4), torch.ones(1, 3).long(), 2)


def test_eager_head_evaluation_has_no_gradient_graph():
    head = torch.nn.Linear(4, 7).requires_grad_(False)
    hidden = torch.randn(1, 4, 4, requires_grad=True)
    labels = torch.tensor([[1, 2, -100, 4]])
    expected = eager_head_sum(head, hidden, labels, 2).detach()
    with torch.no_grad():
        actual = eager_head_sum(head, hidden, labels, 2)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
    assert not actual.requires_grad


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_eager_head_accumulates_variable_valid_token_counts(precision):
    from types import SimpleNamespace

    from examples.huggingface.torch_context import chunked_token_loss

    class Body(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(4, 6)

        def forward(self, features, **kwargs):
            return SimpleNamespace(last_hidden_state=self.projection(features))

    torch.manual_seed(14)
    body = Body()
    head = torch.nn.Linear(6, 7).requires_grad_(False)
    if precision == "bf16":
        from examples.huggingface.mixed_precision import reference_projections

        reference_projections(head)
    reference_body, reference_head = copy.deepcopy(body), copy.deepcopy(head)
    model = SimpleNamespace(model=body, lm_head=head)
    batches = [
        (torch.randn(1, 4, 4), torch.tensor([[1, 2, -100, 3]])),
        (torch.randn(1, 5, 4), torch.tensor([[1, -100, 4, 2, 6]])),
        (torch.randn(1, 3, 4), torch.full((1, 3), -100)),
    ]
    total = sum(int(labels[:, 1:].ne(-100).sum()) for _, labels in batches)
    for features, labels in batches:
        count = int(labels[:, 1:].ne(-100).sum())
        loss = chunked_token_loss(
            model, {"features": features}, labels, 2, head_policy="eager"
        )
        (loss * count / total).backward()
        logits = reference_head(reference_body(features).last_hidden_state[:, :-1]).float()
        expected = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 7),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        (expected / total).backward()
    for actual, expected in zip(body.parameters(), reference_body.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-4, atol=1e-6)
