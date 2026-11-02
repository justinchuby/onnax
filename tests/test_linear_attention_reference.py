# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Stateful, grouped-head gated-delta derivatives against independent autograd."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional independent oracle.

import numpy as np
import pytest

from examples.huggingface.linear_attention_reference import gated_delta, pullback


def inputs(*, batch=2, length=17, query_heads=2, value_heads=4):
    rng = np.random.default_rng(20261102)
    shapes = [(batch, length, query_heads * 3)] * 2 + [
        (batch, length, value_heads * 5),
        (batch, value_heads, 3, 5),
        (batch, length, value_heads),
        (batch, length, value_heads),
    ]
    values = [rng.normal(0, 0.2, shape).astype(np.float32) for shape in shapes]
    values[4] = -rng.uniform(0.01, 0.3, shapes[4]).astype(np.float32)
    values[5] = rng.uniform(0.2, 0.8, shapes[5]).astype(np.float32)
    return values


def reference(values, output_cotangent, state_cotangent, *, query_heads, value_heads, scale):
    torch = pytest.importorskip("torch")
    leaves = [torch.tensor(x, requires_grad=True) for x in values]
    q, k, v, state, decay, beta = leaves
    batch, length, _ = q.shape
    repeats = value_heads // query_heads
    q = q.reshape(batch, length, query_heads, 3).repeat_interleave(repeats, dim=2)
    k = k.reshape(batch, length, query_heads, 3).repeat_interleave(repeats, dim=2)
    v = v.reshape(batch, length, value_heads, 5)
    outputs = []
    for index in range(length):
        faded = torch.exp(decay[:, index])[..., None, None] * state
        key = k[:, index]
        residual = v[:, index] - (faded.transpose(-1, -2) @ key[..., None]).squeeze(-1)
        state = (
            faded + beta[:, index, :, None, None] * key[..., :, None] * residual[..., None, :]
        )
        outputs.append(((q[:, index] * scale)[..., None, :] @ state).squeeze(-2))
    output = torch.stack(outputs, dim=1).reshape(values[2].shape)
    loss = (output * torch.from_numpy(output_cotangent)).sum() + (
        state * torch.from_numpy(state_cotangent)
    ).sum()
    gradients = torch.autograd.grad(loss, leaves)
    return output.detach().numpy(), state.detach().numpy(), [x.numpy() for x in gradients]


@pytest.mark.parametrize("query_heads,value_heads", [(1, 1), (1, 3), (1, 4), (2, 4)])
@pytest.mark.parametrize("length,chunk_size", [(1, 1), (3, 2), (17, 4), (17, 32)])
@pytest.mark.parametrize("cotangent_mode", ["both", "output", "state"])
def test_all_six_gradients_match_autograd(
    query_heads, value_heads, length, chunk_size, cotangent_mode
):
    values = inputs(length=length, query_heads=query_heads, value_heads=value_heads)
    original = [x.tobytes() for x in values]
    rng = np.random.default_rng(71)
    gy = rng.normal(size=values[2].shape).astype(np.float32)
    gs = rng.normal(size=values[3].shape).astype(np.float32)
    if cotangent_mode == "output":
        gs.fill(0)
    if cotangent_mode == "state":
        gy.fill(0)
    options = dict(q_num_heads=query_heads, kv_num_heads=value_heads, scale=0.7)
    actual_y, actual_s = gated_delta(*values, **options)
    gradients, ledger = pullback(*values, gy, gs, **options, chunk_size=chunk_size)
    expected_y, expected_s, expected_gradients = reference(
        values, gy, gs, query_heads=query_heads, value_heads=value_heads, scale=0.7
    )
    np.testing.assert_allclose(actual_y, expected_y, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(actual_s, expected_s, rtol=1e-4, atol=1e-6)
    for actual, expected, primal in zip(gradients, expected_gradients, values, strict=True):
        assert actual.dtype == np.float32 and actual.shape == primal.shape
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)
    assert [x.tobytes() for x in values] == original
    assert ledger["recomputed_tokens"] == length


def test_carry_gradient_composes_across_external_chunks():
    values = inputs(length=11)
    split = 4
    options = dict(q_num_heads=2, kv_num_heads=4, scale=0.7)
    rng = np.random.default_rng(19)
    gy = rng.normal(size=values[2].shape).astype(np.float32)
    gs = rng.normal(size=values[3].shape).astype(np.float32)
    full, _ = pullback(*values, gy, gs, **options, chunk_size=3)
    left = [x[:, :split] if i != 3 else x for i, x in enumerate(values)]
    _, boundary = gated_delta(*left, **options)
    right = [x[:, split:] if i != 3 else boundary for i, x in enumerate(values)]
    after, _ = pullback(*right, gy[:, split:], gs, **options, chunk_size=2)
    before, _ = pullback(*left, gy[:, :split], after[3], **options, chunk_size=2)
    for index, expected in enumerate(full):
        actual = (
            before[3] if index == 3 else np.concatenate((before[index], after[index]), axis=1)
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)


def test_chunk_ledger_counts_boundary_and_local_state_storage():
    values = inputs(length=129)
    _, ledger = pullback(
        *values,
        np.ones_like(values[2]),
        np.ones_like(values[3]),
        q_num_heads=2,
        kv_num_heads=4,
        chunk_size=16,
    )
    assert ledger["saved_boundary_states"] == 9
    assert ledger["maximum_recomputed_states"] == 16
    assert ledger["retained_state_bytes"] == 25 * values[3].nbytes
    assert ledger["retained_state_bytes"] < ledger["full_state_history_bytes"] / 4
