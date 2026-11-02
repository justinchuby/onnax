# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""FP32 gated-delta recurrence and a chunk-recomputed independent pullback.

This example-layer oracle establishes the state/gradient contract for the
Mobius Qwen3.5 training task. Inputs follow its LinearAttention layout.
"""

import math

import numpy as np


def _prepare(query, key, value, state, decay, beta, q_num_heads, kv_num_heads, scale):
    arrays = (query, key, value, state, decay, beta)
    if any(not isinstance(x, np.ndarray) or x.dtype != np.float32 for x in arrays):
        raise TypeError("The reference uses FP32 NumPy arrays")
    if (
        any(
            isinstance(h, bool) or not isinstance(h, int) or h < 1
            for h in (q_num_heads, kv_num_heads)
        )
        or kv_num_heads % q_num_heads
    ):
        raise ValueError("Value heads must be a positive multiple of query/key heads")
    if query.ndim != 3 or key.shape != query.shape or value.ndim != 3:
        raise ValueError("Expected rank-three query/key/value and matching query/key shapes")
    batch, length, width = query.shape
    if (
        value.shape[:2] != (batch, length)
        or width % q_num_heads
        or value.shape[2] % kv_num_heads
    ):
        raise ValueError("Batch/sequence and head widths must agree")
    key_width, value_width = width // q_num_heads, value.shape[2] // kv_num_heads
    if state.shape != (batch, kv_num_heads, key_width, value_width):
        raise ValueError(
            "Recurrent state shape must be [batch,value_heads,key_width,value_width]"
        )
    if decay.shape != (batch, length, kv_num_heads) or beta.shape != decay.shape:
        raise ValueError("Decay and beta must have one scalar per batch/token/value head")
    if not math.isfinite(scale):
        raise ValueError("Scale must be finite")
    repeats = kv_num_heads // q_num_heads
    q = np.repeat(query.reshape(batch, length, q_num_heads, key_width), repeats, axis=2)
    k = np.repeat(key.reshape(batch, length, q_num_heads, key_width), repeats, axis=2)
    v = value.reshape(batch, length, kv_num_heads, value_width)
    return q, k, v, np.float32(scale), repeats


def _step(q, k, v, state, decay, beta, scale):
    faded = np.exp(decay)[..., None, None] * state
    residual = v - np.einsum("bhij,bhi->bhj", faded, k)
    present = faded + beta[..., None, None] * k[..., :, None] * residual[..., None, :]
    output = np.einsum("bhi,bhij->bhj", q * scale, present)
    return output, present


def gated_delta(
    query, key, value, state, decay, beta, *, q_num_heads, kv_num_heads, scale=1.0
):
    """Return sequence output and final state, with the source state held immutable."""
    q, k, v, scale, _ = _prepare(
        query, key, value, state, decay, beta, q_num_heads, kv_num_heads, scale
    )
    outputs = np.empty_like(v)
    current = state.copy()
    for index in range(query.shape[1]):
        outputs[:, index], current = _step(
            q[:, index],
            k[:, index],
            v[:, index],
            current,
            decay[:, index],
            beta[:, index],
            scale,
        )
    return outputs.reshape(value.shape), current


def pullback(
    query,
    key,
    value,
    state,
    decay,
    beta,
    output_cotangent,
    state_cotangent,
    *,
    q_num_heads,
    kv_num_heads,
    scale=1.0,
    chunk_size=32,
):
    """Differentiate both outputs and replay each chunk from its saved boundary.

    Return gradients in query/key/value/state/decay/beta order plus a state-memory
    ledger. Only chunk boundaries survive the forward pass; the reverse pass
    reconstructs one chunk's states at a time. Ledger bytes count retained state
    arrays and omit input/output/gradient storage and temporary arithmetic arrays.
    """
    q, k, v, scale, repeats = _prepare(
        query, key, value, state, decay, beta, q_num_heads, kv_num_heads, scale
    )
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("Chunk size must be a positive integer")
    if output_cotangent.shape != value.shape or state_cotangent.shape != state.shape:
        raise ValueError("Cotangents must match the sequence output and final state")
    if output_cotangent.dtype != np.float32 or state_cotangent.dtype != np.float32:
        raise TypeError("Cotangents use FP32")
    length = query.shape[1]
    starts = list(range(0, length, chunk_size))
    boundaries, current = [], state.copy()
    for start in starts:
        boundaries.append(current)
        for index in range(start, min(start + chunk_size, length)):
            _, current = _step(
                q[:, index],
                k[:, index],
                v[:, index],
                current,
                decay[:, index],
                beta[:, index],
                scale,
            )
    del current
    dq, dk, dv = np.zeros_like(q), np.zeros_like(k), np.zeros_like(v)
    dg, db = np.zeros_like(decay), np.zeros_like(beta)
    go = output_cotangent.reshape(v.shape)
    carry = state_cotangent.copy()
    maximum_local_states = 0
    for chunk in reversed(range(len(starts))):
        start = starts[chunk]
        end = min(start + chunk_size, length)
        local = [boundaries[chunk]]
        for index in range(start, end):
            _, present = _step(
                q[:, index],
                k[:, index],
                v[:, index],
                local[-1],
                decay[:, index],
                beta[:, index],
                scale,
            )
            local.append(present)
        # local[0] aliases an already-counted boundary.
        maximum_local_states = max(maximum_local_states, len(local) - 1)
        for index in reversed(range(start, end)):
            previous, present = local[index - start : index - start + 2]
            qt, kt, vt, gt, bt, cotangent = (
                q[:, index],
                k[:, index],
                v[:, index],
                decay[:, index],
                beta[:, index],
                go[:, index],
            )
            faded = np.exp(gt)[..., None, None] * previous
            residual = vt - np.einsum("bhij,bhi->bhj", faded, kt)
            total = carry + (qt * scale)[..., :, None] * cotangent[..., None, :]
            dq[:, index] = np.einsum("bhij,bhj->bhi", present, cotangent) * scale
            transposed_product = np.einsum("bhij,bhi->bhj", total, kt)
            dr = bt[..., None] * transposed_product
            dv[:, index] = dr
            db[:, index] = np.sum(transposed_product * residual, axis=-1)
            dk[:, index] = bt[..., None] * np.einsum(
                "bhij,bhj->bhi", total, residual
            ) - np.einsum("bhij,bhj->bhi", faded, dr)
            dfaded = total - kt[..., :, None] * dr[..., None, :]
            dg[:, index] = np.sum(dfaded * faded, axis=(-2, -1))
            carry = np.exp(gt)[..., None, None] * dfaded
        del local, previous, present
    batch, _, _, width = q.shape
    target = (batch, length, q_num_heads, repeats, width)
    gradients = (
        dq.reshape(target).sum(axis=3).reshape(query.shape),
        dk.reshape(target).sum(axis=3).reshape(key.shape),
        dv.reshape(value.shape),
        carry,
        dg,
        db,
    )
    ledger = {
        "chunk_size": chunk_size,
        "saved_boundary_states": len(boundaries),
        "maximum_recomputed_states": maximum_local_states,
        "state_bytes": state.nbytes,
        "retained_state_bytes": (len(boundaries) + maximum_local_states) * state.nbytes,
        "full_state_history_bytes": (length + 1) * state.nbytes,
        "recomputed_tokens": length,
    }
    return gradients, ledger
