# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Identify session outputs that need copies for PyTorch version tracking."""

import heapq
from collections.abc import Iterable, Sequence

import numpy as np


def _byte_bounds(array: np.ndarray) -> tuple[int, int]:
    lower = upper = int(array.__array_interface__["data"][0])
    if array.flags.c_contiguous or array.flags.f_contiguous:
        return lower, lower + array.nbytes
    for length, stride in zip(array.shape, array.strides, strict=True):
        extent = (length - 1) * stride
        lower += min(0, extent)
        upper += max(0, extent)
    return lower, upper + array.itemsize


def output_aliases(arrays: Sequence[np.ndarray], feeds: Iterable[np.ndarray]) -> list[bool]:
    """Flag read-only outputs and exact overlaps with feeds or other outputs.

    A byte-interval sweep narrows the candidates before exact NumPy overlap
    checks. Disjoint strided views retain their original ownership decisions.
    """
    feeds = tuple(feeds)
    # Direct checks avoid sweep setup for scalar and small result lists.
    if len(arrays) <= 4:
        return [
            not value.flags.writeable
            or any(np.shares_memory(value, feed) for feed in feeds)
            or any(i != j and np.shares_memory(value, other) for j, other in enumerate(arrays))
            for i, value in enumerate(arrays)
        ]
    aliases = [not value.flags.writeable for value in arrays]
    entries: list[tuple[int, int, int | None, np.ndarray]] = []
    for position, array in enumerate((*arrays, *feeds)):
        if array.size:
            lower, upper = _byte_bounds(array)
            entries.append((lower, upper, position if position < len(arrays) else None, array))
    entries.sort(key=lambda entry: entry[0])
    active: dict[int, tuple[int | None, np.ndarray]] = {}
    ends: list[tuple[int, int]] = []
    for serial, (lower, upper, index, array) in enumerate(entries):
        while ends and ends[0][0] <= lower:
            _, expired = heapq.heappop(ends)
            del active[expired]
        for other_index, other in active.values():
            if (index is None or aliases[index]) and (
                other_index is None or aliases[other_index]
            ):
                continue
            if np.shares_memory(array, other):
                if index is not None:
                    aliases[index] = True
                if other_index is not None:
                    aliases[other_index] = True
        active[serial] = index, array
        heapq.heappush(ends, (upper, serial))
    return aliases
