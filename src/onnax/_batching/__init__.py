# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Private batching implementation; use onnax.vmap or VmapPass."""

from __future__ import annotations

from ._registry import _BatchingRegistry as _BatchingRegistry
from ._transform import VmapPass as VmapPass
