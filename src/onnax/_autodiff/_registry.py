# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Separate typed forward and reverse rule contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Generic, Protocol, TypeVar

import onnx_ir as ir

from onnax import _rule_registry

_ContextT = TypeVar("_ContextT")
_ContextT_contra = TypeVar("_ContextT_contra", contravariant=True)


class _JvpRule(Protocol[_ContextT_contra]):
    def __call__(
        self,
        context: _ContextT_contra,
        node: ir.Node,
        tangents: Mapping[int, ir.Value],
        /,
    ) -> Sequence[ir.Value]: ...


class _VjpRule(Protocol[_ContextT_contra]):
    def __call__(
        self,
        context: _ContextT_contra,
        node: ir.Node,
        cotangents: Sequence[ir.Value],
        input_indices: Sequence[int],
        /,
    ) -> Mapping[int, ir.Value]: ...


class _JvpRegistry(_rule_registry._RuleRegistry[_JvpRule[_ContextT]], Generic[_ContextT]):
    """Instance-local JVP rules, independent of reverse-mode coverage."""

    _kind = "JVP"


class _VjpRegistry(_rule_registry._RuleRegistry[_VjpRule[_ContextT]], Generic[_ContextT]):
    """Instance-local VJP rules, independent of forward-mode coverage."""

    _kind = "VJP"
