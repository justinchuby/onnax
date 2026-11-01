# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Global decorator registration for external transformation rules."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from threading import RLock
from typing import Protocol, TypeVar

import onnx_ir as ir

import onnax
from onnax._rule_registry import _RuleRegistry


class JvpRule(Protocol):
    def __call__(
        self, ctx: onnax.AutodiffContext, node: ir.Node, tangents: Mapping[int, ir.Value], /
    ) -> Sequence[ir.Value]: ...


class VjpRule(Protocol):
    def __call__(
        self,
        ctx: onnax.AutodiffContext,
        node: ir.Node,
        cotangents: Sequence[ir.Value],
        input_indices: Sequence[int],
        /,
    ) -> Mapping[int, ir.Value]: ...


class BatchingRule(Protocol):
    def __call__(self, ctx: onnax.BatchingContext, node: ir.Node, /) -> None: ...


_JvpRuleT = TypeVar("_JvpRuleT", bound=JvpRule)
_VjpRuleT = TypeVar("_VjpRuleT", bound=VjpRule)
_BatchingRuleT = TypeVar("_BatchingRuleT", bound=BatchingRule)


class _RuleSet:
    """Internal storage for global decorator registrations.

    Rules are isolated per collection. Both version bounds are inclusive and
    required. Later registrations override overlapping versions, including builtins.
    Transform invocations take a snapshot; later registrations cannot change
    an in-progress transform. Callbacks operate on a cloned graph, never the source.

    JVP returns one tangent per node output; VJP returns exactly the requested
    input indices. Batching callbacks update the node and register its outputs.
    Custom outputs must have tensor dtype and rank annotations if the inference
    engine cannot infer them. Rules do not install execution kernels or schemas.
    All custom inputs are treated as data; select intended floating inputs
    explicitly. Subgraphs, randomness, and overloaded calls remain restricted.
    """

    def __init__(self) -> None:
        self._jvp: _RuleRegistry[JvpRule] = _RuleRegistry()
        self._vjp: _RuleRegistry[VjpRule] = _RuleRegistry()
        self._batching: _RuleRegistry[BatchingRule] = _RuleRegistry()
        self._jvp._kind = "JVP"
        self._vjp._kind = "VJP"
        self._batching._kind = "batching"

    def jvp(
        self, domain: str, op_type: str, *, since_version: int, until_version: int
    ) -> Callable[[JvpRule], JvpRule]:
        """Register a forward rule while retaining the decorated callable."""
        return self._jvp.rule(
            domain, op_type, since_version=since_version, until_version=until_version
        )

    def vjp(
        self, domain: str, op_type: str, *, since_version: int, until_version: int
    ) -> Callable[[VjpRule], VjpRule]:
        """Register a reverse rule while retaining the decorated callable."""
        return self._vjp.rule(
            domain, op_type, since_version=since_version, until_version=until_version
        )

    def batching(
        self, domain: str, op_type: str, *, since_version: int, until_version: int
    ) -> Callable[[BatchingRule], BatchingRule]:
        """Register a batching rule using the public BatchingContext helpers."""
        return self._batching.rule(
            domain, op_type, since_version=since_version, until_version=until_version
        )

    def copy(self) -> _RuleSet:
        """Return an independent snapshot of this collection."""
        result = _RuleSet()
        result._jvp.extend(self._jvp)
        result._vjp.extend(self._vjp)
        result._batching.extend(self._batching)
        return result


_BUILTINS = _RuleSet()


def _builtin(mode: str, operators: frozenset[str]):
    """Declare builtin families with the same decorator path as extensions."""

    def decorate(rule):
        for op in sorted(operators):
            getattr(_BUILTINS, mode)(
                "", op, since_version=20 if op == "Gelu" else 18, until_version=28
            )(rule)
        return rule

    return decorate


def _load_builtin_rules() -> _RuleSet:
    """Import declarative families once without circular transform imports."""
    import importlib

    for package, modules in (
        (
            "_autodiff",
            (
                "_elementwise",
                "_extrema",
                "_linalg",
                "_layout",
                "_reductions",
                "_normalization",
                "_attention",
                "_rotary",
            ),
        ),
        (
            "_batching",
            ("_elementwise", "_linalg", "_nn", "_indexing", "_layout", "_reductions"),
        ),
    ):
        for module in modules:
            importlib.import_module(f"onnax.{package}.{module}")
    return _BUILTINS


_LOCK = RLock()
_EXTENSIONS = _RuleSet()


def _snapshot() -> _RuleSet:
    with _LOCK:
        return _EXTENSIONS.copy()


def _register(mode, domain, op_type, since_version, until_version):
    def decorate(rule):
        global _EXTENSIONS
        with _LOCK:
            candidate = _EXTENSIONS.copy()
            getattr(candidate, mode)(
                domain, op_type, since_version=since_version, until_version=until_version
            )(rule)
            _EXTENSIONS = candidate
        return rule

    return decorate


def register_jvp(
    domain: str, op_type: str, *, since_version: int, until_version: int
) -> Callable[[_JvpRuleT], _JvpRuleT]:
    """Globally register a forward rule for an inclusive, explicit version interval.

    Returns the decorated callable unchanged. Later rules override extensions or
    builtins within the new interval. Invalid registrations leave existing rules intact.
    Transforms snapshot the global registry at invocation, so plugin imports
    require no changes to jvp(model) or JvpPass()(model).
    """
    return _register("jvp", domain, op_type, since_version, until_version)


def register_vjp(
    domain: str, op_type: str, *, since_version: int, until_version: int
) -> Callable[[_VjpRuleT], _VjpRuleT]:
    """Globally register a reverse rule, used by both vjp and grad.

    The callback returns exactly the requested input-index contributions.
    Registration preserves the callable; later rules override overlapping versions.
    """
    return _register("vjp", domain, op_type, since_version, until_version)


def register_batching(
    domain: str, op_type: str, *, since_version: int, until_version: int
) -> Callable[[_BatchingRuleT], _BatchingRuleT]:
    """Globally register a batching callback using public context helpers.

    The callback updates the cloned node and registers every output. Use
    ctx.elementwise(node) only when the operator obeys elementwise broadcasting.
    Later registrations override overlapping versions, including builtin rules.
    Rules do not install execution kernels or shape inference schemas.
    """
    return _register("batching", domain, op_type, since_version, until_version)
