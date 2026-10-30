# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Private operator-rule dispatch shared by batching and autodiff.

Registrations use inclusive, finite version intervals. Both ends are required:
a new opset must not silently acquire a rule validated only for older opsets.
Overlapping registrations (including duplicates) are errors, even when they use
the same callable. There is deliberately no override. Public decorators combine
these instance-local registries into a process-global extension registry.

Dispatch uses the explicit ``Node.version`` when present, as elsewhere in the
IR; otherwise it requires the caller's enclosing graph/function opset imports.
It never consults the model's imports implicitly, falls back to another domain,
or chooses the nearest supported version across a gap. The standard ONNX domain
aliases ``""`` and ``"ai.onnx"`` are equivalent. Conflicting alias imports are
rejected rather than resolved by dictionary insertion order.

This utility checks dispatch applicability, not operator schemas, shapes, local
function bodies, or rule-specific input constraints. Transformations must check
those preconditions themselves before invoking a rule.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

import onnx_ir as ir

from onnax._diagnostics import _argument_context, _node_context

_RuleT = TypeVar("_RuleT", bound=Callable[..., object])


def _normalize_domain(domain: str) -> str:
    return "" if domain == "ai.onnx" else domain


def _check_version(version: int, *, context: str = "opset registration/lookup") -> None:
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError(
            f"Opset version must be a positive integer, got {_argument_context(version)} ({context}). "
            "Supply an integer >= 1 in the registration range or enclosing opset import."
        )


def _effective_opset(node: ir.Node, opset_imports: Mapping[str, int]) -> int:
    """Resolve the node version without borrowing imports from another scope."""
    domain = _normalize_domain(node.domain)
    context = _node_context(node, opset_imports, "opset lookup")
    if domain == "" and "" in opset_imports and "ai.onnx" in opset_imports:
        _check_version(opset_imports[""], context=context)
        _check_version(opset_imports["ai.onnx"], context=context)
        if opset_imports[""] != opset_imports["ai.onnx"]:
            raise ValueError(
                f"Conflicting opset imports for ''={_argument_context(opset_imports[''])} and "
                f"'ai.onnx'={_argument_context(opset_imports['ai.onnx'])}: {context}. "
                "Use one standard-domain import, or give both aliases the same version."
            )
    if node.version is not None:
        version = node.version
    elif domain in opset_imports:
        version = opset_imports[domain]
    elif domain == "" and "ai.onnx" in opset_imports:
        version = opset_imports["ai.onnx"]
    else:
        raise ValueError(
            f"No opset import for domain {node.domain!r} while resolving "
            f"{_node_context(node, opset_imports, 'rule lookup')}. "
            f"Available imports: {_argument_context(dict(opset_imports))}. Add the operator's actual "
            "version to its enclosing graph/function imports; do not borrow another domain."
        )
    _check_version(version, context=context)
    return version


@dataclass(frozen=True)
class _Registration(Generic[_RuleT]):
    since_version: int
    until_version: int
    rule: _RuleT


class _RuleRegistry(Generic[_RuleT]):
    """Typed, instance-local registry; use the mode-specific subclasses.

    ``register`` installs a callable directly; ``rule`` is its decorator form. It rejects
    invalid ranges and any overlap atomically. Adjacent ranges are allowed.
    ``lookup`` returns a rule or raises ``NotImplementedError`` for missing or
    unsupported registrations; invalid version/import information raises
    ``ValueError``. Nonempty node overloads are rejected because a plain operator
    rule cannot safely stand in for a local function implementation.
    """

    _kind = "operator"

    def __init__(self) -> None:
        self._rules: dict[tuple[str, str], list[_Registration[_RuleT]]] = {}

    def register(
        self,
        domain: str,
        op_type: str,
        rule: _RuleT,
        *,
        since_version: int,
        until_version: int,
    ) -> None:
        """Register a callable for exactly the inclusive supported interval."""
        _check_version(
            since_version,
            context=f"{self._kind} {_argument_context(domain)}::{_argument_context(op_type)} since_version",
        )
        _check_version(
            until_version,
            context=f"{self._kind} {_argument_context(domain)}::{_argument_context(op_type)} until_version",
        )
        if since_version > until_version:
            raise ValueError(
                f"since_version={since_version} must not exceed until_version={until_version} "
                f"for {_argument_context(domain)}::{_argument_context(op_type)}. "
                "Supply an inclusive range in ascending order."
            )
        if not isinstance(domain, str) or not isinstance(op_type, str):
            raise TypeError(
                f"Rule domain and operator name must be strings; got {type(domain).__name__}/{type(op_type).__name__}. Use a domain string and an ONNX operator name."
            )
        if not op_type:
            raise ValueError(
                f"Operator name must not be empty (domain={_argument_context(domain)}). "
                "Register the exact ONNX operator name, such as 'Add'."
            )
        if not callable(rule):
            raise TypeError(
                f"An operator rule must be callable for {_argument_context(domain)}::"
                f"{_argument_context(op_type)}; "
                f"got {type(rule).__name__}. Pass the rule function, not its result."
            )
        key = (_normalize_domain(domain), op_type)
        registrations = self._rules.get(key, [])
        for existing in registrations:
            if (
                since_version <= existing.until_version
                and existing.since_version <= until_version
            ):
                raise ValueError(
                    f"Overlapping {self._kind} rules for {_argument_context(key)}: "
                    f"[{since_version}, {until_version}] overlaps "
                    f"[{existing.since_version}, {existing.until_version}]. "
                    "Overrides are not supported; remove the duplicate registration or use non-overlapping version intervals."
                )
        self._rules.setdefault(key, []).append(
            _Registration(since_version, until_version, rule)
        )
        self._rules[key].sort(key=lambda entry: entry.since_version)

    def rule(
        self, domain: str, op_type: str, *, since_version: int, until_version: int
    ) -> Callable[[_RuleT], _RuleT]:
        """Decorate a rule without replacing the decorated function."""

        def decorate(rule: _RuleT) -> _RuleT:
            self.register(
                domain, op_type, rule, since_version=since_version, until_version=until_version
            )
            return rule

        return decorate

    def extend(self, other: _RuleRegistry[_RuleT]) -> None:
        """Atomically merge rules; registrations are immutable snapshots."""
        merged = type(self)()
        merged._kind = self._kind
        merged._rules = {key: list(entries) for key, entries in self._rules.items()}
        for (domain, op), entries in other._rules.items():
            for entry in entries:
                merged.register(
                    domain,
                    op,
                    entry.rule,
                    since_version=entry.since_version,
                    until_version=entry.until_version,
                )
        self._rules = merged._rules

    def lookup(self, node: ir.Node, opset_imports: Mapping[str, int]) -> _RuleT:
        """Resolve a rule using the imports of the node's enclosing scope."""
        version = _effective_opset(node, opset_imports)
        if node.overload:
            raise NotImplementedError(
                f"No {self._kind} rule for overloaded operator {node.op_identifier()!r}: "
                f"{_node_context(node, opset_imports, self._kind)}. "
                "Inline the local function body and ensure its operators have supported rules; "
                "do not remove the overload to dispatch a different implementation."
            )
        key = (_normalize_domain(node.domain), node.op_type)
        registrations = self._rules.get(key, ())
        for entry in registrations:
            if entry.since_version <= version <= entry.until_version:
                return entry.rule
        supported = ", ".join(
            f"[{entry.since_version}, {entry.until_version}]" for entry in registrations
        )
        raise NotImplementedError(
            f"No {self._kind} rule for {_argument_context(key)} at opset {version}. "
            f"Registered intervals: {supported or 'none'}. "
            f"{_node_context(node, opset_imports, self._kind)}. "
            + (
                "Use a model authored for a registered interval, or implement and validate "
                "a rule for this version in a separate registry; do not just relabel imports."
                if supported
                else "Implement and validate a rule for this exact domain/operator, or "
                "express the computation with supported operators preserving its semantics."
            )
        )
