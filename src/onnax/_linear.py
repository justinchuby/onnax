# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Optional algebraic factoring of elementwise linear islands."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import onnx_ir as ir

from onnax._diagnostics import _argument_context, _value_context
from onnax._rule_registry import _effective_opset

_FLOATS = frozenset((ir.DataType.FLOAT, ir.DataType.DOUBLE))
_OPS = frozenset(("Add", "Sub", "Mul", "Div", "Neg"))


@dataclass(eq=False)
class _Coefficient:
    op: str
    inputs: tuple[_Term, ...]


# None represents a typed scalar one, materialized only for rewritten islands.
_Term: TypeAlias = "ir.Value | _Coefficient | None"


@dataclass
class _Expression:
    terms: dict[ir.Value, _Term]
    steps: int = 0
    hoisted: bool = False


def _expression(
    node: ir.Node,
    dependent: set[ir.Value],
    expressions: dict[ir.Value, _Expression],
    max_terms: int,
) -> _Expression | None:
    def expr(value: ir.Value) -> _Expression:
        return expressions.get(value, _Expression({value: None}))

    a = node.inputs[0]
    assert a is not None
    left = expr(a)
    if node.op_type == "Neg":
        return _Expression(
            {basis: _Coefficient("Neg", (c,)) for basis, c in left.terms.items()},
            min(2, left.steps + 1),
            left.hoisted or any(c is not None for c in left.terms.values()),
        )
    b = node.inputs[1]
    assert b is not None
    right = expr(b)
    if node.op_type in ("Mul", "Div"):
        if a in dependent and b not in dependent:
            terms: dict[ir.Value, _Term] = {
                basis: b
                if c is None and node.op_type == "Mul"
                else _Coefficient(node.op_type, (c, b))
                for basis, c in left.terms.items()
            }
            return _Expression(
                terms,
                min(2, left.steps + 1),
                left.hoisted or any(c is not None for c in left.terms.values()),
            )
        if node.op_type == "Mul" and b in dependent and a not in dependent:
            return _Expression(
                {
                    basis: a if c is None else _Coefficient("Mul", (c, a))
                    for basis, c in right.terms.items()
                },
                min(2, right.steps + 1),
                right.hoisted or any(c is not None for c in right.terms.values()),
            )
    elif a in dependent and b in dependent:
        if len(left.terms.keys() | right.terms.keys()) > max_terms:
            return None
        terms = dict(left.terms)
        for basis, coefficient in right.terms.items():
            if basis in terms:
                terms[basis] = _Coefficient(node.op_type, (terms[basis], coefficient))
            else:
                terms[basis] = (
                    coefficient
                    if node.op_type == "Add"
                    else _Coefficient("Neg", (coefficient,))
                )
        return _Expression(
            terms,
            min(2, left.steps + right.steps + 1),
            left.hoisted or right.hoisted or bool(left.terms.keys() & right.terms.keys()),
        )
    return None


class HoistLinearCoefficientsPass(ir.passes.FunctionalPass):
    """Factor elementwise linear islands relative to selected runtime inputs.

    For example, ``(a * v) * b + c * v`` becomes ``v * (a * b + c)`` when
    ``a``, ``b``, and ``c`` do not depend on the selected input ``v``. Subsequent
    CSE can share those coefficients across calls unrolled with different
    directions. Coefficients remain runtime computations; they are not cached
    across executions or changes to the other inputs.

    This is an explicit opt-in to floating-point reassociation, not a bitwise
    equivalence guarantee. Rounding, overflow, underflow, signed zeros, and
    non-finite behavior can change, including for finite inputs. Numerically
    validate representative inputs after applying this pass. It does not prove
    that a complete graph is linear in the selections: nonlinear operations,
    products of dependent values, and dependent denominators are opaque bases.

    Only main-graph FLOAT/DOUBLE Add/Sub/Mul/Div/Neg at standard ONNX opsets
    18 through 28 are factored. Div requires an independent denominator.
    Broadcasting is retained; shapes may be symbolic or unknown. MatMul,
    custom-domain operators, local functions, and subgraphs are not rewritten.
    Overloaded calls are opaque even when named like standard arithmetic.
    Models defining a bare main-domain function named like an emitted operator
    are left unchanged, so factoring cannot accidentally call that function.
    Subgraph outputs are conservatively treated as dependent because subgraphs
    may capture selected inputs. Model/function metadata, interfaces, and
    initializers are preserved in a clone; the source model is unchanged.

    No optimizer dependency is required. This pass does not perform CSE or
    general constant folding. Apply it after differentiation/composition and
    before an optional ONNXScript optimizer on a deployment copy.

    Args:
        inputs: Nonempty sequence of unique FLOAT/DOUBLE graph-input names.
            Coefficients must be independent of every selected input.
        max_terms: Maximum number of distinct dependent bases represented per
            island. Larger unions become opaque boundaries, bounding expansion.
            At least two arithmetic steps are required to rewrite an island.

    Example::

        from onnax import HoistLinearCoefficientsPass

        # hvp is an onnax.jvp(onnax.grad(model)) result.
        factored = HoistLinearCoefficientsPass(inputs=["q_tangent"])(hvp).model
    """

    def __init__(self, inputs: Sequence[str], *, max_terms: int = 64) -> None:
        if (
            not isinstance(inputs, Sequence)
            or isinstance(inputs, (str, bytes))
            or not inputs
            or any(not isinstance(name, str) for name in inputs)
            or len(set(inputs)) != len(inputs)
        ):
            raise ValueError(
                "HoistLinearCoefficientsPass inputs must be a nonempty sequence of unique "
                f"graph-input names; got {_argument_context(inputs)}. "
                "Pass a list such as ['q_tangent']."
            )
        if isinstance(max_terms, bool) or not isinstance(max_terms, int) or max_terms < 1:
            raise ValueError(
                f"HoistLinearCoefficientsPass max_terms must be an integer >= 1; "
                f"got {_argument_context(max_terms)}. Use 64 for the default expansion bound."
            )
        self.inputs = tuple(inputs)
        self.max_terms = max_terms

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        result = model.clone(deep_copy=True)
        graph = result.graph
        available = {v.name: v for v in graph.inputs}
        dependent: set[ir.Value] = set()
        for name in self.inputs:
            if name not in available:
                raise ValueError(
                    f"HoistLinearCoefficientsPass: unknown input {name!r}; available graph "
                    f"inputs: {list(available)!r}. Select an exact runtime input name; "
                    "initializers cannot be selected."
                )
            value = available[name]
            if not isinstance(value.type, ir.TensorType) or value.dtype not in _FLOATS:
                raise TypeError(
                    f"HoistLinearCoefficientsPass: selected input {_value_context(value)} "
                    "must be a FLOAT or DOUBLE tensor. Select an existing supported floating "
                    "input; integer, half, and complex arithmetic are not factored."
                )
            dependent.add(value)
        # Generated nodes must resolve to standard arithmetic, not to a local
        # function with the same bare identifier. Inlining such functions first
        # is the caller's explicit choice; this pass preserves definitions.
        if any(
            (domain, op, "") in result.functions
            for domain in ("", "ai.onnx")
            for op in _OPS | {"Constant"}
        ):
            return ir.passes.PassResult(result, False)
        original_nodes = list(graph)
        try:
            graph.sort()
        except ValueError as error:
            error.add_note(
                "HoistLinearCoefficientsPass requires an acyclic graph. Correct cyclic "
                "or unresolved producer references in the source model, then retry."
            )
            raise
        nodes = list(graph)
        modified = nodes != original_nodes
        expressions: dict[ir.Value, _Expression] = {}
        dtypes: dict[ir.Value, ir.DataType] = {}
        included: set[ir.Node] = set()
        for node in nodes:
            if any(v in dependent for v in node.inputs if v is not None) or any(
                a.type in (ir.AttributeType.GRAPH, ir.AttributeType.GRAPHS)
                for a in node.attributes.values()
            ):
                dependent.update(node.outputs)
            if (
                node.domain not in ("", "ai.onnx")
                or node.overload
                or node.op_identifier() in result.functions
                or node.op_type not in _OPS
                or len(node.outputs) != 1
                or len(node.inputs) != (1 if node.op_type == "Neg" else 2)
                or any(v is None for v in node.inputs)
            ):
                continue
            version = _effective_opset(node, graph.opset_imports)
            if not 18 <= version <= 28:
                continue
            first = node.inputs[0]
            assert first is not None
            dtype = dtypes.get(first, first.dtype)
            if (
                dtype not in _FLOATS
                or node.outputs[0].dtype not in (None, dtype)
                or any(dtypes.get(v, v.dtype) != dtype for v in node.inputs if v is not None)
            ):
                continue
            assert dtype is not None
            dtypes[node.outputs[0]] = dtype
            if node.outputs[0] not in dependent:
                continue
            expression = _expression(node, dependent, expressions, self.max_terms)
            if expression is not None:
                expressions[node.outputs[0]] = expression
                included.add(node)

        tape = ir.tape.Tape(graph)
        replacements: dict[ir.Value, ir.Value] = {}
        coefficients: dict[tuple[_Coefficient, ir.DataType], ir.Value] = {}
        ones: dict[ir.DataType, ir.Value] = {}

        domain = "" if "" in graph.opset_imports else "ai.onnx"

        def resolved(term: _Term, dtype: ir.DataType) -> ir.Value:
            if isinstance(term, ir.Value):
                return term
            if term is None:
                if dtype not in ones:
                    ones[dtype] = tape.op(
                        "Constant", [], {"value": ir.tensor(1.0, dtype=dtype)}, domain=domain
                    )
                    ones[dtype].dtype = dtype
                    ones[dtype].shape = ir.Shape([])
                return ones[dtype]
            return coefficients[term, dtype]

        def emit(term: _Term, dtype: ir.DataType) -> ir.Value:
            pending = [term] if isinstance(term, _Coefficient) else []
            while pending:
                current = pending[-1]
                key = (current, dtype)
                if key in coefficients:
                    pending.pop()
                    continue
                missing = [
                    v
                    for v in current.inputs
                    if isinstance(v, _Coefficient) and (v, dtype) not in coefficients
                ]
                if missing:
                    pending.extend(missing)
                    continue
                coefficients[key] = tape.op(
                    current.op, [resolved(v, dtype) for v in current.inputs], domain=domain
                )
                coefficients[key].dtype = dtype
                pending.pop()
            return resolved(term, dtype)

        for node in nodes:
            if len(node.outputs) != 1:
                continue
            value = node.outputs[0]
            if value not in expressions:
                continue
            expression = expressions[value]
            if (
                expression.steps < 2
                or not expression.hoisted
                or (
                    not value.is_graph_output()
                    and all(use.node in included for use in value.uses())
                )
            ):
                continue
            dtype = dtypes[value]
            assert dtype in _FLOATS
            parts = [
                tape.op("Mul", [replacements.get(basis, basis), emit(c, dtype)], domain=domain)
                for basis, c in expression.terms.items()
            ]
            replacement = parts[0]
            for part in parts[1:]:
                replacement = tape.op("Add", [replacement, part], domain=domain)
            replacement.name = value.name
            replacement.type = value.type or ir.TensorType(dtype)
            replacement.shape = ir.Shape(value.shape) if value.shape is not None else None
            replacement.doc_string = value.doc_string
            replacement.metadata_props.update(value.metadata_props)
            value.replace_all_uses_with(replacement, replace_graph_outputs=True)
            replacements[value] = replacement
            modified = True
        if replacements:
            for node in reversed(nodes):
                if node in included and all(
                    not v.uses() and not v.is_graph_output() for v in node.outputs
                ):
                    graph.remove(node, safe=True)
            graph.sort()
        return ir.passes.PassResult(result, modified)
