# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Split a flat VJP at its primal/derivative boundary to save activations."""

import onnx_ir as ir
from onnx_ir.passes.common import IdentityEliminationPass
from onnxscript.rewriter import RewriteRuleClassBase


class _ComposeDerivativeTransposes(RewriteRuleClassBase):
    """Compose pullback layouts using values already available to backward."""

    def __init__(self, primal, derivative, available):
        super().__init__()
        self.primal, self.derivative, self.available = primal, derivative, available

    def pattern(self, op, x, perm):
        return op.Transpose(x, perm=perm)

    def check(self, context, x, perm):
        parent = x.producer()
        if (
            context.root not in self.derivative
            or parent is None
            or parent.domain
            or parent.op_type != "Transpose"
            or perm.is_ref()
            or perm.type != ir.AttributeType.INTS
        ):
            return False
        origin = parent.inputs[0]
        if origin.producer() in self.primal and origin not in self.available:
            return False
        first = parent.attributes.get("perm")
        if first is not None and (first.is_ref() or first.type != ir.AttributeType.INTS):
            return False
        second = list(perm.as_ints())
        first = (
            list(first.as_ints()) if first is not None else list(reversed(range(len(second))))
        )
        return (
            len(first) == len(second)
            and sorted(first) == list(range(len(first)))
            and sorted(second) == list(range(len(second)))
        )

    def rewrite(self, op, x, perm):
        parent = x.producer()
        first = parent.attributes.get_ints("perm", list(reversed(range(len(perm.as_ints())))))
        composed = [first[index] for index in perm.as_ints()]
        if composed == list(range(len(composed))):
            return op.Identity(parent.inputs[0])
        return op.Transpose(parent.inputs[0], perm=composed)


def split_vjp(model, joint, *, optimize_layout=False):
    """Split top-level nodes, retaining opaque local functions in both models."""
    primal_count = len(model.graph)
    original = list(model.graph)
    nodes = list(joint.graph)
    if [n.op_identifier() for n in nodes[:primal_count]] != [
        n.op_identifier() for n in original
    ]:
        raise ValueError("VJP primal prefix differs from the source graph")
    joint = joint.clone()
    nodes = list(joint.graph)
    derivative = set(nodes[primal_count:])
    if optimize_layout:
        expected_primal = nodes[:primal_count]
        primal = set(nodes[:primal_count])
        available = (
            set(joint.graph.inputs)
            | set(joint.graph.initializers.values())
            | {
                value
                for node in primal
                for value in node.outputs
                if any(use.node in derivative for use in value.uses())
            }
        )
        _ComposeDerivativeTransposes.rule(primal, derivative, available).apply_to_model(joint)
        nodes = list(joint.graph)
        if nodes[:primal_count] != expected_primal:
            raise ValueError("Layout rewrite changed the VJP primal prefix")
        derivative = set(nodes[primal_count:])
    boundary = [
        v
        for n in nodes[:primal_count]
        for v in n.outputs
        if any(use.node in derivative for use in v.uses())
    ]
    names = tuple(v.name for v in boundary)
    forward = model.clone()
    values = {v.name: v for n in forward.graph for v in n.outputs}
    for name in names:
        value = values[name]
        if value not in forward.graph.outputs:
            forward.graph.outputs.append(value)
    backward = joint.clone()
    primal = list(backward.graph)[:primal_count]
    by_name = {v.name: v for n in primal for v in n.outputs}
    outputs = list(backward.graph.outputs)[len(model.graph.outputs) :]
    backward.graph.outputs.clear()
    backward.graph.outputs.extend(outputs)
    for name in names:
        value = by_name[name]
        residual = ir.val(name, dtype=value.dtype, shape=value.shape)
        value.replace_all_uses_with(residual)
        backward.graph.inputs.append(residual)
    backward.graph.remove(primal, safe=True)
    if optimize_layout:
        IdentityEliminationPass()(backward)
    return forward, backward, names
