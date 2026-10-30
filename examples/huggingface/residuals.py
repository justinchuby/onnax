# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Split a flat VJP at its primal/derivative boundary to save activations."""

import onnx_ir as ir


def split_vjp(model, joint):
    """Return forward/backward graphs and the exact activation boundary."""
    if model.functions:
        raise ValueError("Activation splitting expects an expanded flat graph")
    primal_count = len(model.graph)
    original = list(model.graph)
    nodes = list(joint.graph)
    if [n.op_identifier() for n in nodes[:primal_count]] != [
        n.op_identifier() for n in original
    ]:
        raise ValueError("VJP primal prefix differs from the source graph")
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
    return forward, backward, names
