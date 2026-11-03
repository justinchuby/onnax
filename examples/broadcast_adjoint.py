# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Package runtime broadcast adjoints as portable, shape-specializable functions."""

import json

import onnx_ir as ir

from onnax import AutodiffContext

DOMAIN = "pkg.onnax"
NAME = "SumToShape_v1"
REGION = "onnax.unbroadcast.region"
INTERFACE = "onnax.unbroadcast.interface"


def _body(nodes, inputs, output, opset):
    body = ir.GraphView(inputs, [output], nodes=nodes, opset_imports={"": opset}).clone()
    for value, name in zip(body.inputs, ("Gradient", "Reference"), strict=True):
        value.name, value.type, value.shape, value.const_value = name, None, None, None
        value.metadata_props.clear()
    for i, node in enumerate(body):
        node.name = f"node_{i}"
        node.metadata_props.clear()
        for j, value in enumerate(node.outputs):
            value.name, value.type, value.shape = f"value_{i}_{j}", None, None
            value.metadata_props.clear()
    return ir.Function(DOMAIN, NAME, graph=body, attributes=[])


def _canonical(opset):
    gradient = ir.val("gradient", dtype=ir.DataType.FLOAT)
    reference = ir.val("reference", dtype=ir.DataType.FLOAT)
    graph = ir.Graph([gradient, reference], [], nodes=[], opset_imports={"": opset})
    ctx = AutodiffContext(graph)
    output = ctx.unbroadcast(gradient, reference)
    return _body(list(graph), [gradient, reference], output, opset)


def fuse_broadcast_adjoints(model):
    """Clone and fuse untouched dynamic unbroadcast regions before optimization.

    Each call sums leading and singleton broadcast axes and restores the reference
    shape. The second input supplies only its shape. Standard ONNX function bodies
    preserve execution on runtimes without a native implementation.
    """
    result = model.clone()
    graph = result.graph
    groups = {}
    for node in graph:
        if REGION in node.metadata_props:
            groups.setdefault(node.metadata_props[REGION], []).append(node)
    if not groups:
        return result
    canonical = _canonical(graph.opset_imports[""])
    signature = ir.to_proto(canonical).SerializeToString()
    values = ir.convenience.create_value_mapping(graph)
    for nodes in groups.values():
        descriptor = nodes[0].metadata_props.get(INTERFACE)
        if descriptor is None:
            continue
        try:
            spec = json.loads(descriptor)
        except json.JSONDecodeError:
            continue
        if (
            not isinstance(spec, dict)
            or not isinstance(spec.get("inputs"), list)
            or len(spec["inputs"]) != 2
            or not all(isinstance(name, str) for name in spec["inputs"])
            or not isinstance(spec.get("output"), str)
        ):
            continue
        names = [*spec["inputs"], spec["output"]]
        if any(name not in values for name in names):
            continue
        inputs = [values[name] for name in spec["inputs"]]
        output = values[spec["output"]]
        if len(set(inputs)) != 2:
            continue
        produced = {value for node in nodes for value in node.outputs}
        if output not in produced or any(
            value is not None and value not in produced and value not in inputs
            for node in nodes
            for value in node.inputs
        ):
            continue
        if any(
            value is not output
            and (value in graph.outputs or any(use.node not in nodes for use in value.uses()))
            for value in produced
        ):
            continue
        # Metadata identifies candidates; the complete portable body proves the
        # contract. Edited or partially optimized regions stay unchanged.
        body = _body(nodes, inputs, output, graph.opset_imports[""])
        if ir.to_proto(body).SerializeToString() != signature:
            continue
        key = canonical.identifier()
        if (
            key in result.functions
            and ir.to_proto(result.functions[key]).SerializeToString() != signature
        ):
            raise ValueError("An incompatible pkg.onnax SumToShape_v1 function already exists")
        result.functions[key] = canonical
        call = ir.Node(DOMAIN, NAME, inputs, num_outputs=1, version=1)
        graph.insert_before(nodes[0], call)
        replacement = call.outputs[0]
        replacement.name, replacement.type, replacement.shape = (
            output.name,
            output.type,
            output.shape,
        )
        output.replace_all_uses_with(replacement, replace_graph_outputs=True)
        values[output.name] = replacement
        graph.remove(nodes, safe=True)
        graph.opset_imports[DOMAIN] = 1
    return result
