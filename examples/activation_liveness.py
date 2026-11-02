# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Account for statically shaped tensor lifetimes in serialized ONNX node order."""

import argparse
import json
import math

import onnx_ir as ir

_VIEW_OPS = {"Identity", "Reshape", "Flatten", "Squeeze", "Unsqueeze", "Transpose"}


def tensor_bytes(value):
    """Return packed tensor bytes from fixed-width dtype and static dimensions."""
    if value.dtype is None or value.dtype in (ir.DataType.UNDEFINED, ir.DataType.STRING):
        raise ValueError(f"Expected a fixed-width tensor dtype: {value.name}")
    if value.shape is None or any(
        isinstance(d, bool) or not isinstance(d, int) or d < 0 for d in value.shape
    ):
        raise ValueError(f"Expected a static nonnegative shape: {value.name}")
    return (math.prod(value.shape) * value.dtype.bitwidth + 7) // 8


def _captures(graph):
    defined = (
        set(graph.inputs)
        | set(graph.initializers.values())
        | {value for node in graph for value in node.outputs}
    )
    reads = []
    for node in graph:
        reads.extend(value for value in node.inputs if value is not None)
        for attr in node.attributes.values():
            if attr.type == ir.AttributeType.GRAPH:
                reads.extend(_captures(attr.as_graph()))
            elif attr.type == ir.AttributeType.GRAPHS:
                for child in attr.as_graphs():
                    reads.extend(_captures(child))
    reads.extend(graph.outputs)
    return list(dict.fromkeys(value for value in reads if value not in defined))


def _axes(node, name, count, ranks):
    result = list(node.attributes.get_ints(name, [0] * count))
    if len(result) != count:
        raise ValueError(f"Expected {count} Scan {name}")
    for index, (axis, rank) in enumerate(zip(result, ranks, strict=True)):
        if not -rank <= axis < rank:
            raise ValueError(f"Scan {name} axis outside tensor rank")
        result[index] = axis % rank
    return result


def _scan(node, *, alias_views, opsets):
    version = opsets.get(node.domain, opsets.get("", opsets.get("ai.onnx", 0)))
    if version < 9:
        raise ValueError("Scan memory analysis requires opset 9 or later")
    count = node.attributes.get_int("num_scan_inputs")
    states = len(node.inputs) - count
    body = node.attributes.get_graph("body")
    if count < 1 or states < 0 or any(value is None for value in node.inputs):
        raise ValueError("Expected Scan inputs and a positive num_scan_inputs")
    if len(body.inputs) != len(node.inputs) or len(body.outputs) != len(node.outputs):
        raise ValueError("Scan body interface differs from the outer node")
    if len(node.outputs) < states:
        raise ValueError("Scan has fewer outputs than carry states")
    for value in [*node.inputs, *node.outputs, *body.inputs, *body.outputs]:
        tensor_bytes(value)
    inputs = list(node.inputs[states:])
    outputs = list(node.outputs[states:])
    input_axes = _axes(node, "scan_input_axes", count, [len(v.shape) for v in inputs])
    output_axes = _axes(
        node, "scan_output_axes", len(outputs), [len(v.shape) for v in outputs]
    )
    lengths = [value.shape[axis] for value, axis in zip(inputs, input_axes, strict=True)]
    if len(set(lengths)) != 1:
        raise ValueError("Scan input sequence lengths differ")
    length = lengths[0]
    for name, size in (
        ("scan_input_directions", count),
        ("scan_output_directions", len(outputs)),
    ):
        directions = node.attributes.get_ints(name, [0] * size)
        if len(directions) != size or any(direction not in (0, 1) for direction in directions):
            raise ValueError(f"Expected binary Scan {name}")
    for outer, inner, axis in [
        *zip(inputs, body.inputs[states:], input_axes, strict=True),
        *zip(outputs, body.outputs[states:], output_axes, strict=True),
    ]:
        dimensions = list(outer.shape)
        steps = dimensions.pop(axis)
        if steps != length or dimensions != list(inner.shape) or outer.dtype != inner.dtype:
            raise ValueError("Scan slice shape/dtype differs from its body")
    for index in range(states):
        values = (
            node.inputs[index],
            node.outputs[index],
            body.inputs[index],
            body.outputs[index],
        )
        if any(
            value.shape != values[0].shape or value.dtype != values[0].dtype
            for value in values
        ):
            raise ValueError("Scan carry shape/dtype must remain invariant")
    captures = _captures(body)
    details = _analyze_graph(
        body, alias_views=alias_views, scan_bodies=True, captures=captures, opsets=opsets
    )
    # Follow carry aliases across iterations. A state forwarding another state
    # can retain an older owned buffer even when its own body output is a view.
    state_inputs = {
        group: index for index, group in enumerate(details["input_groups"][:states])
    }
    tokens = [None] * states
    previous = 0
    # In N links a passthrough chain reaches an owned result, a borrowed value,
    # or a cycle containing only the original borrowed states.
    for step in range(min(max(length - 1, 0), states)):
        tokens = [
            tokens[state_inputs[group]]
            if group in state_inputs
            else None
            if details["storage_groups"][group]["external"]
            else (step, group)
            for group in details["output_groups"][:states]
        ]
        previous = max(
            previous,
            sum(
                details["storage_groups"][token[1]]["bytes"]
                for token in set(tokens)
                if token is not None
            ),
        )
    constants = sum(
        tensor_bytes(value) for value in body.initializers.values() if value not in body.inputs
    )
    workspace = constants + previous + details["peak_activation_bytes"] if length else 0
    return {
        "steps": length,
        "carry_bytes": sum(tensor_bytes(value) for value in node.outputs[:states]),
        "retained_sequence_output_bytes": sum(tensor_bytes(value) for value in outputs),
        "borrowed_slice_bytes": sum(tensor_bytes(value) for value in body.inputs[states:]),
        "previous_iteration_carry_bytes": previous,
        "body_initializer_bytes": constants,
        "body_temporary_peak_bytes": details["peak_activation_bytes"],
        "body_workspace_bytes": workspace,
        "captures": [value.name for value in captures],
        "body": details,
    }


def analyze(model, *, alias_views=False, scan_bodies=False):
    """Account for live tensor storage before releasing each node's last-use inputs.

    External inputs/initializers persist for the invocation. Graph outputs persist
    to graph end. View aliases share storage only when dtype and byte size match.
    Node workspaces and runtime fusion/scheduling are outside this accounting.
    With scan_bodies, sequence outputs are preallocated, input slices borrow
    storage, and iteration carries and body temporaries are counted recursively.
    """
    if model.functions:
        raise ValueError("Inline local functions before liveness analysis")
    return _analyze_graph(model.graph, alias_views=alias_views, scan_bodies=scan_bodies)


def _analyze_graph(graph, *, alias_views, scan_bodies, captures=(), opsets=None):
    opsets = {**({} if opsets is None else opsets), **graph.opset_imports}
    nodes = list(graph)
    values, ids, groups, records = [], {}, [], []
    scans = {}

    def add(value, birth, external, group=None):
        if value in ids:
            return ids[value]
        size = tensor_bytes(value)
        index = len(values)
        ids[value] = index
        values.append(value)
        if group is None:
            group = len(groups)
            groups.append(
                {
                    "bytes": size,
                    "birth": birth,
                    "last_use": birth,
                    "external": external,
                    "values": [],
                }
            )
        groups[group]["values"].append(index)
        records.append(
            {
                "id": index,
                "name": value.name,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "bytes": size,
                "birth": birth,
                "last_use": birth,
                "group": group,
            }
        )
        return index

    for value in [*graph.inputs, *graph.initializers.values(), *captures]:
        add(value, -1, True)
    for index, node in enumerate(nodes):
        if any(
            attr.type in (ir.AttributeType.GRAPH, ir.AttributeType.GRAPHS)
            for attr in node.attributes.values()
        ):
            if not scan_bodies or node.domain not in ("", "ai.onnx") or node.op_type != "Scan":
                raise ValueError("Flatten control flow before liveness analysis")
            scans[index] = _scan(node, alias_views=alias_views, opsets=opsets)
        captured = _captures(node.attributes.get_graph("body")) if index in scans else []
        for value in [*node.inputs, *captured]:
            if value is not None:
                if value not in ids:
                    raise ValueError(
                        f"Expected topological order and declared inputs: {value.name}"
                    )
                records[ids[value]]["last_use"] = index
        for value in node.outputs:
            if value in ids:
                raise ValueError("Each value needs one definition")
            group = None
            if (
                alias_views
                and not node.domain
                and node.op_type in _VIEW_OPS
                and node.inputs
                and node.inputs[0] is not None
            ):
                source = node.inputs[0]
                if source.dtype == value.dtype and tensor_bytes(source) == tensor_bytes(value):
                    group = records[ids[source]]["group"]
            add(value, index, False, group)
    for output in graph.outputs:
        if output not in ids:
            raise ValueError("Expected a defined graph output")
        records[ids[output]]["last_use"] = len(nodes)
    for record in records:
        group = groups[record["group"]]
        group["last_use"] = max(group["last_use"], record["last_use"])
    external = sum(group["bytes"] for group in groups if group["external"])
    live = set()
    peak = 0
    peak_node = None
    peak_groups = []
    peak_workspace = 0
    timeline = []
    for index, node in enumerate(nodes):
        live.update(
            records[ids[v]]["group"]
            for v in node.outputs
            if not groups[records[ids[v]]["group"]]["external"]
        )
        amount = sum(groups[g]["bytes"] for g in live)
        workspace = 0
        if index in scans and scans[index]["steps"]:
            detail = scans[index]
            # Final carry storage becomes the last body result; sequence output
            # buffers are allocated before iteration and filled by copies.
            workspace = max(0, detail["body_workspace_bytes"] - detail["carry_bytes"])
            amount += workspace
        if amount > peak:
            peak, peak_node, peak_groups = amount, index, sorted(live)
            peak_workspace = workspace
        timeline.append(
            {
                "node": index,
                "op_type": node.op_type,
                "activation_bytes": amount,
                **({"scan_workspace_bytes": workspace} if index in scans else {}),
            }
        )
        live = {g for g in live if groups[g]["last_use"] > index}
    return {
        "version": 1,
        "alias_views": alias_views,
        "nodes": len(nodes),
        "persistent_external_bytes": external,
        "peak_activation_bytes": peak,
        "peak_total_tensor_bytes": external + peak,
        "peak_node": peak_node,
        "peak_groups": [groups[g] for g in peak_groups],
        "peak_scan_workspace_bytes": peak_workspace,
        "storage_groups": groups,
        "output_groups": [records[ids[value]]["group"] for value in graph.outputs],
        "input_groups": [records[ids[value]]["group"] for value in graph.inputs],
        "scans": [{"node": index, **detail} for index, detail in scans.items()],
        "values": records,
        "timeline": timeline,
        "scope": "Serialized node order; outputs allocated before last-use input release; graph outputs retained; tensor storage only",
        **(
            {
                "scan_allocation": "Preallocated sequence outputs; borrowed input slices; previous carry and body temporaries coexist; final carry takes the last body result"
            }
            if scan_bodies
            else {}
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--alias-views", action="store_true")
    parser.add_argument("--scan-bodies", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(
                ir.load(args.model), alias_views=args.alias_views, scan_bodies=args.scan_bodies
            ),
            indent=2,
        )
    )
