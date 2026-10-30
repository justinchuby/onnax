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


def analyze(model, *, alias_views=False):
    """Account for live tensor storage before releasing each node's last-use inputs.

    External inputs/initializers persist for the invocation. Graph outputs persist
    to graph end. View aliases share storage only when dtype and byte size match.
    Node workspaces and runtime fusion/scheduling are outside this accounting.
    """
    if model.functions:
        raise ValueError("Inline local functions before liveness analysis")
    graph = model.graph
    nodes = list(graph)
    values, ids, groups, records = [], {}, [], []

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

    for value in [*graph.inputs, *graph.initializers.values()]:
        add(value, -1, True)
    for index, node in enumerate(nodes):
        if any(
            attr.type in (ir.AttributeType.GRAPH, ir.AttributeType.GRAPHS)
            for attr in node.attributes.values()
        ):
            raise ValueError("Flatten control flow before liveness analysis")
        for value in node.inputs:
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
    timeline = []
    for index, node in enumerate(nodes):
        live.update(
            records[ids[v]]["group"]
            for v in node.outputs
            if not groups[records[ids[v]]["group"]]["external"]
        )
        amount = sum(groups[g]["bytes"] for g in live)
        if amount > peak:
            peak, peak_node, peak_groups = amount, index, sorted(live)
        timeline.append({"node": index, "op_type": node.op_type, "activation_bytes": amount})
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
        "values": records,
        "timeline": timeline,
        "scope": "Serialized node order; outputs allocated before last-use input release; graph outputs retained; tensor storage only",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--alias-views", action="store_true")
    args = parser.parse_args()
    print(json.dumps(analyze(ir.load(args.model), alias_views=args.alias_views), indent=2))
