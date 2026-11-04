# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Build an explicit deterministic replay region within a split block VJP."""

# Each admitted operation is pure for fixed inputs. Randomness, control flow and
# local functions need their own replay-state contract before entering this set.
_REPLAY_OPS = frozenset(
    [
        "Abs",
        "Add",
        "And",
        "Cast",
        "CastLike",
        "Clip",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Cos",
        "Div",
        "Equal",
        "Erf",
        "Exp",
        "Expand",
        "Flatten",
        "Gather",
        "GatherElements",
        "Gemm",
        "Greater",
        "Identity",
        "Less",
        "Log",
        "LayerNormalization",
        "MatMul",
        "Max",
        "Min",
        "Mul",
        "Neg",
        "Not",
        "Or",
        "Pow",
        "Range",
        "Reciprocal",
        "ReduceMax",
        "ReduceMean",
        "ReduceSum",
        "Relu",
        "RMSNormalization",
        "Reshape",
        "Shape",
        "Sigmoid",
        "Sin",
        "Slice",
        "Softmax",
        "Split",
        "Sqrt",
        "Squeeze",
        "Swish",
        "Sub",
        "Tanh",
        "Transpose",
        "Unsqueeze",
        "Where",
    ]
)


def replay_region(primal, boundaries, selected):
    """Return topologically ordered replay nodes, stopping at retained boundaries.

    Inputs and frozen initializers remain invocation inputs. A shared producer
    executes once even when several selected values depend on it. Unknown or
    stateful operations are rejected before either split graph is changed.
    """
    selected = tuple(selected)
    boundary = {value.name: value for value in boundaries}
    if len(set(selected)) != len(selected):
        raise ValueError("Recomputed activation names must be unique")
    if not set(selected) <= boundary.keys():
        raise ValueError("Recompute names must identify saved forward values")
    retained = {value for name, value in boundary.items() if name not in selected}
    members, needed, visiting = set(primal), set(), set()

    def visit(value):
        if value in retained:
            return
        node = value.producer()
        if node not in members or node in needed:
            return
        if node in visiting:
            raise ValueError("Replay dependencies contain a cycle")
        if (
            node.domain
            or node.op_type not in _REPLAY_OPS
            or any(attr.type.name in ("GRAPH", "GRAPHS") for attr in node.attributes.values())
        ):
            raise ValueError(
                f"Replay requires a deterministic operation: {node.op_identifier()}"
            )
        visiting.add(node)
        for parent in node.inputs:
            if parent is not None:
                visit(parent)
        visiting.remove(node)
        needed.add(node)

    for name in selected:
        visit(boundary[name])
    return tuple(node for node in primal if node in needed)


def describe_replay(primal, boundaries, selected):
    """Describe one block-backward invocation using stable ONNX value names."""
    selected = tuple(selected)
    nodes = replay_region(primal, boundaries, selected)
    members = set(nodes)
    leaves = dict.fromkeys(
        value.name
        for node in nodes
        for value in node.inputs
        if value is not None and value.producer() not in members
    )
    return {
        "version": 1,
        "recomputed_values": list(selected),
        "retained_values": [value.name for value in boundaries if value.name not in selected],
        "replay_inputs": list(leaves),
        "nodes": [
            {
                "name": node.name,
                "op_type": node.op_type,
                "outputs": [value.name for value in node.outputs],
            }
            for node in nodes
        ],
        "invocations_per_backward": int(bool(nodes)),
        "release_policy": "Replay temporaries follow their final consumer in the backward graph",
    }
