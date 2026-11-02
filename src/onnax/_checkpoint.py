# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Named checkpoint boundaries and differentiable subgraph interfaces."""

from dataclasses import dataclass

import onnx_ir as ir


@dataclass(frozen=True)
class Block:
    """One bounded primal graph and its differentiable interface."""

    name: str
    model: ir.Model
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    active_inputs: tuple[str, ...]
    active_outputs: tuple[str, ...]


def _ancestors(value):
    pending, result = [value], set()
    while pending:
        producer = pending.pop().producer()
        if producer is not None and producer not in result:
            result.add(producer)
            pending.extend(v for v in producer.inputs if v is not None)
    return result


def partition(model, parameters, *, boundaries, input_gradients=()):
    """Partition a flat graph at named values, preserving every crossing edge.

    ``boundaries`` maps stable block names to intermediate value names in execution
    order. The final block contains the remaining ancestors of all graph outputs.
    Selected parameters and data gradients determine which interfaces need VJPs.
    The source graph is neither mutated nor assigned decoder-specific metadata.
    """
    if any(
        attr.type in (ir.AttributeType.GRAPH, ir.AttributeType.GRAPHS)
        for node in model.graph
        for attr in node.attributes.values()
    ):
        raise ValueError("Checkpoint partitioning expects a flat graph")
    values = ir.convenience.create_value_mapping(model.graph)
    goals = list(boundaries.items())
    if len({name for name, _ in goals}) != len(goals) or any(
        not isinstance(name, str) or not name or name == "suffix" for name, _ in goals
    ):
        raise ValueError("Checkpoint block names must be unique, nonempty and not 'suffix'")
    if len({value for _, value in goals}) != len(goals):
        raise ValueError("Checkpoint boundary values must be unique")
    unknown = set(parameters) | set(input_gradients) | {v for _, v in goals}
    unknown -= values.keys()
    if unknown:
        raise ValueError(f"Unknown checkpoint values: {sorted(unknown)}")
    goals = [(name, [value]) for name, value in goals]
    goals.append(("suffix", [value.name for value in model.graph.outputs]))
    groups: list[tuple[str, set[ir.Node]]] = []
    seen: set[ir.Node] = set()
    for name, targets in goals:
        ancestors = set().union(*(_ancestors(values[target]) for target in targets))
        if not seen.issubset(ancestors):
            raise ValueError(
                f"Checkpoint boundary {targets} does not follow the preceding group"
            )
        members = ancestors - seen
        if members:
            groups.append((name, members))
        seen = ancestors
    owner = {node: index for index, (_, nodes) in enumerate(groups) for node in nodes}
    floating = {
        ir.DataType.FLOAT,
        ir.DataType.DOUBLE,
        ir.DataType.FLOAT16,
        ir.DataType.BFLOAT16,
    }
    active = {values[name] for name in (*parameters, *input_gradients)}
    for node in model.graph:
        if any(value in active for value in node.inputs if value is not None):
            active.update(value for value in node.outputs if value.dtype in floating)
    blocks = []
    for index, (name, members) in enumerate(groups):
        nodes = [node for node in model.graph if node in members]
        incoming = dict.fromkeys(
            value.name
            for node in nodes
            for value in node.inputs
            if value is not None
            and value.producer() not in members
            and value.name not in model.graph.initializers
        )
        outgoing = [
            value.name
            for node in nodes
            for value in node.outputs
            if value in model.graph.outputs
            or any(owner.get(use.node, index) > index for use in value.uses())
        ]
        for value_name in (*incoming, *outgoing):
            value = values[value_name]
            if not value.name or value.dtype is None or value.shape is None:
                raise ValueError(
                    f"Checkpoint boundary needs a name, dtype and rank: {value_name}"
                )
            if value.dtype in (ir.DataType.FLOAT16, ir.DataType.BFLOAT16):
                raise TypeError("Floating checkpoint boundaries require FLOAT or DOUBLE")
        graph = ir.convenience.extract(model.graph, list(incoming), outgoing)
        if len(graph) != len(nodes):
            raise ValueError(f"Checkpoint extraction changed the node set for {name}")
        graph.name = name
        # Extension rules can emit companion functions absent from primal calls
        # (e.g. NF4 InputGrad). Keep those definitions available to VJP generation.
        piece = ir.Model(
            graph,
            ir_version=model.ir_version,
            functions=[function.clone() for function in model.functions.values()],
        )
        piece.metadata_props.update(model.metadata_props)
        blocks.append(
            Block(
                name,
                piece,
                tuple(incoming),
                tuple(outgoing),
                tuple(value for value in incoming if values[value] in active),
                tuple(value for value in outgoing if values[value] in active),
            )
        )
    if set().union(*(nodes for _, nodes in groups)) != seen:
        raise AssertionError("Checkpoint node ownership is incomplete")
    return blocks
