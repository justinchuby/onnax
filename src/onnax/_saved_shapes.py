# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Replace shape-only saved tensors with runtime shape-vector boundaries."""

import onnx_ir as ir
from onnx_ir.passes.common import IdentityEliminationPass


def compress(forward, backward, residuals, *, outputs):
    """Return private graphs, boundary names and an inspectable compression plan.

    ``outputs`` are the original forward results, which keep their interfaces.
    Shape vectors are computed from actual forward tensors; symbolic dimension
    declarations never substitute for runtime values. Nested captures are allowed.
    """
    residuals, outputs = tuple(residuals), set(outputs)
    if len(set(residuals)) != len(residuals):
        raise ValueError("Residual names must be unique")
    if not outputs <= {value.name for value in forward.graph.outputs}:
        raise ValueError("Original outputs must be present in the forward graph")
    forward, backward = forward.clone(), backward.clone()
    originals = ir.convenience.create_value_mapping(forward.graph)
    inputs = {value.name: value for value in backward.graph.inputs}
    if not set(residuals) <= inputs.keys() or not set(residuals) <= {
        value.name for value in forward.graph.outputs
    }:
        raise ValueError("Residuals must connect forward outputs to backward inputs")
    graphs = list(backward.graphs())
    positions = {
        node: (index, offset)
        for index, graph in enumerate(graphs)
        for offset, node in enumerate(graph)
    }
    returned = {value for graph in graphs for value in graph.outputs}
    occupied = {
        value.name
        for model in (forward, backward)
        for graph in model.graphs()
        for value in ir.convenience.create_value_mapping(graph).values()
    }
    names, records = [], []
    for name in residuals:
        value = inputs[name]
        uses = sorted(
            (use for use in value.uses() if use.node in positions),
            key=lambda use: (*positions[use.node], use.idx),
        )
        if (
            value in returned
            or not uses
            or not all(
                use.idx == 0
                and use.node.domain == ""
                and use.node.op_type == "Shape"
                and all(
                    key in {"start", "end"}
                    and not attr.is_ref()
                    and attr.type == ir.AttributeType.INT
                    for key, attr in use.node.attributes.items()
                )
                and (not use.node.attributes or forward.graph.opset_imports.get("", 0) >= 15)
                for use in uses
            )
        ):
            names.append(name)
            continue
        replacements = {}
        mappings = []
        for use in uses:
            node = use.node
            key = tuple(sorted((key, attr.as_int()) for key, attr in node.attributes.items()))
            if key not in replacements:
                stem = f"{name}.shape"
                candidate, counter = stem, 0
                while candidate in occupied:
                    counter += 1
                    candidate = f"{stem}_{counter}"
                occupied.add(candidate)
                rank = originals[name].shape
                if rank is not None:
                    bounds = dict(key)
                    length = len(
                        range(len(rank))[slice(bounds.get("start"), bounds.get("end"))]
                    )
                    shape = ir.Shape([length])
                else:
                    shape = node.outputs[0].shape or ir.Shape([None])
                generated = ir.val(
                    candidate,
                    dtype=ir.DataType.INT64,
                    shape=None if shape is None else ir.Shape(shape),
                )
                shape_node = ir.Node(
                    "",
                    "Shape",
                    [originals[name]],
                    attributes=dict(node.attributes),
                    outputs=[generated],
                )
                forward.graph.append(shape_node)
                forward.graph.outputs.append(generated)
                saved = ir.val(
                    candidate,
                    dtype=ir.DataType.INT64,
                    shape=None if shape is None else ir.Shape(shape),
                )
                backward.graph.inputs.append(saved)
                replacements[key] = saved
                names.append(candidate)
                mappings.append({"name": candidate, "attributes": dict(key)})
            # Preserve output names, including Scan body outputs. The ordinary
            # optimizer can remove internal identities after this boundary edit.
            node.op_type = "Identity"
            node.version = None
            node.attributes.clear()
            node.replace_input_with(0, replacements[key])
        backward.graph.inputs.remove(value)
        if name not in outputs:
            forward.graph.outputs.remove(originals[name])
        records.append({"tensor": name, "shape_vectors": mappings, "consumers": len(uses)})
    IdentityEliminationPass()(backward)
    return forward, backward, tuple(names), {"version": 1, "compressed": records}
