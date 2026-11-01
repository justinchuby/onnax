# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Package portable attention pullbacks for optional tiled runtime execution."""

import hashlib
import json
import math
import struct

import numpy as np
import onnx_ir as ir

DOMAIN = "ai.onnax"
REGION = "onnax.attention_vjp.region"
INTERFACE = "onnax.attention_vjp.interface"


def fuse_attention_vjp(model):
    """Return a model whose eligible pullbacks have standard ONNX function bodies.

    Apply immediately after VJP generation, before folding or residual splitting.
    The runtime can implement this function contract with a tiled kernel.
    """
    result = model.clone()
    graph = result.graph
    groups = {}
    for node in graph:
        if REGION in node.metadata_props:
            groups.setdefault(node.metadata_props[REGION], []).append(node)
    values = ir.convenience.create_value_mapping(graph)
    for nodes in groups.values():
        if INTERFACE not in nodes[0].metadata_props:
            continue
        spec = json.loads(nodes[0].metadata_props[INTERFACE])
        if any(name not in values for name in [*spec["inputs"], *spec["outputs"].values()]):
            continue
        inputs = [values[name] for name in spec["inputs"]]
        if len(set(inputs)) != len(inputs):
            continue
        q, _, v, _, _ = inputs
        rank, hq, hk = spec["rank"], spec["hq"], spec["hk"]
        attrs = spec["attributes"]
        if q.shape is None or v.shape is None or q.dtype != ir.DataType.FLOAT:
            continue
        d = q.shape[-1]
        if not isinstance(d, int):
            continue
        if rank == 3:
            d //= hq
        if d not in (16, 32, 64, 128) or v.shape[-1] != (d * hk if rank == 3 else d):
            continue
        if attrs.get("softcap", 0) != 0 or attrs.get("softmax_precision", 1) != 1:
            continue
        if any(attrs.get(key, -1) != -1 for key in ("left_window_size", "right_window_size")):
            continue
        mask = values[spec["mask"]] if spec["mask"] else None
        kind = 0 if mask is None else 1 if mask.dtype == ir.DataType.BOOL else 2
        if mask is not None and mask.dtype not in (ir.DataType.BOOL, ir.DataType.FLOAT):
            continue
        selected = {int(i): values[name] for i, name in spec["outputs"].items()}
        # A region may capture its own constants, but no additional external values.
        boundary = set(inputs + ([mask] if mask is not None else []))
        produced = {value for n in nodes for value in n.outputs}
        if any(
            value is not None and value not in boundary | produced
            for n in nodes
            for value in n.inputs
        ):
            continue
        outputs = list(selected.values())
        if any(
            value not in outputs
            and (value in graph.outputs or any(use.node not in nodes for use in value.uses()))
            for value in produced
        ):
            continue
        if mask is None:
            name = "attention_vjp_mask"
            while name in values:
                name += "_"
            mask = ir.val(name, const_value=ir.tensor(np.array([True])))
            values[name] = mask
            mask.dtype = ir.DataType.BOOL
            mask.shape = ir.Shape([1])
            graph.register_initializer(mask)
        inputs.append(mask)
        body = ir.GraphView(
            inputs, outputs, nodes=nodes, opset_imports={"": graph.opset_imports[""]}
        ).clone()
        for index, value in enumerate(body.inputs):
            value.name = ("Q", "K", "V", "Y", "G", "Mask")[index]
            value.const_value = None
        for index, n in enumerate(body):
            n.name = f"node_{index}"
            n.metadata_props.clear()
            for j, value in enumerate(n.outputs):
                value.name = f"value_{index}_{j}"
                value.shape = None
        # Formal output annotations are inferred by the host graph.
        scale = float(attrs.get("scale", 1 / math.sqrt(d)))
        bits = struct.unpack("I", struct.pack("f", scale))[0]
        signature = hashlib.sha256(
            ir.to_proto(
                ir.Function(DOMAIN, "body", graph=body, attributes=[])
            ).SerializeToString()
        ).hexdigest()[:16]
        name = f"AttentionVJP_v1_r{rank}_q{hq}_k{hk}_d{d}_s{bits:08x}_c{attrs.get('is_causal', 0)}_o{''.join(map(str, selected))}_m{kind}_{signature}"
        function = ir.Function(DOMAIN, name, graph=body, attributes=[])
        result.functions[function.identifier()] = function
        call = ir.Node(DOMAIN, name, inputs, num_outputs=len(outputs), version=1)
        graph.insert_before(nodes[0], call)
        for old, new in zip(outputs, call.outputs, strict=True):
            new.name, new.type, new.shape = old.name, old.type, old.shape
            old.replace_all_uses_with(new, replace_graph_outputs=True)
        graph.remove(nodes, safe=True)
        graph.opset_imports[DOMAIN] = 1
    return result
