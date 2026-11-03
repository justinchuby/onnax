# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Package portable attention pullbacks as ordinary ONNX local functions."""

import hashlib
import json
import math
import struct

import numpy as np
import onnx_ir as ir

from onnax._autodiff._attention import vjp as _attention_vjp
from onnax._autodiff._context import AutodiffContext

DOMAIN = "pkg.onnax"
REGION = "onnax.attention_vjp.region"
INTERFACE = "onnax.attention_vjp.interface"


def _descriptor(text):
    try:
        spec = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(spec, dict):
        return None
    inputs, outputs, attrs = (spec.get(key) for key in ("inputs", "outputs", "attributes"))
    if (
        not isinstance(inputs, list)
        or len(inputs) != 5
        or not all(isinstance(name, str) for name in inputs)
        or not isinstance(outputs, dict)
        or not outputs
        or any(key not in ("0", "1", "2") for key in outputs)
        or not all(isinstance(name, str) for name in outputs.values())
        or list(outputs) != sorted(outputs)
        or len(set(outputs.values())) != len(outputs)
        or not isinstance(attrs, dict)
        or any(not isinstance(key, str) for key in attrs)
        or not all(type(value) in (int, float) for value in attrs.values())
        or spec.get("rank") not in (3, 4)
        or any(type(spec.get(key)) is not int or spec[key] <= 0 for key in ("hq", "hk"))
        or spec["hq"] % spec["hk"]
        or not (spec.get("mask") is None or isinstance(spec.get("mask"), str))
    ):
        return None
    integer_attrs = {
        "q_num_heads",
        "kv_num_heads",
        "is_causal",
        "softmax_precision",
        "left_window_size",
        "right_window_size",
    }
    if set(attrs) - integer_attrs - {"scale", "softcap"}:
        return None
    if any(key in attrs and type(attrs[key]) is not int for key in integer_attrs):
        return None
    if any(isinstance(value, float) and not math.isfinite(value) for value in attrs.values()):
        return None
    return spec


def _signature(nodes, inputs, outputs, opset):
    body = ir.GraphView(inputs, outputs, nodes=nodes, opset_imports={"": opset}).clone()
    for index, value in enumerate(body.inputs):
        value.name = f"input_{index}"
        value.type = value.shape = value.const_value = None
        value.metadata_props.clear()
    for index, node in enumerate(body):
        node.name = f"node_{index}"
        node.metadata_props.clear()
        for slot, value in enumerate(node.outputs):
            value.name = f"value_{index}_{slot}"
            value.type = value.shape = None
            value.metadata_props.clear()
    return ir.to_proto(
        ir.Function(DOMAIN, "body", graph=body, attributes=[])
    ).SerializeToString()


def _matches_contract(nodes, inputs, outputs, mask, attrs, opset, selected):
    # Regenerate from typed boundary values, then compare the complete topology,
    # attributes and constants after removing incidental names and annotations.
    originals = inputs + ([mask] if mask is not None else [])
    formals = [
        ir.val(f"formal_{index}", type=value.type, shape=value.shape)
        for index, value in enumerate(originals)
    ]
    q, k, v, y, g = formals[:5]
    graph = ir.Graph(formals, [], nodes=[], opset_imports={"": opset})
    primal = ir.node("Attention", [q, k, v, *formals[5:]], outputs=[y], attributes=attrs)
    try:
        result = _attention_vjp(AutodiffContext(graph), primal, [g], selected)
    except (ValueError, TypeError, NotImplementedError):
        return False
    return _signature(nodes, originals, outputs, opset) == _signature(
        list(graph), formals, [result[index] for index in selected], opset
    )


def _fuse_attention_vjp(model: ir.Model, *, recompute: bool) -> ir.passes.PassResult:
    """Return a model whose eligible pullbacks have standard ONNX function bodies.

    Apply immediately after VJP generation, before folding or residual splitting.
    The runtime can implement this function contract with a tiled kernel.
    With recompute=True, eligible FP32 heads from 32 through 256 use a
    v2 body that recomputes its primal output from Q/K/V
    and omits Y from the interface, enabling fused forward/backward kernels.
    """
    result = model.clone()
    modified = False
    graph = result.graph
    groups: dict[str, list[ir.Node]] = {}
    for node in graph:
        if REGION in node.metadata_props:
            groups.setdefault(node.metadata_props[REGION], []).append(node)
    values = ir.convenience.create_value_mapping(graph)
    for nodes in groups.values():
        if INTERFACE not in nodes[0].metadata_props:
            continue
        spec = _descriptor(nodes[0].metadata_props[INTERFACE])
        if spec is None:
            continue
        if any(name not in values for name in [*spec["inputs"], *spec["outputs"].values()]):
            continue
        inputs = [values[name] for name in spec["inputs"]]
        if len(set(inputs)) != len(inputs):
            continue
        q, k, v, y, g = inputs
        rank, hq, hk = spec["rank"], spec["hq"], spec["hk"]
        attrs = spec["attributes"]
        if any(
            value.shape is None or len(value.shape) != rank or value.dtype != ir.DataType.FLOAT
            for value in (q, k, v)
        ):
            continue
        assert q.shape is not None and k.shape is not None and v.shape is not None
        assert q.dtype == ir.DataType.FLOAT
        if rank == 4 and (q.shape[1] != hq or k.shape[1] != hk or v.shape[1] != hk):
            continue
        if rank == 3 and (attrs.get("q_num_heads") != hq or attrs.get("kv_num_heads") != hk):
            continue
        d = q.shape[-1]
        if not isinstance(d, int):
            continue
        if rank == 3:
            if d % hq:
                continue
            d //= hq
        expected_width = d * hk if rank == 3 else d
        if (not 8 <= d <= 256 or d % 8 != 0) or v.shape[-1] != expected_width:
            continue
        if isinstance(k.shape[-1], int) and k.shape[-1] != expected_width:
            continue
        if attrs.get("softcap", 0) != 0 or attrs.get("softmax_precision", 1) != 1:
            continue
        if any(attrs.get(key, -1) != -1 for key in ("left_window_size", "right_window_size")):
            continue
        if spec["mask"] is not None and spec["mask"] not in values:
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
        if any(output not in produced for output in outputs):
            continue
        region_ops = {node.op_type for node in nodes if node.domain in ("", "ai.onnx")}
        if recompute:
            region_ops.add("Attention")
        if any(
            (domain, op, "") in result.functions
            for domain in ("", "ai.onnx")
            for op in region_ops
        ):
            continue
        if any(
            value not in outputs
            and (value in graph.outputs or any(use.node not in nodes for use in value.uses()))
            for value in produced
        ):
            continue
        # The pullback seed has the primal output's layout. Dynamic reshape
        # paths can leave its inferred type absent; keep this function boundary
        # typed so a native EP can claim it without expanding the portable body.
        expected_shape = y.shape if y.shape is not None else q.shape
        if g.dtype is not None and g.dtype != q.dtype:
            continue
        if g.shape is not None and (
            len(g.shape) != len(expected_shape)
            or any(
                isinstance(a, int) and isinstance(b, int) and a != b
                for a, b in zip(g.shape, expected_shape, strict=True)
            )
        ):
            continue
        if not _matches_contract(
            nodes, inputs, outputs, mask, attrs, graph.opset_imports[""], list(selected)
        ):
            continue
        g.type = ir.TensorType(q.dtype)
        g.shape = ir.Shape(
            expected_shape
            if g.shape is None
            else [
                b if isinstance(b, int) else a
                for a, b in zip(g.shape, expected_shape, strict=True)
            ]
        )
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
        use_recompute = recompute and 32 <= d <= 256
        if use_recompute:
            query, key, value, primal, _seed, formal_mask = body.inputs
            recomputed = ir.val("recomputed_Y", type=primal.type, shape=primal.shape)
            forward_attrs = {
                "scale": float(attrs.get("scale", 1 / math.sqrt(d))),
                "is_causal": attrs.get("is_causal", 0),
            }
            if rank == 3:
                forward_attrs.update(q_num_heads=hq, kv_num_heads=hk)
            forward = ir.node(
                "Attention",
                [query, key, value] + ([formal_mask] if kind else []),
                outputs=[recomputed],
                attributes=forward_attrs,
            )
            body.insert_before(body[0], forward)
            primal.replace_all_uses_with(recomputed)
            body.inputs.remove(primal)
            inputs.remove(y)
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
        version = 2 if use_recompute else 1
        name = f"AttentionVJP_v{version}_r{rank}_q{hq}_k{hk}_d{d}_s{bits:08x}_c{attrs.get('is_causal', 0)}_o{''.join(map(str, selected))}_m{kind}_{signature}"
        function = ir.Function(DOMAIN, name, graph=body, attributes=[])
        existing = result.functions.get(function.identifier())
        if (
            existing is not None
            and ir.to_proto(existing).SerializeToString()
            != ir.to_proto(function).SerializeToString()
        ):
            raise ValueError("An incompatible pkg.onnax attention VJP function already exists")
        result.functions[function.identifier()] = function
        call = ir.Node(DOMAIN, name, inputs, num_outputs=len(outputs), version=1)
        graph.insert_before(nodes[0], call)
        for old, new in zip(outputs, call.outputs, strict=True):
            new.name, new.type, new.shape = old.name, old.type, old.shape
            old.replace_all_uses_with(new, replace_graph_outputs=True)
            if old.name is not None:
                values[old.name] = new
        graph.remove(nodes, safe=True)
        graph.opset_imports[DOMAIN] = 1
        modified = True
    return ir.passes.PassResult(result, modified)


class FuseAttentionVJPPass(ir.passes.FunctionalPass):
    """Package attention pullbacks as ordinary ONNX local functions.

    Apply to an onnax VJP before ONNXScript optimization or residual splitting.
    The pass recognizes intact FP32, cache-free attention pullbacks in the main
    graph with equal Q/K/V head widths from 8 through 256 in multiples of eight. It supports
    rank-3/rank-4 layouts, grouped query heads, causal attention and Boolean or
    additive masks. Each ``pkg.onnax::AttentionVJP`` function contains the
    portable derivative; runtimes can select a native implementation.

    Args:
        recompute: Recompute the primal output inside the function for head
            widths from 32 through 256, using the five-input v2 contract.
            The default v1 contract has six inputs including the primal output.

    Returns a cloned IR model in ``PassResult``. ``modified`` records whether
    any pullbacks were packaged. Complete region bodies are matched before
    replacement; the source model, parameter names and interfaces are preserved.

    Example::

        derivative = onnax.vjp(model, inputs=["q", "k", "v"])
        deployed = onnax.FuseAttentionVJPPass(recompute=True)(derivative).model
    """

    def __init__(self, *, recompute: bool = False) -> None:
        if not isinstance(recompute, bool):
            raise TypeError("FuseAttentionVJPPass recompute must be a bool")
        self.recompute = recompute

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return _fuse_attention_vjp(model, recompute=self.recompute)
