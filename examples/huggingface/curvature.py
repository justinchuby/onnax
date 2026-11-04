# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Export a matrix-free causal-token Gauss-Newton operator as ordinary ONNX."""

import json

import numpy as np
import onnx_ir as ir
from onnx_ir.passes import common

import onnax
from examples.huggingface.optimization import optimize_training_graph
from onnax._inference import _infer_shapes
from onnax._residuals import split_vjp

INTERFACE = "onnax.example.causal_ggn"
DOMAIN = "pkg.onnax"


def causal_ggn(model, parameters, *, output="logits", optimize=True):
    """Compose JVP, masked cross-entropy curvature and VJP at an FP32 parameter point.

    The source output has shape [batch, sequence, vocabulary]. Labels use causal
    shifting and -100 for ignored targets. The result retains primal inputs and
    appends parameter directions and labels. Its outputs are directional curvature
    and one product per selected parameter. Recorded names resolve collisions.
    """
    source = _infer_shapes(model)
    common.NameFixPass()(source)
    source.graph.name = source.graph.name or "causal_ggn"
    names = tuple(parameters)
    inputs = {value.name: value for value in source.graph.inputs}
    if not names or len(set(names)) != len(names) or not set(names) <= inputs.keys():
        raise ValueError("Select unique runtime parameter input names")
    if any(inputs[name].dtype != ir.DataType.FLOAT for name in names):
        raise TypeError("Causal GGN parameters must use FLOAT")
    outputs = {value.name: value for value in source.graph.outputs}
    if output not in outputs:
        raise ValueError(f"Select a model output from {list(outputs)}")
    logits = outputs[output]
    if logits.dtype != ir.DataType.FLOAT or logits.shape is None or len(logits.shape) != 3:
        raise TypeError("Causal GGN expects FLOAT logits[batch, sequence, vocabulary]")
    forward = onnax.jvp(source, inputs=names, outputs=[output])
    joint = onnax.vjp(source, inputs=names, outputs=[output])
    forward_info = json.loads(forward.graph.metadata_props["onnax.autodiff.interface"])
    reverse_info = json.loads(joint.graph.metadata_props["onnax.autodiff.interface"])
    _, backward, residuals = split_vjp(source, joint)
    values = ir.convenience.create_value_mapping(forward.graph)
    selected = list(dict.fromkeys([output, forward_info["derivative_outputs"][0], *residuals]))
    forward.graph.outputs.clear()
    forward.graph.outputs.extend(values[name] for name in selected)

    result = source.clone()
    result.graph.outputs.clear()
    result.graph.remove(list(result.graph), safe=False)
    occupied = set(inputs) | set(result.graph.initializers)

    def fresh(stem):
        name, index = stem, 0
        while name in occupied:
            index += 1
            name = f"{stem}_{index}"
        occupied.add(name)
        return name

    bindings = {value.name: value for value in result.graph.inputs}
    for seed in list(forward.graph.inputs)[len(source.graph.inputs) :]:
        value = ir.val(seed.name, dtype=seed.dtype, shape=seed.shape)
        result.graph.inputs.append(value)
        occupied.add(value.name)
        bindings[value.name] = value
    labels = ir.val(fresh("labels"), dtype=ir.DataType.INT64, shape=list(logits.shape)[:2])
    result.graph.inputs.append(labels)
    for derivative in (forward, backward):
        for identifier, function in derivative.functions.items():
            existing = result.functions.get(identifier)
            if existing is not None and ir.serde.serialize_function(
                existing
            ) != ir.serde.serialize_function(function):
                raise ValueError(f"Conflicting local function: {identifier}")
            result.functions[identifier] = function
    result.graph.opset_imports[DOMAIN] = 1

    def call(derivative, stem, arguments):
        name, index = stem, 0
        while (DOMAIN, name, "") in result.functions:
            index += 1
            name = f"{stem}_{index}"
        graph = derivative.graph.clone()
        arguments = dict(arguments)
        for value in list(graph.initializers.values()):
            if value.name in source.graph.initializers:
                bound = result.graph.initializers[value.name]
            else:
                bound = ir.val(
                    fresh(name + "." + value.name), dtype=value.dtype, shape=value.shape
                )
                bound.const_value = value.const_value
                result.graph.register_initializer(bound)
            arguments[value.name] = bound
            del graph.initializers[value.name]
            value.const_value = None
            if value not in graph.inputs:
                graph.inputs.append(value)
        function = ir.Function(DOMAIN, name, graph=graph, attributes=[])
        result.functions[function.identifier()] = function
        returned = [
            ir.val(fresh(name + "." + value.name), dtype=value.dtype, shape=value.shape)
            for value in graph.outputs
        ]
        result.graph.append(
            ir.Node(
                DOMAIN,
                name,
                [arguments[value.name] for value in graph.inputs],
                outputs=returned,
            )
        )
        return dict(zip((value.name for value in graph.outputs), returned, strict=True))

    primal = call(forward, "CausalGGNForward", bindings)
    tape = ir.tape.Tape(result.graph)

    def op(name, *values, **attrs):
        return tape.op(name, list(values), attrs)

    def constant(value):
        return op("Constant", value=ir.tensor(np.asarray(value)))

    targets = op(
        "Slice",
        labels,
        constant(np.array([1], np.int64)),
        constant(np.array([np.iinfo(np.int64).max], np.int64)),
        constant(np.array([1], np.int64)),
    )
    valid = op(
        "Cast",
        op("Not", op("Equal", targets, constant(np.int64(-100)))),
        to=int(ir.DataType.FLOAT),
    )
    count = op("Max", op("ReduceSum", valid, keepdims=0), constant(np.float32(1)))
    mask = op("Pad", valid, constant(np.array([0, 0, 0, 1], np.int64)))
    mask = op("Unsqueeze", mask, constant(np.array([-1], np.int64)))
    probability = op("Softmax", primal[output], axis=-1)
    tangent = primal[forward_info["derivative_outputs"][0]]
    center = op(
        "ReduceSum",
        op("Mul", probability, tangent),
        constant(np.array([-1], np.int64)),
        keepdims=1,
    )
    cotangent = op(
        "Div", op("Mul", op("Mul", probability, op("Sub", tangent, center)), mask), count
    )
    curvature = op("ReduceSum", op("Mul", tangent, cotangent), keepdims=0)
    curvature.name, curvature.dtype, curvature.shape = (
        fresh("directional_curvature"),
        ir.DataType.FLOAT,
        ir.Shape([]),
    )
    products = call(
        backward,
        "CausalGGNPullback",
        {
            **bindings,
            **{name: primal[name] for name in residuals},
            reverse_info["seeds"][0]: cotangent,
        },
    )
    result.graph.outputs.extend(
        [curvature, *[products[name] for name in reverse_info["derivative_outputs"]]]
    )
    result = common.InlinePass()(result).model
    if optimize:
        result = optimize_training_graph(result)
    common.NameFixPass()(result)
    result.graph.sort()
    result.graph.metadata_props[INTERFACE] = json.dumps(
        {
            "version": 1,
            "parameters": names,
            "directions": forward_info["seeds"],
            "labels": labels.name,
            "curvature": result.graph.outputs[0].name,
            "products": [value.name for value in list(result.graph.outputs)[1:]],
            "reduction": "mean_valid_shifted_tokens",
        }
    )
    return result


def batch_directions(model):
    """Share the primal batch and parameters across a leading axis of directions."""
    interface = json.loads(model.graph.metadata_props[INTERFACE])
    directions = set(interface["directions"])
    return onnax.vmap(
        model,
        in_axes=tuple(0 if value.name in directions else None for value in model.graph.inputs),
    )
