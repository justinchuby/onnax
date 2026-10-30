# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Packed blockwise NF4 reference kernels and frozen-weight differentiation rules."""

import math

import numpy as np
import onnx_ir as ir

import onnax

DOMAIN = "onnax.example.nf4"
# NormalFloat4 codebook used by bitsandbytes (high nibble first).
CODEBOOK = np.array(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=np.float32,
)
CODEBOOK.flags.writeable = False


def _blocks(shape, block_size):
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size < 2
        or block_size % 2
    ):
        raise ValueError("Expected a positive even block size")
    if len(shape) != 2 or any(
        isinstance(d, bool) or not isinstance(d, int) or d < 1 for d in shape
    ):
        raise ValueError("Expected a positive rank-two matrix shape")
    return (math.prod(shape) + block_size - 1) // block_size


def quantize(weight, *, block_size=64):
    """Pack row-major NF4 codes with one FP32 absmax per block."""
    weight = np.asarray(weight)
    if (
        weight.dtype != np.float32
        or weight.ndim != 2
        or not weight.size
        or not np.all(np.isfinite(weight))
    ):
        raise ValueError("Expected a finite nonempty rank-two FP32 weight")
    blocks = _blocks(weight.shape, block_size)
    flat = weight.reshape(-1)
    padded = np.pad(flat, (0, blocks * block_size - flat.size)).reshape(blocks, block_size)
    scales = np.max(np.abs(padded), axis=1)
    normalized = padded / np.where(scales > 0, scales, np.float32(1))[:, None]
    codes = (
        np.argmin(np.abs(normalized[:, :, None] - CODEBOOK), axis=-1)
        .astype(np.uint8)
        .reshape(-1)
    )
    return ((codes[::2] << 4) | codes[1::2]).copy(), scales


def dequantize(packed, scales, shape, *, block_size=64):
    """Independent NumPy NF4 decoding for validation."""
    packed = np.asarray(packed)
    scales = np.asarray(scales)
    size = math.prod(shape)
    blocks = _blocks(shape, block_size)
    if (
        packed.dtype != np.uint8
        or packed.shape != (blocks * block_size // 2,)
        or scales.dtype != np.float32
        or scales.shape != (blocks,)
        or np.any(scales < 0)
        or not np.all(np.isfinite(scales))
    ):
        raise ValueError("Invalid packed NF4 buffers")
    codes = np.empty(packed.size * 2, np.uint8)
    codes[::2] = packed >> 4
    codes[1::2] = packed & 15
    values = CODEBOOK[codes].reshape(blocks, block_size) * scales[:, None]
    return values.reshape(-1)[:size].reshape(shape)


def _function(k, n, block_size, *, transpose=False, opset=18):
    blocks = math.ceil(k * n / block_size)
    x = ir.val("x", dtype=ir.DataType.FLOAT)
    p = ir.val("packed", dtype=ir.DataType.UINT8, shape=[blocks * block_size // 2])
    s = ir.val("scales", dtype=ir.DataType.FLOAT, shape=[blocks])
    tape = ir.tape.Tape()

    def constant(value, dtype=ir.DataType.INT64):
        return tape.op(
            "Constant", [], {"value": ir.tensor(np.asarray(value, dtype=dtype.numpy()))}
        )

    integer = tape.op("Cast", [p], {"to": int(ir.DataType.INT64)})
    sixteen = constant(16)
    high = tape.op("Div", [integer, sixteen])
    low = tape.op("Mod", [integer, sixteen])
    axes = constant([1])
    pairs = tape.op(
        "Concat",
        [tape.op("Unsqueeze", [high, axes]), tape.op("Unsqueeze", [low, axes])],
        {"axis": 1},
    )
    indices = tape.op("Reshape", [pairs, constant([-1, block_size])])
    decoded = tape.op("Gather", [constant(CODEBOOK, ir.DataType.FLOAT), indices], {"axis": 0})
    scaled = tape.op("Mul", [decoded, tape.op("Unsqueeze", [s, axes])])
    flat = tape.op("Reshape", [scaled, constant([-1])])
    trimmed = tape.op("Slice", [flat, constant([0]), constant([k * n]), constant([0])])
    weight = tape.op("Reshape", [trimmed, constant([k, n])])
    if transpose:
        weight = tape.op("Transpose", [weight], {"perm": [1, 0]})
    output = tape.op("MatMul", [x, weight])
    output.dtype = ir.DataType.FLOAT
    output.shape = None
    name = f"{'InputGrad' if transpose else 'MatMul'}_{k}_{n}_{block_size}"
    return ir.Function(
        DOMAIN,
        name,
        graph=ir.Graph(
            [x, p, s], [output], nodes=tape.nodes, opset_imports={"": opset}, name=name
        ),
        attributes=[],
    )


def _rules(forward_name, backward_name):
    @onnax.register_vjp(DOMAIN, forward_name, since_version=1, until_version=1)
    def vjp(ctx, node, cotangents, input_indices):
        if list(input_indices) != [0]:
            raise ValueError(
                "NF4 base weights and scales are frozen; differentiate activations"
            )
        output = ctx.op(
            backward_name, cotangents[0], ctx.input(node, 1), ctx.input(node, 2), domain=DOMAIN
        )
        output.dtype = ir.DataType.FLOAT
        output.shape = ir.Shape(ctx.input(node, 0).shape)
        return {0: output}

    for name in (forward_name, backward_name):

        def register(op):
            @onnax.register_jvp(DOMAIN, op, since_version=1, until_version=1)
            def jvp(ctx, node, tangents):
                if set(tangents) != {0}:
                    raise ValueError("NF4 base weights and scales are frozen")
                output = ctx.op(
                    op, tangents[0], ctx.input(node, 1), ctx.input(node, 2), domain=DOMAIN
                )
                output.dtype = node.outputs[0].dtype
                output.shape = ir.Shape(node.outputs[0].shape)
                return [output]

        register(name)


def lower(model, *, names=("base.W", "base.C"), block_size=64):
    """Replace frozen FP32 matrix initializers with NF4 function calls."""
    from onnx_ir.passes.common import ShapeInferencePass

    if isinstance(names, str) or not names or len(set(names)) != len(names):
        raise ValueError("Select nonempty unique frozen initializer names")
    result = ShapeInferencePass()(model.clone()).model
    existing = {
        v.name
        for v in (
            *result.graph.inputs,
            *result.graph.initializers.values(),
            *(v for node in result.graph for v in node.outputs),
        )
    }
    reserved = {name + suffix for name in names for suffix in (".nf4", ".absmax")}
    if existing.intersection(reserved):
        raise ValueError("NF4 buffer name collision")
    manifest = {}
    for name in names:
        if name not in result.graph.initializers:
            raise ValueError(f"Missing frozen initializer: {name}")
        weight = result.graph.initializers[name]
        if weight in result.graph.inputs or weight.const_value is None:
            raise ValueError("NF4 base weights must be stored frozen initializers")
        array = weight.const_value.numpy()
        packed, scales = quantize(array, block_size=block_size)
        k, n = array.shape
        opset = result.graph.opset_imports[""]
        forward = _function(k, n, block_size, opset=opset)
        backward = _function(k, n, block_size, transpose=True, opset=opset)
        result.functions[forward.identifier()] = forward
        result.functions[backward.identifier()] = backward
        _rules(forward.name, backward.name)
        q = ir.val(name + ".nf4", const_value=ir.tensor(packed))
        s = ir.val(name + ".absmax", const_value=ir.tensor(scales))
        result.graph.register_initializer(q)
        result.graph.register_initializer(s)
        users = [node for node in result.graph if weight in node.inputs]
        if not users or any(
            node.domain or node.op_type != "MatMul" or node.inputs[1] is not weight
            for node in users
        ):
            raise ValueError("NF4 weights must be used as right-hand MatMul operands")
        for node in users:
            operand = node.inputs[0]
            if (
                operand is None
                or operand.dtype != ir.DataType.FLOAT
                or operand.shape is None
                or len(operand.shape) < 1
            ):
                raise ValueError("Expected rank-one or higher FP32 activations for NF4 MatMul")
            node.domain = DOMAIN
            node.op_type = forward.name
            node.version = 1
            node.resize_inputs(3)
            node.replace_input_with(1, q)
            node.replace_input_with(2, s)
            node.outputs[0].dtype = ir.DataType.FLOAT
            node.outputs[0].shape = ir.Shape([*node.inputs[0].shape[:-1], n])
        del result.graph.initializers[name]
        manifest[name] = {
            "packed": q.name,
            "scales": s.name,
            "shape": [k, n],
            "block_size": block_size,
            "packed_bytes": packed.nbytes,
            "scale_bytes": scales.nbytes,
            "original_bytes": array.nbytes,
        }
    result.graph.opset_imports[DOMAIN] = 1
    return result, manifest


def register_rules(model):
    """Install global rules for NF4 functions loaded from this example's exports."""
    for function in model.functions.values():
        if function.domain != DOMAIN or not function.name.startswith("MatMul_"):
            continue
        parts = function.name.split("_")
        if len(parts) != 4:
            raise ValueError("Unknown NF4 reference function format")
        k, n, block_size = map(int, parts[1:])
        _blocks((k, n), block_size)
        backward = f"InputGrad_{k}_{n}_{block_size}"
        if (DOMAIN, backward, "") not in model.functions:
            raise ValueError("Load the NF4 forward export with input-gradient functions")
        _rules(function.name, backward)
