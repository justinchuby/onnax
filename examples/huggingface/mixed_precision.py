# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""BF16 Qwen projection compute with FP32 boundaries and surrogate derivatives."""

import json
from types import MethodType

import onnx_ir as ir
import torch

import onnax
from examples.huggingface.validation import TorchLora
from examples.lora.bf16_rules import DOMAIN, OP, matmul_function


def precision_projections(model, *, precision="bf16"):
    """Lower matrix projections; keep attention, normalization and loss in FP32.

    Each product rounds its operands and result to the selected half dtype. The shared registered
    rules compute straight-through adjoints in FP32 using rounded operands.
    Parameters and external optimizer state remain FP32. Packed NF4 decoder
    calls retain their storage and supply half forward/FP32 derivative rules.
    """
    if precision not in ("bf16", "fp16"):
        raise ValueError("Expected bf16 or fp16 projection precision")
    suffix = "FP16" if precision == "fp16" else "BF16"
    result = model.clone()
    from examples.lora.nf4 import DOMAIN as NF4_DOMAIN

    if any(
        node.domain == NF4_DOMAIN
        and node.op_type.startswith(("MatMul_", "MatMulBF16_", "MatMulFP16_"))
        and not node.op_type.startswith("MatMul" + suffix + "_")
        for node in result.graph
    ):
        raise ValueError(
            f"Lower NF4 projections with precision={precision!r} before conversion"
        )
    packed_count = sum(
        node.domain == NF4_DOMAIN and node.op_type.startswith("MatMul" + suffix + "_")
        for node in result.graph
    )
    function = matmul_function(result.graph.opset_imports[""], precision=precision)
    if function.identifier() in result.functions:
        raise ValueError("The model already defines the selected half product function")
    context = onnax.AutodiffContext(result.graph)
    count = 0
    for node in list(result.graph):
        if node.domain or node.op_type != "MatMul":
            continue
        a, b = node.inputs
        if b.shape is None or len(b.shape) != 2:
            continue
        if (
            a.shape is None
            or len(a.shape) < 2
            or a.dtype != ir.DataType.FLOAT
            or b.dtype != ir.DataType.FLOAT
            or not all(isinstance(d, int) and d > 0 for d in b.shape)
        ):
            raise ValueError(
                "Half-precision projections require FP32 activations and a static matrix"
            )
        k, n = b.shape
        tape = ir.tape.Tape()

        def constant(value, tape=tape):
            return tape.op(
                "Constant", [], {"value": ir.tensor(value, dtype=ir.DataType.INT64)}
            )

        flat = tape.op("Reshape", [a, constant([-1, k])]) if len(a.shape) > 2 else a
        if flat is not a:
            flat.dtype, flat.shape = ir.DataType.FLOAT, ir.Shape(["projection_rows", k])
        product = tape.op(
            OP if precision == "bf16" else "MatMulFP16", [flat, b], domain=DOMAIN
        )
        product.dtype, product.shape = ir.DataType.FLOAT, ir.Shape(["projection_rows", n])
        output = product
        if len(a.shape) > 2:
            prefix = tape.op("Slice", [tape.op("Shape", [a]), constant([0]), constant([-1])])
            shape = tape.op("Concat", [prefix, constant([n])], {"axis": 0})
            output = tape.op("Reshape", [product, shape])
        for operation in tape.nodes:
            operation.name = context.name("bf16.projection")
            for value in operation.outputs:
                value.name = context.name("bf16.value")
        output.name = node.outputs[0].name
        output.dtype, output.shape = node.outputs[0].dtype, node.outputs[0].shape
        ir.convenience.replace_nodes_and_values(
            result.graph, node, [node], tape.nodes, [node.outputs[0]], [output]
        )
        count += 1
    if count == 0 and packed_count == 0:
        raise ValueError("No FP32 matrix projections found")
    result.functions[function.identifier()] = function
    result.graph.opset_imports[DOMAIN] = 1
    result.metadata_props["onnax.example.precision"] = json.dumps(
        {
            "version": 1,
            "projection_compute": "bfloat16" if precision == "bf16" else "float16",
            "projection_products": count + packed_count,
            "attention_and_normalization": "float32",
            "parameters_and_storage": (
                "FP32 adapters; packed NF4 decoder; FP32 embedding and head"
                if packed_count
                else "float32"
            ),
            "adjoints": "float32 straight-through using rounded operands",
        }
    )
    result.graph.sort()
    return result


def bf16_projections(model):
    """Apply Half-precision projections with FP32 adjoints."""
    return precision_projections(model, precision="bf16")


class _Product(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        qa, qb = a.to(torch.bfloat16), b.to(torch.bfloat16)
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(qa.float(), qb.float())
        return (qa @ qb).float()

    @staticmethod
    def backward(ctx, gradient):
        a, b = ctx.saved_tensors
        da = gradient @ b.T if ctx.needs_input_grad[0] else None
        db = (
            a.reshape(-1, a.shape[-1]).T @ gradient.reshape(-1, b.shape[-1])
            if ctx.needs_input_grad[1]
            else None
        )
        return da, db


class _FP16Product(_Product):
    @staticmethod
    def forward(ctx, a, b):
        qa, qb = a.to(torch.float16), b.to(torch.float16)
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(qa.float(), qb.float())
        return (qa @ qb).float()


def reference_projections(reference, *, precision="bf16"):
    """Apply the selected half-forward/FP32-adjoint policy to Transformers."""
    if precision not in ("bf16", "fp16"):
        raise ValueError("Expected bf16 or fp16 projection precision")
    product = _Product if precision == "bf16" else _FP16Product

    def linear(module, x):
        value = product.apply(x, module.weight.T)
        return value if module.bias is None else value + module.bias

    def lora(module, x):
        delta = product.apply(product.apply(x, module.A), module.B)
        return module.base(x) + module.scale * delta

    for module in reference.modules():
        if isinstance(module, torch.nn.Linear):
            module.forward = MethodType(linear, module)
        elif isinstance(module, TorchLora):
            module.forward = MethodType(lora, module)
    return reference
