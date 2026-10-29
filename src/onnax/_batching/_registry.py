# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Typed batching dispatch and supported operator sets."""

from __future__ import annotations

from typing import Generic, Protocol, TypeVar

import onnx_ir as ir

from onnax import _rule_registry

_ContextT = TypeVar("_ContextT")
_ContextT_contra = TypeVar("_ContextT_contra", contravariant=True)


class _BatchingRule(Protocol[_ContextT_contra]):
    def __call__(self, context: _ContextT_contra, node: ir.Node, /) -> None: ...


class _BatchingRegistry(
    _rule_registry._RuleRegistry[_BatchingRule[_ContextT]], Generic[_ContextT]
):
    """Instance-local batching rules with explicit supported opset intervals."""

    _kind = "batching"


_UNARY = frozenset(
    {
        "Abs",
        "Acos",
        "Acosh",
        "Asin",
        "Asinh",
        "Atan",
        "Atanh",
        "BitwiseNot",
        "Cast",
        "Ceil",
        "Celu",
        "Cos",
        "Cosh",
        "Elu",
        "Erf",
        "Exp",
        "Floor",
        "Gelu",
        "HardSigmoid",
        "HardSwish",
        "Identity",
        "IsInf",
        "IsNaN",
        "Log",
        "LeakyRelu",
        "Mish",
        "Neg",
        "Not",
        "Reciprocal",
        "Relu",
        "Round",
        "Selu",
        "Shrink",
        "Sigmoid",
        "Sign",
        "Sin",
        "Sinh",
        "Softplus",
        "Softsign",
        "Sqrt",
        "Tan",
        "Tanh",
        "ThresholdedRelu",
    }
)

_BINARY = frozenset(
    {
        "Add",
        "And",
        "BitShift",
        "BitwiseAnd",
        "BitwiseOr",
        "BitwiseXor",
        "Div",
        "Equal",
        "Greater",
        "GreaterOrEqual",
        "Less",
        "LessOrEqual",
        "Mul",
        "Mod",
        "Or",
        "Pow",
        "Sub",
        "Xor",
    }
)

_VARIADIC = frozenset({"Max", "Mean", "Min", "Sum"})

_REDUCTIONS = frozenset(
    {
        "ReduceL1",
        "ReduceL2",
        "ReduceLogSum",
        "ReduceLogSumExp",
        "ReduceMax",
        "ReduceMean",
        "ReduceMin",
        "ReduceProd",
        "ReduceSum",
        "ReduceSumSquare",
    }
)

_NATIVE_BATCH = frozenset(
    {
        "Conv",
        "ConvTranspose",
        "AveragePool",
        "MaxPool",
        "LpPool",
        "GlobalAveragePool",
        "GlobalMaxPool",
        "GlobalLpPool",
        "InstanceNormalization",
        "BatchNormalization",
        "LRN",
        "GridSample",
        "ConvInteger",
        "QLinearConv",
        "DepthToSpace",
        "SpaceToDepth",
    }
)

_RANDOM = frozenset(
    {
        "Bernoulli",
        "Multinomial",
        "RandomNormal",
        "RandomNormalLike",
        "RandomUniform",
        "RandomUniformLike",
    }
)

_MULTI_OUTPUT = frozenset({"Split", "TopK", "LayerNormalization", "Dropout"})
