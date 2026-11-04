# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Optional ONNXScript rewrites for saved-value layouts."""

import onnx_ir as ir
from onnxscript.rewriter import RewriteRuleClassBase


class _ComposeDerivativeTransposes(RewriteRuleClassBase):
    """Compose pullback layouts using values already available to backward."""

    def __init__(self, primal, derivative, available):
        super().__init__()
        self.primal, self.derivative, self.available = primal, derivative, available

    def pattern(self, op, x, perm):
        return op.Transpose(x, perm=perm)

    def check(self, context, x, perm):
        parent = x.producer()
        if (
            context.root not in self.derivative
            or parent is None
            or parent.domain
            or parent.op_type != "Transpose"
            or perm.is_ref()
            or perm.type != ir.AttributeType.INTS
        ):
            return False
        origin = parent.inputs[0]
        if origin.producer() in self.primal and origin not in self.available:
            return False
        first = parent.attributes.get("perm")
        if first is not None and (first.is_ref() or first.type != ir.AttributeType.INTS):
            return False
        second = list(perm.as_ints())
        first = (
            list(first.as_ints()) if first is not None else list(reversed(range(len(second))))
        )
        return (
            len(first) == len(second)
            and sorted(first) == list(range(len(first)))
            and sorted(second) == list(range(len(second)))
        )

    def rewrite(self, op, x, perm):
        parent = x.producer()
        first = parent.attributes.get_ints("perm", list(reversed(range(len(perm.as_ints())))))
        composed = [first[index] for index in perm.as_ints()]
        if composed == list(range(len(composed))):
            return op.Identity(parent.inputs[0])
        return op.Transpose(parent.inputs[0], perm=composed)
