# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Propagate arithmetic intervals from LoRA inputs to predictions, loss and gradients."""

import numpy as np
import onnx
import onnx_ir as ir

from examples.lora import intervals as bounds


def propagate(model, feeds, *, nonlinear_ulps=4):
    """Evaluate the supported example graph using only inputs and initializers."""
    if set(feeds) != {value.name for value in model.graph.inputs}:
        raise ValueError("Interval feeds must match graph inputs")
    for value in model.graph.inputs:
        array = np.asarray(feeds[value.name])
        if array.dtype != np.float32 or tuple(array.shape) != tuple(value.shape):
            raise ValueError(
                f"Expected static FP32 input shape {value.shape} for {value.name}"
            )
    values = {}
    for name, array in {
        **{
            name: value.const_value.numpy() for name, value in model.graph.initializers.items()
        },
        **feeds,
    }.items():
        array = np.asarray(array)
        if array.dtype.kind in "iu" and np.any(np.abs(array.astype(object)) > 2**53):
            raise ValueError("Integer constants must be exactly representable in FP64")
        values[name] = bounds.Interval.point(array)
    for node in model.graph:
        if node.domain or len(node.outputs) != 1:
            raise ValueError("Expected standard-domain single-output example operators")
        args = [values[value.name] for value in node.inputs if value is not None]
        op = node.op_type
        if op == "Cast":
            dtype = ir.DataType(node.attributes["to"].as_int())
            if dtype == ir.DataType.BFLOAT16:
                result = bounds.bf16(args[0])
            elif dtype == ir.DataType.FLOAT:
                result = args[0]
                if node.inputs[0].dtype not in (ir.DataType.FLOAT, ir.DataType.BFLOAT16):
                    result = bounds.fp32(result)
            else:
                raise ValueError(f"Unsupported interval Cast: {dtype}")
        elif op == "MatMul":
            result = bounds.matmul(*args)
            if node.outputs[0].dtype == ir.DataType.BFLOAT16:
                result = bounds.bf16(result)
        elif op in ("Add", "Sub", "Mul", "Div"):
            result = bounds.elementwise(*args, op, same_input=node.inputs[0] is node.inputs[1])
        elif op in ("Tanh", "Exp"):
            result = bounds.nonlinear(args[0], op, ulps=nonlinear_ulps)
        elif op == "Neg":
            result = -args[0]
        elif op == "Transpose":
            result = args[0].transpose(tuple(node.attributes["perm"].as_ints()))
        elif op == "GreaterOrEqual":
            # Boolean endpoints mean definitely true / possibly true.
            result = bounds.Interval(
                args[0].lower >= args[1].upper,
                args[0].upper >= args[1].lower,
            )
        elif op == "Where":
            condition, yes, no = args
            result = bounds.Interval(
                np.minimum(
                    np.where(condition.upper, yes.lower, np.inf),
                    np.where(condition.lower, np.inf, no.lower),
                ),
                np.maximum(
                    np.where(condition.upper, yes.upper, -np.inf),
                    np.where(condition.lower, -np.inf, no.upper),
                ),
            )
        elif op == "ReduceMean":
            if len(args) != 1 or node.attributes["keepdims"].as_int() != 0:
                raise ValueError("Expected a scalar mean over all coordinates")
            result = bounds.mean(args[0])
        elif op == "Expand":
            shape = args[1]
            if np.any(shape.lower != shape.upper) or np.any(
                shape.lower != np.floor(shape.lower)
            ):
                raise ValueError("Expected exact integer expansion dimensions")
            dimensions = tuple(int(n) for n in shape.lower)
            result = bounds.Interval(
                np.broadcast_to(args[0].lower, dimensions),
                np.broadcast_to(args[0].upper, dimensions),
            )
        else:
            raise ValueError(f"Unsupported interval operator: {op}")
        values[node.outputs[0].name] = result
    return values


class GraphInspector:
    """Check composed enclosures and local nonlinear budgets on an instrumented copy."""

    def __init__(self, model, *, session_factory, nonlinear_ulps=4):
        self.model = model.clone()
        self.nonlinear_ulps = nonlinear_ulps
        observed = model.clone()
        self.original_names = [value.name for value in observed.graph.outputs]
        self.names = list(self.original_names)
        tape = ir.tape.Tape(observed.graph)
        for node in list(observed.graph):
            value = node.outputs[0]
            if value.dtype not in (ir.DataType.FLOAT, ir.DataType.BFLOAT16):
                continue
            if value.name in self.names:
                continue
            if value.dtype == ir.DataType.BFLOAT16:
                output = tape.op("Cast", [value], {"to": int(ir.DataType.FLOAT)})
                output.dtype, output.shape = ir.DataType.FLOAT, ir.Shape(value.shape)
            else:
                output = value
            observed.graph.outputs.append(output)
            self.names.append(value.name)
        onnx.checker.check_model(ir.to_proto(observed), full_check=True)
        self.executor = session_factory(observed)

    def check(self, feeds, original, *, reference=None):
        enclosure = propagate(self.model, feeds, nonlinear_ulps=self.nonlinear_ulps)
        observed = self.executor.run(None, feeds)
        values = dict(zip(self.names, observed, strict=True))
        unchanged = all(
            a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
            for a, b in zip(original, observed[: len(self.original_names)], strict=True)
        )
        checks = {name: enclosure[name].check(value) for name, value in values.items()}
        observed_inputs = {
            **{
                name: value.const_value.numpy()
                for name, value in self.model.graph.initializers.items()
            },
            **feeds,
            **values,
        }
        arithmetic_checks = {}
        for node in self.model.graph:
            op = node.op_type
            if op not in ("Add", "Sub", "Mul", "Div", "MatMul", "ReduceMean"):
                continue
            args = [bounds.Interval.point(observed_inputs[v.name]) for v in node.inputs]
            if op == "MatMul":
                local = bounds.matmul(*args)
                if node.outputs[0].dtype == ir.DataType.BFLOAT16:
                    local = bounds.bf16(local)
            elif op == "ReduceMean":
                local = bounds.mean(args[0])
            else:
                local = bounds.elementwise(
                    *args, op, same_input=node.inputs[0] is node.inputs[1]
                )
            arithmetic_checks[node.outputs[0].name] = local.check(values[node.outputs[0].name])
        nonlinear_checks = {}
        for node in self.model.graph:
            if node.op_type in ("Tanh", "Exp"):
                local = bounds.nonlinear(
                    bounds.Interval.point(values[node.inputs[0].name]),
                    node.op_type,
                    ulps=self.nonlinear_ulps,
                )
                nonlinear_checks[node.outputs[0].name] = local.check(
                    values[node.outputs[0].name]
                )
        floating = [
            *values.values(),
            *feeds.values(),
            *(
                value.const_value.numpy()
                for value in self.model.graph.initializers.values()
                if value.dtype == ir.DataType.FLOAT
            ),
        ]
        normal_operands = all(
            np.all((value == 0) | (np.abs(value) >= 2**-126)) for value in floating
        )
        reference_checks = {}
        if reference is not None:
            reference_checks = {
                name: enclosure[name].check(value)
                for name, value in zip(self.original_names, reference, strict=True)
            }
        return {
            "original_outputs_equal": unchanged,
            "normal_or_zero_operands": normal_operands,
            "reference_outputs": reference_checks,
            "outputs": {name: checks[name] for name in self.original_names},
            "intermediates": checks,
            "nonlinear_budgets": nonlinear_checks,
            "arithmetic_budgets": arithmetic_checks,
            "all_pass": unchanged
            and normal_operands
            and all(not row["violations"] for row in arithmetic_checks.values())
            and all(not row["violations"] for row in checks.values())
            and all(not row["violations"] for row in nonlinear_checks.values())
            and all(not row["violations"] for row in reference_checks.values()),
        }
