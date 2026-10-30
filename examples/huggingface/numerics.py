# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Compare reference/runtime kernels at identical FP32 inputs.

Run: python -m examples.huggingface.numerics --output report.json
"""

import argparse
import json
import math
import platform

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
from onnx.reference import ReferenceEvaluator

from examples.lora.train import session


def audit():
    results = {}
    for op, values in [("Sigmoid", [-18, -40, 18]), ("LogSoftmax", [0, -200])]:
        x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[len(values)])
        tape = ir.tape.Tape()
        y = tape.op(op, [x], {"axis": -1} if op == "LogSoftmax" else {})
        y.name, y.dtype, y.shape = "y", x.dtype, x.shape
        graph = ir.Model(
            ir.Graph([x], [y], nodes=tape.nodes, opset_imports={"": 18}, name="audit"),
            ir_version=10,
        )
        proto = ir.serde.serialize_model(graph)
        feeds = {"x": np.array(values, np.float32)}
        if op == "Sigmoid":
            expected = [
                math.exp(-abs(v)) / (1 + math.exp(-abs(v)))
                if v < 0
                else 1 / (1 + math.exp(-v))
                for v in values
            ]
        else:
            correction = math.log1p(math.exp(-200))
            expected = [-correction, -200 - correction]
        with np.errstate(divide="ignore"):
            reference = ReferenceEvaluator(proto).run(None, feeds)[0]
        runtime = session(proto.SerializeToString()).run(None, feeds)[0]

        def encode(array):
            return [float(v) if math.isfinite(float(v)) else str(float(v)) for v in array]

        results[op] = {
            "inputs_fp32": values,
            "scalar_math_oracle": expected,
            "onnx_reference": encode(reference),
            "ort_cpu": encode(runtime),
        }
    return {
        "versions": {
            "numpy": np.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "platform": platform.platform(),
        },
        "dtype": "float32",
        "ort_graph_optimization": "disabled",
        "operators": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.output, "w") as stream:
        json.dump(audit(), stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
