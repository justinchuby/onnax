# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Account for the exported recomputation graph using captured input shapes."""

import argparse
import json
import os

import onnx_ir as ir
from onnx_ir.passes.common import InlinePass, RemoveUnusedFunctionsPass, ShapeInferencePass

from examples.activation_liveness import analyze
from examples.huggingface.block_runtime import digest, load_case
from examples.huggingface.optimization import optimize_training_graph


def report(directory, captures, name):
    record, block, feeds, _ = load_case(directory, captures, name)
    model = ir.load(os.path.join(directory, block["model"]))
    bindings = {}
    for value in model.graph.inputs:
        array = feeds[value.name]
        if value.shape is not None:
            if len(value.shape) != array.ndim:
                raise ValueError("Captured input rank differs from model declaration")
            for dim, size in zip(value.shape, array.shape, strict=True):
                if isinstance(dim, int):
                    if dim != size:
                        raise ValueError("Captured input size differs from model declaration")
                elif dim.value is not None:
                    if dim.value in bindings and bindings[dim.value] != size:
                        raise ValueError("Captured symbolic dimensions conflict")
                    bindings[dim.value] = size
        value.shape = ir.Shape(array.shape)
    for graph in model.graphs():
        for value in ir.convenience.create_value_mapping(graph).values():
            if value.shape is not None:
                value.shape = ir.Shape(
                    [
                        dim if isinstance(dim, int) else bindings.get(dim.value, dim)
                        for dim in value.shape
                    ]
                )
    # This is a logical fallback expansion of the already generated derivative.
    # Input values remain dynamic; only their shapes specialize the graph.
    RemoveUnusedFunctionsPass()(model)
    InlinePass()(model)
    ShapeInferencePass()(model)
    model = optimize_training_graph(model)
    for graph in model.graphs():
        for value in ir.convenience.create_value_mapping(graph).values():
            if value.const_value is not None:
                value.shape = ir.Shape(value.const_value.shape)
    ShapeInferencePass()(model)
    try:
        liveness = {"status": "complete", **analyze(model, scan_bodies=True)}
    except ValueError as error:
        liveness = {"status": "unresolved", "reason": str(error)}
    return {
        "version": 1,
        "block": name,
        "bundle_identity": record["identity"],
        "executed_model_sha256": record["files"][block["model"]],
        "input_arrays": block["input_arrays"],
        "dimensions": bindings,
        "liveness": liveness,
        "policy": "Exported recomputation/VJP, local ONNX functions expanded and existing ONNXScript training optimizer applied with concrete shapes; input values remain dynamic; preallocated Scan outputs; separate tensor storage per value",
        "source_sha256": {"block_liveness": digest(__file__)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--captures", required=True)
    parser.add_argument("--block", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = report(args.bundle, args.captures, args.block)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"block": args.block, "status": result["liveness"]["status"]}))


if __name__ == "__main__":
    main()
