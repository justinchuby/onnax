# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Report a VJP's saved forward values and block-local tensor live set."""

import argparse
import hashlib
import json
import os

import onnx_ir as ir
from onnx_ir.passes.common import RemoveUnusedFunctionsPass, ShapeInferencePass
from onnxscript import optimizer

import onnax
from examples.activation_liveness import analyze, tensor_bytes
from examples.huggingface.optimization import optimize_training_graph
from examples.huggingface.residuals import split_vjp


def _record(value, category, returned, origin=None):
    try:
        size = tensor_bytes(value)
    except ValueError:
        size = None
    producer = (value if origin is None else origin).producer()
    return {
        "name": value.name,
        "category": category,
        "dtype": str(value.dtype) if value.dtype is not None else None,
        "shape": None
        if value.shape is None
        else [dim if isinstance(dim, int) else str(dim) for dim in value.shape],
        "bytes": size,
        "forward_output": value.name in returned,
        "producer": None
        if producer is None
        else {
            "name": producer.name,
            "domain": producer.domain,
            "op_type": producer.op_type,
        },
        "backward_consumers": [
            {"op_type": use.node.op_type, "input_index": use.idx} for use in value.uses()
        ],
    }


def report(model, *, inputs, outputs, parameters=(), dimensions=None, optimize=False):
    """Analyze the generated pullback on a copy of a source ONNX IR model.

    Parameters are explicit trainable graph inputs. Initializers are reported
    separately from data, cotangents and computed saved values. Byte counts
    describe logical tensors; a runtime allocator supplies physical peaks.
    """
    source = model.clone()
    inputs, outputs = tuple(inputs), tuple(outputs)
    dimensions = {} if dimensions is None else dict(dimensions)
    if any(
        not isinstance(name, str)
        or not name
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        for name, size in dimensions.items()
    ):
        raise ValueError("Dimensions must map nonempty names to nonnegative integers")
    for graph in source.graphs():
        for value in ir.convenience.create_value_mapping(graph).values():
            if value.shape is not None:
                value.shape = ir.Shape(
                    [
                        dim if isinstance(dim, int) else dimensions.get(str(dim), dim)
                        for dim in value.shape
                    ]
                )
    parameters = set(parameters)
    initializers = set(source.graph.initializers)
    source_inputs = {value.name for value in source.graph.inputs}
    if not parameters <= source_inputs:
        raise ValueError("Trainable parameter names must be source graph inputs")
    if not parameters <= set(inputs):
        raise ValueError("Trainable parameters must be selected VJP inputs")
    ShapeInferencePass()(source)
    joint = onnax.vjp(source, inputs=list(inputs), outputs=list(outputs))
    _, backward, residuals = split_vjp(source, joint, optimize_layout=True)
    RemoveUnusedFunctionsPass()(backward)
    ShapeInferencePass()(backward)
    optimizer.remove_unused_nodes(backward)
    if optimize:
        backward = optimize_training_graph(backward)
        for graph in backward.graphs():
            for value in ir.convenience.create_value_mapping(graph).values():
                if value.const_value is not None:
                    value.shape = ir.Shape(value.const_value.shape)
        ShapeInferencePass()(backward)
    # Preserve the mathematical interface in the source; trim only this
    # accounting copy so unused declared inputs do not inflate the live set.
    returned = {value.name for value in source.graph.outputs}
    residuals = set(residuals)
    source_values = ir.convenience.create_value_mapping(source.graph)
    records = []
    for value in dict.fromkeys(
        [*backward.graph.inputs, *backward.graph.initializers.values()]
    ):
        if not value.uses() and value not in backward.graph.outputs:
            if value in backward.graph.inputs:
                backward.graph.inputs.remove(value)
            if value.name in backward.graph.initializers:
                del backward.graph.initializers[value.name]
            continue
        category = (
            "saved_activation"
            if value.name in residuals
            else "trainable_parameter"
            if value.name in parameters
            else "data_input"
            if value.name in source_inputs
            else "frozen_initializer"
            if value.name in initializers and value.name in backward.graph.initializers
            else "folded_constant"
            if value.name in backward.graph.initializers
            else "cotangent"
        )
        records.append(_record(value, category, returned, source_values.get(value.name)))
    categories = {}
    for category in (
        "trainable_parameter",
        "frozen_initializer",
        "folded_constant",
        "data_input",
        "saved_activation",
        "cotangent",
    ):
        members = [row for row in records if row["category"] == category]
        categories[category] = {
            "count": len(members),
            "known_bytes": sum(row["bytes"] for row in members if row["bytes"] is not None),
            "unknown_values": [row["name"] for row in members if row["bytes"] is None],
        }
    extra = [
        row
        for row in records
        if row["category"] == "saved_activation" and not row["forward_output"]
    ]
    try:
        liveness = {"status": "complete", **analyze(backward)}
        liveness["peak_by_category"] = {
            **{name: row["known_bytes"] for name, row in categories.items()},
            "backward_temporaries": liveness["peak_activation_bytes"],
        }
    except ValueError as error:
        liveness = {"status": "unresolved", "reason": str(error)}
    return {
        "version": 1,
        "source_graph": source.graph.name,
        "selected_inputs": list(inputs),
        "selected_outputs": list(outputs),
        "dimensions": dimensions,
        "optimization": "training_pipeline" if optimize else "none",
        "categories": categories,
        "values": records,
        "extra_forward_outputs": {
            "known_bytes": sum(row["bytes"] for row in extra if row["bytes"] is not None),
            "unknown_values": [row["name"] for row in extra if row["bytes"] is None],
        },
        "backward_liveness": liveness,
        "scope": "One generated block VJP; logical tensor bytes, separate storage per value; other blocks and runtime workspaces are outside this live set",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?")
    parser.add_argument("--bundle")
    parser.add_argument("--block")
    parser.add_argument("--inputs", nargs="+")
    parser.add_argument("--outputs", nargs="+")
    parser.add_argument("--parameters", nargs="*")
    parser.add_argument("--dimension", action="append", default=[], metavar="NAME=SIZE")
    parser.add_argument("--output", required=True)
    parser.add_argument("--optimize", action="store_true")
    args = parser.parse_args()
    provenance = {}
    if args.bundle:
        if args.model or args.inputs or args.outputs or args.parameters:
            parser.error("A bundle selects the model and gradient interface")
        with open(os.path.join(args.bundle, "interface.json"), encoding="utf-8") as stream:
            interface = json.load(stream)
        if interface.get("version") != 1 or interface.get("kind") != "checkpointed_vjp":
            parser.error("Expected a version 1 checkpointed ONNX execution bundle")
        matches = [
            (index, row)
            for index, row in enumerate(interface["blocks"])
            if row["name"] == args.block
        ]
        if len(matches) != 1:
            parser.error("Select a unique --block name from the bundle interface")
        index, block = matches[0]
        filename = f"block_{index}_forward.onnx"
        if filename not in interface["files"] or not block.get("backward"):
            parser.error("Selected block has no derivative model interface")
        args.model = os.path.join(args.bundle, filename)
        args.inputs, args.outputs = block["active_inputs"], block["active_outputs"]
        args.parameters = [
            name for name in args.inputs if name in interface["specification"]["parameters"]
        ]
        provenance = {"bundle_identity": interface["identity"], "block": block["name"]}
    elif not args.model or not args.inputs or not args.outputs or args.block:
        parser.error("Supply a model with --inputs/--outputs, or --bundle with --block")
    with open(args.model, "rb") as stream:
        provenance["onnx_file_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    dimensions = {}
    for item in args.dimension:
        try:
            name, size = item.split("=", 1)
            if name in dimensions:
                parser.error(f"Duplicate dimension: {name}")
            dimensions[name] = int(size)
        except ValueError:
            parser.error(f"Expected NAME=SIZE: {item}")
    result = report(
        ir.load(args.model),
        inputs=args.inputs,
        outputs=args.outputs,
        parameters=args.parameters or (),
        dimensions=dimensions,
        optimize=args.optimize,
    )
    result["provenance"] = provenance
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
