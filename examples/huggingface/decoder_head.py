# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Frozen-head identity and portable artifacts for decoder task adapters."""

import contextlib
import hashlib
import json
import os
import shutil
import tempfile

import numpy as np
import onnx
import onnx_ir as ir

import onnax
from examples.huggingface.chunked_loss import ChunkedHead
from examples.huggingface.serialization import SharedWeights


def file_digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(head):
    weight = head.model.graph.initializers["head.weight"].const_value
    array = weight.numpy()
    if head.weight_layout == "hidden_vocab":
        array = array.T
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "version": 1,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "projection_precision": head.projection_precision,
                "loss": "hard_cross_entropy",
                "shift": 1,
                "ignore_index": -100,
                "reduction": "valid_shifted_token_mean",
            },
            sort_keys=True,
        ).encode()
    )
    digest.update(array.data)
    return digest.hexdigest()


def _inference(directory, head_model):
    model = ir.load(os.path.join(directory, "forward.onnx"))
    hidden = model.graph.outputs[0]
    if hidden.shape is None or len(hidden.shape) != 3:
        raise ValueError("Expected a rank-three decoder hidden-state output")
    context = onnax.AutodiffContext(model.graph)
    source = head_model.graph.initializers["head.weight"].const_value
    if head_model.metadata_props.get("onnax.example.head.weight_layout") != "vocab_hidden":
        raise ValueError("Decoder tasks store head weights in source layout")
    vocab, width = source.shape
    weight = ir.val(context.name("head.weight"), const_value=source)
    model.graph.register_initializer(weight)
    tape = ir.tape.Tape()

    def constant(values):
        return tape.op("Constant", [], {"value": ir.tensor(np.array(values, np.int64))})

    flat = tape.op("Reshape", [hidden, constant([-1, width])])
    product = tape.op("Gemm", [flat, weight], {"transB": 1})
    prefix = tape.op("Slice", [tape.op("Shape", [hidden]), constant([0]), constant([2])])
    shape = tape.op("Concat", [prefix, constant([vocab])], {"axis": 0})
    logits = tape.op("Reshape", [product, shape])
    for node in tape.nodes:
        node.name = context.name("head.inference")
        for value in node.outputs:
            value.name = context.name("head.value")
    logits.name, logits.dtype = "logits", ir.DataType.FLOAT
    logits.shape = ir.Shape([*hidden.shape[:-1], vocab])
    model.graph.extend(tape.nodes)
    model.graph.outputs.clear()
    model.graph.outputs.append(logits)
    path = os.path.join(directory, "inference.onnx")
    ir.save(model, path)
    onnx.checker.check_model(path, full_check=True)
    return {"model": "inference.onnx", "sha256": file_digest(path)}


def export(head, directory):
    with SharedWeights(directory, data_name="head-weights.data") as weights:
        record = head.export_to(weights)
    files = [record[name] for name in ("loss", "gradient", "forward")]
    files.append(weights.data_name)
    record["files"] = {name: file_digest(os.path.join(directory, name)) for name in files}
    record["identity"] = identity(head)
    # Load the exported head so all inference tensor references are relative to this directory.
    projected = ir.load(os.path.join(directory, record["loss"]))
    inference = _inference(directory, projected)
    return record, inference


def validate_file(directory, name, expected):
    if not isinstance(name, str) or os.path.basename(name) != name:
        raise ValueError("Head artifacts must use filenames within the execution directory")
    path = os.path.join(directory, name)
    if file_digest(path) != expected:
        raise ValueError("Frozen-head artifact hash mismatch")


def load(directory, record, *, session_factory):
    with contextlib.ExitStack() as resources:
        directory_owner = tempfile.TemporaryDirectory(prefix="onnax-decoder-head-")
        resources.callback(directory_owner.cleanup)
        snapshots = directory_owner.name
        for name, expected in record["files"].items():
            if not isinstance(name, str) or os.path.basename(name) != name:
                raise ValueError(
                    "Head artifacts must use filenames within the execution directory"
                )
            shutil.copyfile(os.path.join(directory, name), os.path.join(snapshots, name))
            validate_file(snapshots, name, expected)
        head = ChunkedHead.from_record(snapshots, record, session_factory=session_factory)
        resources.callback(head.close)
        if identity(head) != record["identity"]:
            raise ValueError("Frozen-head identity differs from its task record")
        head.directory = directory_owner
        resources.pop_all()
        return head
