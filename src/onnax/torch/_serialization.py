# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Owned ONNX snapshots with streaming, shared external tensor payloads."""

from __future__ import annotations

import hashlib
import os

import numpy as np
import onnx_ir as ir


def file_digest(path: str) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_model(model: ir.Model, directory: str, stem: str, *, reuse: str | None = None) -> str:
    """Snapshot tensors independently of caller buffers, one payload at a time."""
    result = model.clone()
    data_name = stem + ".data"
    with open(os.path.join(directory, data_name), "wb") as stream:
        tensors: dict[int, tuple[ir.TensorProtocol, ir.TensorProtocol]] = {}

        def convert(tensor: ir.TensorProtocol) -> ir.TensorProtocol:
            cached = tensors.get(id(tensor))
            if cached is not None and cached[0] is tensor:
                return cached[1]
            output: ir.TensorProtocol
            if tensor.dtype == ir.DataType.STRING:
                output = ir.tensor(np.array(tensor.numpy(), copy=True), name=tensor.name)
            elif (
                isinstance(tensor, ir.ExternalTensor)
                and reuse is not None
                and os.path.realpath(tensor.base_dir) == os.path.realpath(reuse)
                and os.path.basename(tensor.location) == tensor.location
                and tensor.location != data_name
            ):
                output = ir.ExternalTensor(
                    tensor.location,
                    tensor.offset,
                    tensor.length,
                    tensor.dtype,
                    shape=ir.Shape(tensor.shape),
                    name=tensor.name or "",
                    base_dir=directory,
                )
            elif tensor.nbytes >= 64 * 1024:
                array = np.ascontiguousarray(tensor.numpy())
                offset = stream.tell()
                stream.write(array.reshape(-1).view(np.uint8).data)
                output = ir.ExternalTensor(
                    data_name,
                    offset,
                    array.nbytes,
                    tensor.dtype,
                    shape=ir.Shape(array.shape),
                    name=tensor.name or "",
                    base_dir=directory,
                )
            elif isinstance(tensor, ir.ExternalTensor):
                output = ir.tensor(np.array(tensor.numpy(), copy=True), name=tensor.name)
            else:
                output = tensor
            tensors[id(tensor)] = tensor, output
            return output

        def convert_graph(graph):
            for value in graph.initializers.values():
                assert value.const_value is not None
                value.const_value = convert(value.const_value)
            for node in graph:
                for name, attr in list(node.attributes.items()):
                    if attr.is_ref():
                        continue
                    if attr.type == ir.AttributeType.TENSOR:
                        node.attributes[name] = ir.AttrTensor(name, convert(attr.as_tensor()))
                    elif attr.type == ir.AttributeType.TENSORS:
                        node.attributes[name] = ir.AttrTensors(
                            name, [convert(t) for t in attr.as_tensors()]
                        )
                    elif attr.type == ir.AttributeType.GRAPH:
                        convert_graph(attr.as_graph())
                    elif attr.type == ir.AttributeType.GRAPHS:
                        for nested in attr.as_graphs():
                            convert_graph(nested)

        for graph in (result.graph, *(f.graph for f in result.functions.values())):
            convert_graph(graph)
    path = os.path.join(directory, stem + ".onnx")
    ir.save(result, path)
    return path


def source_identity(directory: str, specification: dict) -> str:
    """Hash graph structure, functions, frozen payloads and the selected interface."""
    import json

    digest = hashlib.sha256()
    for name in ("model.onnx", "model.data"):
        digest.update(name.encode())
        digest.update(bytes.fromhex(file_digest(os.path.join(directory, name))))
    digest.update(json.dumps(specification, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()
