# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Execute a flat ONNX tensor graph through reusable subgraph sessions."""

import os
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import onnx
import onnx_ir as ir
from onnx_ir.passes.common import RemoveUnusedNodesPass

from examples.huggingface.serialization import save_graph
from examples.lora.train import session


@dataclass(frozen=True)
class _Segment:
    executor: Any
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


class SegmentedSession:
    """Keep the model's named interface while exchanging segment boundaries in NumPy."""

    def __init__(self, model: ir.Model, *, target_nodes=1024, session_factory=session):
        if (
            isinstance(target_nodes, bool)
            or not isinstance(target_nodes, int)
            or target_nodes < 1
        ):
            raise ValueError("target_nodes must be a positive integer")
        if any(
            attribute.type in (ir.AttributeType.GRAPH, ir.AttributeType.GRAPHS)
            for node in model.graph
            for attribute in node.attributes.values()
        ):
            raise ValueError("Control-flow graphs require an unsplit session")
        start = time.perf_counter()
        self.directory = tempfile.TemporaryDirectory(prefix="onnax-segments-")
        directory = self.directory.name
        source = os.path.join(directory, "source.onnx")
        inferred = os.path.join(directory, "inferred.onnx")
        save_graph(model, source, data_name="source.data")
        # Values crossing a cut become graph inputs and need their inferred types.
        onnx.shape_inference.infer_shapes_path(source, inferred, strict_mode=True)
        typed = ir.load(inferred)
        typed.graph.sort()
        self.input_specs = {}
        for value in typed.graph.inputs:
            if not value.name or value.dtype is None or value.shape is None:
                raise ValueError("Graph inputs need names, tensor types and declared ranks")
            self.input_specs[value.name] = (np.dtype(value.dtype.numpy()), tuple(value.shape))
        self.inputs = tuple(value.name for value in typed.graph.inputs)
        self.outputs = tuple(value.name for value in typed.graph.outputs)
        self.required_inputs = {
            value.name for value in typed.graph.inputs if value.const_value is None
        }
        self.constants = {
            value.name: value.const_value
            for value in typed.graph.outputs
            if value.const_value is not None
        }
        inputs = set(typed.graph.inputs)
        outputs = set(typed.graph.outputs)
        nodes = list(typed.graph)
        # Unknown-rank intermediates stay inside a segment. ONNX graph inputs
        # require shape fields, so extend a proposed cut through their last use.
        positions = {node: index for index, node in enumerate(nodes)}
        changes = [0] * (len(nodes) + 2)
        for index, node in enumerate(nodes):
            for value in node.outputs:
                if value.shape is not None and value.dtype is not None:
                    continue
                if value in outputs:
                    raise ValueError(f"Model output needs a tensor shape: {value.name}")
                last = max((positions[use.node] for use in value.uses()), default=index)
                if last > index:
                    changes[index + 1] += 1
                    changes[last + 1] -= 1
        legal, active = [], 0
        for change in changes:
            active += change
            legal.append(active == 0)
        self.segments = []
        first = 0
        while first < len(nodes):
            end = min(first + target_nodes, len(nodes))
            while not legal[end]:
                end += 1
            block = nodes[first:end]
            first = end
            selected = set(block)
            external, initializers, boundary, seen = [], [], [], set()
            for node in block:
                for value in node.inputs:
                    if value is None or value in seen or value.producer() in selected:
                        continue
                    seen.add(value)
                    if value.const_value is not None:
                        initializers.append(value)
                    if value in inputs or value.const_value is None:
                        external.append(value)
                boundary.extend(
                    value
                    for value in node.outputs
                    if value in outputs
                    or any(use.node not in selected for use in value.uses())
                )
            if not boundary:
                continue
            for value in (*external, *boundary):
                if not value.name or value.dtype is None:
                    raise ValueError(f"Segment boundary needs a name and tensor type: {value}")
            view = ir.GraphView(
                external,
                boundary,
                nodes=block,
                initializers=initializers,
                opset_imports=dict(typed.graph.opset_imports),
                name=f"{typed.graph.name or 'model'}_part{len(self.segments)}",
            )
            part = ir.Model(
                view.clone(),
                ir_version=typed.ir_version,
                functions=list(typed.functions.values()),
            )
            RemoveUnusedNodesPass()(part)
            index = len(self.segments)
            path = os.path.join(directory, f"part{index}.onnx")
            save_graph(part, path, data_name=f"part{index}.data")
            self.segments.append(
                _Segment(
                    session_factory(path),
                    tuple(value.name for value in external),
                    tuple(value.name for value in boundary),
                )
            )
        self.construction_seconds = time.perf_counter() - start
        self.consumers = Counter(name for part in self.segments for name in part.inputs)
        self.last_boundary_bytes = 0

    def run(self, output_names, feeds):
        requested = self.outputs if not output_names else tuple(output_names)
        unknown = set(requested) - set(self.outputs)
        if unknown:
            raise ValueError(f"Unknown outputs: {sorted(unknown)}")
        unknown = set(feeds) - set(self.inputs)
        if unknown:
            raise ValueError(f"Unknown inputs: {sorted(unknown)}")
        missing = self.required_inputs - set(feeds)
        if missing:
            raise ValueError(f"Missing inputs: {sorted(missing)}")
        for name, value in feeds.items():
            dtype, shape = self.input_specs[name]
            if not isinstance(value, np.ndarray) or value.dtype != dtype:
                raise TypeError(f"Input {name} requires a NumPy array with dtype {dtype}")
            if len(value.shape) != len(shape) or any(
                isinstance(size, int) and size >= 0 and size != actual
                for size, actual in zip(shape, value.shape)
            ):
                raise ValueError(f"Input {name} has shape {value.shape}, expected {shape}")
        working = dict(feeds)
        remaining = self.consumers.copy()
        keep = set(requested)
        self.last_boundary_bytes = 0
        for part in self.segments:
            supplied = {name: working[name] for name in part.inputs if name in working}
            self.last_boundary_bytes += sum(value.nbytes for value in supplied.values())
            values = part.executor.run(list(part.outputs), supplied)
            self.last_boundary_bytes += sum(value.nbytes for value in values)
            working.update(
                (name, value)
                for name, value in zip(part.outputs, values, strict=True)
                if remaining[name] or name in keep
            )
            for name in part.inputs:
                remaining[name] -= 1
                if not remaining[name] and name not in keep:
                    working.pop(name, None)
        return [
            working[name]
            if name in working
            else np.array(self.constants[name].numpy(), copy=True)
            for name in requested
        ]
