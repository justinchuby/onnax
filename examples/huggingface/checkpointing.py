# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Partition Qwen graphs at preserved decoder hidden-state boundaries."""

import json
import os
import tempfile
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import onnx
import onnx_ir as ir
import torch
from torch.autograd.function import once_differentiable

import onnax
from examples.huggingface.bridge import OnnxModule
from examples.huggingface.optimization import optimize_training_graph
from examples.huggingface.qwen import DECODER_BOUNDARIES
from examples.huggingface.serialization import SharedWeights
from examples.lora.train import session


@dataclass(frozen=True)
class Block:
    """One bounded primal graph and its differentiable interface."""

    name: str
    model: ir.Model
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    active_inputs: tuple[str, ...]
    active_outputs: tuple[str, ...]


def _ancestors(value):
    pending, result = [value], set()
    while pending:
        producer = pending.pop().producer()
        if producer is not None and producer not in result:
            result.add(producer)
            pending.extend(v for v in producer.inputs if v is not None)
    return result


def partition(model, parameters, *, layers=1):
    """Extract prefix, decoder groups and suffix, including shared captures.

    Each node belongs to the earliest boundary that depends on it. Every edge
    crossing a group is an explicit output/input pair, including shape values,
    masks and RoPE operands reused after decomposition or CSE.
    """
    if isinstance(layers, bool) or not isinstance(layers, int) or layers < 1:
        raise ValueError("checkpoint layers must be a positive integer")
    if len(model.graph.outputs) != 1:
        raise ValueError("Checkpoint body requires one hidden-state output")
    if any(
        attr.type in (ir.AttributeType.GRAPH, ir.AttributeType.GRAPHS)
        for node in model.graph
        for attr in node.attributes.values()
    ):
        raise ValueError("Decoder checkpointing expects a flat graph")
    record = json.loads(model.metadata_props[DECODER_BOUNDARIES])
    if record.get("version") != 1:
        raise ValueError("Unsupported decoder boundary metadata version")
    values = ir.convenience.create_value_mapping(model.graph)
    hidden = record["hidden_states"]
    exits = hidden[1:]
    goals = [("prefix", hidden[0])]
    for start in range(0, len(exits), layers):
        stop = min(start + layers, len(exits))
        goals.append((f"decoder_{start}_{stop}", exits[stop - 1]))
    goals.append(("suffix", model.graph.outputs[0].name))
    groups, seen = [], set()
    for name, goal in goals:
        ancestors = _ancestors(values[goal])
        if not seen.issubset(ancestors):
            raise ValueError(f"Decoder boundary {goal} does not follow the preceding group")
        members = ancestors - seen
        if members:
            groups.append((name, members))
        seen = ancestors
    owner = {node: index for index, (_, nodes) in enumerate(groups) for node in nodes}
    floating = {
        ir.DataType.FLOAT,
        ir.DataType.DOUBLE,
        ir.DataType.FLOAT16,
        ir.DataType.BFLOAT16,
    }
    active = {values[name] for name in parameters} | {values[hidden[0]]}
    for node in model.graph:
        if any(value in active for value in node.inputs if value is not None):
            active.update(value for value in node.outputs if value.dtype in floating)
    blocks = []
    for index, (name, members) in enumerate(groups):
        nodes = [node for node in model.graph if node in members]
        incoming = dict.fromkeys(
            value.name
            for node in nodes
            for value in node.inputs
            if value is not None
            and value.producer() not in members
            and value.name not in model.graph.initializers
        )
        outgoing = [
            value.name
            for node in nodes
            for value in node.outputs
            if value in model.graph.outputs
            or any(owner.get(use.node, index) > index for use in value.uses())
        ]
        graph = ir.convenience.extract(model.graph, list(incoming), outgoing)
        if len(graph) != len(nodes):
            raise ValueError(f"Checkpoint extraction changed the node set for {name}")
        graph.name = name
        # Extension rules can emit companion functions absent from primal calls
        # (e.g. NF4 InputGrad). Keep those definitions available to VJP generation.
        piece = ir.Model(
            graph,
            ir_version=model.ir_version,
            functions=[function.clone() for function in model.functions.values()],
        )
        piece.metadata_props.update(model.metadata_props)
        blocks.append(
            Block(
                name,
                piece,
                tuple(incoming),
                tuple(outgoing),
                tuple(value for value in incoming if values[value] in active),
                tuple(value for value in outgoing if values[value] in active),
            )
        )
    if set().union(*(nodes for _, nodes in groups)) != seen:
        raise AssertionError("Checkpoint node ownership is incomplete")
    return blocks


@dataclass(frozen=True)
class _Execution:
    block: Block
    forward: object
    backward: object
    seeds: tuple[str, ...]
    gradients: tuple[str, ...]
    derivative: ir.Model | None


class _GroupFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, runtime, execution, invocation, *inputs):
        ctx.runtime, ctx.execution, ctx.invocation = runtime, execution, invocation
        ctx.save_for_backward(*inputs)
        feeds = {
            name: tensor.detach().contiguous().numpy()
            for name, tensor in zip(execution.block.inputs, inputs, strict=True)
        }
        arrays = execution.forward.run(list(execution.block.outputs), feeds)
        outputs = []
        for index, array in enumerate(arrays):
            if any(
                np.shares_memory(array, other) for other in (*feeds.values(), *arrays[:index])
            ):
                array = array.copy()
            outputs.append(torch.from_numpy(array))
        ctx.shapes = [value.shape for value in outputs]
        ctx.mark_non_differentiable(
            *(
                value
                for name, value in zip(execution.block.outputs, outputs, strict=True)
                if name not in execution.block.active_outputs
            )
        )
        ctx.set_materialize_grads(False)
        return tuple(outputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, *upstreams):
        runtime, execution = ctx.runtime, ctx.execution
        block = execution.block
        feeds = {
            name: value.detach().contiguous().numpy()
            for name, value in zip(block.inputs, ctx.saved_tensors, strict=True)
        }
        gradients = dict(zip(block.outputs, upstreams, strict=True))
        shapes = dict(zip(block.outputs, ctx.shapes, strict=True))
        for name, seed in zip(block.active_outputs, execution.seeds, strict=True):
            value = gradients[name]
            feeds[seed] = (
                np.zeros(shapes[name], dtype=np.float32)
                if value is None
                else value.detach().contiguous().numpy()
            )
        start = time.perf_counter()
        values = execution.backward.run(list(execution.gradients), feeds)
        elapsed = time.perf_counter() - start
        derivatives = dict(zip(block.active_inputs, values, strict=True))
        if ctx.invocation["remaining"] == 0:
            ctx.invocation.update(remaining=ctx.invocation["count"], elapsed=0.0)
        ctx.invocation["elapsed"] += elapsed
        ctx.invocation["remaining"] -= 1
        if ctx.invocation["remaining"] == 0:
            runtime.timings["backward"].append(ctx.invocation["elapsed"])
        return (
            None,
            None,
            None,
            *(
                torch.from_numpy(derivatives[name]) if name in derivatives else None
                for name in block.inputs
            ),
        )


class CheckpointedModule(OnnxModule):
    """Execute grouped ONNX VJPs while retaining decoder boundary checkpoints."""

    def __init__(self, model, params, *, layers=1, optimize=True, session_factory=session):
        torch.nn.Module.__init__(self)
        self._init_parameters(model, params, None)
        self.layers = layers
        self.blocks = partition(model, params, layers=layers)
        self.runtime_directory = tempfile.TemporaryDirectory(prefix="onnax-checkpoints-")
        self.executions = []
        parameter_gradients = {}
        self.last_residual_bytes = 0
        self.timings = {name: [] for name in ("forward", "backward", "exchange")}
        with SharedWeights(self.runtime_directory.name) as weights:
            for block in self.blocks:
                forward = session_factory(weights.save(block.model, block.name + "_forward"))
                backward, seeds, gradients, derivative = None, (), (), None
                if block.active_inputs and block.active_outputs:
                    derivative = onnax.vjp(
                        block.model,
                        inputs=list(block.active_inputs),
                        outputs=list(block.active_outputs),
                    )
                    seeds = tuple(v.name for v in derivative.graph.inputs[len(block.inputs) :])
                    outputs = list(derivative.graph.outputs)[len(block.outputs) :]
                    gradients = tuple(value.name for value in outputs)
                    parameter_gradients.update(
                        (name, gradient)
                        for name, gradient in zip(block.active_inputs, gradients, strict=True)
                        if name in self.names
                    )
                    derivative.graph.outputs.clear()
                    derivative.graph.outputs.extend(outputs)
                    if optimize:
                        derivative = optimize_training_graph(derivative)
                    backward = session_factory(
                        weights.save(derivative, block.name + "_backward")
                    )
                self.executions.append(
                    _Execution(block, forward, backward, seeds, gradients, derivative)
                )
        self.gradients = [parameter_gradients[name] for name in self.names]
        self.forward_uses = Counter(
            name for execution in self.executions for name in execution.block.inputs
        )

    def export(self, directory, *, forward_model=None):
        os.makedirs(directory, exist_ok=True)
        result = self._bound_forward(forward_model)
        record = {"version": 1, "layers": self.layers, "body": "body.onnx", "blocks": []}
        with SharedWeights(directory) as weights:
            paths = [weights.save(result, "forward"), weights.save(self.model, "body")]
            for execution in self.executions:
                block = execution.block
                forward = block.name + "_forward"
                paths.append(weights.save(block.model, forward))
                backward = None
                if execution.derivative is not None:
                    backward = block.name + "_backward"
                    paths.append(weights.save(execution.derivative, backward))
                record["blocks"].append(
                    {
                        "name": block.name,
                        "forward": forward + ".onnx",
                        "backward": backward + ".onnx" if backward is not None else None,
                        "inputs": block.inputs,
                        "outputs": block.outputs,
                        "active_inputs": block.active_inputs,
                        "active_outputs": block.active_outputs,
                        "seeds": execution.seeds,
                        "gradients": execution.gradients,
                    }
                )
        for path in paths:
            onnx.checker.check_model(path, full_check=True)
        self._export_parameters(directory)
        with open(os.path.join(directory, "checkpoint_plan.json"), "w") as stream:
            json.dump(record, stream, indent=2)
        path = os.path.join(directory, "interface.json")
        with open(path) as stream:
            interface = json.load(stream)
        interface.update(
            kind="checkpointed_vjp", model="body.onnx", plan="checkpoint_plan.json"
        )
        for parameter in interface["parameters"]:
            parameter["model"] = next(
                row["backward"]
                for row in record["blocks"]
                if parameter["name"] in row["active_inputs"]
            )
        with open(path, "w") as stream:
            json.dump(interface, stream, indent=2)
        return result

    @classmethod
    def load(cls, directory, *, session_factory=session):
        """Reuse the exported forward/VJP modules without regenerating derivatives."""
        with open(os.path.join(directory, "checkpoint_plan.json")) as stream:
            record = json.load(stream)
        if record.get("version") != 1:
            raise ValueError("Unsupported checkpoint execution plan version")
        with open(os.path.join(directory, "interface.json")) as stream:
            interface = json.load(stream)
        if interface.get("kind") != "checkpointed_vjp":
            raise ValueError("Expected an exported checkpointed VJP interface")
        model = ir.load(os.path.join(directory, record["body"]))
        with np.load(os.path.join(directory, "adapter.npz")) as values:
            if values.files != [row["name"] for row in interface["parameters"]]:
                raise ValueError("Exported adapter names differ from the parameter interface")
            parameters = {name: values[name] for name in values.files}
        result = cls.__new__(cls)
        torch.nn.Module.__init__(result)
        result._init_parameters(model, parameters, None)
        if result.frozen != interface["frozen"] or result.output != interface["output"]:
            raise ValueError("Exported checkpoint body differs from the frozen interface")
        result.layers = record["layers"]
        result.last_residual_bytes = 0
        result.timings = {name: [] for name in ("forward", "backward", "exchange")}
        result.executions = []
        parameter_gradients = {}
        for row in record["blocks"]:
            path = os.path.join(directory, row["forward"])
            block = Block(
                row["name"],
                ir.load(path),
                *(
                    tuple(row[name])
                    for name in ("inputs", "outputs", "active_inputs", "active_outputs")
                ),
            )
            forward = session_factory(path)
            backward = derivative = None
            if row["backward"] is not None:
                path = os.path.join(directory, row["backward"])
                backward, derivative = session_factory(path), ir.load(path)
            result.executions.append(
                _Execution(
                    block,
                    forward,
                    backward,
                    tuple(row["seeds"]),
                    tuple(row["gradients"]),
                    derivative,
                )
            )
            parameter_gradients.update(
                (name, gradient)
                for name, gradient in zip(block.active_inputs, row["gradients"], strict=True)
                if name in parameters
            )
        result.blocks = [execution.block for execution in result.executions]
        result.gradients = [parameter_gradients[name] for name in result.names]
        if result.gradients != [row["gradient"] for row in interface["parameters"]]:
            raise ValueError("Checkpoint gradient mapping differs from the interface")
        result.forward_uses = Counter(
            name for execution in result.executions for name in execution.block.inputs
        )
        return result

    def _execute(self, feeds, *, save):
        values = {**feeds, **dict(zip(self.names, self.adapters, strict=True))}
        remaining = self.forward_uses.copy()
        checkpoint_bytes = {}
        invocation = {"elapsed": 0.0, "remaining": 0, "count": 0}
        for execution in self.executions:
            block = execution.block
            inputs = tuple(values[name] for name in block.inputs)
            if save and any(value.requires_grad for value in inputs):
                invocation["remaining"] += 1
                invocation["count"] += 1
                result = _GroupFunction.apply(self, execution, invocation, *inputs)
                for name, value in zip(block.inputs, inputs, strict=True):
                    if name not in self.names and name not in self.data_names:
                        checkpoint_bytes[name] = value.numel() * value.element_size()
            else:
                arguments = {
                    name: value.detach().contiguous().numpy()
                    for name, value in zip(block.inputs, inputs, strict=True)
                }
                arrays = execution.forward.run(list(block.outputs), arguments)
                result = tuple(torch.from_numpy(value) for value in arrays)
            values.update(zip(block.outputs, result, strict=True))
            for name in block.inputs:
                remaining[name] -= 1
                if remaining[name] == 0 and name != self.output:
                    values.pop(name, None)
        self.last_residual_bytes = sum(checkpoint_bytes.values())
        return values[self.output]

    def forward(self, **inputs):
        if set(inputs) != set(self.data_names):
            raise ValueError(
                f"Expected ONNX data inputs {self.data_names}, got {tuple(inputs)}"
            )
        if any(value.device.type != "cpu" for value in inputs.values()):
            raise ValueError("This bridge executes CPU input tensors")
        start = time.perf_counter()
        # Snapshot caller-owned data once; grouped saved tensors share these snapshots.
        feeds = {name: value.detach().clone().contiguous() for name, value in inputs.items()}
        exchanged = time.perf_counter()
        result = self._execute(feeds, save=torch.is_grad_enabled())
        self.timings["exchange"].append(exchanged - start)
        self.timings["forward"].append(time.perf_counter() - exchanged)
        return result
