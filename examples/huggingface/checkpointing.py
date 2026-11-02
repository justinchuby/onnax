# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Partition Qwen graphs at preserved decoder hidden-state boundaries."""

import json
import os
import tempfile
import time
from collections import Counter

import numpy as np
import onnx
import onnx_ir as ir
import torch

import onnax
from examples.huggingface.bridge import OnnxModule
from examples.huggingface.optimization import optimize_training_graph
from examples.huggingface.qwen import DECODER_BOUNDARIES
from examples.huggingface.serialization import SharedWeights
from examples.lora.train import session
from onnax._checkpoint import Block
from onnax._checkpoint import partition as partition_graph
from onnax.torch._checkpoint_execution import _Execution, execute


def partition(model, parameters, *, layers=1):
    """Translate preserved Qwen decoder metadata into named checkpoint boundaries."""
    if isinstance(layers, bool) or not isinstance(layers, int) or layers < 1:
        raise ValueError("checkpoint layers must be a positive integer")
    if len(model.graph.outputs) != 1:
        raise ValueError("Checkpoint body requires one hidden-state output")
    record = json.loads(model.metadata_props[DECODER_BOUNDARIES])
    if record.get("version") != 1:
        raise ValueError("Unsupported decoder boundary metadata version")
    hidden = record["hidden_states"]
    exits = hidden[1:]
    goals = {"prefix": hidden[0]}
    for start in range(0, len(exits), layers):
        stop = min(start + layers, len(exits))
        goals[f"decoder_{start}_{stop}"] = exits[stop - 1]
    return partition_graph(model, parameters, boundaries=goals, input_gradients=[hidden[0]])


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

    def export(self, directory, *, forward_model=None, head=None):
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
            if head is not None:
                record["head"] = head.export_to(weights)
                paths.extend(
                    os.path.join(directory, record["head"][name])
                    for name in ("loss", "gradient", "forward")
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

    @property
    def parameter_names(self):
        return self.names

    @property
    def trainable(self):
        return self.adapters

    @property
    def output_names(self):
        return (self.output,)

    def _execute(self, feeds, *, save):
        return execute(self, feeds, save=save)[0]

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
