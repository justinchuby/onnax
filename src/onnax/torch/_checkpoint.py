# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Model-independent named-boundary checkpoint execution."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Self

import numpy as np
import onnx
import onnx_ir as ir
import torch
from onnx_ir.passes import common

import onnax
from onnax._checkpoint import partition
from onnax.torch._checkpoint_execution import _Execution, execute
from onnax.torch._module import (
    OnnxModule,
    SessionFactory,
    _Model,
    _ort_session,
    _validate,
    _validate_bundle_specs,
)
from onnax.torch._serialization import SharedWeights, file_digest, source_identity


class CheckpointedModule(OnnxModule):
    """Recompute block-local activations from explicit named checkpoints.

    ``boundaries`` maps block names to intermediate ONNX value names, in forward
    order. Every edge crossing a block is retained, including shared captures.
    Parameters, data gradients and session factories follow ``OnnxModule``.
    Optimizers and task losses remain external. ``gradient_names`` maps block
    names to their input-to-gradient-output mappings; PyTorch sums contributions
    when a parameter is shared by several blocks.
    """

    def __init__(
        self,
        model: _Model,
        parameters: Mapping[str, torch.Tensor | np.ndarray] | None = None,
        *,
        boundaries: Mapping[str, str],
        outputs: Sequence[str] | None = None,
        input_gradients: Sequence[str] | None = None,
        session_factory: SessionFactory | None = None,
        optimize: Callable[[ir.Model], ir.Model] | None = None,
    ) -> None:
        self.boundaries = dict(boundaries)
        super().__init__(
            model,
            parameters,
            outputs=outputs,
            input_gradients=input_gradients,
            session_factory=session_factory,
            optimize=optimize,
        )

    def _build(self, model, parameters, outputs, input_gradients, session_factory, optimize):
        source, initial, specification, identity, _, _ = self._prepare(
            model, parameters, outputs, input_gradients
        )
        self._configure_parameters(source, initial, specification, identity, {}, {})
        self.blocks = partition(
            source,
            self.parameter_names,
            boundaries=self.boundaries,
            input_gradients=self.input_gradient_names,
        )
        self.executions: list[_Execution] = []
        factory = session_factory or _ort_session
        with SharedWeights(self._directory.name) as weights:
            for index, block in enumerate(self.blocks):
                forward = block.model.clone()
                if optimize is not None:
                    forward = optimize(forward)
                if tuple(v.name for v in forward.graph.outputs) != block.outputs:
                    raise ValueError(
                        "Optimization must preserve checkpoint output names and order"
                    )
                path = weights.save(forward, f"block_{index}_forward")
                executor = factory(path)
                self._sessions.append(executor)
                backward = derivative = None
                seeds: tuple[str, ...] = ()
                gradients: tuple[str, ...] = ()
                if block.active_inputs and block.active_outputs:
                    derivative = onnax.vjp(
                        block.model,
                        inputs=list(block.active_inputs),
                        outputs=list(block.active_outputs),
                    )
                    interface = json.loads(
                        derivative.graph.metadata_props["onnax.autodiff.interface"]
                    )
                    seeds = tuple(interface["seeds"])
                    gradients = tuple(interface["derivative_outputs"])
                    values = {v.name: v for v in derivative.graph.outputs}
                    derivative.graph.outputs.clear()
                    derivative.graph.outputs.extend(values[name] for name in gradients)
                    if optimize is not None:
                        derivative = optimize(derivative)
                    if tuple(v.name for v in derivative.graph.outputs) != gradients:
                        raise ValueError(
                            "Optimization must preserve checkpoint gradient names and order"
                        )
                    common.RemoveUnusedNodesPass()(derivative)
                    derivative.graph.metadata_props["onnax.torch.source_identity"] = identity
                    path = weights.save(derivative, f"block_{index}_backward")
                    backward = factory(path)
                    self._sessions.append(backward)
                self.executions.append(
                    _Execution(
                        block,
                        executor,
                        backward,
                        seeds,
                        gradients,
                        derivative,
                        tuple(v.name for v in forward.graph.inputs if v.name is not None),
                    )
                )
        self._constant_outputs = {
            v.name: torch.from_numpy(np.array(v.const_value.numpy(), copy=True))
            for v in source.graph.outputs
            if v.const_value is not None
        }
        self._configure_execution()

    def _configure_execution(self):
        self.forward_uses = Counter(
            name for execution in self.executions for name in execution.block.inputs
        )
        self.gradient_names = {
            execution.block.name: dict(
                zip(execution.block.active_inputs, execution.gradients, strict=True)
            )
            for execution in self.executions
            if execution.gradients
        }
        self.timings: dict[str, list[float]] = {
            name: [] for name in ("forward", "backward", "exchange")
        }
        self.last_residual_bytes = 0

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Run blocks in order, retaining their inputs for block-local recomputation."""
        self._ensure_open()
        if set(inputs) != set(self.data_names):
            raise ValueError(f"Expected data inputs {self.data_names}; got {tuple(inputs)}")
        for name, value in inputs.items():
            _validate(name, value, self._input_specs[name])
            if value.requires_grad and name not in self.input_gradient_names:
                raise ValueError(f"Enable input_gradients for differentiable input {name!r}")
        for name, value in self.parameters_by_name().items():
            _validate(name, value, self._input_specs[name])
        start = time.perf_counter()
        # Snapshot loader buffers once; save differentiable originals for version checks.
        feeds = {
            name: value if value.requires_grad else value.detach().clone().contiguous()
            for name, value in inputs.items()
        }
        exchanged = time.perf_counter()
        result = execute(self, feeds, save=torch.is_grad_enabled())
        for name, value in zip(self.output_names, result, strict=True):
            _validate(name, value, self._output_specs[name])
        self.timings["exchange"].append(exchanged - start)
        self.timings["forward"].append(time.perf_counter() - exchanged)
        return result[0] if len(result) == 1 else result

    def export(self, directory: str | os.PathLike[str]) -> ir.Model:
        """Export the source, bound forward model and reusable per-block VJPs."""
        result = super().export(directory)
        path = os.path.join(directory, "interface.json")
        with open(path, encoding="utf-8") as stream:
            interface = json.load(stream)
        interface.update(
            kind="checkpointed_vjp",
            boundaries=self.boundaries,
            blocks=[
                {
                    "name": execution.block.name,
                    "inputs": execution.block.inputs,
                    "outputs": execution.block.outputs,
                    "active_inputs": execution.block.active_inputs,
                    "active_outputs": execution.block.active_outputs,
                    "seeds": execution.seeds,
                    "gradients": execution.gradients,
                    "backward": execution.derivative is not None,
                }
                for execution in self.executions
            ],
        )
        for name in interface["files"]:
            if name.endswith(".onnx"):
                onnx.checker.check_model(os.path.join(directory, name), full_check=True)
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(interface, stream, indent=2)
        return result

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike[str],
        *,
        session_factory: SessionFactory | None = None,
    ) -> Self:
        """Restore saved blocks and VJPs without regenerating derivatives."""
        directory = os.fspath(directory)
        with open(os.path.join(directory, "interface.json"), encoding="utf-8") as stream:
            interface = json.load(stream)
        if interface.get("version") != 1 or interface.get("kind") != "checkpointed_vjp":
            raise ValueError("Expected a version 1 checkpointed ONNX bundle")
        for name, digest in interface["files"].items():
            if (
                os.path.basename(name) != name
                or file_digest(os.path.join(directory, name)) != digest
            ):
                raise ValueError(f"Bundle file integrity mismatch: {name!r}")
        if source_identity(directory, interface["specification"]) != interface["identity"]:
            raise ValueError("Bundle source identity mismatch")
        module = cls.__new__(cls)
        torch.nn.Module.__init__(module)
        module._closed = False
        module._sessions = []
        module.executions = []
        module._directory = tempfile.TemporaryDirectory(prefix="onnax-torch-checkpoints-")
        try:
            module._restore(directory, interface, session_factory or _ort_session)
        except BaseException:
            module.close()
            raise
        return module

    def _restore(self, directory, interface, factory):
        for name in interface["files"]:
            shutil.copyfile(
                os.path.join(directory, name), os.path.join(self._directory.name, name)
            )
        source = ir.load(os.path.join(self._directory.name, "model.onnx"))
        _validate_bundle_specs(source, interface)
        with np.load(
            os.path.join(self._directory.name, "parameters.npz"), allow_pickle=False
        ) as arrays:
            initial = {
                name: arrays[f"arr_{index}"].copy()
                for index, name in enumerate(interface["specification"]["parameters"])
            }
        self._configure_parameters(
            source, initial, interface["specification"], interface["identity"], {}, {}
        )
        self.boundaries = dict(interface["boundaries"])
        self.blocks = partition(
            source,
            self.parameter_names,
            boundaries=self.boundaries,
            input_gradients=self.input_gradient_names,
        )
        if len(self.blocks) != len(interface["blocks"]):
            raise ValueError("Checkpoint plan differs from the source graph")
        for index, (block, row) in enumerate(
            zip(self.blocks, interface["blocks"], strict=True)
        ):
            if (
                any(
                    row[key] != list(getattr(block, key))
                    for key in ("inputs", "outputs", "active_inputs", "active_outputs")
                )
                or row["name"] != block.name
            ):
                raise ValueError("Checkpoint plan differs from the source graph")
            forward_path = os.path.join(self._directory.name, f"block_{index}_forward.onnx")
            forward = factory(forward_path)
            self._sessions.append(forward)
            backward = derivative = None
            if row["backward"]:
                path = os.path.join(self._directory.name, f"block_{index}_backward.onnx")
                derivative = ir.load(path)
                recorded = json.loads(
                    derivative.graph.metadata_props["onnax.autodiff.interface"]
                )
                if (
                    derivative.graph.metadata_props.get("onnax.torch.source_identity")
                    != interface["identity"]
                    or recorded["seeds"] != row["seeds"]
                    or recorded["derivative_outputs"] != row["gradients"]
                ):
                    raise ValueError("Checkpoint derivative interface differs from the bundle")
                backward = factory(path)
                self._sessions.append(backward)
            elif (
                row["seeds"]
                or row["gradients"]
                or (block.active_inputs and block.active_outputs)
            ):
                raise ValueError("Missing checkpoint derivative")
            self.executions.append(
                _Execution(
                    block,
                    forward,
                    backward,
                    tuple(row["seeds"]),
                    tuple(row["gradients"]),
                    derivative,
                    tuple(
                        v.name
                        for v in ir.load(forward_path).graph.inputs
                        if v.name is not None
                    ),
                )
            )
        self._constant_outputs = {
            v.name: torch.from_numpy(np.array(v.const_value.numpy(), copy=True))
            for v in source.graph.outputs
            if v.const_value is not None
        }
        self._configure_execution()
        if self.gradient_names != interface["gradients"]:
            raise ValueError("Checkpoint gradient mapping differs from the bundle")

    def close(self) -> None:
        """Release all block sessions and owned graph files."""
        self.executions = []
        self.blocks = []
        super().close()
