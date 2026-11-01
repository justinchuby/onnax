# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Model-independent parameter ownership and a reusable NumPy session boundary."""

from __future__ import annotations

import contextlib
import functools
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, Self

import numpy as np
import onnx
import onnx_ir as ir
import torch
from onnx.helper import find_min_ir_version_for  # ruff: ignore[banned-api] - Schema compatibility query, not graph construction.
from onnx_ir.passes import common
from torch.autograd.function import once_differentiable

import onnax
from onnax._inference import _infer_shapes
from onnax.torch._serialization import file_digest, save_model, source_identity

_FLOATS = (ir.DataType.FLOAT, ir.DataType.DOUBLE)
_Model = ir.Model | ir.Graph | ir.GraphView | onnx.ModelProto | str | os.PathLike[str]


class Session(Protocol):
    """A session owns returned NumPy buffers for as long as they are referenced."""

    def run(
        self, output_names: Sequence[str] | None, input_feed: Mapping[str, np.ndarray], /
    ) -> Sequence[np.ndarray]: ...


SessionFactory = Callable[[str], Session]


def _ort_session(path: str) -> Session:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise ImportError("Install onnax[torch] or supply session_factory") from error
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    return ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])


def _as_model(model: _Model) -> ir.Model:
    if isinstance(model, ir.Model):
        return model.clone()
    if isinstance(model, onnx.ModelProto):
        return ir.from_proto(model)
    if isinstance(model, (ir.Graph, ir.GraphView)):
        imports = [
            onnx.OperatorSetIdProto(domain=domain, version=version)
            for domain, version in model.opset_imports.items()
        ]
        version = max(10, find_min_ir_version_for(imports, ignore_unknown=True))
        return ir.Model(model.clone(deep_copy=True), ir_version=version)
    return ir.load(os.fspath(model))


@functools.lru_cache(maxsize=32)
def _dtype(code: int) -> torch.dtype:
    return torch.from_numpy(np.empty((), dtype=ir.DataType(code).numpy())).dtype


def _validate_gradient_interface(directory, seeds, gradients, inputs, outputs):
    if set(gradients) != set(inputs) or set(seeds) != set(outputs):
        raise ValueError("Derivative interface does not match the selected inputs and outputs")
    model = ir.load(os.path.join(directory, "derivative.onnx"))
    recorded = json.loads(model.graph.metadata_props.get("onnax.torch.interface", "null"))
    if recorded != {"seeds": seeds, "gradients": gradients}:
        raise ValueError("Cached derivative name mapping differs from the bundle interface")


def _derivative_mismatch() -> None:
    raise ValueError("Bundle derivative belongs to a different source graph")


def _spec(value: ir.Value) -> dict:
    if value.dtype is None or value.shape is None:
        raise ValueError(f"Annotate dtype and rank for ONNX value {value.name!r}")
    return {
        "dtype": int(value.dtype),
        "shape": [d if isinstance(d, int) else None for d in value.shape],
    }


def _validate(name: str, value: torch.Tensor, spec: dict) -> None:
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise ValueError(f"{name!r} must be a CPU torch.Tensor for the NumPy session boundary")
    expected = torch.from_numpy(np.empty((), dtype=ir.DataType(spec["dtype"]).numpy())).dtype
    if value.dtype != expected:
        raise TypeError(f"{name!r} has dtype {value.dtype}; ONNX expects {expected}")
    if value.ndim != len(spec["shape"]) or any(
        n is not None and n != actual for n, actual in zip(spec["shape"], value.shape)
    ):
        raise ValueError(
            f"{name!r} has shape {tuple(value.shape)}; ONNX expects {spec['shape']}"
        )


class _Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, *values):
        ctx.module = module
        ctx.set_materialize_grads(False)
        count = len(module.data_names)
        data, params = values[:count], values[count:]
        # Version checks cover parameters and differentiable caller inputs. Other
        # data buffers are snapshotted so a loader can reuse them after forward.
        ctx.save_for_backward(*params, *(v for v in data if v.requires_grad))
        ctx.feeds = {
            name: value.detach().contiguous().numpy().copy()
            for name, value in zip(module.data_names, data, strict=True)
        }
        feeds = {
            **ctx.feeds,
            **dict(
                zip(
                    module.parameter_names,
                    (p.detach().contiguous().numpy() for p in params),
                    strict=True,
                )
            ),
        }
        arrays = module._run(0, module.output_names, feeds)
        # Identity/view outputs must not expose saved backward inputs or aliases
        # invisible to PyTorch storage/version tracking. Keep ordinary owned
        # runtime outputs as zero-copy NumPy views.
        aliases = [
            not value.flags.writeable
            or any(np.shares_memory(value, feed) for feed in feeds.values())
            or any(i != j and np.shares_memory(value, other) for j, other in enumerate(arrays))
            for i, value in enumerate(arrays)
        ]
        arrays = [
            value.copy() if alias else value
            for value, alias in zip(arrays, aliases, strict=True)
        ]
        ctx.output_info = {
            name: (array.shape, array.dtype)
            for name, array in zip(module.output_names, arrays, strict=True)
        }
        outputs = tuple(torch.from_numpy(array) for array in arrays)
        for name, value in zip(module.output_names, outputs, strict=True):
            _validate(name, value, module._output_specs[name])
        nondifferentiable = [
            v
            for name, v in zip(module.output_names, outputs, strict=True)
            if name not in module._seeds
        ]
        if nondifferentiable:
            ctx.mark_non_differentiable(*nondifferentiable)
        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, *upstream):
        module = ctx.module
        module._ensure_open()
        saved = ctx.saved_tensors  # Trigger PyTorch's in-place version checks.
        params = saved[: len(module.parameter_names)]
        feeds = {
            **ctx.feeds,
            **dict(
                zip(
                    module.parameter_names,
                    (p.detach().contiguous().numpy() for p in params),
                    strict=True,
                )
            ),
        }
        for name, value in zip(module.output_names, upstream, strict=True):
            if name in module._seeds:
                shape, dtype = ctx.output_info[name]
                feeds[module._seeds[name]] = (
                    np.zeros(shape, dtype=dtype)
                    if value is None
                    else value.detach().contiguous().numpy()
                )
        names = (*module.data_names, *module.parameter_names)
        wanted = [
            name
            for i, name in enumerate(names, 1)
            if ctx.needs_input_grad[i] and name in module.gradient_names
        ]
        arrays = (
            module._run(1, [module.gradient_names[name] for name in wanted], feeds)
            if wanted
            else []
        )
        gradients = dict(zip(wanted, arrays, strict=True))
        for name, array in gradients.items():
            _validate(name, torch.from_numpy(array), module._input_specs[name])
        return (
            None,
            *(
                torch.from_numpy(gradients[name]) if name in gradients else None
                for name in names
            ),
        )


class OnnxModule(torch.nn.Module):
    """Own ONNX parameters and connect reusable forward/VJP sessions to autograd.

    Parameters are named CPU FLOAT/DOUBLE tensors. Selected initializer parameters
    become explicit graph inputs. Floating data inputs propagate gradients by
    default. One selected output returns a Tensor; multiple outputs return a tuple.
    A supplied session factory receives an owned ONNX file path, including external
    tensor payloads. The module owns sessions and files until ``close()``.
    """

    _onnax_identity: torch.Tensor

    def __init__(
        self,
        model: _Model,
        parameters: Mapping[str, torch.Tensor | np.ndarray] | None = None,
        *,
        outputs: Sequence[str] | None = None,
        input_gradients: Sequence[str] | None = None,
        session_factory: SessionFactory | None = None,
        optimize: Callable[[ir.Model], ir.Model] | None = None,
    ) -> None:
        super().__init__()
        self._closed = False
        self._sessions: list[Session] = []
        self._directory = tempfile.TemporaryDirectory(prefix="onnax-torch-")
        try:
            self._build(
                model, parameters or {}, outputs, input_gradients, session_factory, optimize
            )
        except BaseException:
            self.close()
            raise

    def _build(self, model, parameters, outputs, input_gradients, session_factory, optimize):
        source = _as_model(model)
        source.graph.name = source.graph.name or "onnax_torch"
        values = ir.convenience.create_value_mapping(source.graph)
        initial = {}
        for name, array in parameters.items():
            if name not in values or (
                name not in source.graph.initializers
                and values[name] not in source.graph.inputs
            ):
                raise ValueError(f"Parameter {name!r} must be an ONNX input or initializer")
            value = values[name]
            tensor = torch.as_tensor(array)
            if tensor.device.type != "cpu" or tensor.dtype not in (
                torch.float32,
                torch.float64,
            ):
                raise TypeError(f"Parameter {name!r} must be CPU float32 or float64")
            if value.dtype is None:
                value.dtype = (
                    ir.DataType.FLOAT if tensor.dtype == torch.float32 else ir.DataType.DOUBLE
                )
            if value.shape is None:
                value.shape = ir.Shape(tensor.shape)
            _validate(name, tensor, _spec(value))
            value.shape = ir.Shape(tensor.shape)
            if name in source.graph.initializers:
                del source.graph.initializers[name]
                value.const_value = None
                if value not in source.graph.inputs:
                    source.graph.inputs.append(value)
            initial[name] = tensor.detach().contiguous().clone()
        source = _infer_shapes(source)
        available = {v.name: v for v in source.graph.outputs}
        selected = list(available) if outputs is None else list(outputs)
        if (
            not selected
            or len(set(selected)) != len(selected)
            or any(n not in available for n in selected)
        ):
            raise ValueError(f"Select unique ONNX outputs from {list(available)}")
        source.graph.outputs.clear()
        source.graph.outputs.extend(available[n] for n in selected)
        common.RemoveUnusedNodesPass()(source)
        inputs = {v.name: v for v in source.graph.inputs}
        if any(not isinstance(n, str) or not n for n in (*inputs, *selected)):
            raise ValueError("ONNX graph inputs and selected outputs need nonempty names")
        data_names = [n for n in inputs if n not in initial]
        differentiable = (
            [n for n in data_names if inputs[n].dtype in _FLOATS]
            if input_gradients is None
            else list(input_gradients)
        )
        if len(set(differentiable)) != len(differentiable) or any(
            n not in data_names or inputs[n].dtype not in _FLOATS for n in differentiable
        ):
            raise ValueError("input_gradients must select unique FLOAT/DOUBLE data inputs")
        specification = {
            "parameters": list(initial),
            "outputs": selected,
            "input_gradients": differentiable,
        }
        save_model(source, self._directory.name, "model")
        identity = source_identity(self._directory.name, specification)
        source = ir.load(os.path.join(self._directory.name, "model.onnx"))
        float_outputs = [
            v.name for v in source.graph.outputs if v.dtype in _FLOATS and v.name is not None
        ]
        gradient_inputs = [*differentiable, *initial]
        if gradient_inputs and any(
            v.dtype in (ir.DataType.FLOAT16, ir.DataType.BFLOAT16)
            for v in source.graph.outputs
        ):
            raise TypeError("Differentiable outputs use FLOAT or DOUBLE boundaries")
        backward = None
        seeds: dict[str, str] = {}
        gradients: dict[str, str] = {}
        if gradient_inputs and float_outputs:
            backward = onnax.vjp(source, inputs=gradient_inputs, outputs=float_outputs)
            interface = json.loads(backward.graph.metadata_props["onnax.autodiff.interface"])
            seeds = dict(zip(float_outputs, interface["seeds"], strict=True))
            gradients = dict(
                zip(gradient_inputs, interface["derivative_outputs"], strict=True)
            )
            by_name = {v.name: v for v in backward.graph.outputs}
            backward.graph.outputs.clear()
            backward.graph.outputs.extend(by_name[n] for n in gradients.values())
        execution = source.clone()
        if optimize is not None:
            execution = optimize(execution)
            backward = optimize(backward) if backward is not None else None
        if [v.name for v in execution.graph.outputs] != selected:
            raise ValueError("Optimization must preserve selected output names and order")
        common.RemoveUnusedNodesPass()(execution)
        save_model(execution, self._directory.name, "execution", reuse=self._directory.name)
        if backward is not None:
            if [v.name for v in backward.graph.outputs] != list(gradients.values()):
                raise ValueError("Optimization must preserve gradient output names and order")
            common.RemoveUnusedNodesPass()(backward)
            backward.graph.metadata_props["onnax.torch.source_identity"] = identity
            backward.graph.metadata_props["onnax.torch.interface"] = json.dumps(
                {"seeds": seeds, "gradients": gradients}
            )
            save_model(
                backward, self._directory.name, "derivative", reuse=self._directory.name
            )
        self._configure(
            source,
            initial,
            specification,
            identity,
            seeds,
            gradients,
            session_factory or _ort_session,
        )

    def _configure(self, source, initial, specification, identity, seeds, gradients, factory):
        self.parameter_names = tuple(specification["parameters"])
        self.output_names = tuple(specification["outputs"])
        self.input_gradient_names = tuple(specification["input_gradients"])
        self.gradient_names = dict(gradients)
        self._seeds = dict(seeds)
        self._specification = specification
        self._identity = identity
        self._input_specs = {v.name: _spec(v) for v in source.graph.inputs}
        self.data_names = tuple(n for n in self._input_specs if n not in self.parameter_names)
        self._output_specs = {v.name: _spec(v) for v in source.graph.outputs}
        for name in self.parameter_names:
            _validate(name, torch.as_tensor(initial[name]), self._input_specs[name])
        self.trainable = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.as_tensor(initial[n]).detach().contiguous().clone())
                for n in self.parameter_names
            ]
        )
        self.register_buffer(
            "_onnax_identity", torch.tensor(list(bytes.fromhex(identity)), dtype=torch.uint8)
        )
        expected_inputs = [*self.input_gradient_names, *self.parameter_names]
        expected_outputs = [v.name for v in source.graph.outputs if v.dtype in _FLOATS]
        if expected_inputs and expected_outputs:
            _validate_gradient_interface(
                self._directory.name, seeds, gradients, expected_inputs, expected_outputs
            )
        elif seeds or gradients:
            raise ValueError("Unexpected derivative interface for a forward-only module")
        self._session_inputs = []
        for stem in ("execution", "derivative") if gradients else ("execution",):
            model = ir.load(os.path.join(self._directory.name, stem + ".onnx"))
            allowed = set(self._input_specs) | set(seeds.values())
            names = [v.name for v in model.graph.inputs]
            if not set(names) <= allowed:
                raise ValueError("Execution graph introduced unknown inputs")
            self._session_inputs.append(names)
            self._sessions.append(factory(os.path.join(self._directory.name, stem + ".onnx")))

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("OnnxModule is closed")

    def _run(self, index, names, feeds):
        self._ensure_open()
        values = list(
            self._sessions[index].run(
                list(names), {n: feeds[n] for n in self._session_inputs[index]}
            )
        )
        if len(values) != len(names) or any(not isinstance(v, np.ndarray) for v in values):
            raise TypeError(
                "Session.run must return one owned NumPy array per requested output"
            )
        return values

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Execute named ONNX data inputs with external PyTorch losses and optimizers."""
        self._ensure_open()
        if set(inputs) != set(self.data_names):
            raise ValueError(f"Expected data inputs {self.data_names}; got {tuple(inputs)}")
        for name, value in inputs.items():
            _validate(name, value, self._input_specs[name])
            if value.requires_grad and name not in self.input_gradient_names:
                raise ValueError(f"Enable input_gradients for differentiable input {name!r}")
        for name, value in zip(self.parameter_names, self.trainable, strict=True):
            _validate(name, value, self._input_specs[name])
        outputs = _Function.apply(self, *(inputs[n] for n in self.data_names), *self.trainable)
        return outputs[0] if len(self.output_names) == 1 else outputs

    def parameters_by_name(self) -> dict[str, torch.nn.Parameter]:
        """Return parameters under their stable ONNX names, including dotted names."""
        return dict(zip(self.parameter_names, self.trainable, strict=True))

    def arrays(self) -> dict[str, np.ndarray]:
        """Snapshot current parameters for external adapter formats."""
        return {n: p.detach().numpy().copy() for n, p in self.parameters_by_name().items()}

    def _apply(self, fn, recurse=True):
        for dtype in {p.dtype for p in self.trainable} or {torch.float32}:
            probe = fn(torch.empty(0, dtype=dtype))
            if probe.device.type != "cpu" or probe.dtype != dtype:
                raise ValueError(
                    "OnnxModule keeps the ONNX graph's dtype and CPU parameter boundary"
                )
        return super()._apply(fn, recurse)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        identity = state_dict.get(prefix + "_onnax_identity")
        if not isinstance(identity, torch.Tensor) or not torch.equal(
            identity.cpu(), self._onnax_identity.cpu()
        ):
            raise ValueError(
                "Checkpoint ONNX graph, frozen weights or parameter interface differs"
            )
        for i, name in enumerate(self.parameter_names):
            value = state_dict.get(prefix + f"trainable.{i}")
            if value is not None:
                _validate(name, value, self._input_specs[name])
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def export(self, directory: str | os.PathLike[str]) -> ir.Model:
        """Write a reloadable bundle and a forward model with current parameters bound."""
        self._ensure_open()
        directory = os.fspath(directory)
        if os.path.exists(directory) and os.listdir(directory):
            raise FileExistsError("Export into a new or empty directory")
        os.makedirs(directory, exist_ok=True)
        files = {}
        for name in os.listdir(self._directory.name):
            if name.endswith((".onnx", ".data")):
                shutil.copyfile(
                    os.path.join(self._directory.name, name), os.path.join(directory, name)
                )
                files[name] = file_digest(os.path.join(directory, name))
        arrays = self.arrays()
        np.savez(
            os.path.join(directory, "parameters.npz"),
            *(arrays[n] for n in self.parameter_names),
        )
        files["parameters.npz"] = file_digest(os.path.join(directory, "parameters.npz"))
        model = ir.load(os.path.join(self._directory.name, "model.onnx"))
        inputs = {v.name: v for v in model.graph.inputs}
        for name, array in self.arrays().items():
            value = inputs[name]
            model.graph.inputs.remove(value)
            value.const_value = ir.tensor(array, name=name)
            model.graph.register_initializer(value)
        save_model(model, directory, "forward", reuse=self._directory.name)
        for name in ("forward.onnx", "forward.data"):
            files[name] = file_digest(os.path.join(directory, name))
        onnx.checker.check_model(os.path.join(directory, "forward.onnx"), full_check=True)
        with open(os.path.join(directory, "interface.json"), "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "version": 1,
                    "identity": self._identity,
                    "specification": self._specification,
                    "seeds": self._seeds,
                    "gradients": self.gradient_names,
                    "files": files,
                },
                stream,
                indent=2,
            )
        return ir.load(os.path.join(directory, "forward.onnx"))

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike[str],
        *,
        session_factory: SessionFactory | None = None,
    ) -> Self:
        """Load exported forward and derivative modules without generating new derivatives."""
        directory = os.fspath(directory)
        with open(os.path.join(directory, "interface.json"), encoding="utf-8") as stream:
            interface = json.load(stream)
        if interface.get("version") != 1:
            raise ValueError("Unsupported onnax.torch bundle version")
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
        module._directory = tempfile.TemporaryDirectory(prefix="onnax-torch-")
        try:
            for name in interface["files"]:
                shutil.copyfile(
                    os.path.join(directory, name), os.path.join(module._directory.name, name)
                )
            source = ir.load(os.path.join(module._directory.name, "model.onnx"))
            if interface["gradients"]:
                derivative = ir.load(os.path.join(module._directory.name, "derivative.onnx"))
                if (
                    derivative.graph.metadata_props.get("onnax.torch.source_identity")
                    != interface["identity"]
                ):
                    _derivative_mismatch()
            with np.load(
                os.path.join(module._directory.name, "parameters.npz"), allow_pickle=False
            ) as values:
                initial = {
                    name: values[f"arr_{i}"].copy()
                    for i, name in enumerate(interface["specification"]["parameters"])
                }
            module._configure(
                source,
                initial,
                interface["specification"],
                interface["identity"],
                interface["seeds"],
                interface["gradients"],
                session_factory or _ort_session,
            )
        except BaseException:
            module.close()
            raise
        return module

    def close(self) -> None:
        """Release owned sessions and temporary graph files; repeated calls are safe."""
        if getattr(self, "_closed", True):
            return
        self._closed = True
        sessions, self._sessions = self._sessions, []
        errors = []
        session: Session | None = None
        try:
            seen = set()
            for session in reversed(sessions):
                if id(session) not in seen:
                    seen.add(id(session))
                    close = getattr(session, "close", None)
                    if close is not None:
                        try:
                            close()
                        except Exception as error:
                            errors.append(error)
            session = close = None
        finally:
            sessions.clear()
            self._directory.cleanup()
        if errors:
            raise ExceptionGroup("Session cleanup failed", errors)

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()
