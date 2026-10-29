# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Input/output selection and serialized composition contracts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

import onnx_ir as ir

from ._common import _FLOATS, _INTERFACE, _MARKER
from ._registry import _JvpRegistry, _VjpRegistry

if TYPE_CHECKING:
    from ._context import _Context


def _select(
    values: Sequence[ir.Value], names: Sequence[str] | None, what: str
) -> list[ir.Value]:
    if isinstance(names, str):
        raise TypeError(f"{what} must be a sequence of names, not a string.")
    if names is None:
        selected = [value for value in values if value.dtype in _FLOATS]
    else:
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate names in {what}: {names!r}.")
        by_name = {value.name: value for value in values}
        selected = []
        for name in names:
            if name not in by_name:
                raise ValueError(f"Unknown {what} name {name!r}.")
            selected.append(by_name[name])
    if not selected:
        raise ValueError(f"Select at least one FLOAT or DOUBLE tensor in {what}.")
    for value in selected:
        if not isinstance(value.type, ir.TensorType) or value.dtype not in _FLOATS:
            raise TypeError(
                f"Autodiff {what} {value.name!r} must be a FLOAT or DOUBLE tensor, "
                f"got {value.type}."
            )
        if value.shape is None:
            raise ValueError(
                f"Selected {what} {value.name!r} needs known-rank shape metadata; "
                "dimension sizes may be symbolic."
            )
    return selected


def _composition_defaults(
    model: ir.Model, registry: _JvpRegistry[_Context] | _VjpRegistry[_Context]
) -> tuple[Sequence[str] | None, Sequence[str] | None]:
    """Authorize a bounded composition, not arbitrary removal of the marker."""
    props = model.graph.metadata_props
    if _MARKER not in props and "onnx_ir.autodiff" not in props:
        if _INTERFACE in props:
            raise NotImplementedError("Higher-order interface has no autodiff marker.")
        return None, None
    message = (
        "Higher-order differentiation supports only one JvpPass over a generated "
        "GradPass or VjpPass model with versioned interface metadata."
    )
    if "onnx_ir.autodiff" in props or not isinstance(registry, _JvpRegistry):
        raise NotImplementedError(message)
    try:
        info = json.loads(props.get(_INTERFACE, "null"))
    except (ValueError, TypeError) as e:
        raise NotImplementedError(f"{message} Invalid interface metadata.") from e
    if (
        not isinstance(info, dict)
        or info.get("version") != 1
        or info.get("modes") not in (["grad"], ["vjp"])
        or props[_MARKER] != info["modes"][0]
    ):
        raise NotImplementedError(message)
    graph_inputs = {v.name for v in model.graph.inputs}
    graph_outputs = {v.name for v in model.graph.outputs}
    for key, available in (
        ("source_inputs", graph_inputs),
        ("source_outputs", graph_outputs),
        ("seeds", graph_inputs),
        ("derivative_outputs", graph_outputs),
    ):
        names = info.get(key)
        if (
            not isinstance(names, list)
            or any(not isinstance(n, str) or n not in available for n in names)
            or len(set(names)) != len(names)
        ):
            raise NotImplementedError(f"{message} Invalid interface field {key!r}.")
    by_name = {v.name: v for v in model.graph.inputs}
    default_inputs = [
        name
        for name in info["source_inputs"]
        if by_name[name].dtype in _FLOATS and not by_name[name].is_initializer()
    ]
    return default_inputs, info["derivative_outputs"]


def _record_interface(
    source: ir.Model,
    result: ir.Model,
    mode: str,
    xs: Sequence[ir.Value],
    ys: Sequence[ir.Value],
) -> None:
    """Record actual collision-resolved names, retained by ONNX serialization."""
    graph = result.graph
    previous = source.graph.metadata_props.get(_INTERFACE)
    prior = json.loads(previous) if previous is not None else None
    info = {
        "version": 1,
        "modes": [*(prior["modes"] if prior else []), mode],
        "source_inputs": (
            prior["source_inputs"] if prior else [v.name for v in source.graph.inputs]
        ),
        "source_outputs": (
            prior["source_outputs"] if prior else [v.name for v in source.graph.outputs]
        ),
        "seeds": [
            *(prior["seeds"] if prior else []),
            *(v.name for v in graph.inputs[len(source.graph.inputs) :]),
        ],
        "derivative_outputs": [v.name for v in graph.outputs[len(source.graph.outputs) :]],
        "selection": {
            "inputs": [v.name for v in xs],
            "outputs": [v.name for v in ys],
        },
    }
    graph.metadata_props[_INTERFACE] = json.dumps(info, sort_keys=True)
    graph.metadata_props[_MARKER] = mode
