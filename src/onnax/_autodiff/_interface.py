# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Input/output selection and serialized composition contracts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

import onnx_ir as ir

from onnax._diagnostics import _argument_context, _value_context

from ._common import _FLOATS, _INTERFACE, _MARKER
from ._registry import _JvpRegistry, _VjpRegistry

if TYPE_CHECKING:
    from ._context import _Context


def _select(
    values: Sequence[ir.Value],
    names: Sequence[str] | None,
    what: str,
    *,
    mode: str = "Autodiff",
) -> list[ir.Value]:
    available = [value.name for value in values]
    choices = f"Available {what}: {available!r}."
    if names is not None and (
        not isinstance(names, Sequence)
        or isinstance(names, (str, bytes))
        or any(not isinstance(name, str) for name in names)
    ):
        raise TypeError(
            f"{mode}: {what} must be a sequence of names, got {_argument_context(names)}. "
            f"Pass a list such as ['name'], not a bare string or non-string entries. {choices}"
        )
    if names is None:
        selected = [value for value in values if value.dtype in _FLOATS]
    else:
        if len(set(names)) != len(names):
            raise ValueError(
                f"{mode}: Duplicate names in {what}: {_argument_context(names)}. "
                f"Select each name once in the desired derivative order. {choices}"
            )
        by_name = {value.name: value for value in values}
        selected = []
        for name in names:
            if name not in by_name:
                raise ValueError(
                    f"{mode}: Unknown {what} name {_argument_context(name)} "
                    f"in selection {_argument_context(names)}. {choices} "
                    "Use exact graph names, including collision-resolved derivative suffixes."
                )
            selected.append(by_name[name])
    if not selected:
        raise ValueError(
            f"{mode}: Select at least one FLOAT or DOUBLE tensor in {what}; "
            f"selection={_argument_context(names)}. "
            f"{choices} Eligible floating tensors: "
            f"{[v.name for v in values if isinstance(v.type, ir.TensorType) and v.dtype in _FLOATS]!r}. "
            "Pass a nonempty selection of eligible names; non-floating differentiation "
            "requires a separately authored supported real-valued model."
        )
    for value in selected:
        if not isinstance(value.type, ir.TensorType) or value.dtype not in _FLOATS:
            raise TypeError(
                f"{mode}: Autodiff {what} {value.name!r} must be a FLOAT or DOUBLE tensor, "
                f"got {_value_context(value)} in selection {_argument_context(names)}. {choices} "
                "Select a supported floating tensor. Integer/boolean, half-precision, "
                "and complex derivatives are unsupported; casting the selection alone "
                "does not preserve the original computation's derivative."
            )
        if value.shape is None:
            raise ValueError(
                f"{mode}: Selected {what} {value.name!r} needs known-rank shape metadata; "
                f"got {_value_context(value)}. Annotate this source value's shape "
                "(dimension sizes may be symbolic); inference could not determine its rank."
            )
    return selected


def _composition_defaults(
    model: ir.Model, registry: _JvpRegistry[_Context] | _VjpRegistry[_Context]
) -> tuple[Sequence[str] | None, Sequence[str] | None]:
    """Authorize a bounded composition, not arbitrary removal of the marker."""
    props = model.graph.metadata_props
    if _MARKER not in props and "onnx_ir.autodiff" not in props:
        if _INTERFACE in props:
            raise NotImplementedError(
                f"Higher-order interface {_INTERFACE!r} has no autodiff marker {_MARKER!r} "
                f"in graph {model.graph.name!r}. Regenerate the derivative from the "
                "original model using grad/vjp and preserve both metadata entries "
                "through serialization; do not fabricate or remove markers."
            )
        return None, None
    message = (
        "Higher-order differentiation supports only one JvpPass over a generated "
        "GradPass or VjpPass model with versioned interface metadata. "
        f"Requested {registry._kind} on graph {model.graph.name!r}, "
        f"marker={_argument_context(props.get(_MARKER, props.get('onnx_ir.autodiff')))}."
    )
    remedy = (
        "Regenerate from the original model with grad or vjp, preserve the generated "
        "interface metadata, then apply jvp once. Do not remove markers to bypass "
        "unsupported compositions."
    )
    if "onnx_ir.autodiff" in props or not isinstance(registry, _JvpRegistry):
        raise NotImplementedError(f"{message} {remedy}")
    try:
        info = json.loads(props.get(_INTERFACE, "null"))
    except (ValueError, TypeError) as e:
        raise NotImplementedError(
            f"{message} Invalid interface metadata {_INTERFACE!r}: {e}. {remedy}"
        ) from e
    if not isinstance(info, dict):
        raise NotImplementedError(
            f"{message} Invalid interface metadata: expected a JSON object, "
            f"got {type(info).__name__}. {remedy}"
        )
    if info.get("version") != 1:
        raise NotImplementedError(
            f"{message} Invalid interface metadata version={_argument_context(info.get('version'))}; "
            f"supported version is 1. {remedy}"
        )
    if info.get("modes") not in (["grad"], ["vjp"]):
        raise NotImplementedError(
            f"{message} Got modes={_argument_context(info.get('modes'))}; expected ['grad'] or ['vjp']. "
            f"Forward-over-forward, reverse-over-forward/reverse and third derivatives "
            f"are unsupported. {remedy}"
        )
    if props[_MARKER] != info["modes"][0]:
        raise NotImplementedError(
            f"{message} Interface modes={info['modes']!r} disagree with the marker. {remedy}"
        )
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
            raise NotImplementedError(
                f"{message} Invalid interface field {key!r}={_argument_context(names)}; "
                f"expected a list of unique names from {sorted(available, key=str)!r}. {remedy}"
            )
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
