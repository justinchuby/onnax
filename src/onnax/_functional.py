# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Container-preserving functional graph transformations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import overload

import onnx
import onnx_ir as ir

from onnax._autodiff import GradPass, JvpPass, VjpPass
from onnax._batching import VmapPass

_Input = ir.Model | ir.Graph | ir.GraphView | onnx.ModelProto
_Output = ir.Model | ir.Graph | onnx.ModelProto


def _apply(model: _Input, transform: ir.passes.PassBase) -> _Output:
    if isinstance(model, ir.Model):
        return transform(model).model
    if isinstance(model, onnx.ModelProto):
        return ir.to_proto(transform(ir.from_proto(model)).model)
    if isinstance(model, (ir.Graph, ir.GraphView)):
        graph = model.clone(deep_copy=True) if isinstance(model, ir.GraphView) else model
        wrapped = ir.Model(graph, ir_version=onnx.IR_VERSION)
        return transform(wrapped).model.graph
    raise TypeError(
        "Expected onnx_ir.Model, onnx_ir.Graph, onnx_ir.GraphView or onnx.ModelProto; "
        f"got {type(model).__name__}"
    )


# Generated protobuf attributes can match IR protocols during static checking.
# Keep ModelProto overloads first so its return type stays precise.
@overload
def vmap(
    model: onnx.ModelProto,
    in_axes: Sequence[int | None],
    *,
    axis_size: int | None = None,
    out_axes: int | Sequence[int] = 0,
) -> onnx.ModelProto: ...


@overload
def vmap(
    model: ir.Model,
    in_axes: Sequence[int | None],
    *,
    axis_size: int | None = None,
    out_axes: int | Sequence[int] = 0,
) -> ir.Model: ...


@overload
def vmap(
    model: ir.Graph | ir.GraphView,
    in_axes: Sequence[int | None],
    *,
    axis_size: int | None = None,
    out_axes: int | Sequence[int] = 0,
) -> ir.Graph: ...


def vmap(
    model: _Input,
    in_axes: Sequence[int | None],
    *,
    axis_size: int | None = None,
    out_axes: int | Sequence[int] = 0,
) -> _Output:
    """Map independent examples, returning a new graph in the input container type.

    ``model`` describes one example and is never modified.
    All original outputs are retained, each with a new example axis.

    Args:
        model: Single-example model, graph or graph view at standard ONNX
            opset 18 through 28. Graphs carry their own opset imports.
        in_axes: One axis per graph input, in graph-input order. An integer
            inserts an example axis in the resulting input; ``None`` keeps that
            input shared. Negative axes index the resulting tensor. At least
            one input must be mapped. Initializers remain shared; mapped inputs
            cannot have initializer defaults.
        axis_size: Optional positive fixed batch length. Otherwise infer it at
            runtime from the first mapped input. All mapped lengths must agree
            and be positive; the graph does not assert this contract at runtime.
        out_axes: Axis to insert in every output, or one axis per graph output.
            Even outputs independent of mapped inputs are repeated.

    Returns:
        A source-preserving result with ordinary batched ONNX nodes.
        ``onnx_ir.Model``, ``onnx_ir.Graph`` and ``onnx.ModelProto`` inputs retain
        their container type; ``onnx_ir.GraphView`` returns an independent Graph. Shared parameter gradients can be mapped
        without reducing over examples::

            gradients = grad(model, inputs=["weight"], output="loss")
            per_example = vmap(gradients, in_axes=(0, 0))
            # Sample and target inputs are mapped; weight is a stored initializer.

    See :class:`VmapPass` for the complete operator/control restrictions.
    Mapped controls, ragged outputs, random operators and general nested mapping
    remain unsupported. Errors identify the failed input/control and corrective
    action without modifying the source model; inference failures retain their
    original cause.
    """
    return _apply(model, VmapPass(in_axes, axis_size=axis_size, out_axes=out_axes))


@overload
def jvp(
    model: onnx.ModelProto,
    *,
    inputs: Sequence[str] | None = None,
    outputs: Sequence[str] | None = None,
) -> onnx.ModelProto: ...


@overload
def jvp(
    model: ir.Model,
    *,
    inputs: Sequence[str] | None = None,
    outputs: Sequence[str] | None = None,
) -> ir.Model: ...


@overload
def jvp(
    model: ir.Graph | ir.GraphView,
    *,
    inputs: Sequence[str] | None = None,
    outputs: Sequence[str] | None = None,
) -> ir.Graph: ...


def jvp(
    model: _Input,
    *,
    inputs: Sequence[str] | None = None,
    outputs: Sequence[str] | None = None,
) -> _Output:
    """Build a graph computing primal outputs and a Jacobian-vector product.

    Preserves the input container type
    (``onnx_ir.Model``, ``onnx_ir.Graph`` or ``onnx.ModelProto``);
    ``onnx_ir.GraphView`` returns an independent Graph. The result retains all
    original inputs/outputs and appends tangent seed inputs and tangent outputs.
    Seeds are runtime graph inputs, one per selected input, with exactly its
    primal runtime shape/dtype (no broadcasting). Names use ``<name>_tangent``
    with collision-resolving suffixes; inspect the returned graph interface.

    Args:
        model: Source model, graph or graph view, never modified.
        inputs: Ordered unique graph input/initializer names to differentiate.
            Defaults to non-initializer FLOAT/DOUBLE graph inputs. Explicitly
            selected initializers stay stored and acquire only a tangent input;
            they are not promoted to runtime primal inputs.
        outputs: Ordered unique graph output names to differentiate. Defaults to
            FLOAT/DOUBLE outputs. Tangent outputs follow this selection order.

    On a generated ``grad``/``vjp`` model, omitted selections instead use original
    non-initializer floating inputs and the generated derivative outputs. Existing
    cotangent seeds stay fixed unless explicitly selected. One such
    forward-over-reverse composition is supported, including after serialization::

        gradient = grad(model, inputs=["weight"], output="loss")
        hvp = jvp(gradient, inputs=["weight"])
        # Feed the appended weight_tangent input; the last output is H @ direction.

    The versioned ``onnax.autodiff.interface`` graph metadata records actual seed
    names, derivative output names and selections. See :class:`JvpPass` for that
    schema and the full opset, FLOAT/DOUBLE, shape, domain and operator contract.
    Unsupported active paths and other higher-order compositions fail explicitly.
    Selection errors list available names; operator failures identify mode,
    domain/operator, node and effective opset. Original inference errors propagate
    with graph/boundary context in their Python exception notes.
    """
    return _apply(model, JvpPass(inputs=inputs, outputs=outputs))


@overload
def vjp(
    model: onnx.ModelProto,
    *,
    inputs: Sequence[str] | None = None,
    outputs: Sequence[str] | None = None,
) -> onnx.ModelProto: ...


@overload
def vjp(
    model: ir.Model,
    *,
    inputs: Sequence[str] | None = None,
    outputs: Sequence[str] | None = None,
) -> ir.Model: ...


@overload
def vjp(
    model: ir.Graph | ir.GraphView,
    *,
    inputs: Sequence[str] | None = None,
    outputs: Sequence[str] | None = None,
) -> ir.Graph: ...


def vjp(
    model: _Input,
    *,
    inputs: Sequence[str] | None = None,
    outputs: Sequence[str] | None = None,
) -> _Output:
    """Build a graph computing primal outputs and a vector-Jacobian product.

    Preserves the input container type (``onnx_ir.Model``, ``onnx_ir.Graph``
    or ``onnx.ModelProto``); ``onnx_ir.GraphView`` returns an independent Graph.
    Retains all original inputs/outputs, appends one runtime cotangent seed input
    per selected output, and appends one input cotangent output per selected
    input. Contributions from all selected outputs and repeated uses are summed.
    Disconnected selected inputs produce correctly shaped zeros.

    Args:
        model: Source model, graph or graph view, never modified.
        inputs: Ordered unique graph input/initializer names. Defaults to
            non-initializer FLOAT/DOUBLE graph inputs. Initializers remain fixed
            unless selected and keep their stored primal values when selected.
        outputs: Ordered unique FLOAT/DOUBLE graph output names. Defaults to all
            floating graph outputs. Determines seed-input order, not retained
            primal-output order.

    Seeds must exactly match their selected output's runtime shape and dtype.
    Seed inputs and derivative outputs use ``<name>_cotangent`` with collision-
    resolving suffixes; inspect the graph or its versioned interface metadata::

        pullback = vjp(model, inputs=["weight"], outputs=["prediction"])
        # Supply prediction_cotangent at runtime; read weight_cotangent.

    See :class:`VjpPass` and :class:`JvpPass` for the shared numerical/operator
    restrictions. A subsequent ``jvp`` is the only supported higher-order mode;
    reverse-over-reverse and reverse-over-forward remain explicitly rejected.
    """
    return _apply(model, VjpPass(inputs=inputs, outputs=outputs))


@overload
def grad(
    model: onnx.ModelProto,
    *,
    inputs: Sequence[str] | None = None,
    output: str | None = None,
) -> onnx.ModelProto: ...


@overload
def grad(
    model: ir.Model,
    *,
    inputs: Sequence[str] | None = None,
    output: str | None = None,
) -> ir.Model: ...


@overload
def grad(
    model: ir.Graph | ir.GraphView,
    *,
    inputs: Sequence[str] | None = None,
    output: str | None = None,
) -> ir.Graph: ...


def grad(
    model: _Input,
    *,
    inputs: Sequence[str] | None = None,
    output: str | None = None,
) -> _Output:
    """Build a scalar-loss gradient graph with an implicit unit cotangent seed.

    Preserves the input container type (``onnx_ir.Model``, ``onnx_ir.Graph``
    or ``onnx.ModelProto``); ``onnx_ir.GraphView`` returns an independent Graph.
    Retains every original input and primal output and appends gradients in
    selected-input order. The implicit unit cotangent adds no seed inputs.

    Args:
        model: Source model, graph or graph view, never modified.
        inputs: Ordered unique input/initializer names to differentiate.
            Defaults to non-initializer FLOAT/DOUBLE graph inputs. Select stored
            parameters explicitly; selection does not change their primal values
            or promote them to graph inputs.
        output: Name of one rank-zero FLOAT/DOUBLE graph output. If omitted,
            there must be exactly one floating output and it must be rank zero;
            a length-one vector is not scalar.

    Gradients use collision-resolved ``<input>_cotangent`` names, recorded in
    ``onnax.autodiff.interface``. Parameter differentiation and composition::

        gradient = grad(model, inputs=["weight", "bias"], output="loss")
        per_example = vmap(gradient, in_axes=(0, 0))
        hvp = jvp(gradient, inputs=["weight", "bias"])

    Here sample/target are the two runtime inputs and parameters are stored
    initializers. For runtime parameters add ``None`` entries to ``in_axes``.
    ``vmap(grad(...))`` retains per-example gradients, not gradients of a
    batch-reduced loss. See :class:`GradPass` and :class:`JvpPass` for supported
    operators, real-domain/shape contracts and the bounded HVP composition.
    """
    return _apply(model, GradPass(inputs=inputs, output=output))
