# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Active-path validation and forward/reverse transformation orchestration."""

from __future__ import annotations

from collections.abc import Sequence

import onnx
import onnx_ir as ir

from onnax import _rule_registry
from onnax._autodiff import _elementwise, _layout, _linalg, _reductions
from onnax._autodiff._common import _FLOATS, _data_indices, _input
from onnax._autodiff._context import _Context
from onnax._autodiff._interface import _composition_defaults, _record_interface, _select
from onnax._autodiff._registry import _JvpRegistry, _VjpRegistry
from onnax._inference import _infer_shapes


def _prepare(
    model: ir.Model,
    inputs: Sequence[str] | None,
    outputs: Sequence[str] | None,
    registry: _JvpRegistry[_Context] | _VjpRegistry[_Context],
) -> tuple[ir.Model, list[ir.Value], list[ir.Value], list[ir.Node], set[ir.Value]]:
    default_inputs, default_outputs = _composition_defaults(model, registry)
    if inputs is None and default_inputs is not None:
        inputs = default_inputs
    if outputs is None and default_outputs is not None:
        outputs = default_outputs
    if model.graph.opset_imports.get("", 0) < 18:
        raise ValueError("Autodiff requires the standard ONNX opset >= 18; convert first.")
    if model.graph.opset_imports[""] > 28:
        raise ValueError("Autodiff supports opsets 18 through 28 only.")
    result = model.clone()
    graph = result.graph
    graph.sort()
    for graph_like in (graph, *result.functions.values()):
        for node in graph_like:
            if any(
                a.type in (ir.AttributeType.GRAPH, ir.AttributeType.GRAPHS)
                for a in node.attributes.values()
            ):
                raise NotImplementedError(
                    "Autodiff does not support control flow or subgraphs."
                )
            if node.domain == "" and (
                node.op_type.startswith("Random")
                or node.op_type in ("Dropout", "Multinomial", "Bernoulli")
            ):
                raise NotImplementedError(
                    f"Autodiff does not support randomness: {node.op_type}."
                )
    result = _infer_shapes(result)
    graph = result.graph
    candidates = list(graph.inputs)
    if inputs is None:
        candidates = [v for v in candidates if not v.is_initializer()]
    else:
        candidates.extend(v for v in graph.initializers.values() if v not in candidates)
    xs = _select(candidates, inputs, "inputs")
    ys = _select(list(graph.outputs), outputs, "outputs")
    nodes = list(graph)
    needed = set(ys)
    for node in reversed(nodes):
        if any(y in needed for y in node.outputs):
            needed.update(
                _input(node, i)
                for i in _data_indices(node)
                if i < len(node.inputs) and node.inputs[i] is not None
            )
    active = set(xs)
    relevant_nodes = []
    for node in nodes:
        indices = [
            i for i in _data_indices(node) if i < len(node.inputs) and node.inputs[i] in active
        ]
        if not indices or not any(y in needed for y in node.outputs):
            continue
        if node.domain or node.overload or node.op_identifier() in result.functions:
            raise NotImplementedError(
                f"No autodiff rule for active operator {node.domain!r}::{node.op_type} "
                f"(node {node.name!r}). Inline local functions before differentiation."
            )
        try:
            registry.lookup(node, graph.opset_imports)
        except NotImplementedError as e:
            raise NotImplementedError(
                f"No autodiff rule for active operator {node.op_type!r}: {e}"
            ) from e
        version = _rule_registry._effective_opset(node, graph.opset_imports)
        schema = onnx.defs.get_schema(node.op_type, version, "")
        if (
            schema.since_version
            != onnx.defs.get_schema(node.op_type, graph.opset_imports[""], "").since_version
        ):
            raise NotImplementedError(
                f"Node version for {node.op_type} selects a different schema than the graph opset."
            )
        if node.op_type == "Pow":
            base, exponent = _input(node), _input(node, 1)
            if exponent.dtype in _FLOATS and exponent.dtype != base.dtype:
                raise TypeError("Differentiating Pow requires matching floating-point dtypes.")
        for value in [*(_input(node, i) for i in indices), *node.outputs]:
            if not isinstance(value.type, ir.TensorType) or value.dtype not in _FLOATS:
                raise TypeError(
                    f"Active value {value.name!r} at {node.op_type} must be FLOAT or DOUBLE. "
                    "Integer/boolean casts and complex differentiation are not supported."
                )
        active.update(y for y in node.outputs if y in needed)
        relevant_nodes.append(node)
    return result, xs, ys, relevant_nodes, active


class JvpPass(ir.passes.FunctionalPass):
    """Linearize a model in forward mode, preserving the source model.

    Given selected inputs ``x`` and outputs ``y = f(x)``, produce a new model
    computing ``(all_primal_outputs, J_f(x) @ tangent)``. All original initializers
    and functions are retained; original inputs and primal outputs keep their
    order. Append one tangent input per selected input and one tangent output per
    selected output, in selection order. Seeds must have exactly the corresponding
    primal tensor's runtime shape and dtype; broadcasting seeds is not supported.
    Names use ``<primal>_tangent`` with collision-avoiding numeric suffixes.

    Args:
        inputs: Ordered, unique names of graph inputs or initializers to
            differentiate. By default, select FLOAT and DOUBLE graph inputs that
            are not initializers, in graph-input order. Initializers are held
            constant unless explicitly selected; selecting a parameter does not
            change its stored primal value or promote it to a graph input.
        outputs: Ordered, unique names of graph outputs to differentiate. By
            default, select all FLOAT and DOUBLE outputs in graph-output order.
            Each selection must contain at least one tensor.

    Support contract (also applies to :class:`VjpPass` and :class:`GradPass`):
        * Standard ONNX opsets 18 through 28; real FLOAT/DOUBLE tensor derivatives
          only. Original opset imports are preserved, not silently upgraded.
          Shape inference runs on a clone using
          :func:`onnx_shape_inference.infer_symbolic_shapes` with ``policy="refine"``.
          Existing symbolic shape metadata is reused; no training runtime is required.
          Selected graph inputs/outputs need known ranks so their derivative
          interfaces are valid ONNX; all dimension sizes may be symbolic.
        * Elementwise: Identity, Add, Sub, Mul, Div, Pow, Neg, Abs, Exp, Log,
          Sqrt, Reciprocal, Sin, Cos, Tan, Sinh, Cosh, Tanh, Sigmoid, Relu,
          LeakyRelu, Softplus, Softsign, Erf.
        * Linear algebra: MatMul (vectors, matrices, broadcast batches), Gemm
          (including transposes and broadcast bias).
        * Reductions: ReduceSum and ReduceMean, including negative/runtime axes,
          keepdims, absent/empty axes, and noop_with_empty_axes.
        * Layout/indexing: Reshape, Flatten, Squeeze, Unsqueeze, Transpose,
          Expand, Concat, Split, Gather (including repeated/negative indices).
          Also floating-point Cast, Where, Softmax, and LogSoftmax.
          Forward mode additionally supports ScatterND with reduction="add";
          its data/updates are differentiable and its indices are structural.
          Pow accepts an inactive integer exponent; floating operands must have
          matching dtypes.
        * Reverse-mode broadcast reduction uses runtime shapes, including
          singleton/leading axes and zero-length dimensions. MatMul, Gather, and
          default Transpose need known ranks, but dimension sizes may be symbolic.
        * Shape, size, axes, indices, split sizes, and boolean conditions are
          structural, not differentiable tensor values. Comparisons are treated
          as fixed branch predicates. Where differentiates only its chosen data
          branch; it does not differentiate the condition.
        * Every unsupported operator on an active path from selected inputs to
          selected outputs raises NotImplementedError, rather than silently
          returning zero. Inactive paths can contain unsupported/custom operators.
          Control flow/subgraphs and random operators (including Dropout) are
          rejected. Inline active local-function calls with InlinePass first.
        * One forward-over-reverse composition is supported: apply JvpPass to a
          GradPass or VjpPass model, including after serialization. By default
          differentiate the original non-initializer FLOAT/DOUBLE inputs and
          only the generated gradient/cotangent outputs. Explicit selections
          use actual graph value names, including collision-resolved suffixes;
          parameters and existing cotangent seeds stay fixed unless selected.
          All existing inputs/outputs are preserved and new seeds are appended.
          Graph metadata ``onnax.autodiff.interface`` is JSON with ``version=1``,
          ``modes``, ``source_inputs``, ``source_outputs``, ``seeds``,
          ``derivative_outputs``, and the latest ``selection`` of inputs/outputs.
          Source lists contain the full original graph interfaces, seeds list
          all appended seed inputs, and derivative_outputs lists only the latest
          appended outputs, in selection order (reverse: selected inputs;
          forward: selected outputs). These actual, collision-resolved names
          disambiguate generated interfaces.
          Other repeated modes, third derivatives, and legacy marker-only models
          remain rejected. Support is checked by dispatching the ordinary
          generated graph, not by an operator-level higher-order flag.
        * Forward-over-reverse covers smooth arithmetic/nonlinearities, matrix
          operations, broadcasting, reductions, layouts, and Gather's additive
          ScatterND pullback. Shape/index paths remain structural even when
          computed at runtime. Unsupported active generated operators still fail
          precisely (for example Sign in the pullback of Abs). Piecewise
          rules such as Relu/Where retain fixed-predicate conventions away from
          boundaries; no second derivative at a nonsmooth point is claimed.
          Complex numbers, sequences, sparse tensors,
          integer/boolean casts on active data paths, and half-precision
          differentiation are not supported.
        * Derivatives assume finite inputs within the operators' differentiable
          real domains: in particular, positive bases for Pow, positive inputs
          for Log/Sqrt, nonzero divisors, and nonempty ReduceMean groups.
          Inactive integral Pow exponents also support negative bases and, for
          nonnegative exponents, zero bases (including the ONNX ``0**0 == 1``).
          No domain checks are inserted. At zero, Abs and Relu use derivative
          zero and LeakyRelu uses alpha. Where predicates are fixed, including
          at branch boundaries; no derivative of a discontinuous predicate is
          claimed. Floating Cast uses the usual real-valued identity derivative,
          not the derivative of rounding/overflow in finite-precision arithmetic.

    Example::

        import onnx_ir as ir
        from onnax import JvpPass

        x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[3])
        tape = ir.tape.Tape()
        y = tape.op("Mul", [x, x])
        y.name = "squared"
        model = ir.Model(ir.Graph([x], [y], nodes=tape.nodes,
                                 opset_imports={"": 18}, name="square"), ir_version=9)
        linearized = JvpPass(inputs=["x"])(model).model
        # Inputs: x, x_tangent; outputs: squared, squared_tangent.
        # At x=[1, 2, 3], tangent=[1, 1, 1], the latter is [2, 4, 6].
        print(model)       # Original graph, unchanged.
        print(linearized)  # Original Mul plus the executable derivative graph.

    See Also:
        ``python examples/autodiff.py`` prints original/derivative graphs
        and executes parameter gradients, prediction sensitivities, and
        forward-over-reverse parameter Hessian-vector products.
    """

    def __init__(
        self, inputs: Sequence[str] | None = None, outputs: Sequence[str] | None = None
    ) -> None:
        self.inputs = inputs
        self.outputs = outputs

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        registry = _JvpRegistry[_Context]()
        for family in (_elementwise, _linalg, _layout, _reductions):
            for op in sorted(family.OPS):
                registry.register("", op, family.jvp, since_version=18, until_version=28)
        result, xs, ys, nodes, _ = _prepare(model, self.inputs, self.outputs, registry)
        b = _Context(result.graph)
        tangents = {x: b.seed(x, "tangent") for x in xs}
        for node in nodes:
            d = {
                i: tangents[_input(node, i)]
                for i in _data_indices(node)
                if node.inputs[i] in tangents
            }
            values = registry.lookup(node, result.graph.opset_imports)(b, node, d)
            if len(values) != len(node.outputs):
                raise ValueError(
                    f"JVP rule for {node.op_type} returned an invalid output arity."
                )
            tangents.update(zip(node.outputs, values))
        for y in ys:
            b.output(tangents[y] if y in tangents else b.like(y), y, "tangent")
        _record_interface(model, result, "jvp", xs, ys)
        return ir.passes.PassResult(result, modified=True)


class VjpPass(ir.passes.FunctionalPass):
    """Transpose a model's linearization in reverse mode.

    Compute ``(all_primal_outputs, J_f(x).T @ cotangent)`` in a cloned model.
    Append one cotangent input per selected output, then append one cotangent
    output per selected input. Selection, dtype, shape, operator, and support
    semantics are shared with :class:`JvpPass`. Input/output names use
    ``<primal>_cotangent`` with collision-avoiding suffixes; selection order is
    authoritative. Original inputs/outputs retain their original order.

    Contributions from fan-out, repeated operands, and multiple selected outputs
    are summed. Broadcast contributions are reduced to each input's runtime
    shape. Disconnected selected inputs produce correctly shaped zeros, not
    missing outputs. For example, selecting ``outputs=["prediction", "loss"]``
    appends those two seed inputs in that order, and selecting
    ``inputs=["bias", "weight"]`` appends those two gradients in that order.

    Args:
        inputs: Ordered input/initializer names; see :class:`JvpPass`.
        outputs: Ordered output names; see :class:`JvpPass`.

    Example::

        from onnax import VjpPass

        pullback = VjpPass(inputs=["x"], outputs=["squared"])(model).model
        # For the model in JvpPass's example:
        # Inputs: x, squared_cotangent; outputs: squared, x_cotangent.
        # At x=[1, 2, 3], seed=[1, 2, 3], x_cotangent=[2, 8, 18].
    """

    def __init__(
        self, inputs: Sequence[str] | None = None, outputs: Sequence[str] | None = None
    ) -> None:
        self.inputs = inputs
        self.outputs = outputs

    def _transform(self, model: ir.Model, scalar: bool = False) -> ir.passes.PassResult:
        registry = _VjpRegistry[_Context]()
        for family in (_elementwise, _linalg, _layout, _reductions):
            for op in sorted(family.OPS - {"ScatterND"}):
                registry.register("", op, family.vjp, since_version=18, until_version=28)
        result, xs, ys, nodes, active = _prepare(model, self.inputs, self.outputs, registry)
        if scalar and (len(ys) != 1 or ys[0].shape is None or len(ys[0].shape) != 0):
            raise ValueError("GradPass requires one selected rank-zero (scalar) output.")
        b = _Context(result.graph)
        cotangents = {y: b.scalar(y, 1) if scalar else b.seed(y, "cotangent") for y in ys}
        for node in reversed(nodes):
            if not any(y in cotangents for y in node.outputs):
                continue
            gs = [cotangents[y] if y in cotangents else b.like(y) for y in node.outputs]
            indices = [i for i in _data_indices(node) if node.inputs[i] in active]
            contributions = registry.lookup(node, result.graph.opset_imports)(
                b, node, gs, indices
            )
            if set(contributions) != set(indices):
                raise ValueError(
                    f"VJP rule for {node.op_type} returned invalid input contributions."
                )
            for i in indices:
                x, contribution = _input(node, i), contributions[i]
                cotangents[x] = (
                    b.op("Add", cotangents[x], contribution)
                    if x in cotangents
                    else contribution
                )
        for x in xs:
            b.output(cotangents[x] if x in cotangents else b.like(x), x, "cotangent")
        _record_interface(model, result, "grad" if scalar else "vjp", xs, ys)
        return ir.passes.PassResult(result, modified=True)

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return self._transform(model)


class GradPass(VjpPass):
    """Differentiate a scalar loss using a VJP with an implicit unit seed.

    Retain all original inputs and primal outputs, append gradients in selected
    input order, and add no seed inputs. ``output`` must name a rank-zero
    FLOAT/DOUBLE graph output, not a length-one vector. If omitted, the model
    must have exactly one floating-point output, and it must be rank zero.
    Gradient outputs use ``<input>_cotangent`` names as in :class:`VjpPass`.
    See :class:`JvpPass` for the shared support and domain contract.

    Args:
        inputs: Ordered input/initializer names; see :class:`JvpPass`.
        output: Scalar graph output to differentiate.

    Example::

        import onnx_ir as ir
        from onnax import GradPass, JvpPass

        prediction = ir.val("prediction", dtype=ir.DataType.DOUBLE, shape=["batch"])
        target = ir.val("target", dtype=ir.DataType.DOUBLE, shape=["batch"])
        tape = ir.tape.Tape()
        error = tape.op("Sub", [prediction, target])
        squared = tape.op("Mul", [error, error])
        loss = tape.op("ReduceMean", [squared], {"keepdims": 0})
        loss.name = "mse"
        model = ir.Model(ir.Graph([prediction, target], [loss], nodes=tape.nodes,
                                 opset_imports={"": 18}, name="mse"), ir_version=9)
        differentiated = GradPass(inputs=["prediction"])(model).model
        # Outputs: mse, prediction_cotangent.
        # The gradient is 2 * (prediction - target) / batch.
        print(model)
        print(differentiated)
        hvp = JvpPass(inputs=["prediction"])(differentiated).model
        # Feed prediction_tangent=v; the last output is 2 * v / batch.
        # No Hessian is materialized; the saved graph uses ordinary ONNX ops.
        print(hvp)
    """

    def __init__(self, inputs: Sequence[str] | None = None, output: str | None = None) -> None:
        super().__init__(inputs=inputs, outputs=None if output is None else [output])

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return self._transform(model, scalar=True)
