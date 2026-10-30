# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Active-path validation and forward/reverse transformation orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import onnx
import onnx_ir as ir

from onnax import _rule_registry
from onnax._autodiff._common import _FLOATS, _data_indices, _input
from onnax._autodiff._context import _Context
from onnax._autodiff._interface import _composition_defaults, _record_interface, _select
from onnax._autodiff._registry import _JvpRegistry, _VjpRegistry
from onnax._diagnostics import _argument_context, _node_context, _value_context
from onnax._inference import _infer_shapes
from onnax._rules import _load_builtin_rules, _snapshot

_OPSET_MIN = {"Gelu": 20}


def _validate_rule_value(
    value: ir.Value, reference: ir.Value, graph: ir.Graph, mode: str, node: ir.Node
) -> None:
    context = _node_context(node, graph.opset_imports, mode)
    producer = value.producer()
    if not (
        (producer is not None and producer.graph is graph)
        or value in graph.inputs
        or (value.name is not None and graph.initializers.get(value.name) is value)
    ):
        raise ValueError(
            f"{context}: rule returned {_value_context(value)} from outside the transformed graph. Build values with ctx.op/ctx.constant or use ctx.input and the supplied seeds."
        )
    if value.dtype is not None and value.dtype != reference.dtype:
        raise TypeError(
            f"{context}: rule returned dtype {value.dtype}, expected {reference.dtype} for {_value_context(reference)}. Return a derivative with the primal dtype; use ctx.scalar for typed constants."
        )


def _prepare(
    model: ir.Model,
    inputs: Sequence[str] | None,
    outputs: Sequence[str] | None,
    registry: _JvpRegistry[_Context] | _VjpRegistry[_Context],
    *,
    mode: str,
) -> tuple[ir.Model, list[ir.Value], list[ir.Value], list[ir.Node], set[ir.Value]]:
    default_inputs, default_outputs = _composition_defaults(model, registry)
    if inputs is None and default_inputs is not None:
        inputs = default_inputs
    if outputs is None and default_outputs is not None:
        outputs = default_outputs
    version = model.graph.opset_imports.get("", 0)
    try:
        supported = bool(18 <= version <= 28)
    except (TypeError, ValueError) as error:
        message = (
            f"{mode}: invalid standard-domain opset {_argument_context(version)} "
            f"in graph {model.graph.name!r}; expected a positive integer, supported "
            "18 through 28. Supply the actual integer opset version and validate "
            "the model; do not relabel an unsupported model's imports."
        )
        if isinstance(error, TypeError):
            raise TypeError(message) from error
        raise ValueError(message) from error
    if not supported:
        raise ValueError(
            f"{mode}: Autodiff requires the standard ONNX opset >= 18 and <= 28; "
            f"graph {model.graph.name!r} imports {_argument_context(model.graph.opset_imports.get(''))}. "
            "Use a model authored for supported opsets 18 through 28. For an older "
            "model, convert with ONNX's version converter only where semantics are "
            "preserved and validate the result. Do not relabel imports or assume "
            "a newer model can safely be downgraded."
        )
    _rule_registry._check_version(
        version, context=f"{mode} graph {model.graph.name!r}, supported opsets 18 through 28"
    )
    result = model.clone()
    graph = result.graph
    try:
        graph.sort()
    except ValueError as error:
        error.add_note(
            f"{mode}: could not topologically sort graph {graph.name!r}. "
            "Correct cyclic or unresolved producer references in the source graph."
        )
        raise
    for graph_like in (graph, *(function.graph for function in result.functions.values())):
        for node in graph_like:
            if any(
                a.type in (ir.AttributeType.GRAPH, ir.AttributeType.GRAPHS)
                for a in node.attributes.values()
            ):
                raise NotImplementedError(
                    f"{_node_context(node, graph_like.opset_imports, mode)}: "
                    "Autodiff does not support control flow or subgraphs. "
                    "Use a semantics-equivalent supported tensor graph when possible, "
                    "or differentiate the required branch/body separately outside this "
                    "transform; no derivative through branch/loop control is provided."
                )
            if node.domain == "" and (
                node.op_type.startswith("Random")
                or node.op_type in ("Dropout", "Multinomial", "Bernoulli")
            ):
                raise NotImplementedError(
                    f"{_node_context(node, graph_like.opset_imports, mode)}: "
                    f"Autodiff does not support randomness: {node.op_type}. "
                    "Supply externally generated samples as explicit inputs to a "
                    "deterministic graph only if differentiating with samples held "
                    "fixed is intended; stochastic/distributional derivatives are unsupported."
                )
    result = _infer_shapes(result, context=mode)
    graph = result.graph
    candidates = list(graph.inputs)
    if inputs is None:
        candidates = [v for v in candidates if not v.is_initializer()]
    else:
        candidates.extend(v for v in graph.initializers.values() if v not in candidates)
    xs = _select(candidates, inputs, "inputs", mode=mode)
    ys = _select(list(graph.outputs), outputs, "outputs", mode=mode)
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
        context = _node_context(node, graph.opset_imports, mode)
        if (
            node.overload
            or (
                node.op_identifier() in result.functions
                and (node.domain, node.op_type) not in registry._rules
            )
            or (node.domain and (node.domain, node.op_type) not in registry._rules)
        ):
            raise NotImplementedError(
                f"{context}: No autodiff rule for active operator "
                f"{node.domain!r}::{node.op_type}. "
                + (
                    "Inline local functions before differentiation with "
                    "onnx_ir.passes.common.InlinePass and ensure the body uses supported rules."
                    if node.op_identifier() in result.functions or node.overload
                    else "Express the custom computation with semantics-equivalent supported "
                    "standard ONNX operators, or register a matching rule with register_jvp/register_vjp."
                )
            )
        version = _rule_registry._effective_opset(node, graph.opset_imports)
        try:
            registry.lookup(node, graph.opset_imports)
        except NotImplementedError as e:
            if node.domain:
                raise NotImplementedError(
                    f"{context}: {e} Register the matching domain/operator/version for this mode; do not relabel imports."
                ) from e
            minimum = _OPSET_MIN.get(node.op_type)
            if minimum is not None and not minimum <= version <= 28:
                raise NotImplementedError(
                    f"{context}: {node.op_type} autodiff supports opsets {minimum} through 28; "
                    f"got effective opset {version}. Author or re-export the model using "
                    "a supported operator schema, or use a semantics-equivalent supported "
                    "decomposition. Do not merely relabel the model's opset imports."
                ) from e
            if node.version is not None and not 18 <= node.version <= 28:
                raise NotImplementedError(
                    f"{context}: supported {mode} opsets are 18 through 28. "
                    "Use a node and graph authored for a supported opset, or implement "
                    "and validate support for this schema; do not relabel the imports."
                ) from e
            raise NotImplementedError(
                f"{context}: No autodiff rule for active operator {node.op_type!r}. "
                "Use a semantics-equivalent supported tensor graph or a separate derivative "
                "implementation for this operator; unsupported active paths cannot be "
                "treated as zero. See JvpPass's operator/support contract."
            ) from e
        if not node.domain and node.op_identifier() not in result.functions:
            schema = onnx.defs.get_schema(node.op_type, version, "")
            graph_schema = onnx.defs.get_schema(node.op_type, graph.opset_imports[""], "")
            if schema.since_version != graph_schema.since_version:
                raise NotImplementedError(
                    f"{context}: Node version for {node.op_type} (schema {schema.since_version}) "
                    f"selects a different schema than the graph opset {graph.opset_imports['']} "
                    f"(schema {graph_schema.since_version}). Rewrite/convert the node to the "
                    "graph's schema and validate it; serialization cannot preserve per-node "
                    "schema overrides."
                )
        if not node.domain and node.op_type == "Pow":
            base, exponent = _input(node), _input(node, 1)
            if exponent.dtype in _FLOATS and exponent.dtype != base.dtype:
                raise TypeError(
                    f"{context}: Differentiating Pow requires matching floating-point dtypes; "
                    f"base={_value_context(base)}, exponent={_value_context(exponent)}. "
                    "Use matching FLOAT/FLOAT or DOUBLE/DOUBLE operands in the source "
                    "computation, or leave an integer exponent inactive. Do not cast seeds "
                    "alone or make an active exponent inactive to bypass this restriction."
                )
        for value in [*(_input(node, i) for i in indices), *node.outputs]:
            if not isinstance(value.type, ir.TensorType) or value.dtype not in _FLOATS:
                raise TypeError(
                    f"{context}: Active value {_value_context(value)} must be FLOAT or DOUBLE. "
                    "Integer/boolean casts and complex differentiation are not supported; "
                    "neither are half-precision derivatives. Select a supported floating "
                    "path or use a separately authored real FLOAT/DOUBLE model. Casting "
                    "seeds cannot make this original path differentiable."
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
        * Elementwise: Identity, Add, Sub, Mul, Div, Sum, Mean, Max, Min, Pow, Neg, Abs, Exp, Log,
          Sqrt, Reciprocal, Sin, Cos, Tan, Asin, Acos, Atan, Asinh, Acosh, Atanh, Sinh, Cosh, Tanh, Sigmoid, Sign, Relu,
          LeakyRelu, PRelu, Elu, Selu, HardSigmoid, HardSwish, Softplus, Softsign, Erf.
          Elu and Selu use the
          nonpositive branch's slope at zero (alpha and alpha * gamma).
          Gelu is supported from opset 20 with approximate="none" (default)
          or approximate="tanh", including forward-over-reverse products.
          PRelu differentiates both data and its unidirectionally broadcast
          slope tensor, including selected stored parameters.
        * Linear algebra: MatMul (vectors, matrices, broadcast batches), Gemm
          (including transposes and broadcast bias).
        * Reductions: ReduceSum, ReduceMean, ReduceL1, ReduceLogSum, and ReduceSumSquare,
          including negative/runtime axes,
          keepdims, absent/empty axes, and noop_with_empty_axes.
        * Layout/indexing: Reshape, Flatten, Squeeze, Unsqueeze, Transpose,
          Expand, Concat, Split, Gather (including repeated/negative indices).
          Also floating-point Cast, Where, Softmax, and LogSoftmax.
          Forward mode additionally supports ScatterND with reduction="add";
          its data/updates are differentiable and its indices are structural.
          Pow accepts an inactive integer exponent; floating operands must have
          matching dtypes.
        * Reverse-mode broadcast reduction uses runtime shapes, including
          singleton/leading axes and zero-length dimensions. Elementwise, Where,
          Expand, and Gemm bias pullbacks use concrete shapes to skip unnecessary
          reductions or emit fixed axes; symbolic dimensions retain runtime handling.
          MatMul, Gather, and
          default Transpose need known ranks, but dimension sizes may be symbolic.
        * Shape, size, axes, indices, split sizes, and boolean conditions are
          structural, not differentiable tensor values. Comparisons are treated
          as fixed branch predicates. Where differentiates only its chosen data
          branch; it does not differentiate the condition. Max/Min split a tied
          extremum's derivative equally across all operand occurrences, including
          inactive and repeated operands; their equality predicates remain fixed
          under higher-order differentiation.
        * Every unsupported operator on an active path from selected inputs to
          selected outputs raises NotImplementedError, rather than silently
          returning zero. Inactive paths can contain unsupported/custom operators.
          Control flow/subgraphs and random operators (including Dropout) are
          rejected. Custom/local-function calls need a globally registered rule
          for this mode, or must be inlined with InlinePass first.
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
          precisely. Abs pullbacks support Sign with a zero-derivative convention. Piecewise
          rules such as Relu/Where retain fixed-predicate conventions away from
          boundaries; no second derivative at a nonsmooth point is claimed.
          Complex numbers, sequences, sparse tensors,
          integer/boolean casts on active data paths, and half-precision
          differentiation are not supported.
        * Derivatives assume finite inputs within the operators' differentiable
          real domains: in particular, positive bases for Pow, positive inputs
          for Log/Sqrt, inputs strictly between -1 and 1 for Asin/Acos/Atanh,
          inputs greater than 1 for Acosh, positive reduced sums for ReduceLogSum,
          nonzero divisors, and nonempty ReduceMean groups.
          Inactive integral Pow exponents also support negative bases and, for
          nonnegative exponents, zero bases (including the ONNX ``0**0 == 1``).
          No domain checks are inserted. At zero, Abs and Relu use derivative
          zero and LeakyRelu uses alpha. ReduceL1 follows Abs at zero.
          Max/Min's equal-share rule at ties is a convention, not a claim of
          classical differentiability. Sign uses zero derivative everywhere,
          including its discontinuity at zero; no classical derivative is claimed there. PRelu uses its slope tensor for the data
          partial and zero for the slope partial. HardSigmoid uses zero at both
          clipping boundaries; HardSwish uses zero at the lower boundary and one
          at the upper boundary. Their HVP curvature is zero at those boundaries.
          Where predicates are fixed, including
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
        self,
        inputs: Sequence[str] | None = None,
        outputs: Sequence[str] | None = None,
    ) -> None:
        self.inputs = inputs
        self.outputs = outputs

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        registry = _JvpRegistry[_Context]()
        registry.extend(_load_builtin_rules()._jvp)
        registry.extend(_snapshot()._jvp)
        result, xs, ys, nodes, _ = _prepare(
            model, self.inputs, self.outputs, registry, mode="JVP"
        )
        b = _Context(result.graph)
        tangents = {x: b.seed(x, "tangent") for x in xs}
        for node in nodes:
            d = {
                i: tangents[_input(node, i)]
                for i in _data_indices(node)
                if node.inputs[i] in tangents
            }
            try:
                values = registry.lookup(node, result.graph.opset_imports)(b, node, d)
            except (ValueError, TypeError, NotImplementedError) as error:
                error.add_note(_node_context(node, result.graph.opset_imports, "JVP"))
                raise
            if not isinstance(values, Sequence) or any(
                not isinstance(value, ir.Value) for value in values
            ):
                raise TypeError(
                    f"{_node_context(node, result.graph.opset_imports, 'JVP')}: rule must return a sequence of IR Values, got {type(values).__name__}. Return one tangent per output."
                )
            if len(values) != len(node.outputs):
                raise ValueError(
                    f"{_node_context(node, result.graph.opset_imports, 'JVP')}: "
                    f"JVP rule for {node.op_type} returned an invalid output arity: "
                    f"got {len(values)}, expected {len(node.outputs)}. "
                    "Fix the registered rule to return one tangent per primal node output."
                )
            for reference, value in zip(node.outputs, values):
                _validate_rule_value(value, reference, result.graph, "JVP", node)
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
        self,
        inputs: Sequence[str] | None = None,
        outputs: Sequence[str] | None = None,
    ) -> None:
        self.inputs = inputs
        self.outputs = outputs

    def _transform(self, model: ir.Model, scalar: bool = False) -> ir.passes.PassResult:
        registry = _VjpRegistry[_Context]()
        registry.extend(_load_builtin_rules()._vjp)
        registry.extend(_snapshot()._vjp)
        mode = "grad (VJP)" if scalar else "VJP"
        result, xs, ys, nodes, active = _prepare(
            model, self.inputs, self.outputs, registry, mode=mode
        )
        if scalar and (len(ys) != 1 or ys[0].shape is None or len(ys[0].shape) != 0):
            raise ValueError(
                "GradPass requires one selected rank-zero (scalar) output; "
                f"got {[_value_context(y) for y in ys]}. "
                f"Available scalar FLOAT/DOUBLE outputs: "
                f"{[v.name for v in result.graph.outputs if v.dtype in _FLOATS and v.shape is not None and len(v.shape) == 0]!r}. "
                "Select one with output='name'. For a vector-valued result, use jvp/vjp "
                "with explicit tangent/cotangent seeds; reduce to a scalar only if that "
                "reduction defines the intended loss."
            )
        b = _Context(result.graph)
        cotangents = {y: b.scalar(y, 1) if scalar else b.seed(y, "cotangent") for y in ys}
        for node in reversed(nodes):
            if not any(y in cotangents for y in node.outputs):
                continue
            gs = [cotangents[y] if y in cotangents else b.like(y) for y in node.outputs]
            indices = [i for i in _data_indices(node) if node.inputs[i] in active]
            try:
                contributions = registry.lookup(node, result.graph.opset_imports)(
                    b, node, gs, indices
                )
            except (ValueError, TypeError, NotImplementedError) as error:
                error.add_note(_node_context(node, result.graph.opset_imports, mode))
                raise
            if not isinstance(contributions, Mapping) or any(
                not isinstance(value, ir.Value) for value in contributions.values()
            ):
                raise TypeError(
                    f"{_node_context(node, result.graph.opset_imports, mode)}: rule must return a mapping of input indices to IR Values, got {type(contributions).__name__}. Return exactly the requested input contributions."
                )
            if any(
                isinstance(index, bool) or not isinstance(index, int)
                for index in contributions
            ):
                raise TypeError(
                    f"{_node_context(node, result.graph.opset_imports, mode)}: VJP rule input indices must be integers, excluding bool; got {_argument_context(list(contributions))}. Return exactly the requested integer input indices."
                )
            if set(contributions) != set(indices):
                raise ValueError(
                    f"{_node_context(node, result.graph.opset_imports, mode)}: "
                    f"VJP rule for {node.op_type} returned invalid input contributions: "
                    f"got indices {sorted(contributions)!r}, expected {sorted(indices)!r}. "
                    "Fix the registered rule to return one contribution for each active data "
                    "input index, excluding structural/inactive inputs."
                )
            for i in indices:
                x, contribution = _input(node, i), contributions[i]
                _validate_rule_value(contribution, x, result.graph, mode, node)
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

    def __init__(
        self,
        inputs: Sequence[str] | None = None,
        output: str | None = None,
    ) -> None:
        super().__init__(inputs=inputs, outputs=None if output is None else [output])

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        return self._transform(model, scalar=True)
