# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Source validation, family registration, and batching orchestration."""

from __future__ import annotations

from collections.abc import Sequence

import onnx
import onnx_ir as ir
from onnx_shape_inference import OpUsageError, ShapeInferenceError

from onnax import _rule_registry
from onnax._batching._context import _axis, _Context
from onnax._batching._registry import _RANDOM, _BatchingRegistry
from onnax._diagnostics import _argument_context, _node_context, _value_context
from onnax._inference import _infer_shapes
from onnax._rules import _load_builtin_rules, _RuleSet, _snapshot


class VmapPass(ir.passes.FunctionalPass):
    """Batch independent applications of a single-example model.

    The source model describes *one* example. This functional pass inserts a
    batch axis into each mapped input and every output. It
    emits ordinary ONNX tensor operations, not a Loop or a custom Map operator.
    Input names and node-output names are preserved, and the source model is not
    modified. A directly returned input or initializer may need a new output name.

    Args:
        in_axes: One entry per graph input, in graph-input order. An integer
            selects where to insert the batch axis; ``None`` leaves an input
            shared between examples. Negative axes index the resulting input.
            At least one input must be mapped.
        axis_size: Optional fixed, positive number of independent examples.
            When omitted, the batch length is inferred at runtime from the first
            mapped input. All mapped inputs must have the same batch length;
            this is an input contract, not a runtime assertion.
        out_axes: Where to insert the batch axis in each output. An integer
            applies to all outputs; a sequence specifies one axis per output.
            Negative axes index the resulting output. Outputs independent of
            mapped inputs are repeated along this axis.

    Example:
        Map vectors while sharing a projection matrix::

            import onnx_ir as ir
            from onnax import VmapPass

            # Source inputs: x: [D], weight: [D, H]; output: y: [H].
            model = ir.load("single_example.onnx")
            batched = VmapPass(in_axes=(0, None), out_axes=-1)(model).model
            # Batched inputs: x: [B, D], weight: [D, H]; output: y: [H, B].
            # The same model accepts different positive runtime values of B.
            ir.save(batched, "batched.onnx")

        See ``examples/vmap.py`` for runnable image-classification and
        recommendation examples that save and print the before/after graphs,
        and compare batched execution with independent examples.

    .. rubric:: Semantics

    Internally, mapped values carry a leading batch dimension. Broadcasting
    pads singleton dimensions *after* that dimension, while reductions and
    transpose permutations refer only to example dimensions. Shape and Size
    describe one example and produce shared values internally. Shared outputs
    are repeated when adapting graph outputs.

    Operators with an existing native N dimension, such as Conv and pooling,
    temporarily merge B and N, run the ordinary kernel, and restore both
    dimensions using runtime shapes. Normalization rules must not mix statistics
    across B: BatchNormalization requires inference mode when mapped, while
    InstanceNormalization and LayerNormalization preserve example statistics.
    MatMul, MatMulInteger, and QLinearMatMul promote/squeeze vector operands;
    Gemm lowers to MatMul and arithmetic. Einsum introduces a distinct batch
    label. Indexing and scattering pair matching examples, rather than forming
    cross-example products.

    .. rubric:: Supported operators

    Mapped tensor paths have rules for the following standard-domain families.
    An operator must also exist in the source model's imported opset; the pass
    does not upgrade the opset or add runtime extensions.

    * Unary math and predicates: Abs, Acos, Acosh, Asin, Asinh, Atan, Atanh,
      Ceil, Cos, Cosh, Erf, Exp, Floor, Identity, IsInf, IsNaN, Log, Neg, Not,
      Reciprocal, Round, Sign, Sin, Sinh, Sqrt, Tan, and Cast/CastLike.
    * Activations: Celu, Clip, Elu, Gelu (opset >= 20), HardSigmoid, HardSwish,
      LeakyRelu, Mish, PRelu, Relu, Selu, Shrink, Sigmoid, Softplus, Softsign,
      Tanh, and ThresholdedRelu.
    * Broadcasting arithmetic/logical: Add, And, Div, Equal, Greater,
      GreaterOrEqual, Less, LessOrEqual, Max, Mean, Min, Mul, Or, Pow, Sub,
      Sum, Where, Xor; BitShift, BitwiseAnd, BitwiseNot, BitwiseOr,
      BitwiseXor, and Mod.
    * Linear algebra: MatMul, Gemm, Einsum, Det, MatMulInteger, QLinearMatMul.
    * Image/native-N kernels: Conv, ConvTranspose, ConvInteger, QLinearConv,
      AveragePool, MaxPool (without indices), LpPool, GlobalAveragePool,
      GlobalMaxPool, GlobalLpPool, GridSample, DepthToSpace, SpaceToDepth.
    * Normalization: BatchNormalization (inference), InstanceNormalization,
      LayerNormalization (including statistics outputs),
      MeanVarianceNormalization, LRN, LpNormalization, Softmax, LogSoftmax,
      Hardmax; Dropout in inference mode, including its optional mask.
    * Shapes/layout: Reshape, Expand, Flatten, Squeeze, Unsqueeze, Shape, Size,
      Transpose, Concat, Split, Tile, Slice, Pad, Resize, Trilu, EyeLike,
      ConstantOfShape, tensor-valued Constant and numeric literal Constants.
    * Indexing: Gather, GatherElements, GatherND, ScatterElements, ScatterND,
      OneHot, TopK, ArgMax, ArgMin, CumSum.
    * Reductions: ReduceL1, ReduceL2, ReduceLogSum, ReduceLogSumExp, ReduceMax,
      ReduceMean, ReduceMin, ReduceProd, ReduceSum, ReduceSumSquare.
    * Quantization: QuantizeLinear and DequantizeLinear with shared scalar or
      per-axis scale/zero-point parameters (not blocked quantization).

    Deterministic operators whose inputs are all shared need no mapped rule:
    they remain shared if their outputs have tensor types and known ranks,
    and their standard ONNX schema is available. This permits, for example,
    shared NonZero results, even though mapped NonZero can produce ragged
    outputs and is rejected. Random operators and subgraphs are still rejected.

    .. rubric:: Controls and restrictions

    * Requires a valid, topologically sorted source graph, standard-domain
      opset 18 through 28, tensor graph inputs with known ranks, and uniform example
      shapes. Dimensions may be symbolic, unknown, or zero. Empty *batches*
      are outside the input contract. Shape inference does not replace source
      model validation.
    * Initializers are shared. Mapped inputs cannot have initializer defaults.
      All mapped inputs must have equal positive batch lengths, even if a
      singleton batch would happen to broadcast. Equality is not asserted
      inside the generated graph.
    * Reduction axes may be shared runtime 1-D INT64 tensors, including
      shape-dependent axes emitted by reverse-mode unbroadcasting. With
      ``keepdims=1`` their length may vary between calls; with ``keepdims=0``
      their length must be statically known so output rank stays fixed.
      Axes must be unique and valid in example coordinates. Mapped axes are
      rejected even if their values happen to agree at runtime. Negative axes,
      empty axes, ``keepdims``, and ``noop_with_empty_axes`` retain example
      semantics; the batch dimension is never reduced.
    * Squeeze/Unsqueeze axes must be shared constant 1-D INT64 tensors (or
      omitted where allowed). CumSum's axis must be a shared constant integer
      scalar. Omitted Squeeze axes require a static example shape; B=1 never
      causes the batch axis to be squeezed.
    * Reshape/Expand targets are shared 1-D INT64 tensors of known length;
      values may be runtime inputs. Reshape preserves per-example ``0``,
      ``-1``, and ``allowzero`` semantics. Flatten never folds B into an
      example dimension. ConstantOfShape also needs a shared shape vector
      of known length.
    * Slice controls and Pad controls are shared; explicit axes must be
      constants. Slice without axes needs a known starts-vector length.
      Resize controls, Tile repeats, Split sizes, TopK K, OneHot depth/values,
      and Trilu's diagonal are shared. GatherND needs a known index depth
      or annotated output rank.
    * Conv/native-N parameters, normalization parameters, and quantization
      scales/zero points must be shared; GridSample's data and grid may each
      be mapped or shared. Clip bounds must be shared scalars. Dropout's
      training mode must be omitted or a constant false scalar.
    * Control flow, unregistered custom/local function calls, random/training Dropout,
      mapped sequences/optionals, ragged outputs, and operators without a
      mapped rule raise :class:`onnx_ir.passes.PreconditionError`. Inline
      supported local functions first; unused function definitions remain
      unchanged. General nested mapping is not guaranteed.

    Before batching, :func:`onnx_shape_inference.infer_symbolic_shapes` runs on
    an isolated clone with ``policy="refine"``: declared annotations are retained
    and missing information is refined. Inference errors propagate; there is
    no alternate inference engine. Known ranks are still required at rule
    boundaries, and truly varying/ragged shapes remain unsupported.

    In particular, ``VmapPass([0, 0])(GradPass(inputs=["weight", "bias"],
    output="loss")(model).model)`` maps a scalar single-example loss and its
    parameter gradients when the source has sample/target inputs and shared
    parameter initializers. Runtime parameters instead use ``in_axes=None``.
    Each gradient has a distinct example axis, not a sum or mean over examples.
    This covers supported first-order arithmetic, smooth activations, linear
    algebra, reductions, and layout/index paths when every generated node meets
    the batching rules; it is not a guarantee for every autodiff graph.
    The inference engine can leave runtime-axis reductions unranked; provide
    valid source rank annotations at such boundaries rather than relying on
    a second inference engine. Composite reductions with empty axes and
    ``noop_with_empty_axes=1`` still apply their non-reduction steps (for
    example, ReduceLogSum takes Log).
    See ``examples/per_example_gradients.py`` for a runnable regression example.

    Names and graph/model metadata are preserved, except directly returned
    inputs/initializers can need renamed output adapters. Failure never
    modifies the source model.
    """

    def __init__(
        self,
        in_axes: Sequence[int | None],
        *,
        axis_size: int | None = None,
        out_axes: int | Sequence[int] = 0,
    ) -> None:
        if axis_size is not None and (
            isinstance(axis_size, bool) or not isinstance(axis_size, int) or axis_size <= 0
        ):
            raise ValueError(
                f"axis_size must be a positive integer or None, got {_argument_context(axis_size)}. "
                "Use a fixed batch length >= 1, or None to infer it from a mapped input."
            )
        try:
            self.in_axes = tuple(in_axes)
        except TypeError as error:
            raise TypeError(
                f"in_axes must be a sequence of integers/None, got {_argument_context(in_axes)}. "
                "Pass one entry per graph input, such as (0, None)."
            ) from error
        try:
            self.out_axes = out_axes if isinstance(out_axes, int) else tuple(out_axes)
        except TypeError as error:
            raise TypeError(
                f"out_axes must be an integer or sequence of integers, got {_argument_context(out_axes)}. "
                "Use 0 for leading output batch axes, or one axis per graph output."
            ) from error
        self.axis_size = axis_size

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        rules = _snapshot()
        version = model.graph.opset_imports.get("", 0)
        try:
            supported = bool(18 <= version <= 28)
        except (TypeError, ValueError) as error:
            message = (
                f"vmap: invalid standard-domain opset {_argument_context(version)} "
                f"in graph {model.graph.name!r}; expected a positive integer, supported "
                "18 through 28. Supply the actual integer opset version and validate "
                "the model; do not relabel an unsupported model's imports."
            )
            if isinstance(error, TypeError):
                raise TypeError(message) from error
            raise ValueError(message) from error
        if not supported:
            raise ir.passes.PreconditionError(
                f"VmapPass requires standard-domain opset >= 18 and <= 28; "
                f"graph {model.graph.name!r} imports {_argument_context(model.graph.opset_imports.get(''))}. "
                "Use a model authored for supported opsets 18 through 28. If converting "
                "an older model, use a semantics-preserving ONNX version converter and "
                "validate the result; do not simply relabel the import or downgrade a newer model."
            )
        _rule_registry._check_version(
            version, context=f"vmap graph {model.graph.name!r}, supported opsets 18 through 28"
        )
        if len(self.in_axes) != len(model.graph.inputs):
            raise ir.passes.PreconditionError(
                f"in_axes={_argument_context(self.in_axes)} must have one entry per graph input; "
                f"expected {len(model.graph.inputs)} for "
                f"{[value.name for value in model.graph.inputs]!r}, got {len(self.in_axes)}. "
                "Supply axes in graph-input order, using None for shared inputs."
            )
        if all(axis is None for axis in self.in_axes):
            raise ir.passes.PreconditionError(
                f"At least one input must be mapped; in_axes={_argument_context(self.in_axes)} "
                f"for {[value.name for value in model.graph.inputs]!r}. "
                "Set an input's entry to an insertion axis such as 0, or use the "
                "original model directly when all inputs are shared."
            )
        if isinstance(self.out_axes, int):
            out_axes = (self.out_axes,) * len(model.graph.outputs)
        else:
            out_axes = self.out_axes
        if len(out_axes) != len(model.graph.outputs):
            raise ir.passes.PreconditionError(
                f"out_axes={_argument_context(out_axes)} must have one entry per graph output; "
                f"expected {len(model.graph.outputs)} for "
                f"{[value.name for value in model.graph.outputs]!r}, got {len(out_axes)}. "
                "Include retained primal and derivative outputs, or pass one integer for all."
            )
        available = set(model.graph.inputs) | set(model.graph.initializers.values())
        for node in model.graph:
            if (
                node.op_identifier() in model.functions
                and (node.domain, node.op_type) not in rules._batching._rules
            ):
                raise ir.passes.PreconditionError(
                    f"{_node_context(node, model.graph.opset_imports, 'vmap')}: "
                    "VmapPass does not support local function calls. Inline functions first "
                    "with onnx_ir.passes.common.InlinePass, then ensure their bodies use "
                    "supported batching operators."
                )
            if any(value is not None and value not in available for value in node.inputs):
                raise ir.passes.PreconditionError(
                    f"{_node_context(node, model.graph.opset_imports, 'vmap')}: "
                    "VmapPass requires a topologically sorted graph without outer-scope values; "
                    f"unavailable inputs: {[v.name for v in node.inputs if v is not None and v not in available]!r}. "
                    "Define these values as graph inputs/initializers or earlier node outputs; "
                    "sort a clone with graph.sort() if only node order is wrong."
                )
            available.update(node.outputs)
        if any(value not in available for value in model.graph.outputs):
            raise ir.passes.PreconditionError(
                f"Graph {model.graph.name!r} output is not defined in the graph: "
                f"{[v.name for v in model.graph.outputs if v not in available]!r}. "
                "Return a graph input, initializer, or node output defined in this graph."
            )
        try:
            result = _infer_shapes(model, context="vmap")
        except (OpUsageError, ShapeInferenceError) as e:
            raise ir.passes.PreconditionError(
                f"Shape inference failed while preparing batching rules for graph "
                f"{model.graph.name!r} (vmap, opset {version}): {e}. "
                "Correct the reported operator inputs/attributes or incompatible source "
                "shape annotations, then retry."
            ) from e
        _Batcher(result.graph, self.axis_size, rules).run(self.in_axes, out_axes)
        return ir.passes.PassResult(result, modified=True)


class _Batcher(_Context):
    def __init__(
        self, graph: ir.Graph, axis_size: int | None, rules: _RuleSet | None = None
    ) -> None:
        super().__init__(graph, axis_size)
        self.registry = _BatchingRegistry[_Context]()
        self.registry.extend(_load_builtin_rules()._batching)
        if rules is not None:
            self.registry.extend(rules._batching)
        self.external = {} if rules is None else rules._batching._rules

    def _node(self, node: ir.Node) -> None:
        entries = self.external.get((node.domain, node.op_type), ())
        use_extension = bool(entries)
        if entries and not node.domain:
            version = _rule_registry._effective_opset(node, self.graph.opset_imports)
            use_extension = any(
                entry.since_version <= version <= entry.until_version for entry in entries
            )
        if use_extension and not node.overload:
            if not node.domain and node.op_type in _RANDOM:
                raise ir.passes.PreconditionError(
                    "External rules cannot bypass random-operator restrictions. Supply independent samples as mapped inputs."
                )
            if any(
                a.is_ref() or a.type in (ir.AttributeType.GRAPH, ir.AttributeType.GRAPHS)
                for a in node.attributes.values()
            ):
                raise ir.passes.PreconditionError(
                    "Extension batching rules cannot bypass subgraph restrictions. Inline the body first."
                )
            returned = self.registry.lookup(node, self.graph.opset_imports)(self, node)
            if returned is not None:
                raise TypeError(
                    "External batching rule must return None after registering outputs. Use ctx.register or ctx.register_outputs."
                )
            if any(value not in self.values for value in node.outputs):
                raise ValueError(
                    "External batching rule did not register every output. Call ctx.register_outputs with each output's example rank and mapping status."
                )
            return
        if (
            node.domain != ""
            or node.overload
            or any(
                attr.is_ref() or attr.type in (ir.AttributeType.GRAPH, ir.AttributeType.GRAPHS)
                for attr in node.attributes.values()
            )
        ):
            raise ir.passes.PreconditionError(
                "VmapPass does not support custom/function calls or subgraphs. "
                "Inline supported local functions with onnx_ir.passes.common.InlinePass; "
                "for custom operators or control flow, use a semantics-equivalent supported "
                "tensor graph or execute examples independently."
            )
        inputs = [value for value in node.inputs if value is not None]
        mapped = any(self.mapped[value] for value in inputs)
        op = node.op_type
        if op in _RANDOM:
            raise ir.passes.PreconditionError(
                f"VmapPass has no batching rule for random operator {op}. "
                "Generate independent random samples outside the transformed graph and "
                "supply them as mapped tensor inputs, or execute the original examples independently."
            )
        version = _rule_registry._effective_opset(node, self.graph.opset_imports)
        if not 18 <= version <= 28:
            raise ir.passes.PreconditionError(
                f"No batching rule for {op!r} at opset {version}; supported opsets are 18 through 28. "
                "Use a node and graph authored for a supported opset; do not relabel "
                "versions to bypass this check."
            )
        if op == "Gelu" and version < 20:
            raise ir.passes.PreconditionError(
                f"Gelu requires standard-domain opset >= 20 (supported 20 through 28), "
                f"got {version}. Use a Gelu model authored at opset 20 through 28, "
                "or express its exact/approximate formula with supported operators."
            )
        try:
            schema = onnx.defs.get_schema(op, version, "")
            graph_schema = onnx.defs.get_schema(op, self.graph.opset_imports[""], "")
        except onnx.defs.SchemaError as e:
            raise ir.passes.PreconditionError(
                f"No standard ONNX schema for operator {op} at node opset {version} "
                f"and graph opset {self.graph.opset_imports['']}: {e}. "
                "Check the operator spelling/domain and use a schema available at both "
                "versions; install an ONNX version providing the model's schemas if needed."
            ) from e
        if schema.since_version != graph_schema.since_version:
            raise ir.passes.PreconditionError(
                f"Node version for {op} (opset {version}, schema {schema.since_version}) "
                f"selects a different schema than the graph opset "
                f"{self.graph.opset_imports['']} (schema {graph_schema.since_version}). "
                "Rewrite/convert the node to the graph's schema and validate it before "
                "retrying; serialization cannot preserve a per-node schema override."
            )
        if (
            not mapped
            and op not in ("Constant", "Dropout")
            and all(
                isinstance(value.type, ir.TensorType) and value.shape is not None
                for value in node.outputs
            )
        ):
            for output in node.outputs:
                assert output.shape is not None
                self._register(output, len(output.shape), False)
            return
        try:
            rule = self.registry.lookup(node, self.graph.opset_imports)
        except NotImplementedError as e:
            raise ir.passes.PreconditionError(
                f"{e} For operators requiring per-example/ragged outputs, execute "
                "the source examples independently rather than forcing a uniform batch."
            ) from e
        rule(self, node)

    def run(self, in_axes: Sequence[int | None], out_axes: Sequence[int]) -> None:
        nodes = list(self.graph)
        self.before = nodes[0] if nodes else None
        for value in self.graph.initializers.values():
            if value.shape is None:
                raise ir.passes.PreconditionError(
                    f"Initializers must have known ranks; got {_value_context(value)}. "
                    "Attach shape metadata matching this initializer's stored tensor."
                )
            self._register(value, len(value.shape), False)
        for value, axis in zip(self.graph.inputs, in_axes):
            if not isinstance(value.type, ir.TensorType) or value.shape is None:
                raise ir.passes.PreconditionError(
                    f"Graph inputs must be tensors with known ranks; got {_value_context(value)}. "
                    "Annotate the input with TensorType and a shape (symbolic dimensions "
                    "are allowed); sequences/optionals cannot be batched."
                )
            rank = len(value.shape)
            if axis is None:
                self._register(value, rank, False)
                continue
            if value.is_initializer():
                raise ir.passes.PreconditionError(
                    f"Mapped inputs cannot have initializer defaults: {_value_context(value)}, "
                    f"in_axes={_argument_context(axis)}. Use None if this input is truly shared, or explicitly "
                    "remove its initializer default from a source clone and supply its mapped "
                    "value at runtime."
                )
            axis = _axis(
                axis, rank + 1, what=f"in_axes for input {value.name!r}, inserted rank"
            )
            self._register(value, rank, True)
            dims = list(value.shape)
            dims.insert(axis, dims.pop(0))
            value.shape = ir.Shape(dims)
            self.values[value] = self._transpose(
                value, [axis, *(i for i in range(rank + 1) if i != axis)]
            )
            if self.batch_source is None:
                self.batch_source = self.values[value]
        for node in nodes:
            self.before = node
            context = _node_context(node, self.graph.opset_imports, "vmap")
            try:
                self._node(node)
            except ir.passes.PreconditionError as error:
                raise ir.passes.PreconditionError(f"{context}: {error}") from error
            except (ValueError, TypeError, NotImplementedError) as error:
                error.add_note(context)
                raise
        self.before = None
        for i, (output, axis) in enumerate(zip(tuple(self.graph.outputs), out_axes)):
            rank = self.ranks[output]
            axis = _axis(
                axis, rank + 1, what=f"out_axes[{i}] for output {output.name!r}, inserted rank"
            )
            value = self.values[output]
            if not self.mapped[output]:
                value = self._repeat(value)
            perm = list(range(1, rank + 1))
            perm.insert(axis, 0)
            value = self._transpose(value, perm)
            if (
                value is not output
                and not output.is_graph_input()
                and not output.is_initializer()
            ):
                output.name, value.name = value.name, output.name
            self.graph.outputs[i] = value
