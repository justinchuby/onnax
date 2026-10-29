# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Export the fixed-size jelly energy and its onnax-generated derivatives.

Run from the repository root: python examples/jelly/export_models.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path  # ruff: ignore[banned-api] - standalone export CLI.

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
from onnx.reference import ReferenceEvaluator
from onnx_ir.passes.common import CheckerPass
from onnxscript import optimizer

from onnax import grad, jvp
from onnax._inference import _infer_shapes

GRID = 6
MASS = 0.04
EPSILON = 0.015
# ORT Web 1.31.1 JSEP lacks the Reshape/Unsqueeze schemas introduced at 23.
OPSET = 22
OUTPUT = Path(__file__).parent / "public" / "models"


def mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.array(
        [
            [-0.65 + 1.3 * col / (GRID - 1), -0.65 + 1.3 * row / (GRID - 1)]
            for row in range(GRID)
            for col in range(GRID)
        ],
        dtype=np.float32,
    )
    edges = []
    for row in range(GRID):
        for col in range(GRID):
            i = row * GRID + col
            if col + 1 < GRID:
                edges.append((i, i + 1))
            if row + 1 < GRID:
                edges.append((i, i + GRID))
            if row + 1 < GRID and col + 1 < GRID:
                edges.extend(((i, i + GRID + 1), (i + 1, i + GRID)))
    edges = np.asarray(edges, dtype=np.int64)
    incidence = np.zeros((len(edges), len(positions)), dtype=np.float32)
    incidence[np.arange(len(edges)), edges[:, 0]] = 1
    incidence[np.arange(len(edges)), edges[:, 1]] = -1
    return positions, edges, incidence


def energy_model() -> ir.Model:
    """Author only scalar energy; all spatial derivatives come from onnax.

    x = free * q + anchor imposes exact Dirichlet constraints. diagonal/target
    encode backward-Euler inertia, viscous damping and the pointer tether.
    Setting inertia/damping to zero gives the physical energy for force overlays.
    """
    positions, _, incidence = mesh()
    n = len(positions)
    inputs = {
        name: ir.val(name, dtype=ir.DataType.FLOAT, shape=shape)
        for name, shape in (
            ("q", [n, 2]),
            ("free", [n, 2]),
            ("anchor", [n, 2]),
            ("diagonal", [n, 2]),
            ("target", [n, 2]),
            ("load", [n, 2]),
            ("stiffness", []),
        )
    }
    tape = ir.tape.Tape()

    def constant(name: str, data: np.ndarray | float | list[int]) -> ir.Value:
        return tape.initializer(ir.tensor(np.asarray(data), name=name))

    half = constant("half", np.float32(0.5))
    epsilon2 = constant("epsilon_squared", np.float32(EPSILON**2))
    matrix = constant("incidence", incidence)
    rest = np.sqrt(np.sum((incidence @ positions) ** 2, axis=1, keepdims=True) + EPSILON**2)
    rest_value = constant("rest_lengths", rest)
    axes = constant("coordinate_axis", np.array([1], dtype=np.int64))

    def op(name: str, *args: ir.Value, **attrs: int) -> ir.Value:
        return tape.op(name, args, attrs)

    def sum_all(value: ir.Value) -> ir.Value:
        return op("ReduceSum", value, keepdims=0)

    x = op("Add", op("Mul", inputs["free"], inputs["q"]), inputs["anchor"])
    delta = op("MatMul", matrix, x)
    length = op(
        "Sqrt", op("Add", op("ReduceSum", op("Mul", delta, delta), axes, keepdims=1), epsilon2)
    )
    stretch = op("Sub", length, rest_value)
    elastic = op(
        "Mul", op("Mul", half, inputs["stiffness"]), sum_all(op("Mul", stretch, stretch))
    )
    elastic.name = "elastic"
    offset = op("Sub", x, inputs["target"])
    quadratic = op(
        "Mul", half, sum_all(op("Mul", inputs["diagonal"], op("Mul", offset, offset)))
    )
    energy = op("Sub", op("Add", elastic, quadratic), sum_all(op("Mul", inputs["load"], x)))
    energy.name = "energy"
    return ir.Model(
        ir.Graph(
            list(inputs.values()),
            [elastic, energy],
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": OPSET},
            name="jelly",
        ),
        ir_version=10,
    )


def specialize(model: ir.Model, path: Path) -> ir.Model:
    """Fold static shape/broadcast plumbing without replacing any derivatives.

    onnax intentionally emits general runtime unbroadcasting (including NonZero).
    ORT's BASIC optimizer folds this constant shape algebra for this fixed mesh.
    No extended/contrib fusions or manual derivative rules are used.
    """
    model = fold_static_shapes(_infer_shapes(model))
    optimizer.remove_unused_nodes(model)
    metadata = dict(model.graph.metadata_props)
    CheckerPass(full_check=True)(model)
    with tempfile.TemporaryDirectory() as directory:
        optimized = Path(directory) / "specialized.onnx"
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.optimized_model_filepath = str(optimized)
        ort.InferenceSession(
            ir.to_proto(model).SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        result = _infer_shapes(ir.load(optimized))
    result.graph.metadata_props.update(metadata)
    # These are precisely the static-only operations that must disappear for WebGPU.
    forbidden = {"NonZero", "ConstantOfShape", "If", "Loop"}
    for node in result.graph:
        if node.domain or node.op_type in forbidden:
            raise RuntimeError(
                f"WebGPU export contains {node.domain or 'ai.onnx'}::{node.op_type} "
                f"at {node.name!r}. Check fixed input shapes and BASIC constant folding; "
                "do not enable a silent CPU fallback."
            )
    CheckerPass(full_check=True)(result)
    ir.save(result, path)
    onnx.checker.check_model(str(path), full_check=True)
    return result


def fold_static_shapes(model: ir.Model) -> ir.Model:
    """Evaluate only constant subgraphs and concrete Shape/Size queries.

    This is needed before ORT optimization: dynamic NonZero output lengths in
    general unbroadcast rules otherwise obscure subsequent static shape queries.
    Derivative arithmetic with any runtime dependency is left untouched.
    """
    graph = model.graph
    constants = {name: value.const_value.numpy() for name, value in graph.initializers.items()}
    for node in list(graph):
        arrays = None
        source = node.inputs[0] if node.inputs else None
        if node.op_type in ("Shape", "Size") and source is not None:
            shape = (
                constants[source.name].shape
                if source.name in constants
                else source.shape.numpy()
                if source.shape is not None and all(isinstance(d, int) for d in source.shape)
                else None
            )
            if shape is not None:
                if node.op_type == "Size":
                    arrays = [np.asarray(np.prod(shape), dtype=np.int64)]
                else:
                    start = node.attributes.get("start")
                    end = node.attributes.get("end")
                    arrays = [
                        np.asarray(shape, dtype=np.int64)[
                            start.as_int() if start else None : end.as_int() if end else None
                        ]
                    ]
        elif all(value is None or value.name in constants for value in node.inputs):
            feeds = {
                value.name: constants[value.name] for value in node.inputs if value is not None
            }
            arrays = ReferenceEvaluator(ir.serde.serialize_node(node), opsets={"": OPSET}).run(
                None, feeds
            )
        if arrays is not None:
            for value, array in zip(node.outputs, arrays):
                constants[value.name] = array
                replacement = ir.val(value.name, const_value=ir.tensor(array))
                value.replace_all_uses_with(replacement, replace_graph_outputs=True)
                graph.register_initializer(replacement)
            graph.remove(node, safe=True)
    # Recompute intermediates after specialization rather than preserving stale
    # annotations on data-dependent shape plumbing.
    for node in graph:
        for value in node.outputs:
            if not value.is_graph_output():
                value.shape = None
    return _infer_shapes(model)


def hoist_linear_coefficients(model: ir.Model, variable: str) -> ir.Model:
    """Factor linear elementwise islands, hoisting invariant work out of CG.

    This is graph algebra, not a spring-specific replacement HVP. Reassociation
    changes FLOAT32 rounding, so exported equivalence is checked numerically.
    """
    tape = ir.tape.Tape()
    inputs = [ir.val(v.name, dtype=v.dtype, shape=list(v.shape)) for v in model.graph.inputs]
    mapped = dict(zip(model.graph.inputs, inputs))
    for v in model.graph.initializers.values():
        mapped[v] = tape.initializer(v.const_value, name=v.name)
    one = tape.initializer(ir.tensor(np.float32(1), name="linear_one"))
    dependencies, expressions = {}, {}

    def op(name, *values):
        return tape.op(name, values)

    def dependent(value):
        if value not in dependencies:
            node = value.producer()
            dependencies[value] = value.name == variable or (
                node is not None and any(dependent(v) for v in node.inputs if v is not None)
            )
        return dependencies[value]

    def terms(value):
        if value in expressions:
            return expressions[value]
        node = value.producer()
        result = {value: one}
        if node is not None and node.op_type in ("Mul", "Div"):
            a, b = node.inputs
            if dependent(a) and not dependent(b):
                result = {
                    basis: op(node.op_type, coefficient, transform(b))
                    for basis, coefficient in terms(a).items()
                }
            elif node.op_type == "Mul" and dependent(b) and not dependent(a):
                result = {
                    basis: op("Mul", coefficient, transform(a))
                    for basis, coefficient in terms(b).items()
                }
        elif node is not None and node.op_type in ("Add", "Sub"):
            a, b = node.inputs
            if dependent(a) and dependent(b):
                result = dict(terms(a))
                for basis, coefficient in terms(b).items():
                    if basis in result:
                        result[basis] = op(node.op_type, result[basis], coefficient)
                    else:
                        result[basis] = (
                            coefficient if node.op_type == "Add" else op("Neg", coefficient)
                        )
        elif node is not None and node.op_type == "Neg":
            result = {
                basis: op("Neg", coefficient)
                for basis, coefficient in terms(node.inputs[0]).items()
            }
        expressions[value] = result
        return result

    def transform(value):
        if value in mapped:
            return mapped[value]
        node = value.producer()
        expression = terms(value) if dependent(value) else {value: one}
        if value not in expression:
            parts = [
                op("Mul", transform(basis), coefficient)
                for basis, coefficient in expression.items()
            ]
            result = parts[0]
            for part in parts[1:]:
                result = op("Add", result, part)
        else:
            result = tape.op(
                node.op_type,
                [transform(v) if v is not None else None for v in node.inputs],
                node.attributes,
                domain=node.domain,
            )
        mapped[value] = result
        return result

    outputs = [transform(v) for v in model.graph.outputs]
    for before, after in zip(model.graph.outputs, outputs):
        after.name = before.name
    graph = ir.Graph(
        inputs,
        outputs,
        nodes=tape.nodes,
        initializers=tape.initializers,
        opset_imports=dict(model.graph.opset_imports),
        name=model.graph.name,
        metadata_props=dict(model.graph.metadata_props),
    )
    return _infer_shapes(ir.Model(graph, ir_version=model.ir_version))


def runtime_model(model: ir.Model, kind: str) -> ir.Model:
    """Specialize execution outputs and eliminate provably redundant reductions."""
    model = model.clone(deep_copy=True)
    graph = model.graph
    original_outputs = [value.name for value in graph.outputs]
    for node in list(graph):
        if node.op_type != "ReduceSum" or len(node.inputs) < 2:
            continue
        value, axes = node.inputs[:2]
        if value.shape is None or axes.const_value is None:
            continue
        dimensions = list(value.shape)
        reduction_axes = axes.const_value.numpy().flatten().tolist()
        keep = node.attributes.get("keepdims")
        if (
            (keep is None or keep.as_int() == 1)
            and reduction_axes
            and all(dimensions[axis] == 1 for axis in reduction_axes)
        ):
            # Summing a singleton with keepdims is exactly identity, even for
            # non-finite inputs. This removes general unbroadcast scaffolding.
            node.outputs[0].replace_all_uses_with(value, replace_graph_outputs=True)
            graph.remove(node, safe=True)
    for node in list(graph):
        if node.op_type not in ("Reshape", "Expand"):
            continue
        value, shape = node.inputs[:2]
        if shape.const_value is None:
            continue
        target = shape.const_value.numpy().tolist()
        source = value
        producer = source.producer()
        if (
            node.op_type == "Reshape"
            and producer is not None
            and producer.op_type == "Unsqueeze"
        ):
            source = producer.inputs[0]
        if source.shape is not None and list(source.shape) == target:
            node.outputs[0].replace_all_uses_with(source, replace_graph_outputs=True)
            graph.remove(node, safe=True)
    for node in list(graph):
        if node.op_type != "Expand" or node.outputs[0].is_graph_output():
            continue
        output = node.outputs[0]
        consumers = [other for other in graph if output in other.inputs]
        if consumers and all(
            other.op_type in ("Add", "Sub", "Mul", "Div")
            and any(
                v is not output and v is not None and v.shape == output.shape
                for v in other.inputs
            )
            for other in consumers
        ):
            # The other operand already supplies the full broadcast shape.
            output.replace_all_uses_with(node.inputs[0])
            graph.remove(node, safe=True)
    if kind == "gradient":
        # Compose the SAME generated gradient with physical-only parameters.
        # CSE shares spring arithmetic; force overlays need no second dispatch.
        nodes = list(graph)
        tape = ir.tape.Tape(graph)
        mapping = {v.name: v for v in graph.inputs}
        mapping.update({v.name: v for v in graph.initializers.values()})
        for name in ("diagonal", "target"):
            value = ir.val(f"physical_{name}", dtype=ir.DataType.FLOAT, shape=[GRID * GRID, 2])
            graph.inputs.append(value)
            mapping[name] = value
        for node in nodes:
            copied = tape.op(
                node.op_type,
                [mapping[v.name] if v is not None else None for v in node.inputs],
                node.attributes,
                domain=node.domain,
            )
            mapping[node.outputs[0].name] = copied
        physical_gradient = mapping[graph.outputs[-1].name]
        physical_gradient.name = "physical_gradient"
        graph.outputs.append(physical_gradient)
        original_outputs.append("physical_gradient")
    model = optimizer.optimize(model, num_iterations=3, onnx_shape_inference=False)
    if kind == "hvp":
        model = hoist_linear_coefficients(model, "q_tangent")
        model = optimizer.optimize(model, num_iterations=3, onnx_shape_inference=False)
    graph = model.graph
    tape = ir.tape.Tape(graph)
    if kind == "gradient":
        flat = tape.initializer(ir.tensor(np.array([-1], dtype=np.int64), name="pack_shape"))
        parts = [tape.op("Reshape", [value, flat]) for value in graph.outputs]
        output = tape.op("Concat", parts, {"axis": 0})
    else:
        output = tape.op("Identity", [graph.outputs[-1]])
    output.name = "packed"
    graph.outputs.clear()
    graph.outputs.append(output)
    graph.metadata_props["onnax.jelly.runtime_outputs"] = json.dumps(original_outputs)
    interface = graph.metadata_props.pop("onnax.autodiff.interface", None)
    if interface is not None:
        graph.metadata_props["onnax.jelly.source_autodiff_interface"] = interface
    mode = graph.metadata_props.pop("onnax.autodiff", None)
    if mode is not None:
        graph.metadata_props["onnax.jelly.source_autodiff"] = mode
    optimizer.remove_unused_nodes(model)
    return _infer_shapes(model)


def cg_model(hvp: ir.Model) -> ir.Model:
    """Unroll the same eight matrix-free CG iterations, without CPU round trips."""
    inputs = [
        ir.val(value.name, dtype=value.dtype, shape=list(value.shape))
        for value in hvp.graph.inputs
        if value.name != "q_tangent"
    ]
    n = GRID * GRID
    gradient = ir.val("gradient", dtype=ir.DataType.FLOAT, shape=[n, 2])
    regularization = ir.val("regularization", dtype=ir.DataType.FLOAT, shape=[])
    inputs.extend((gradient, regularization))
    tape = ir.tape.Tape()
    shared = {
        value.name: tape.initializer(value.const_value, name=value.name)
        for value in hvp.graph.initializers.values()
    }
    shared.update({value.name: value for value in inputs})

    def op(name, *values, **attrs):
        return tape.op(name, values, attrs)

    def scalar(name, value):
        return tape.initializer(ir.tensor(np.array(value, dtype=np.float32), name=name))

    zero, one = scalar("cg_zero", 0), scalar("cg_one", 1)
    tolerance = scalar("cg_tolerance_squared", 0.02**2)
    minimum = scalar("cg_minimum_squared", 1e-12)
    curvature_floor = scalar("cg_curvature_floor", 1e-8)

    def dot(a, b):
        return op("ReduceSum", op("Mul", a, b), keepdims=0)

    r = op("Neg", gradient)
    d = r
    p = op("Mul", gradient, zero)
    rr = dot(r, r)
    relative = op("Mul", rr, tolerance)
    threshold = op("Where", op("Greater", relative, minimum), relative, minimum)
    false = op("Less", one, zero)
    stopped = false
    negative, iterations = zero, zero
    for iteration in range(8):
        mapping = {**shared, "q_tangent": d}
        for node in hvp.graph:
            copied = tape.op(
                node.op_type,
                [mapping[v.name] if v is not None else None for v in node.inputs],
                node.attributes,
                domain=node.domain,
            )
            mapping[node.outputs[0].name] = copied
        raw = mapping[hvp.graph.outputs[0].name]
        hd = op("Add", raw, op("Mul", regularization, d))
        curvature = dot(d, hd)
        # JSEP WebGPU supports Where, but not the ONNX And/Or/Max kernels.
        active = op("Where", op("Greater", rr, threshold), op("Not", stopped), false)
        bad = op(
            "Where",
            active,
            op("LessOrEqual", curvature, op("Mul", curvature_floor, dot(d, d))),
            false,
        )
        good = op("Where", active, op("Not", bad), false)
        alpha = op("Where", good, op("Div", rr, op("Where", good, curvature, one)), zero)
        if iteration == 0:
            p = op("Where", bad, r, p)
        p = op("Where", good, op("Add", p, op("Mul", alpha, d)), p)
        new_r = op("Sub", r, op("Mul", alpha, hd))
        next_rr = dot(new_r, new_r)
        beta = op("Where", good, op("Div", next_rr, op("Where", active, rr, one)), zero)
        d = op("Add", new_r, op("Mul", beta, d))
        r = new_r
        rr = op("Where", good, next_rr, rr)
        stopped = op("Where", stopped, stopped, bad)
        iterations = op("Add", iterations, op("Where", active, one, zero))
        negative = op("Where", bad, one, negative)
    flat = tape.initializer(ir.tensor(np.array([-1], dtype=np.int64), name="cg_pack_shape"))
    packed = op(
        "Concat",
        op("Reshape", p, flat),
        op("Reshape", iterations, flat),
        op("Reshape", negative, flat),
        axis=0,
    )
    packed.name = "packed"
    model = ir.Model(
        ir.Graph(
            inputs,
            [packed],
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": OPSET},
            name="jelly_matrix_free_cg",
        ),
        ir_version=10,
    )
    model = _infer_shapes(model)
    model = optimizer.optimize(model, num_iterations=3, onnx_shape_inference=False)
    model.graph.metadata_props["onnax.jelly.cg"] = (
        "8 safeguarded CG iterations; HVP from jvp(grad(E))"
    )
    return _infer_shapes(model)


def compose(tape, model, inputs, prefix):
    mapping = {v: inputs[v.name] for v in model.graph.inputs}
    for value in model.graph.initializers.values():
        mapping[value] = tape.initializer(value.const_value, name=f"{prefix}_{value.name}")
    for node in model.graph:
        if len(node.outputs) != 1:
            raise ValueError(
                f"Demo composition requires single-output nodes; found {node.op_type}."
            )
        mapping[node.outputs[0]] = tape.op(
            node.op_type,
            [mapping[v] if v is not None else None for v in node.inputs],
            node.attributes,
            domain=node.domain,
        )
    return mapping


def energy_difference_model() -> ir.Model:
    """Evaluate E(q)-E(reference_q) without subtracting large FLOAT32 totals."""
    source = energy_model()
    tape = ir.tape.Tape()
    inputs = {
        v.name: ir.val(v.name, dtype=v.dtype, shape=list(v.shape)) for v in source.graph.inputs
    }
    inputs["reference_q"] = ir.val(
        "reference_q", dtype=ir.DataType.FLOAT, shape=[GRID * GRID, 2]
    )
    constants = {
        v.name: tape.initializer(v.const_value, name=v.name)
        for v in source.graph.initializers.values()
    }

    def op(name, *values, **attrs):
        return tape.op(name, values, attrs)

    def total(value):
        return op("ReduceSum", value, keepdims=0)

    free, anchor = inputs["free"], inputs["anchor"]
    x = op("Add", op("Mul", free, inputs["q"]), anchor)
    reference = op("Add", op("Mul", free, inputs["reference_q"]), anchor)
    displacement = op("Mul", free, op("Sub", inputs["q"], inputs["reference_q"]))
    b = constants["incidence"]
    edge = op("MatMul", b, x)
    old_edge = op("MatMul", b, reference)
    edge_change = op("MatMul", b, displacement)
    length = op(
        "Sqrt",
        op(
            "Add",
            op("ReduceSum", op("Mul", edge, edge), constants["coordinate_axis"], keepdims=1),
            constants["epsilon_squared"],
        ),
    )
    old_length = op(
        "Sqrt",
        op(
            "Add",
            op(
                "ReduceSum",
                op("Mul", old_edge, old_edge),
                constants["coordinate_axis"],
                keepdims=1,
            ),
            constants["epsilon_squared"],
        ),
    )
    length_sum = op("Add", length, old_length)
    squared_change = op(
        "ReduceSum",
        op("Mul", edge_change, op("Add", edge, old_edge)),
        constants["coordinate_axis"],
        keepdims=1,
    )
    length_change = op("Div", squared_change, length_sum)
    stretch_sum = op(
        "Sub", length_sum, op("Add", constants["rest_lengths"], constants["rest_lengths"])
    )
    elastic_change = op(
        "Mul",
        op("Mul", constants["half"], inputs["stiffness"]),
        total(op("Mul", length_change, stretch_sum)),
    )
    offset_sum = op(
        "Add", op("Sub", x, inputs["target"]), op("Sub", reference, inputs["target"])
    )
    quadratic_change = op(
        "Mul",
        constants["half"],
        total(op("Mul", inputs["diagonal"], op("Mul", displacement, offset_sum))),
    )
    output = op(
        "Sub",
        op("Add", elastic_change, quadratic_change),
        total(op("Mul", inputs["load"], displacement)),
    )
    output.name = "energy_change"
    return _infer_shapes(
        ir.Model(
            ir.Graph(
                list(inputs.values()),
                [output],
                nodes=tape.nodes,
                initializers=tape.initializers,
                opset_imports={"": OPSET},
                name="jelly_energy_difference",
            ),
            ir_version=10,
        )
    )


def step_model(cg: ir.Model, gradient: ir.Model) -> ir.Model:
    """Batch a safeguarded full-step proposal and its generated gradient."""
    tape = ir.tape.Tape()
    inputs = {}
    for value in [*cg.graph.inputs, *gradient.graph.inputs]:
        inputs.setdefault(
            value.name, ir.val(value.name, dtype=value.dtype, shape=list(value.shape))
        )
    mapping = compose(tape, cg, inputs, "cg")
    cg_parts = cg.graph.outputs[0].producer().inputs
    p, iterations, negative = [mapping[v.producer().inputs[0]] for v in cg_parts]

    def op(name, *values, **attrs):
        return tape.op(name, values, attrs)

    def constant(name, value):
        return tape.initializer(ir.tensor(np.float32(value), name=name))

    def dot(a, b):
        return op("ReduceSum", op("Mul", a, b), keepdims=0)

    zero, one, radius = constant("zero", 0), constant("one", 1), constant("radius", 0.45)
    g = inputs["gradient"]
    p = op("Where", op("Less", dot(g, p), zero), p, op("Neg", g))
    length = op("Sqrt", dot(p, p))
    scale = op(
        "Where",
        op("Greater", length, radius),
        op("Div", radius, op("Where", op("Greater", length, zero), length, one)),
        one,
    )
    p = op("Mul", p, scale)
    candidate = op("Add", inputs["q"], p)
    evaluated = compose(tape, gradient, {**inputs, "q": candidate}, "trial")[
        gradient.graph.outputs[0]
    ]
    difference = energy_difference_model()
    energy_change = compose(
        tape, difference, {**inputs, "q": candidate, "reference_q": inputs["q"]}, "difference"
    )[difference.graph.outputs[0]]
    flat = tape.initializer(ir.tensor(np.array([-1], dtype=np.int64), name="step_flat"))
    parts = [
        op("Reshape", v, flat)
        for v in (candidate, evaluated, energy_change, iterations, negative)
    ]
    output = op("Concat", *parts, axis=0)
    output.name = "packed"
    model = ir.Model(
        ir.Graph(
            list(inputs.values()),
            [output],
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": OPSET},
            name="jelly_newton_proposal",
        ),
        ir_version=10,
    )
    model = _infer_shapes(model)
    optimizer.remove_unused_nodes(model)
    model.graph.metadata_props["onnax.jelly.step"] = (
        "CG proposal + generated trial gradient; CPU Armijo check"
    )
    return model


def batch_model(step: ir.Model, gradient: ir.Model) -> ir.Model:
    """Keep the common two-Newton path on GPU, retaining CPU Armijo checks."""
    tape = ir.tape.Tape()
    inputs = {
        v.name: ir.val(v.name, dtype=v.dtype, shape=list(v.shape))
        for v in step.graph.inputs
        if v.name != "gradient"
    }
    initial = compose(tape, gradient, inputs, "initial")
    gradient_value = gradient.graph.outputs[0].producer().inputs[2].producer().inputs[0]
    current_gradient = initial[gradient_value]
    initial_output = initial[gradient.graph.outputs[0]]
    q, regularization = inputs["q"], inputs["regularization"]
    stages = []
    ten = tape.initializer(ir.tensor(np.float32(10), name="regularization_growth"))
    half = tape.initializer(ir.tensor(np.float32(0.5), name="negative_threshold"))
    for i in range(2):
        mapped = compose(
            tape,
            step,
            {**inputs, "q": q, "gradient": current_gradient, "regularization": regularization},
            f"stage{i}",
        )
        parts = step.graph.outputs[0].producer().inputs
        q_value, evaluation, _, _, negative = [v.producer().inputs[0] for v in parts]
        next_gradient = evaluation.producer().inputs[2].producer().inputs[0]
        q, current_gradient = mapped[q_value], mapped[next_gradient]
        flag = tape.op("Greater", [mapped[negative], half])
        grown = tape.op("Mul", [regularization, ten])
        regularization = tape.op("Where", [flag, grown, regularization])
        stages.append(mapped[step.graph.outputs[0]])
    output = tape.op("Concat", [initial_output, *stages], {"axis": 0})
    output.name = "packed"
    model = _infer_shapes(
        ir.Model(
            ir.Graph(
                list(inputs.values()),
                [output],
                nodes=tape.nodes,
                initializers=tape.initializers,
                opset_imports={"": OPSET},
                name="jelly_two_newton_batch",
            ),
            ir_version=10,
        )
    )
    optimizer.remove_unused_nodes(model)
    model.graph.metadata_props["onnax.jelly.batch"] = (
        "Two speculative full-step proposals, both checked by CPU Armijo; rejected paths discarded"
    )
    return model


def export(output: Path = OUTPUT) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    source = energy_model()
    gradient = grad(source, inputs=["q"], output="energy")
    hessian_vector = jvp(gradient, inputs=["q"])
    positions, edges, _ = mesh()
    manifest = {
        "opset": OPSET,
        "grid": GRID,
        "mass": MASS,
        "epsilon": EPSILON,
        "positions": positions.tolist(),
        "edges": edges.tolist(),
        "models": {},
        "runtime": {},
    }
    runtimes = {}
    for name, model in (("energy", source), ("gradient", gradient), ("hvp", hessian_vector)):
        saved = specialize(model, output / f"{name}.onnx")
        manifest["models"][name] = {
            "inputs": [value.name for value in saved.graph.inputs],
            "outputs": [value.name for value in saved.graph.outputs],
            "operators": sorted({node.op_type for node in saved.graph}),
            "nodes": len(saved.graph),
            "graph": graph_summary(saved),
        }
        runtime = runtime_model(saved, name)
        runtimes[name] = runtime
        CheckerPass(full_check=True)(runtime)
        runtime_dir = output / "runtime"
        runtime_dir.mkdir(exist_ok=True)
        ir.save(runtime, runtime_dir / f"{name}.onnx")
        onnx.checker.check_model(str(runtime_dir / f"{name}.onnx"), full_check=True)
        manifest["runtime"][name] = {
            "inputs": [value.name for value in runtime.graph.inputs],
            "outputs": [value.name for value in runtime.graph.outputs],
            "operators": sorted({node.op_type for node in runtime.graph}),
            "nodes": len(runtime.graph),
            "graph": graph_summary(runtime),
        }
    cg = cg_model(runtime)
    CheckerPass(full_check=True)(cg)
    ir.save(cg, output / "runtime" / "cg.onnx")
    onnx.checker.check_model(str(output / "runtime" / "cg.onnx"), full_check=True)
    manifest["runtime"]["cg"] = {
        "inputs": [value.name for value in cg.graph.inputs],
        "outputs": [value.name for value in cg.graph.outputs],
        "operators": sorted({node.op_type for node in cg.graph}),
        "nodes": len(cg.graph),
        "graph": graph_summary(cg),
    }
    step = step_model(cg, runtimes["gradient"])
    CheckerPass(full_check=True)(step)
    ir.save(step, runtime_dir / "step.onnx")
    onnx.checker.check_model(str(runtime_dir / "step.onnx"), full_check=True)
    manifest["runtime"]["step"] = {
        "inputs": [v.name for v in step.graph.inputs],
        "outputs": [v.name for v in step.graph.outputs],
        "operators": sorted({node.op_type for node in step.graph}),
        "nodes": len(step.graph),
        "graph": graph_summary(step),
    }
    batch = batch_model(step, runtimes["gradient"])
    CheckerPass(full_check=True)(batch)
    ir.save(batch, runtime_dir / "batch.onnx")
    onnx.checker.check_model(str(runtime_dir / "batch.onnx"), full_check=True)
    manifest["runtime"]["batch"] = {
        "inputs": [v.name for v in batch.graph.inputs],
        "outputs": [v.name for v in batch.graph.outputs],
        "operators": sorted({node.op_type for node in batch.graph}),
        "nodes": len(batch.graph),
        "graph": graph_summary(batch),
    }
    (output / "mesh.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def graph_summary(model: ir.Model) -> dict:
    """Describe real interfaces and bounded graph excerpts without rerunning inference."""
    nodes = list(model.graph)
    selected = list(range(min(6, len(nodes))))
    selected.extend(i for i in range(max(6, len(nodes) - 6), len(nodes)))
    excerpt = []
    previous = -1
    for index in selected:
        if index > previous + 1:
            excerpt.append(f"... {index - previous - 1} nodes omitted ...")
        node = nodes[index]
        inputs = ", ".join(value.name if value is not None else "" for value in node.inputs)
        outputs = ", ".join(value.name for value in node.outputs)
        excerpt.append(f"{index}: {outputs} = {node.op_type}({inputs})")
        previous = index

    def interface(values):
        return [f"{value.name}: {value.dtype} {value.shape}" for value in values]

    return {
        "inputs": interface(model.graph.inputs),
        "outputs": interface(model.graph.outputs),
        "excerpt": excerpt,
    }


if __name__ == "__main__":
    print(json.dumps(export(), indent=2))
