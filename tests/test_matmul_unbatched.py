# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Vector/matrix adjoints avoid batch reductions and retain empty dimensions."""

import numpy as np
import onnx
import onnx_ir as ir
import pytest
from onnx.reference import ReferenceEvaluator
from test_higher_order import _hvp_check

import onnax


@pytest.mark.parametrize(
    "shapes",
    [
        ((3,), (3,)),
        ((3,), (3, 4)),
        ((2, 3), (3,)),
        ((2, 3), (3, 4)),
        ((0, 3), (3, 4)),
        ((2, 0), (0, 4)),
        ((0,), (0,)),
    ],
)
def test_unbatched_vjp_has_no_runtime_broadcast_restore(shapes):
    a, b = [
        ir.val(name, dtype=ir.DataType.FLOAT, shape=shape)
        for name, shape in zip(("a", "b"), shapes, strict=True)
    ]
    tape = ir.tape.Tape()
    y = tape.op("MatMul", [a, b])
    y.name = "y"
    model = ir.Model(
        ir.Graph([a, b], [y], nodes=tape.nodes, opset_imports={"": 18}, name="unbatched"),
        ir_version=10,
    )
    gradient = onnax.vjp(model, inputs=["a", "b"], outputs=["y"])
    onnx.checker.check_model(ir.to_proto(gradient), full_check=True)
    assert not any(
        node.op_type in {"Shape", "NonZero", "ReduceSum", "Reshape"} for node in gradient.graph
    )
    rng = np.random.default_rng(19)
    av, bv = [rng.normal(size=shape).astype(np.float32) for shape in shapes]
    product = av @ bv
    cotangent = rng.normal(size=product.shape).astype(np.float32)
    values = ReferenceEvaluator(ir.to_proto(gradient)).run(
        None, {"a": av, "b": bv, gradient.graph.inputs[-1].name: cotangent}
    )
    ap = av.reshape(1, -1) if av.ndim == 1 else av
    bp = bv.reshape(-1, 1) if bv.ndim == 1 else bv
    gp = cotangent.reshape(ap.shape[0], bp.shape[1])
    np.testing.assert_allclose(values[-2], (gp @ bp.T).reshape(av.shape), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(values[-1], (ap.T @ gp).reshape(bv.shape), rtol=1e-5, atol=1e-6)


def test_symbolic_row_count_uses_direct_adjoints_after_reload(tmp_path):
    a = ir.val("a", dtype=ir.DataType.FLOAT, shape=["batch", 3])
    b = ir.val("b", dtype=ir.DataType.FLOAT, shape=[3, 4])
    tape = ir.tape.Tape()
    y = tape.op("MatMul", [a, b])
    y.name = "y"
    model = ir.Model(
        ir.Graph([a, b], [y], nodes=tape.nodes, opset_imports={"": 18}, name="dynamic_rows"),
        ir_version=10,
    )
    gradient = onnax.vjp(model, inputs=["a", "b"], outputs=["y"])
    onnx.checker.check_model(ir.to_proto(gradient), full_check=True)
    assert not any(
        node.op_type in {"Shape", "NonZero", "ReduceSum", "Reshape"} for node in gradient.graph
    )
    ir.save(gradient, tmp_path / "gradient.onnx")
    executor = ReferenceEvaluator(ir.to_proto(ir.load(tmp_path / "gradient.onnx")))
    for batch in [0, 1, 7]:
        av = np.arange(batch * 3, dtype=np.float32).reshape(batch, 3) / 10
        bv = np.arange(12, dtype=np.float32).reshape(3, 4) / 10
        g = np.ones((batch, 4), np.float32)
        values = executor.run(None, {"a": av, "b": bv, gradient.graph.inputs[-1].name: g})
        np.testing.assert_allclose(values[-2], g @ bv.T)
        np.testing.assert_allclose(values[-1], av.T @ g)


@pytest.mark.parametrize(
    "shapes",
    [
        ((2, 5, 3), (3, 4)),
        ((2, 1, 5, 3), (3, 4)),
        ((3,), (2, 3, 4)),
        ((2, 5, 3), (3,)),
        ((5, 3), (2, 1, 3, 4)),
        ((0, 5, 3), (3, 4)),
        ((2, 0, 3), (3, 4)),
        ((2, 5, 0), (0, 4)),
        ((5, 3), (0, 3, 4)),
        ((3,), (0, 3, 4)),
        ((2, 5, 1), (1, 1)),
        ((1, 3), (2, 3, 1)),
    ],
)
def test_single_batched_operand_adjoints(shapes):
    a, b = [
        ir.val(name, dtype=ir.DataType.FLOAT, shape=shape)
        for name, shape in zip(("a", "b"), shapes, strict=True)
    ]
    tape = ir.tape.Tape()
    y = tape.op("MatMul", [a, b])
    y.name = "y"
    model = ir.Model(
        ir.Graph([a, b], [y], nodes=tape.nodes, opset_imports={"": 18}, name="batched"),
        ir_version=10,
    )
    gradient = onnax.vjp(model, inputs=["a", "b"], outputs=["y"])
    onnx.checker.check_model(ir.to_proto(gradient), full_check=True)
    assert not any(n.op_type in {"Shape", "NonZero", "Reshape"} for n in gradient.graph)
    rng = np.random.default_rng(29)
    av, bv = [rng.normal(size=s).astype(np.float32) for s in shapes]
    seed = rng.normal(size=(av @ bv).shape).astype(np.float32)
    values = ReferenceEvaluator(ir.to_proto(gradient)).run(
        None, {"a": av, "b": bv, gradient.graph.inputs[-1].name: seed}
    )
    assert values[-2].shape == av.shape and values[-1].shape == bv.shape
    # The adjoint identity uses NumPy's forward directional derivative rather
    # than reproducing the graph's batch-reduction implementation.
    for _ in range(3):
        da, db = [rng.normal(size=s) for s in shapes]
        expected = np.sum((da @ bv + av @ db) * seed)
        actual = np.sum(values[-2] * da) + np.sum(values[-1] * db)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_symbolic_batch_and_sequence_reuse_one_adjoint_graph(tmp_path):
    a = ir.val("a", dtype=ir.DataType.FLOAT, shape=["batch", "sequence", 3])
    b = ir.val("b", dtype=ir.DataType.FLOAT, shape=[3, 4])
    tape = ir.tape.Tape()
    y = tape.op("MatMul", [a, b])
    y.name = "y"
    model = ir.Model(
        ir.Graph([a, b], [y], nodes=tape.nodes, opset_imports={"": 18}, name="dynamic_batch"),
        ir_version=10,
    )
    gradient = onnax.vjp(model, inputs=["a", "b"], outputs=["y"])
    assert not any(n.op_type in {"Shape", "NonZero", "Reshape"} for n in gradient.graph)
    ir.save(gradient, tmp_path / "gradient.onnx")
    onnx.checker.check_model(str(tmp_path / "gradient.onnx"), full_check=True)
    executor = ReferenceEvaluator(str(tmp_path / "gradient.onnx"))
    rng = np.random.default_rng(31)
    bv = rng.normal(size=(3, 4)).astype(np.float32)
    for batch in [0, 1, 4]:
        for sequence in [0, 1, 7]:
            av = rng.normal(size=(batch, sequence, 3)).astype(np.float32)
            seed = rng.normal(size=(batch, sequence, 4)).astype(np.float32)
            actual = executor.run(
                None, {"a": av, "b": bv, gradient.graph.inputs[-1].name: seed}
            )
            np.testing.assert_allclose(actual[-2], seed @ bv.T, rtol=1e-5, atol=1e-6)
            expected = (av.swapaxes(-2, -1) @ seed).sum(axis=0)
            np.testing.assert_allclose(actual[-1], expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("shapes", [((2, 3, 4), (4, 2)), ((3, 4), (2, 1, 4, 2))])
def test_single_batched_operand_hvp(shapes):
    a, b = [
        ir.val(name, dtype=ir.DataType.DOUBLE, shape=shape)
        for name, shape in zip(("a", "b"), shapes, strict=True)
    ]
    tape = ir.tape.Tape()
    y = tape.op("MatMul", [a, b])
    loss = tape.op("ReduceSum", [tape.op("Mul", [y, y])], {"keepdims": 0})
    loss.name = "loss"
    model = ir.Model(
        ir.Graph([a, b], [loss], nodes=tape.nodes, opset_imports={"": 18}, name="batch_hvp"),
        ir_version=10,
    )
    rng = np.random.default_rng(47)
    _hvp_check(model, {"a": rng.normal(size=shapes[0]), "b": rng.normal(size=shapes[1])})
