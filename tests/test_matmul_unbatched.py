# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Vector/matrix adjoints avoid batch reductions and retain empty dimensions."""

import numpy as np
import onnx
import onnx_ir as ir
import pytest
from onnx.reference import ReferenceEvaluator

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
