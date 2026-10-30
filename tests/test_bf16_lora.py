# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Real BF16 ONNX nodes, declared surrogate derivatives and FP32 master updates."""

import json

import numpy as np
import onnx_ir as ir
import pytest
from onnx.reference import ReferenceEvaluator
from onnx_ir.passes.common import InlinePass

import onnax
from examples.lora.bf16 import reference, round_bf16, run
from examples.lora.bf16_rules import DOMAIN, OP, lower_matmuls
from examples.lora.model import PARAMETERS
from examples.lora.train import close, problem


def product_model():
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[1, 1])
    w = ir.val("w", dtype=ir.DataType.FLOAT, shape=[1, 1])
    tape = ir.tape.Tape()
    product = tape.op("MatMul", [x, w])
    loss = tape.op("ReduceSum", [product], {"keepdims": 0})
    loss.name, loss.dtype, loss.shape = "loss", ir.DataType.FLOAT, ir.Shape([])
    return ir.Model(
        ir.Graph(
            [x, w], [loss], nodes=tape.nodes, opset_imports={"": 18}, name="single_product"
        ),
        ir_version=10,
    )


def test_complete_bf16_lora_and_same_backend_baseline(tmp_path):
    report = run(tmp_path)
    assert report["steps"] == 200
    assert report["checkpoint_resume_verified"]
    assert report["master_parameter_bytes"] == 640
    assert report["adam_state_bytes"] == 1280
    assert report["validation_loss"]["final"] < report["validation_loss"]["initial"] * 0.5
    assert np.sqrt(report["validation_loss"]["final"]) < 0.01
    assert report["fp32_training_baseline"]["final_validation_loss"] < 1e-6
    for precision, measured in report["measurement"].items():
        assert measured["precision"] == precision
        assert measured["peak_process_rss_bytes"] > 0
        for metric in measured["timings_ms"].values():
            assert 0 < metric["median"] <= metric["p95"]
    graph = ir.load(tmp_path / "gradient.optimized.onnx")
    assert not graph.functions and all(not node.domain for node in graph.graph)
    # Shape inference annotates actual operands/results; a label alone is not evidence.
    bf16_products = [
        n
        for n in graph.graph
        if n.op_type == "MatMul" and n.inputs[0].dtype == ir.DataType.BFLOAT16
    ]
    assert len(bf16_products) == 4
    assert all(
        n.inputs[1].dtype == n.outputs[0].dtype == ir.DataType.BFLOAT16 for n in bf16_products
    )
    assert all(
        v.dtype == ir.DataType.FLOAT for v in (*graph.graph.inputs, *graph.graph.outputs)
    )
    raw = ir.load(tmp_path / "gradient.onnx")
    assert len(raw.functions) == 1
    assert sum(n.domain == DOMAIN and n.op_type == OP for n in raw.graph) == 4
    interface = json.loads((tmp_path / "interface.json").read_text())
    assert interface["version"] == 1
    assert [p["input"] for p in interface["parameters"]] == list(PARAMETERS)


def test_rounding_oracle_matches_cast_at_ties_and_subnormals():
    dtype = ir.DataType.BFLOAT16.numpy()
    # Nearest-even ties on both parity sides, both signs, signed zeros and subnormals.
    bits = np.array(
        [
            0x00000000,
            0x80000000,
            0x3F808000,
            0x3F818000,
            0xBF808000,
            0xBF818000,
            0x00008000,
            0x00018000,
        ],
        np.uint32,
    )
    values = bits.view(np.float32)
    expected = values.astype(dtype).astype(np.float32)
    np.testing.assert_array_equal(round_bf16(values).view(np.uint32), expected.view(np.uint32))
    rng = np.random.default_rng(42)
    values = rng.normal(size=4096).astype(np.float32)
    np.testing.assert_array_equal(round_bf16(values), values.astype(dtype).astype(np.float32))
    with pytest.raises(ValueError, match="finite"):
        round_bf16(np.array([np.inf], np.float32))


def test_straight_through_jvp_vjp_are_declared_not_literal_rounding_derivatives():
    mixed = lower_matmuls(product_model())
    before = ir.to_proto(mixed).SerializeToString()
    gradient = onnax.grad(mixed, inputs=["x", "w"], output="loss")
    forward = onnax.jvp(mixed, inputs=["x", "w"], outputs=["loss"])
    x, w = np.array([[1.1]], np.float32), np.array([[1.25]], np.float32)
    evaluate = ReferenceEvaluator(ir.to_proto(gradient))
    value, dx, dw = evaluate.run(None, {"x": x, "w": w})
    close(dx, round_bf16(w))
    close(dw, round_bf16(x))
    tangent = ReferenceEvaluator(ir.to_proto(forward)).run(
        None,
        {
            "x": x,
            "w": w,
            forward.graph.inputs[-2].name: np.ones_like(x),
            forward.graph.inputs[-1].name: np.full_like(w, 2),
        },
    )[-1]
    close(tangent, (dx + 2 * dw).sum())
    primal = ReferenceEvaluator(ir.to_proto(mixed))
    plus = primal.run(None, {"x": x + np.float32(1e-5), "w": w})[0]
    minus = primal.run(None, {"x": x - np.float32(1e-5), "w": w})[0]
    assert plus == minus == value
    assert np.count_nonzero(dx) == 1
    assert before == ir.to_proto(mixed).SerializeToString()
    # Inlined raw BF16 paths still reject unsupported generic half differentiation.
    with pytest.raises(TypeError, match="BFLOAT16"):
        onnax.grad(InlinePass()(mixed.clone()).model, inputs=["x"], output="loss")
    with pytest.raises(TypeError, match="BFLOAT16"):
        onnax.jvp(gradient, inputs=["x"])


def test_analytic_straight_through_gradient_matches_smooth_surrogate_differences():
    w, c, x, y, *_rest, p = problem()
    x, y = x[:7], y[:7]
    p[PARAMETERS[1]] = np.full_like(p[PARAMETERS[1]], 0.05)
    a, b = (p[n] for n in PARAMETERS)
    qx, qw, qa, qb, qc = (round_bf16(v).astype(np.float64) for v in (x, w, a, b, c))
    xa = round_bf16(np.dot(qx.astype(np.float32), qa.astype(np.float32))).astype(np.float64)
    adapter = round_bf16(np.dot(xa.astype(np.float32), qb.astype(np.float32))).astype(
        np.float64
    )
    base = round_bf16(np.dot(qx.astype(np.float32), qw.astype(np.float32))).astype(np.float64)
    z0 = (base.astype(np.float32) + np.float32(0.25) * adapter.astype(np.float32)).astype(
        np.float64
    )
    h0 = np.tanh(z0.astype(np.float32)).astype(np.float64)
    qh = round_bf16(h0.astype(np.float32)).astype(np.float64)
    prediction = round_bf16(np.dot(qh.astype(np.float32), qc.astype(np.float32))).astype(
        np.float64
    )

    def smooth(candidate):
        # Freeze rounding offsets at the base point: a smooth local surrogate whose
        # rounding boundaries have identity derivatives, unlike literal quantization.
        av = qa + (candidate[PARAMETERS[0]] - a.astype(np.float64))
        bv = qb + (candidate[PARAMETERS[1]] - b.astype(np.float64))
        xav = xa + (qx @ av - qx @ qa)
        delta = adapter + (xav @ bv - xa @ qb)
        z = z0 + 0.25 * (delta - adapter)
        hidden = h0 + np.tanh(z) - np.tanh(z0)
        rounded_hidden = qh + (hidden - h0)
        predicted = prediction + (rounded_hidden @ qc - qh @ qc)
        return np.mean((predicted - y.astype(np.float64)) ** 2)

    grads = reference(x, y, p, w, c)[2]
    for name in PARAMETERS:
        for coordinate in ((0, 0), (1, 1), (2, 2)):
            plus = {n: v.astype(np.float64) for n, v in p.items()}
            minus = {n: v.astype(np.float64) for n, v in p.items()}
            plus[name][coordinate] += 1e-5
            minus[name][coordinate] -= 1e-5
            close((smooth(plus) - smooth(minus)) / 2e-5, grads[name][coordinate])
