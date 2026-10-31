# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Validate the local derivative oracle, including an intentionally broken rule."""

import os
from unittest.mock import patch

import numpy as np
import onnx_ir as ir
import pytest

pytest.importorskip("mobius")
pytest.importorskip("trl")
pytest.importorskip("torch")


def test_tiny_nf4_all_active_pullbacks():
    from examples.huggingface.local_vjp_audit import run

    report = run(nf4=True)
    assert report["unchecked_active_nodes"] == {}
    assert report["covered_nodes"] == 160
    for state in report["states"]:
        assert state["failed_partials"] == 0
        assert state["operators"]["NF4MatMul"]["nodes"] == 11
        assert state["operators"]["Split"]["input_partials"] > 0
    assert [state["adapter_b"] for state in report["states"]] == [0.0, 0.01]


@pytest.mark.parametrize("bf16", [False, True])
def test_local_oracle_detects_wrong_rule_and_nonfinite_gradient(bf16):
    from examples.huggingface.local_vjp_audit import check, trace
    from examples.huggingface.rms_audit import diagnostic_session
    from onnax._autodiff._registry import _VjpRegistry

    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[2, 3])
    weight = ir.val("weight", const_value=ir.tensor(np.ones((3, 4), np.float32)))
    tape = ir.tape.Tape()
    from examples.lora.bf16_rules import DOMAIN, OP, matmul_function

    y = tape.op(OP if bf16 else "MatMul", [x, weight], domain=DOMAIN if bf16 else "")
    y.name, y.dtype, y.shape = "logits", ir.DataType.FLOAT, ir.Shape([2, 4])
    model = ir.Model(
        ir.Graph([x], [y], nodes=tape.nodes, initializers=[weight], opset_imports={"": 18}),
        ir_version=10,
    )
    if bf16:
        function = matmul_function(18)
        model.functions[function.identifier()] = function
        model.graph.opset_imports[DOMAIN] = 1
    original_lookup = _VjpRegistry.lookup
    original_model = ir.to_proto(model).SerializeToString()

    def broken_lookup(registry, node, imports):
        original = original_lookup(registry, node, imports)

        def broken(context, node, seeds, indices):
            result = original(context, node, seeds, indices)
            return {
                i: context.op("Mul", value, context.scalar(node.inputs[i], 2))
                for i, value in result.items()
            }

        return broken

    with patch.object(_VjpRegistry, "lookup", broken_lookup):
        joint, seed, probes, _ = trace(model, ["x"])
    assert _VjpRegistry.lookup is original_lookup
    assert ir.to_proto(model).SerializeToString() == original_model
    arguments = {"x": np.ones((2, 3), np.float32), seed: np.ones((2, 4), np.float32)}
    from examples.huggingface.serialization import reference_session

    executor = (reference_session if bf16 else diagnostic_session)(
        ir.to_proto(joint).SerializeToString()
    )
    observed = dict(
        zip([v.name for v in joint.graph.outputs], executor.run(None, arguments), strict=True)
    )
    constants = {"weight": weight.const_value}
    report = check(probes, observed, constants)
    assert report["failed_partials"] == 1
    assert report["failures"][0]["max_abs"] == 4
    gradient_name = probes[0][2][0].name
    observed[gradient_name] = np.full((2, 3), np.nan, np.float32)
    with pytest.raises(AssertionError, match="Nonfinite local gradient"):
        check(probes, observed, constants)
    observed[gradient_name] = np.ones((1, 3), np.float32)
    with pytest.raises(AssertionError, match="interface mismatch"):
        check(probes, observed, constants)


@pytest.mark.parametrize("nf4", [False, True])
@pytest.mark.parametrize("provider", ["reference", "mlx"])
def test_bf16_local_pullbacks_at_observed_primals(nf4, provider):
    if provider == "mlx" and os.environ.get("ONNAX_TEST_MLX_EP") != "1":
        pytest.skip("Native MLX opt-in")
    from examples.huggingface.local_vjp_audit import run

    report = run(
        nf4=nf4,
        projection_precision="bf16",
        provider=provider,
        library=os.environ.get("ONNAX_MLX_EP_LIBRARY") if provider == "mlx" else None,
    )
    assert report["unchecked_active_nodes"] == {}
    for state in report["states"]:
        assert state["failed_partials"] == 0, state["failures"]
        assert state["operators"]["MatMulBF16"]["input_partials"] > 0
        if nf4:
            assert state["operators"]["NF4MatMulBF16"]["nodes"] == 11
    assert [state["adapter_b"] for state in report["states"]] == [0.0, 0.01]
