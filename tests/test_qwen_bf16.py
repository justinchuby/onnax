# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Qwen BF16 projections and independently declared FP32 surrogate adjoints."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional workload dependencies.

import json
import os
from types import SimpleNamespace

import onnx
import onnx_ir as ir
import pytest
from onnx.reference import ReferenceEvaluator
from onnx_ir.passes.common import InlinePass, ShapeInferencePass

pytest.importorskip("mobius")
pytest.importorskip("trl")
torch = pytest.importorskip("torch")

from examples.huggingface.joint_loss import JointLoss, causal_loss_graph
from examples.huggingface.mixed_precision import bf16_projections, reference_projections
from examples.huggingface.qwen import build_base, decompose, inject, reference_model
from examples.huggingface.reference_losses import stable_token_loss
from examples.huggingface.validation import add_reference_adapters


def reference_session(source):
    source = onnx.load_model_from_string(source) if isinstance(source, bytes) else str(source)
    return ReferenceEvaluator(source)


@pytest.mark.parametrize("native", [False, True])
def test_qwen_bf16_forward_gradients_and_updates(native):
    if native and os.environ.get("ONNAX_TEST_MLX_EP") != "1":
        pytest.skip("Native MLX opt-in")
    factory = reference_session
    if native:
        from examples.lora.mlx_runtime import register_library, session

        if library := os.environ.get("ONNAX_MLX_EP_LIBRARY"):
            register_library(library)
        factory = session
    torch.set_num_threads(1)
    reference = reference_model()
    source, params, mapping = inject(build_base(reference))
    source = decompose(source)
    before = ir.to_proto(source).SerializeToString()
    graph = bf16_projections(source)
    assert ir.to_proto(source).SerializeToString() == before
    policy = json.loads(graph.metadata_props["onnax.example.precision"])
    assert policy["projection_products"] == 23
    assert sum(node.op_type == "MatMul" for node in graph.graph) == 4
    inlined = ShapeInferencePass()(InlinePass()(graph.clone()).model).model
    onnx.checker.check_model(ir.to_proto(inlined), full_check=True)
    bf16 = [
        node
        for node in inlined.graph
        if node.op_type == "MatMul" and node.inputs[0].dtype == ir.DataType.BFLOAT16
    ]
    assert len(bf16) == 23
    assert all(node.outputs[0].dtype == ir.DataType.BFLOAT16 for node in bf16)
    joint = JointLoss(causal_loss_graph(graph), params, session_factory=factory)
    reference = reference_projections(
        add_reference_adapters(reference, SimpleNamespace(onnx=joint, mapping=mapping))
    )
    onnx.checker.check_model(ir.to_proto(joint.gradient), full_check=True)
    ids = torch.tensor([[1, 4, 5, 6, 2], [1, 7, 2, 0, 0]])
    mask = ids.ne(0).long()
    labels = ids.clone()
    labels[mask == 0] = -100
    inputs = dict(
        input_ids=ids, attention_mask=mask, position_ids=(mask.cumsum(-1) - 1).clamp(min=0)
    )
    executor = factory(ir.to_proto(graph).SerializeToString())
    prediction = executor.run(
        ["logits"],
        {**{name: value.numpy() for name, value in inputs.items()}, **joint.arrays()},
    )[0]
    expected = reference(**inputs, use_cache=False).logits
    torch.testing.assert_close(torch.from_numpy(prediction), expected, rtol=1e-4, atol=1e-6)
    actual_optimizer = torch.optim.AdamW(joint.parameters(), lr=0.003, weight_decay=0)
    expected_optimizer = torch.optim.AdamW(
        [p for p in reference.parameters() if p.requires_grad], lr=0.003, weight_decay=0
    )
    count = labels[:, 1:].ne(-100).sum()
    for step in range(5):
        joint.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        actual_loss = joint(**inputs, labels=labels) / count
        expected_loss = stable_token_loss(reference(**inputs, use_cache=False).logits, labels)
        actual_loss.backward()
        expected_loss.backward()
        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-4, atol=1e-6)
        for row in mapping:
            for key in ("A", "B"):
                actual = joint.adapters[joint.names.index(row[key])]
                expected = getattr(reference.get_submodule(row["module"]), key)
                assert actual.dtype == actual.grad.dtype == torch.float32
                torch.testing.assert_close(
                    actual.grad,
                    expected.grad,
                    rtol=1e-4,
                    atol=1e-6,
                    msg=f"step {step + 1}: {row[key]} gradient",
                )
        actual_optimizer.step()
        expected_optimizer.step()
        for row in mapping:
            for key in ("A", "B"):
                actual = joint.adapters[joint.names.index(row[key])]
                expected = getattr(reference.get_submodule(row["module"]), key)
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=1e-4,
                    atol=1e-6,
                    msg=f"step {step + 1}: {row[key]} parameter",
                )
                assert actual_optimizer.state[actual]["exp_avg"].dtype == torch.float32
                assert actual_optimizer.state[actual]["exp_avg_sq"].dtype == torch.float32
