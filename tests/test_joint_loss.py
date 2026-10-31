# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Qualify combined ONNX loss/gradient execution against an independent model."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional workload dependencies.

import os
from types import SimpleNamespace

import onnx
import onnx_ir as ir
import pytest
from _testing import runtime_session

pytest.importorskip("mobius")
pytest.importorskip("trl")
torch = pytest.importorskip("torch")

from examples.huggingface.joint_loss import JointLoss, causal_loss_graph
from examples.huggingface.qwen import build_base, decompose, inject, reference_model
from examples.huggingface.reference_losses import stable_token_loss
from examples.huggingface.validation import add_reference_adapters


def make_joint(nf4=False, *, session_factory=None):
    reference = reference_model()
    adapted, params, mapping = inject(build_base(reference))
    graph = decompose(adapted)
    if nf4:
        from examples.huggingface.nf4 import lower_decoder

        graph, _ = lower_decoder(graph, reference)
    before = ir.to_proto(graph).SerializeToString()
    options = {} if session_factory is None else {"session_factory": session_factory}
    joint = JointLoss(causal_loss_graph(graph), params, **options)
    assert ir.to_proto(graph).SerializeToString() == before
    onnx.checker.check_model(ir.to_proto(joint.gradient), full_check=True)
    reference = add_reference_adapters(reference, SimpleNamespace(onnx=joint, mapping=mapping))
    return joint, reference, mapping, graph


def batch():
    ids = torch.tensor([[1, 4, 5, 6, 2], [1, 7, 2, 0, 0]])
    mask = ids.ne(0).long()
    labels = ids.clone()
    labels[mask == 0] = -100
    labels[0, 2] = -100
    return dict(
        input_ids=ids,
        attention_mask=mask,
        position_ids=(mask.cumsum(-1) - 1).clamp(min=0),
        labels=labels,
    )


@pytest.mark.parametrize("nf4", [False, True])
def test_joint_qwen_independent_updates(nf4):
    torch.set_num_threads(1)
    joint, reference, mapping, graph = make_joint(nf4)
    sample = batch()
    inputs = {n: v for n, v in sample.items() if n != "labels"}
    predicted = runtime_session(graph).run(
        ["logits"], {**{n: v.numpy() for n, v in inputs.items()}, **joint.arrays()}
    )[0]
    with torch.no_grad():
        expected = reference(**inputs, use_cache=False).logits
    torch.testing.assert_close(torch.from_numpy(predicted), expected, rtol=1e-4, atol=1e-6)
    actual_optimizer = torch.optim.AdamW(joint.parameters(), lr=0.003, weight_decay=0)
    expected_optimizer = torch.optim.AdamW(
        [p for p in reference.parameters() if p.requires_grad], lr=0.003, weight_decay=0
    )
    count = sample["labels"][:, 1:].ne(-100).sum()
    for _ in range(5):
        joint.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        actual_loss = joint(**sample) / count
        expected_loss = stable_token_loss(
            reference(**inputs, use_cache=False).logits, sample["labels"]
        )
        actual_loss.backward()
        expected_loss.backward()
        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-4, atol=1e-6)
        for row in mapping:
            for key in ("A", "B"):
                p = joint.adapters[joint.names.index(row[key])]
                q = getattr(reference.get_submodule(row["module"]), key)
                torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-6)
        actual_optimizer.step()
        expected_optimizer.step()
        for row in mapping:
            for key in ("A", "B"):
                p = joint.adapters[joint.names.index(row[key])]
                q = getattr(reference.get_submodule(row["module"]), key)
                torch.testing.assert_close(p, q, rtol=1e-4, atol=1e-6)


def test_joint_rejects_parameter_mutation_before_backward():
    joint, _, _, _ = make_joint()
    loss = joint(**batch())
    with torch.no_grad():
        joint.adapters[0].add_(0.001)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()


@pytest.mark.skipif(os.environ.get("ONNAX_TEST_MLX_EP") != "1", reason="Native MLX opt-in")
def test_joint_native_nf4_loss_and_gradients():
    from examples.lora.mlx_runtime import register_library, session

    library = os.environ.get("ONNAX_MLX_EP_LIBRARY")
    if library:
        register_library(library)
    torch.set_num_threads(1)
    joint, reference, mapping, _ = make_joint(nf4=True, session_factory=session)
    sample = batch()
    actual = joint(**sample)
    inputs = {name: value for name, value in sample.items() if name != "labels"}
    count = sample["labels"][:, 1:].ne(-100).sum()
    expected = (
        stable_token_loss(reference(**inputs, use_cache=False).logits, sample["labels"])
        * count
    )
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
    for row in mapping:
        for key in ("A", "B"):
            p = joint.adapters[joint.names.index(row[key])]
            q = getattr(reference.get_submodule(row["module"]), key)
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-6)
