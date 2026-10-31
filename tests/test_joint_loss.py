# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Qualify combined ONNX loss/gradient execution against an independent model."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional workload dependencies.

import json
import os
from types import SimpleNamespace

import numpy as np
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


def selected_factory(native):
    if not native:
        return None
    if os.environ.get("ONNAX_TEST_MLX_EP") != "1":
        pytest.skip("Native MLX opt-in")
    from examples.lora.mlx_runtime import register_library, session

    library = os.environ.get("ONNAX_MLX_EP_LIBRARY")
    if library:
        register_library(library)
    return session


@pytest.mark.parametrize("native", [False, True])
def test_variable_microbatches_accumulate_with_upstream_gradient(native):
    torch.set_num_threads(1)
    joint, reference, mapping, _ = make_joint(True, session_factory=selected_factory(native))
    samples = [batch(), {name: value[:1, :3].clone() for name, value in batch().items()}]
    ignored = {name: value[:1, :1].clone() for name, value in batch().items()}
    ignored["labels"].fill_(-100)
    samples.append(ignored)
    count = sum(int(sample["labels"][:, 1:].ne(-100).sum()) for sample in samples)
    executor = joint.executor
    actual_losses, expected_losses = [], []
    for sample in samples:
        actual_losses.append(joint(**sample) / count)
        inputs = {name: value for name, value in sample.items() if name != "labels"}
        expected_losses.append(
            stable_token_loss(
                reference(**inputs, use_cache=False).logits, sample["labels"], num_items=count
            )
        )
    actual, expected = (
        2.5 * torch.stack(actual_losses).sum(),
        2.5 * torch.stack(expected_losses).sum(),
    )
    actual.backward()
    expected.backward()
    assert joint.executor is executor
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
    for row in mapping:
        for key in ("A", "B"):
            p = joint.adapters[joint.names.index(row[key])]
            q = getattr(reference.get_submodule(row["module"]), key)
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("native", [False, True])
def test_joint_all_ignored_labels_have_zero_loss_and_gradients(native):
    torch.set_num_threads(1)
    joint, _, _, _ = make_joint(True, session_factory=selected_factory(native))
    sample = batch()
    sample["labels"].fill_(-100)
    loss = joint(**sample)
    loss.backward()
    assert loss.item() == 0
    for parameter in joint.parameters():
        torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter), rtol=0, atol=0)


def test_joint_validates_labels_and_feeds_before_execution(monkeypatch):
    joint, _, _, _ = make_joint()

    class Unreachable:
        def run(self, *args, **kwargs):
            raise AssertionError("Invalid labels reached the runtime")

    monkeypatch.setattr(joint, "executor", Unreachable())
    sample = batch()
    for labels in (
        torch.full((2, 5), -2),
        torch.full((2, 5), 256),
        torch.zeros(2, 5),
        torch.zeros(2, 4, dtype=torch.int64),
    ):
        with pytest.raises(ValueError, match=r"[Ll]abels"):
            joint(**{**sample, "labels": labels})
    with pytest.raises(ValueError, match="missing"):
        joint(**{name: value for name, value in sample.items() if name != "labels"})
    with pytest.raises(ValueError, match="unexpected"):
        joint(**sample, unknown=torch.zeros(1))


def test_joint_checkpoint_export_and_reload(tmp_path):
    torch.set_num_threads(1)
    joint, _, _, graph = make_joint(True)
    optimizer = torch.optim.AdamW(joint.parameters(), lr=0.003, weight_decay=0)
    sample = batch()

    def update(module, opt):
        opt.zero_grad(set_to_none=True)
        value = module(**sample)
        value.backward()
        opt.step()

    update(joint, optimizer)
    torch.save(
        {"model": joint.state_dict(), "optimizer": optimizer.state_dict()},
        tmp_path / "checkpoint.pt",
    )
    restored, _, _, _ = make_joint(True)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.003, weight_decay=0)
    checkpoint = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    restored.load_state_dict(checkpoint["model"])
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    for _ in range(2):
        update(joint, optimizer)
        update(restored, restored_optimizer)
    for actual, expected in zip(joint.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    directory = tmp_path / "export"
    interface = joint.export(str(directory), forward_model=graph)
    assert json.loads((directory / "interface.json").read_text()) == interface
    assert interface["causal_loss"]["reduction"] == "sum"
    assert interface["kind"] == "loss_gradient"
    feeds = {name: value.numpy() for name, value in sample.items()}
    with np.load(directory / "adapter.npz") as parameters:
        arrays = {name: parameters[name].copy() for name in joint.names}
    expected = joint.executor.run(joint.outputs, {**feeds, **arrays})
    actual = runtime_session(directory / "loss_gradient.onnx").run(None, {**feeds, **arrays})
    for x, y in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(x, y)
    forward_inputs = {name: value for name, value in feeds.items() if name != "labels"}
    expected = runtime_session(graph).run(None, {**forward_inputs, **arrays})
    actual = runtime_session(directory / "forward.onnx").run(None, forward_inputs)
    for x, y in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(x, y)


def test_joint_checkpoint_rejects_base_mismatch():
    joint, _, _, _ = make_joint()
    state = joint.state_dict()
    state["base_identity"] = state["base_identity"].clone()
    state["base_identity"][0] ^= 1
    with pytest.raises(ValueError, match="base model"):
        joint.load_state_dict(state)


def test_joint_checkpoint_precision_policy(tmp_path):
    joint, _, _, _ = make_joint()
    graph = joint.model.clone()
    policy = {"version": 1, "projection_compute": "bfloat16"}
    graph.metadata_props["onnax.example.precision"] = json.dumps(policy)
    precise = JointLoss(graph, joint.arrays())
    assert precise.frozen == joint.frozen
    assert precise.names == joint.names
    with pytest.raises(ValueError, match="precision policy"):
        precise.load_state_dict(joint.state_dict())
    with pytest.raises(ValueError, match="precision policy"):
        joint.load_state_dict(precise.state_dict())
    # JSON formatting does not change the computation contract.
    graph.metadata_props["onnax.example.precision"] = json.dumps(policy, indent=2)
    restored = JointLoss(graph, joint.arrays())
    restored.load_state_dict(precise.state_dict())
    precise.export(tmp_path)
    assert json.loads((tmp_path / "interface.json").read_text())["precision"] == policy
