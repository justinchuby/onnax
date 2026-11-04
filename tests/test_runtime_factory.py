# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Run derivative modules through a caller-selected evaluator."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional training dependencies.

import numpy as np
import onnx
import pytest
from onnx.reference import ReferenceEvaluator

torch = pytest.importorskip("torch")

from examples.huggingface.bridge import OnnxModule
from examples.huggingface.chunked_loss import ChunkedHead
from examples.lora.model import PARAMETERS, adapt, base_model
from examples.lora.train import problem, reference


@pytest.mark.parametrize("saved", [False, True])
def test_external_evaluator_forward_and_vjp(saved):
    executors = []

    def factory(source):
        evaluator = ReferenceEvaluator(
            onnx.load_model_from_string(source)
            if isinstance(source, bytes)
            else onnx.load(source)
        )
        executors.append(evaluator)
        return evaluator

    w, c, x, y, _, _, params = problem()
    model = OnnxModule(
        adapt(base_model(w, c)), params, save_activations=saved, session_factory=factory
    )
    prediction = model(features=torch.tensor(x[:7]))
    loss = (prediction - torch.tensor(y[:7])).square().mean()
    loss.backward()
    expected = reference(x[:7], y[:7], params, w, c)
    np.testing.assert_allclose(prediction.detach(), expected[0], rtol=1e-4, atol=1e-6)
    for name, parameter in zip(PARAMETERS, model.adapters, strict=True):
        np.testing.assert_allclose(parameter.grad, expected[2][name], rtol=1e-4, atol=1e-6)
    assert len(executors) == 2


def test_chunked_head_prediction_and_loss_use_selected_evaluator():
    executors = []

    def factory(source):
        evaluator = ReferenceEvaluator(
            onnx.load_model_from_string(source)
            if isinstance(source, bytes)
            else onnx.load(source)
        )
        executors.append(evaluator)
        return evaluator

    torch.manual_seed(13)
    weight = torch.randn(8, 31)
    hidden = torch.randn(2, 6, 8, requires_grad=True)
    labels = torch.randint(31, (2, 6))
    labels[1, -2:] = -100
    head = ChunkedHead(weight.numpy(), chunk_size=7, session_factory=factory)
    logits = head.forward(hidden)
    torch.testing.assert_close(logits, hidden @ weight, rtol=1e-4, atol=1e-6)
    actual = head(hidden, labels)
    actual.backward()
    expected_hidden = hidden.detach().clone().requires_grad_()
    expected = torch.nn.functional.cross_entropy(
        (expected_hidden @ weight)[:, :-1].reshape(-1, 31), labels[:, 1:].reshape(-1)
    )
    expected.backward()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(hidden.grad, expected_hidden.grad, rtol=1e-4, atol=1e-6)
    assert len(executors) == 2


def test_qwen_body_and_head_use_independent_factories():
    pytest.importorskip("mobius")
    pytest.importorskip("trl")
    from examples.huggingface.sft import prepare
    from examples.lora.train import session

    calls = {"body": [], "head": []}

    def body(source):
        calls["body"].append(source)
        return session(source)

    def head(source):
        calls["head"].append(source)
        return session(source)

    model, _ = prepare(
        mode="chunked", checkpoint_layers=1, session_factory=body, head_session_factory=head
    )
    assert len(calls["head"]) == 1
    ids = torch.tensor([[1, 4, 5, 2]])
    loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids).loss
    loss.backward()
    assert calls["body"] and len(calls["head"]) == 1
    assert all(parameter.grad is not None for parameter in model.onnx.adapters)
    model(input_ids=ids, attention_mask=torch.ones_like(ids))
    assert len(calls["head"]) == 2


@pytest.mark.parametrize("entrypoint", ["benchmark", "sft"])
@pytest.mark.parametrize("mode,provider", [("chunked", "cpu"), ("joint", "mlx")])
def test_compile_head_requires_native_chunked_execution(entrypoint, mode, provider, tmp_path):
    pytest.importorskip("mobius")
    pytest.importorskip("trl")
    from examples.huggingface import benchmark, sft

    with pytest.raises(ValueError, match="compile_head requires MLX chunked"):
        if entrypoint == "benchmark":
            benchmark.measure(mode, provider=provider, compile_head=True)
        else:
            sft.run(str(tmp_path / "unused"), mode=mode, provider=provider, compile_head=True)
    assert not (tmp_path / "unused").exists()
