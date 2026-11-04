# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Keep source-layout weights through loss/VJP export and external updates."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional training dependencies.

import os

import numpy as np
import onnx
import pytest

pytest.importorskip("mobius")
pytest.importorskip("trl")
torch = pytest.importorskip("torch")

from examples.huggingface.chunked_loss import ChunkedHead, head_graph
from examples.huggingface.serialization import SharedWeights
from examples.huggingface.sft import token_loss
from examples.lora.train import session


@pytest.fixture(params=["cpu", "mlx"])
def factory(request):
    if request.param == "cpu":
        return session
    if os.environ.get("ONNAX_TEST_MLX_EP") != "1":
        pytest.skip("Native MLX opt-in")
    from examples.lora.mlx_runtime import register_library
    from examples.lora.mlx_runtime import session as mlx_session

    if path := os.environ.get("ONNAX_MLX_EP_LIBRARY"):
        register_library(path)
    return mlx_session


def test_source_layout_head_gradients_updates_and_reload(tmp_path, factory):
    torch.set_num_threads(1)
    rng = np.random.default_rng(491)
    weight = rng.normal(0, 0.2, (257, 16)).astype(np.float32)
    runtime = ChunkedHead(
        weight, chunk_size=3, weight_layout="vocab_hidden", session_factory=factory
    )
    for graph in (runtime.model, runtime.gradient, runtime.forward_model):
        value = graph.graph.initializers["head.weight"]
        assert list(value.shape) == [257, 16]
        np.testing.assert_array_equal(value.const_value.numpy(), weight)
    directory = tmp_path / "head"
    directory.mkdir()
    with SharedWeights(directory) as writer:
        record = runtime.export_to(writer)
    restored = ChunkedHead.from_record(directory, record, session_factory=factory)
    for key in ("loss", "gradient", "forward"):
        onnx.checker.check_model(str(directory / record[key]), full_check=True)
    actual = torch.nn.Parameter(torch.tensor(rng.normal(size=(2, 7, 16)).astype(np.float32)))
    expected = torch.nn.Parameter(actual.detach().clone())
    labels = torch.tensor([[1, 2, 3, 4, -100, -100, -100], [-100, 3, -100, 5, 6, 7, 8]])
    optimizers = [
        torch.optim.SGD([value], lr=0.1, momentum=0.9) for value in (actual, expected)
    ]
    losses = []
    for _ in range(8):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss = restored(actual, labels, num_items=17)
        reference = token_loss(expected @ torch.from_numpy(weight.T), labels, num_items=17)
        torch.testing.assert_close(loss, reference, rtol=1e-4, atol=1e-6)
        (loss * 2.3).backward()
        (reference * 2.3).backward()
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-4, atol=1e-6)
        for optimizer in optimizers:
            optimizer.step()
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
    torch.testing.assert_close(
        restored.forward(actual),
        actual.detach() @ torch.from_numpy(weight.T),
        rtol=1e-4,
        atol=1e-6,
    )
    masked = torch.full_like(labels, -100)
    optimizers[0].zero_grad(set_to_none=True)
    empty = restored(actual, masked)
    empty.backward()
    torch.testing.assert_close(empty.detach(), torch.zeros_like(empty), rtol=0, atol=0)
    assert torch.count_nonzero(actual.grad) == 0
    assert restored.last_profile["chunks"] == 0


def test_source_layout_is_explicit_and_validated():
    weight = np.ones((257, 16), np.float32)
    with pytest.raises(ValueError, match="weight layout"):
        head_graph(weight, weight_layout="guess")
    with pytest.raises(ValueError, match="FP32"):
        head_graph(weight, weight_layout="vocab_hidden", projection_precision="bf16")


def test_shared_head_terms_preserve_small_gradients(factory):
    weight = np.array([[1, 0], [-1, 0]], np.float32)
    runtime = ChunkedHead(weight, weight_layout="vocab_hidden", session_factory=factory)
    try:
        assert any(node.op_type == "HardCrossEntropyF32" for node in runtime.model.graph)
        assert all(node.op_type != "HardCrossEntropyF32" for node in runtime.gradient.graph)
        values = runtime.executor.run(
            None,
            {
                "hidden": np.array([[20, 0], [20, 0]], np.float32),
                "labels": np.array([0, -100], np.int64),
            },
        )
        tail = np.float32(np.exp(-40))
        expected_loss = np.log1p(tail)
        expected_gradient = np.array([[-2 * tail / (1 + tail), 0], [0, 0]], np.float32)
        np.testing.assert_allclose(values[0], expected_loss, rtol=1e-4, atol=0)
        np.testing.assert_allclose(values[1], expected_gradient, rtol=1e-4, atol=0)
        assert values[1][0, 0] < 0
    finally:
        runtime.close()


@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp16"])
@pytest.mark.parametrize("compiled", [False, True])
def test_shared_head_terms_native_precision(precision, compiled, monkeypatch):
    if os.environ.get("ONNAX_TEST_MLX_EP") != "1":
        pytest.skip("Native MLX opt-in")
    import functools

    from examples.lora.mlx_runtime import register_library
    from examples.lora.mlx_runtime import session as mlx_session

    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "0")
    if path := os.environ.get("ONNAX_MLX_EP_LIBRARY"):
        register_library(path)
    weight = np.array([[1, -1], [0, 0]], np.float32)
    runtime = ChunkedHead(
        weight,
        projection_precision=precision,
        session_factory=functools.partial(mlx_session, enable_compile=compiled),
    )
    try:
        values = runtime.executor.run(
            None,
            {
                "hidden": np.array([[20, 0]], np.float32),
                "labels": np.array([0], np.int64),
            },
        )
        tail = np.float32(np.exp(-40))
        np.testing.assert_allclose(values[0], np.log1p(tail), rtol=1e-4, atol=0)
        np.testing.assert_allclose(
            values[1], np.array([[-2 * tail / (1 + tail), 0]], np.float32), rtol=1e-4, atol=0
        )
    finally:
        runtime.close()
