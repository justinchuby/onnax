# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Gather only supervised causal targets and preserve the complete body pullback."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional training dependencies.

import hashlib
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("mobius")

from examples.huggingface.chunked_loss import ChunkedHead
from examples.huggingface.serialization import SharedWeights
from examples.huggingface.sft import prepare, token_loss
from examples.huggingface.validation import add_reference_parameters
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


class RecordingSession:
    def __init__(self, original):
        self.original = original
        self.feeds = []

    def run(self, outputs, feeds):
        self.feeds.append({name: array.copy() for name, array in feeds.items()})
        return self.original.run(outputs, feeds)


def labels_for(case):
    labels = torch.full((2, 7), -100, dtype=torch.int64)
    if case == "all":
        labels[:] = torch.arange(14).reshape(2, 7) % 13
    elif case == "tail":
        labels[0, [1, 3, 6]] = torch.tensor([2, 3, 4])
        labels[1, [1, 4]] = torch.tensor([5, 6])
    elif case == "single":
        labels[1, 6] = 4
    elif case == "first_only":
        labels[:, 0] = 3
    elif case == "padding":
        labels[:] = torch.tensor(
            [[1, 2, 3, 2, -100, -100, -100], [1, 4, 2, -100, -100, -100, -100]]
        )
    return labels


@pytest.mark.parametrize("case", ["all", "tail", "single", "ignored", "first_only", "padding"])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_loss_hidden_gradient_and_executed_rows(case, noncontiguous, factory):
    torch.set_num_threads(1)
    rng = np.random.default_rng(83)
    weight = rng.normal(0, 0.2, (5, 13)).astype(np.float32)
    head = ChunkedHead(weight, chunk_size=2, session_factory=factory)
    recording = RecordingSession(head.executor)
    head.executor = recording
    if noncontiguous:
        hidden = (
            torch.from_numpy(rng.normal(size=(2, 5, 7)).astype(np.float32))
            .transpose(1, 2)
            .requires_grad_()
        )
    else:
        hidden = torch.from_numpy(
            rng.normal(size=(2, 7, 5)).astype(np.float32)
        ).requires_grad_()
    labels = labels_for(case)
    if noncontiguous:
        storage = torch.full((2, 14), -100, dtype=torch.int64)
        storage[:, ::2] = labels
        labels = storage[:, ::2]
    reference = hidden.detach().clone().requires_grad_()
    loss = head(hidden, labels, num_items=17)
    expected = token_loss(reference @ torch.from_numpy(weight), labels, 17)
    (loss * 0.37).backward()
    (expected * 0.37).backward()
    torch.testing.assert_close(loss, expected, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(hidden.grad, reference.grad, rtol=1e-4, atol=1e-6)
    batches, positions = np.nonzero(labels[:, 1:].numpy() != -100)
    valid = len(batches)
    assert [len(row["labels"]) for row in recording.feeds] == [
        min(2, valid - start) for start in range(0, valid, 2)
    ]
    if valid:
        np.testing.assert_array_equal(
            np.concatenate([row["hidden"] for row in recording.feeds]),
            hidden.detach().numpy()[batches, positions],
        )
        assert all(np.all(row["labels"] != -100) for row in recording.feeds)
    assert head.last_profile["valid_rows"] == valid
    assert head.last_profile["runtime_input_bytes"] == valid * (5 * 4 + 8)
    assert head.forward_executor is None
    assert hidden.grad[:, -1].count_nonzero() == 0


def test_ignored_nonfinite_hidden_skips_executor_and_returns_zero(factory):
    head = ChunkedHead(np.ones((3, 5), np.float32), chunk_size=2, session_factory=factory)
    recording = RecordingSession(head.executor)
    head.executor = recording
    hidden = torch.full((2, 7, 3), float("nan"), requires_grad=True)
    loss = head(hidden, torch.full((2, 7), -100, dtype=torch.int64))
    loss.backward()
    assert loss.item() == 0
    assert hidden.grad.count_nonzero() == 0
    assert not recording.feeds
    assert head.last_profile["execute_seconds"] == 0
    assert head.last_profile["runtime_output_bytes"] == 0


def test_multiple_calls_reuse_session_with_changing_supervision(factory):
    head = ChunkedHead(
        np.full((5, 13), 0.1, np.float32), chunk_size=2, session_factory=factory
    )
    executor = head.executor
    for case in ("ignored", "all", "single", "tail"):
        hidden = torch.randn(2, 7, 5, requires_grad=True)
        head(hidden, labels_for(case)).backward()
        assert head.executor is executor
        assert torch.isfinite(hidden.grad).all()


def test_export_reload_preserves_sparse_scheduling(tmp_path, factory):
    rng = np.random.default_rng(26)
    head = ChunkedHead(
        rng.normal(size=(5, 13)).astype(np.float32), chunk_size=2, session_factory=factory
    )
    with SharedWeights(tmp_path) as weights:
        record = head.export_to(weights)
    (tmp_path / "checkpoint_plan.json").write_text(json.dumps({"version": 1, "head": record}))
    restored = ChunkedHead.load(tmp_path, session_factory=factory)
    for case in ("tail", "ignored", "single"):
        hidden = torch.randn(2, 7, 5)
        results = []
        for current in (head, restored):
            value = hidden.clone().requires_grad_()
            loss = current(value, labels_for(case))
            loss.backward()
            results.append((loss, value.grad))
        for actual, expected in zip(results[0], results[1], strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert restored.last_profile["valid_rows"] == head.last_profile["valid_rows"]


def task_batches():
    batches = []
    for ids in (
        torch.tensor([[1, 4, 5, 6, 7, 8, 2], [1, 9, 10, 2, 0, 0, 0]]),
        torch.tensor([[1, 11, 12, 2]]),
    ):
        mask = ids.ne(0).long()
        labels = ids.masked_fill(mask == 0, -100)
        labels[:, : 4 if ids.shape[1] == 7 else 3] = -100
        batches.append(
            {
                "input_ids": ids,
                "attention_mask": mask,
                "position_ids": (mask.cumsum(-1) - 1).clamp(min=0),
                "labels": labels,
            }
        )
    return batches


def test_tiny_qwen_adapters_updates_and_context_gradients_match_full_head(factory):
    torch.set_num_threads(1)
    full, reference = prepare(mode="full", session_factory=factory)
    chunked, _ = prepare(mode="chunked", chunk_size=2, session_factory=factory)
    initial = full.onnx.arrays()
    rng = np.random.default_rng(17)
    for name in initial:
        if name.endswith(".B"):
            initial[name] = rng.normal(0, 0.01, initial[name].shape).astype(np.float32)
    for model in (full, chunked):
        with torch.no_grad():
            for name, parameter in zip(model.onnx.names, model.onnx.adapters, strict=True):
                parameter.copy_(torch.from_numpy(initial[name]))
    reference = add_reference_parameters(reference, initial, full.mapping)
    frozen = {
        name: hashlib.sha256(parameter.detach().numpy().tobytes()).hexdigest()
        for name, parameter in reference.named_parameters()
        if not parameter.requires_grad
    }
    embeddings = []

    def observe(module, args, output):
        output.requires_grad_(True)
        output.retain_grad()
        embeddings.append(output)
        return output

    handle = reference.get_input_embeddings().register_forward_hook(observe)
    optimizers = [
        torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=0.003,
            eps=1e-5,
            weight_decay=0,
        )
        for model in (full, chunked, reference)
    ]
    data = task_batches()
    count = sum(int(batch["labels"][:, 1:].ne(-100).sum()) for batch in data)
    try:
        for _ in range(3):
            embeddings.clear()
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            losses = []
            for model in (full, chunked, reference):
                total = 0.0
                for batch in data:
                    if model is reference:
                        output = model(
                            **{
                                name: value
                                for name, value in batch.items()
                                if name != "labels"
                            },
                            use_cache=False,
                        )
                        loss = token_loss(output.logits, batch["labels"], count)
                    else:
                        loss = model(**batch, num_items_in_batch=count).loss
                    total += float(loss.detach())
                    loss.backward()
                losses.append(total)
            np.testing.assert_allclose(losses, losses[-1], rtol=1e-4, atol=1e-6)
            assert embeddings[0].grad[0, 0].count_nonzero() > 0
            for row in full.mapping:
                target = reference.get_submodule(row["module"])
                for key in ("A", "B"):
                    grads = [
                        dict(zip(model.onnx.names, model.onnx.adapters, strict=True))[
                            row[key]
                        ].grad
                        for model in (full, chunked)
                    ]
                    for gradient in grads:
                        torch.testing.assert_close(
                            gradient, getattr(target, key).grad, rtol=1e-4, atol=1e-6
                        )
            for optimizer in optimizers:
                optimizer.step()
            for row in full.mapping:
                target = reference.get_submodule(row["module"])
                for key in ("A", "B"):
                    for model in (full, chunked):
                        parameter = dict(
                            zip(model.onnx.names, model.onnx.adapters, strict=True)
                        )[row[key]]
                        torch.testing.assert_close(
                            parameter, getattr(target, key), rtol=1e-4, atol=1e-6
                        )
        assert frozen == {
            name: hashlib.sha256(parameter.detach().numpy().tobytes()).hexdigest()
            for name, parameter in reference.named_parameters()
            if not parameter.requires_grad
        }
    finally:
        handle.remove()
