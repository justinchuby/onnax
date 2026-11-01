# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Decoder checkpoint composition, gradients and autograd lifetime contracts."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional example dependencies.

import os

import numpy as np
import pytest

pytest.importorskip("mobius")
pytest.importorskip("trl")
torch = pytest.importorskip("torch")

from examples.huggingface.serialization import reference_session
from examples.huggingface.sft import prepare


def batch():
    ids = torch.tensor([[1, 4, 5, 6, 2], [1, 7, 2, 0, 0]])
    mask = ids.ne(0).long()
    labels = ids.clone()
    labels[mask == 0] = -100
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "position_ids": (mask.cumsum(-1) - 1).clamp(min=0),
        "labels": labels,
    }


@pytest.mark.parametrize(
    "layers,nf4,precision",
    [(1, False, "fp32"), (2, True, "fp32"), (4, True, "fp32"), (1, True, "bf16")],
)
def test_checkpoint_gradients_and_updates_match_flat(layers, nf4, precision):
    torch.set_num_threads(1)
    options = {"session_factory": reference_session} if precision == "bf16" else {}
    flat, _ = prepare(mode="chunked", nf4=nf4, projection_precision=precision, **options)
    grouped, _ = prepare(
        mode="chunked",
        nf4=nf4,
        projection_precision=precision,
        checkpoint_layers=layers,
        **options,
    )
    rng = np.random.default_rng(42)
    with torch.no_grad():
        for name, parameter in zip(flat.onnx.names, flat.onnx.adapters, strict=True):
            if name.endswith(".B"):
                parameter.copy_(
                    torch.from_numpy(
                        rng.normal(0, 0.01, tuple(parameter.shape)).astype(np.float32)
                    )
                )
    grouped.load_state_dict(flat.state_dict())
    optimizers = [
        torch.optim.AdamW(m.parameters(), lr=0.003, weight_decay=0) for m in (flat, grouped)
    ]
    for _ in range(3):
        losses = []
        for model, optimizer in zip((flat, grouped), optimizers, strict=True):
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch()).loss
            loss.backward()
            losses.append(loss.detach())
        torch.testing.assert_close(*losses, rtol=1e-4, atol=1e-6)
        for a, b in zip(flat.onnx.adapters, grouped.onnx.adapters, strict=True):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-6)
        for optimizer in optimizers:
            optimizer.step()
        for a, b in zip(flat.onnx.adapters, grouped.onnx.adapters, strict=True):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)
    assert grouped.onnx.last_residual_bytes > 0
    assert len(grouped.onnx.timings["backward"]) == 3
    assert grouped.head.forward_executor is None
    with torch.no_grad():
        a = flat(**{k: v for k, v in batch().items() if k != "labels"}).logits
        b = grouped(**{k: v for k, v in batch().items() if k != "labels"}).logits
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def test_checkpoint_retained_backward_and_input_snapshot():
    model, _ = prepare(mode="chunked", checkpoint_layers=1)
    inputs = {k: v for k, v in batch().items() if k != "labels"}
    hidden = model.onnx(**inputs)
    loss = hidden.square().sum()
    inputs["input_ids"].fill_(0)
    loss.backward(retain_graph=True)
    first = [p.grad.clone() for p in model.onnx.adapters]
    loss.backward()
    for p, expected in zip(model.onnx.adapters, first, strict=True):
        torch.testing.assert_close(p.grad, 2 * expected, rtol=1e-4, atol=1e-6)
    assert len(model.onnx.timings["backward"]) == 2


def test_checkpoint_rejects_parameter_update_before_backward():
    model, _ = prepare(mode="chunked", checkpoint_layers=1)
    loss = model(**batch()).loss
    with torch.no_grad():
        model.onnx.adapters[0].add_(0.01)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()


@pytest.mark.parametrize("layers", [-1, True, 1.5])
def test_invalid_checkpoint_layers(layers):
    with pytest.raises(ValueError, match="nonnegative integer"):
        prepare(mode="chunked", checkpoint_layers=layers)


def test_checkpoint_strategy_options():
    with pytest.raises(ValueError, match="require chunked"):
        prepare(mode="joint", checkpoint_layers=1)
    with pytest.raises(ValueError, match="without save_activations"):
        prepare(mode="chunked", checkpoint_layers=1, save_activations=True)


def test_checkpoint_export_load_and_reexport(tmp_path, monkeypatch):
    import onnax
    from examples.huggingface.checkpointing import CheckpointedModule

    model, _ = prepare(mode="chunked", nf4=True, checkpoint_layers=1)
    inputs = {k: v for k, v in batch().items() if k != "labels"}
    expected = model.onnx(**inputs)
    expected.square().sum().backward()
    model.onnx.export(tmp_path)

    def no_new_derivatives(*args, **kwargs):
        raise AssertionError("Loading must execute the exported VJP modules")

    monkeypatch.setattr(onnax, "vjp", no_new_derivatives)
    loaded = CheckpointedModule.load(tmp_path)
    actual = loaded(**inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    for a, b in zip(model.onnx.adapters, loaded.adapters, strict=True):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    weights = list(tmp_path.glob("weights-*.data"))
    assert len(weights) == 1
    before = weights[0].read_bytes()
    loaded.export(tmp_path)
    assert len(list(tmp_path.glob("weights-*.data"))) == 1
    assert weights[0].read_bytes() == before
    torch.testing.assert_close(loaded(**inputs), expected, rtol=0, atol=0)


def test_three_layer_tail_and_frozen_middle(monkeypatch):
    from examples.huggingface import qwen

    original = qwen.tiny_config

    def config():
        result = original()
        result.num_hidden_layers = 3
        result.layer_types = [result.layer_types[0]] * 3
        return result

    monkeypatch.setattr(qwen, "tiny_config", config)
    flat, _ = prepare(mode="chunked")
    grouped, _ = prepare(mode="chunked", checkpoint_layers=2)
    grouped.load_state_dict(flat.state_dict())
    assert [b.name for b in grouped.onnx.blocks] == [
        "prefix",
        "decoder_0_2",
        "decoder_2_3",
        "suffix",
    ]
    for model in (flat, grouped):
        for name, value in zip(model.onnx.names, model.onnx.adapters, strict=True):
            if "layers.1." in name:
                value.requires_grad_(False)
        model(**batch()).loss.backward()
    for a, b in zip(flat.onnx.adapters, grouped.onnx.adapters, strict=True):
        if a.requires_grad:
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-6)
        else:
            assert a.grad is b.grad is None
    grouped.zero_grad(set_to_none=True)
    inputs = batch()
    inputs["labels"].fill_(-100)
    loss = grouped(**inputs).loss
    assert loss.item() == 0
    loss.backward()
    assert all(p.grad is None or p.grad.count_nonzero() == 0 for p in grouped.onnx.adapters)


def test_checkpoint_against_independent_transformers_reference():
    from examples.huggingface.validation import add_reference_adapters, audit

    model, reference = prepare(mode="chunked", nf4=True, checkpoint_layers=1)
    reference = add_reference_adapters(reference, model)
    with torch.no_grad():
        for name, parameter in zip(model.onnx.names, model.onnx.adapters, strict=True):
            if name.endswith(".B"):
                parameter.add_(0.01)
    report = audit(model, reference, batch())
    assert report["strict_gradient_check_passed"]
    assert report["strict_logit_check_passed"]


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_native_checkpoint_gradients(precision):
    if os.environ.get("ONNAX_TEST_MLX_EP") != "1":
        pytest.skip("Native MLX opt-in")
    from examples.lora.mlx_runtime import register_library, session

    if path := os.environ.get("ONNAX_MLX_EP_LIBRARY"):
        register_library(path)
    torch.set_num_threads(1)
    options = dict(
        mode="chunked", nf4=True, projection_precision=precision, session_factory=session
    )
    flat, _ = prepare(**options)
    grouped, _ = prepare(checkpoint_layers=1, **options)
    with torch.no_grad():
        for name, parameter in zip(flat.onnx.names, flat.onnx.adapters, strict=True):
            if name.endswith(".B"):
                parameter.add_(0.01)
    grouped.load_state_dict(flat.state_dict())
    losses = []
    for model in (flat, grouped):
        loss = model(**batch()).loss
        loss.backward()
        losses.append(loss.detach())
    torch.testing.assert_close(*losses, rtol=1e-4, atol=1e-6)
    for a, b in zip(flat.onnx.adapters, grouped.onnx.adapters, strict=True):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-6)
