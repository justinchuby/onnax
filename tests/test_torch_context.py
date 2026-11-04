# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Independent full-logits and bounded PyTorch training paths."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional example dependencies.

import copy

import pytest

pytest.importorskip("mobius")
pytest.importorskip("trl")
torch = pytest.importorskip("torch")

from examples.huggingface.benchmark import measure
from examples.huggingface.qwen import reference_model
from examples.huggingface.sft import token_loss
from examples.huggingface.torch_context import checkpoint_decoder, chunked_token_loss
from examples.huggingface.validation import TorchLora


def model():
    result = reference_model()
    for parameter in result.parameters():
        parameter.requires_grad_(False)
    for layer in result.model.layers:
        for name in ("q_proj", "v_proj"):
            base = getattr(layer.self_attn, name)
            a = torch.randn(base.in_features, 4).numpy() * 0.02
            b = torch.randn(4, base.out_features).numpy() * 0.01
            setattr(layer.self_attn, name, TorchLora(base, a, b, 0.25))
    return result


def inputs():
    ids = torch.tensor([[1, 4, 5, 6, 2], [1, 7, 2, 0, 0]])
    mask = ids.ne(0).long()
    labels = ids.masked_fill(mask == 0, -100)
    return {"input_ids": ids, "attention_mask": mask}, labels


@pytest.mark.parametrize("group_size", [1, 2, 4])
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_grouped_chunked_gradients_and_updates(group_size, precision, attention):
    torch.set_num_threads(1)
    full = model()
    full.set_attn_implementation(attention)
    if precision == "bf16":
        from examples.huggingface.mixed_precision import reference_projections

        reference_projections(full)
    grouped = checkpoint_decoder(copy.deepcopy(full), group_size)
    batch, labels = inputs()
    optimizers = [
        torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=0.003)
        for m in (full, grouped)
    ]
    for _ in range(3):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss = token_loss(full(**batch, use_cache=False).logits, labels)
        # Eight shifted rows exercise a tail chunk and a padded batch boundary.
        bounded = chunked_token_loss(grouped, batch, labels, 7)
        torch.testing.assert_close(loss, bounded, rtol=1e-4, atol=1e-6)
        loss.backward()
        bounded.backward()
        a = [p for p in full.parameters() if p.requires_grad]
        b = [p for p in grouped.parameters() if p.requires_grad]
        for first, second in zip(a, b, strict=True):
            torch.testing.assert_close(first.grad, second.grad, rtol=1e-4, atol=1e-6)
        for optimizer in optimizers:
            optimizer.step()
        for first, second in zip(a, b, strict=True):
            torch.testing.assert_close(first, second, rtol=1e-4, atol=1e-6)
    for p in grouped.parameters():
        if not p.requires_grad:
            assert p.grad is None
    grouped.zero_grad(set_to_none=True)
    zero = chunked_token_loss(grouped, batch, torch.full_like(labels, -100), 1)
    assert zero.item() == 0
    zero.backward()
    assert all(p.grad.count_nonzero() == 0 for p in grouped.parameters() if p.requires_grad)


def test_benchmark_checkpoint_policy_and_input_identity():
    options = dict(samples=1, warmup=0, batch_size=1, sequence_length=5)
    full = measure("pytorch", **options)
    grouped = measure(
        "pytorch", checkpoint_layers=2, torch_chunked=True, chunk_size=3, **options
    )
    assert grouped["input_ids_sha256"] == full["input_ids_sha256"]
    assert grouped["head_chunk_size"] == 3
    assert len(grouped["timing_samples_ms"]) == 1
    sample = grouped["timing_samples_ms"][0]
    assert sample["step"] == pytest.approx(
        sample["forward_and_loss"] + sample["backward"] + sample["adamw"]
    )
    assert grouped["provider"] == "pytorch_cpu"
    assert grouped["checkpoint_layers"] == 2
    assert grouped["gradient_and_optimizer_dtype"] == "float32"
    assert grouped["losses"] == pytest.approx(full["losses"], rel=1e-4, abs=1e-6)
    with pytest.raises(ValueError, match="require pytorch"):
        measure("chunked", torch_device="mps", **options)


@pytest.mark.parametrize("family", ["phi3", "qwen3", "qwen35"])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_per_layer_checkpoints_preserve_family_updates(family, attention):
    import numpy as np

    from examples.huggingface.decoder_families import prepare_graph
    from examples.huggingface.validation import add_reference_parameters

    torch.set_num_threads(1)
    _, arrays, mapping, reference = prepare_graph(family)
    rng = np.random.default_rng(912)
    for row in mapping:
        arrays[row["B"]] = rng.normal(0, 0.01, arrays[row["B"]].shape).astype(np.float32)
    full = add_reference_parameters(reference, arrays, mapping)
    full.set_attn_implementation(attention)
    grouped = copy.deepcopy(full)
    original_layers = list(grouped.model.layers)
    calls = []
    hook = original_layers[0].register_forward_pre_hook(lambda *_: calls.append(1))
    grouped = checkpoint_decoder(grouped, 1)
    assert len(grouped.model.layers) == len(original_layers)
    assert not grouped.training
    batch, labels = inputs()
    optimizers = [
        torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=0.003)
        for m in (full, grouped)
    ]
    frozen = {
        n: p.detach().clone() for n, p in grouped.named_parameters() if not p.requires_grad
    }
    for _ in range(3):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss = token_loss(full(**batch, use_cache=False).logits, labels)
        bounded = chunked_token_loss(grouped, batch, labels, 3)
        torch.testing.assert_close(bounded, loss, rtol=1e-4, atol=1e-6)
        loss.backward()
        bounded.backward()
        first = [p for p in full.parameters() if p.requires_grad]
        second = [p for p in grouped.parameters() if p.requires_grad]
        for a, b in zip(first, second, strict=True):
            torch.testing.assert_close(b.grad, a.grad, rtol=1e-4, atol=1e-6)
        for optimizer in optimizers:
            optimizer.step()
        for a, b in zip(first, second, strict=True):
            torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-6)
    hook.remove()
    assert len(calls) >= 6  # Recompute in eval mode as well as the three forward calls.
    for name, parameter in grouped.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None
            assert torch.equal(parameter, frozen[name])
    with pytest.raises(ValueError, match="already"):
        checkpoint_decoder(grouped, 1)


def test_hybrid_multi_layer_groups_keep_original_layout_on_rejection():
    from examples.huggingface.decoder_families import reference_model

    model = reference_model("qwen35")
    layers = list(model.model.layers)
    with pytest.raises(ValueError, match="one-layer"):
        checkpoint_decoder(model, 2)
    assert list(model.model.layers) == layers
