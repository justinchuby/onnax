# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""PEFT interchange and bounded-token ONNX loss regressions."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional integration dependencies.

import numpy as np
import pytest

pytest.importorskip("mobius")
pytest.importorskip("peft")
pytest.importorskip("trl")
torch = pytest.importorskip("torch")

from peft import PeftModel

from examples.huggingface.chunked_loss import ChunkedHead
from examples.huggingface.peft_adapter import export_adapter, import_adapter
from examples.huggingface.qwen import reference_model
from examples.huggingface.sft import prepare, token_loss


@pytest.mark.parametrize("chunk", [1, 7, 128])
def test_chunked_head_loss_and_hidden_gradient(chunk):
    torch.manual_seed(17)
    torch.set_num_threads(1)
    weight = torch.randn(8, 31)
    inputs = torch.randn(2, 6, 8, requires_grad=True)
    labels = torch.randint(31, (2, 6))
    labels[1, -2:] = -100
    expected = token_loss(inputs @ weight, labels)
    expected.backward()
    actual_inputs = inputs.detach().clone().requires_grad_()
    head = ChunkedHead(weight.numpy(), chunk_size=chunk)
    actual = head(actual_inputs, labels)
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(actual_inputs.grad, inputs.grad, rtol=1e-4, atol=1e-6)
    empty = head(actual_inputs, torch.full_like(labels, -100))
    assert empty.item() == 0


def test_peft_two_way_logits_gradients_and_update(tmp_path):
    torch.set_num_threads(1)
    model, _ = prepare()
    with torch.no_grad():
        for n, p in zip(model.onnx.names, model.onnx.adapters, strict=True):
            if n.endswith(".B"):
                p.add_(0.01)
    export_adapter(model, str(tmp_path / "onnx"))
    peft = PeftModel.from_pretrained(
        reference_model(), str(tmp_path / "onnx"), is_trainable=True
    )
    peft.eval()
    ids = torch.tensor([[1, 4, 5, 6, 2]])
    mask = torch.ones_like(ids)
    labels = ids.clone()
    a = model(input_ids=ids, attention_mask=mask, labels=labels)
    b = peft(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
    torch.testing.assert_close(a.logits, b.logits, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(a.loss, b.loss, rtol=1e-4, atol=1e-6)
    a.loss.backward()
    b.loss.backward()
    params = dict(peft.named_parameters())
    for row in model.mapping:
        for key in ("A", "B"):
            expected = params[
                f"base_model.model.{row['module']}.lora_{key}.default.weight"
            ].grad.T
            actual = model.onnx.adapters[model.onnx.names.index(row[key])].grad
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
    torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0).step()
    torch.optim.AdamW(
        [p for p in peft.parameters() if p.requires_grad], lr=0.003, weight_decay=0
    ).step()
    expected_arrays = model.onnx.arrays()
    peft.save_pretrained(str(tmp_path / "pytorch"))
    with torch.no_grad():
        for p in model.onnx.adapters:
            p.zero_()
    import_adapter(model, str(tmp_path / "pytorch"))
    for name, expected in expected_arrays.items():
        np.testing.assert_allclose(model.onnx.arrays()[name], expected, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(
        model(input_ids=ids, attention_mask=mask).logits,
        peft(input_ids=ids, attention_mask=mask).logits,
        rtol=1e-4,
        atol=1e-6,
    )


def test_chunked_body_full_loss_and_adapter_gradients():
    full, _ = prepare()
    chunked, _ = prepare(mode="chunked", chunk_size=7)
    ids = torch.tensor([[1, 4, 5, 6, 2], [1, 7, 8, 0, 0]])
    mask = ids.ne(0).long()
    labels = ids.clone()
    labels[mask == 0] = -100
    a = full(input_ids=ids, attention_mask=mask, labels=labels)
    b = chunked(input_ids=ids, attention_mask=mask, labels=labels)
    torch.testing.assert_close(a.loss, b.loss, rtol=1e-4, atol=1e-6)
    a.loss.backward()
    b.loss.backward()
    for x, y in zip(full.onnx.adapters, chunked.onnx.adapters, strict=True):
        torch.testing.assert_close(x.grad, y.grad, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(
        a.logits, chunked(input_ids=ids, attention_mask=mask).logits, rtol=1e-4, atol=1e-6
    )


def test_fp64_graph_against_independent_functional_oracle():
    import onnx_ir as ir

    import onnax
    from examples.huggingface.optimization import optimize_training_graph
    from examples.huggingface.precision import double_graph, qwen64
    from examples.lora.train import session

    model, reference = prepare()
    graph = double_graph(model.forward_graph)
    arrays = model.onnx.arrays()
    ids = torch.tensor([[1, 4, 5, 6, 2]])
    batch = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "position_ids": torch.arange(5)[None],
    }
    feeds = {
        **{n: v.numpy() for n, v in batch.items()},
        **{n: v.astype(np.float64) for n, v in arrays.items()},
    }
    prediction = session(ir.to_proto(graph).SerializeToString()).run(["logits"], feeds)[0]
    expected, params = qwen64(reference, model.forward_graph, arrays, batch)
    np.testing.assert_allclose(prediction, expected.detach(), rtol=1e-4, atol=1e-6)
    actual = torch.tensor(prediction, requires_grad=True)
    token_loss(actual, ids).backward()
    gradient = optimize_training_graph(
        onnax.vjp(graph, inputs=list(arrays), outputs=["logits"])
    )
    feeds[gradient.graph.inputs[-1].name] = actual.grad.numpy()
    values = session(ir.to_proto(gradient).SerializeToString()).run(
        [v.name for v in gradient.graph.outputs[2:]], feeds
    )
    token_loss(expected, ids).backward()
    for name, value in zip(arrays, values, strict=True):
        np.testing.assert_allclose(value, params[name].grad, rtol=1e-4, atol=1e-6)


def test_saved_activations_match_recomputed_parameter_updates():
    torch.set_num_threads(1)
    recompute, _ = prepare()
    saved, _ = prepare(save_activations=True)
    ids = torch.tensor([[1, 4, 5, 6, 2], [1, 7, 8, 2, 0]])
    mask = ids.ne(0).long()
    labels = ids.clone()
    labels[mask == 0] = -100
    optimizers = [
        torch.optim.AdamW(m.parameters(), lr=0.003, weight_decay=0) for m in (recompute, saved)
    ]
    for _ in range(3):
        for model, update in zip((recompute, saved), optimizers, strict=True):
            update.zero_grad()
            model(input_ids=ids, attention_mask=mask, labels=labels).loss.backward()
            update.step()
        for a, b in zip(recompute.onnx.adapters, saved.onnx.adapters, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert saved.onnx.residual_names
    assert saved.onnx.last_residual_bytes > 0


def test_precision_conversion_preserves_source_types_and_payloads():
    import onnx_ir as ir

    from examples.huggingface.precision import double_graph

    model, _ = prepare()
    before = ir.to_proto(model.forward_graph).SerializeToString()
    double_graph(model.forward_graph)
    assert ir.to_proto(model.forward_graph).SerializeToString() == before


def test_invalid_labels_and_atomic_adapter_import(tmp_path):
    from safetensors.torch import load_file, save_file

    head = ChunkedHead(np.ones((2, 3), np.float32))
    with pytest.raises(ValueError, match="vocabulary IDs"):
        head(torch.ones(1, 3, 2), torch.tensor([[0, -2, 1]]))
    model, _ = prepare()
    export_adapter(model, str(tmp_path))
    before = model.onnx.arrays()
    values = load_file(str(tmp_path / "adapter_model.safetensors"))
    names = list(values)
    values[names[0]] += 0.25
    values[names[-1]] = values[names[-1]].half()
    save_file(values, str(tmp_path / "adapter_model.safetensors"))
    with pytest.raises(ValueError, match="shape/dtype"):
        import_adapter(model, str(tmp_path))
    for name, value in model.onnx.arrays().items():
        np.testing.assert_array_equal(value, before[name])


@pytest.mark.parametrize("diagnose", [False, True])
def test_peft_gradient_diagnostic_retains_failures(tmp_path, diagnose):
    from examples.huggingface.peft_adapter import roundtrip

    model, reference = prepare()
    name = model.onnx.names[0]
    model.onnx.adapters[0].register_hook(lambda gradient: gradient + 0.01)
    ids = torch.tensor([[1, 4, 5, 2]])
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}
    if diagnose:
        report = roundtrip(model, reference, batch, str(tmp_path), diagnose_rounding=True)
        assert not report["gradient_check_passed"]
        assert report["gradient_failures"][name]["violations"] > 0
        assert not report["update_check_passed"]
    else:
        with pytest.raises(AssertionError):
            roundtrip(model, reference, batch, str(tmp_path))
