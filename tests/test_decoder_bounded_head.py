# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""One bounded output-head path across decoder families and portable exports."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional training dependencies.

import copy
import json
import os

import numpy as np
import pytest

pytest.importorskip("mobius")
pytest.importorskip("trl")
torch = pytest.importorskip("torch")

from examples.huggingface.decoder_sft import OnnxDecoder, batch, prepare
from examples.huggingface.sft import token_loss
from examples.huggingface.validation import add_reference_parameters
from examples.lora.train import session


@pytest.fixture(params=["cpu", "mlx"])
def runtime_factory(request):
    if request.param == "cpu":
        return session
    if os.environ.get("ONNAX_TEST_MLX_EP") != "1":
        pytest.skip("Native MLX opt-in")
    from examples.lora.mlx_runtime import register_library
    from examples.lora.mlx_runtime import session as mlx_session

    if path := os.environ.get("ONNAX_MLX_EP_LIBRARY"):
        register_library(path)
    return mlx_session


@pytest.mark.parametrize("family", ["phi3", "qwen3", "qwen35"])
def test_bounded_family_gradients_updates_and_portable_export(
    tmp_path, family, runtime_factory
):
    torch.set_num_threads(1)
    model, reference = prepare(family, head_chunk_size=3, session_factory=runtime_factory)
    try:
        add_reference_parameters(reference, model.onnx.arrays(), model.mapping)
        data = batch()
        actual_optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
        reference_optimizer = torch.optim.SGD(
            [p for p in reference.parameters() if p.requires_grad], lr=0.03
        )
        for _ in range(3):
            actual_optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            actual = model(**data, num_items_in_batch=17)
            expected = reference(**data)
            loss = token_loss(expected.logits, data["labels"], num_items=17)
            torch.testing.assert_close(actual.loss, loss, rtol=1e-4, atol=1e-6)
            actual.loss.backward()
            loss.backward()
            for row in model.mapping:
                target = reference.get_submodule(row["module"])
                for key in ("A", "B"):
                    parameter = model.onnx.parameters_by_name()[row[key]]
                    torch.testing.assert_close(
                        parameter.grad, getattr(target, key).grad, rtol=1e-4, atol=1e-6
                    )
            actual_optimizer.step()
            reference_optimizer.step()
        with torch.no_grad():
            predicted = model(**{k: v for k, v in data.items() if k != "labels"}).logits
            torch.testing.assert_close(
                predicted, reference(**data).logits, rtol=1e-4, atol=1e-6
            )
        model.export(tmp_path / "export")
        task = json.loads((tmp_path / "export/task.json").read_text())
        assert task["version"] == 2
        inference = runtime_factory(
            str(tmp_path / "export/execution" / task["inference"]["model"])
        )
        values = {name: value.numpy() for name, value in data.items() if name != "labels"}
        np.testing.assert_allclose(
            inference.run(None, values)[0], predicted.numpy(), rtol=1e-4, atol=1e-6
        )
        restored = OnnxDecoder.load(tmp_path / "export", session_factory=runtime_factory)
        try:
            torch.testing.assert_close(
                restored(**data).loss, model(**data).loss, rtol=0, atol=0
            )
            restored.export(tmp_path / "again")
            state = copy.deepcopy(model.state_dict())
            before = {
                name: p.detach().clone()
                for name, p in restored.onnx.parameters_by_name().items()
            }
            state["_head_identity"][0] ^= 1
            with pytest.raises(ValueError, match="frozen-head identity"):
                restored.load_state_dict(state)
            for name, parameter in restored.onnx.parameters_by_name().items():
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        finally:
            restored.close()
    finally:
        model.close()


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5])
def test_invalid_head_size_rejected_before_building(monkeypatch, chunk_size):
    def unexpected_build(*args, **kwargs):
        pytest.fail("Invalid chunk size reached model construction")

    monkeypatch.setattr("examples.huggingface.decoder_sft.prepare_graph", unexpected_build)
    with pytest.raises(ValueError, match="head_chunk_size"):
        prepare("qwen3", head_chunk_size=chunk_size)


@pytest.mark.parametrize("change", ["identity", "vocab", "hidden", "weight_bytes", "path"])
def test_bounded_task_rejects_inconsistent_head(tmp_path, change):
    model, _ = prepare("phi3", head_chunk_size=3)
    try:
        model.export(tmp_path / "export")
    finally:
        model.close()
    directory = tmp_path / "export"
    if change in ("identity", "vocab", "hidden"):
        path = directory / "config.json"
        record = json.loads(path.read_text())
        if change == "identity":
            record["onnax_head"]["identity"] = "00" * 32
        else:
            record["vocab_size" if change == "vocab" else "hidden_size"] += 1
        path.write_text(json.dumps(record))
    elif change == "weight_bytes":
        record = json.loads((directory / "task.json").read_text())
        name = next(name for name in record["head"]["files"] if name.endswith(".data"))
        path = directory / "execution" / name
        data = bytearray(path.read_bytes())
        data[0] ^= 1
        path.write_bytes(data)
    else:
        path = directory / "task.json"
        record = json.loads(path.read_text())
        record["head"]["files"]["../outside.data"] = "00" * 32
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=r"head|Head|shape/rank"):
        OnnxDecoder.load(directory)


def test_loaded_head_owns_artifact_snapshot(tmp_path):
    import shutil

    model, _ = prepare("phi3", head_chunk_size=3)
    try:
        expected = model(**batch()).loss.detach()
        model.export(tmp_path / "export")
    finally:
        model.close()
    restored = OnnxDecoder.load(tmp_path / "export")
    snapshot = restored.head.directory.name
    try:
        shutil.rmtree(tmp_path / "export")
        torch.testing.assert_close(restored(**batch()).loss, expected, rtol=0, atol=0)
        restored(**batch()).loss.backward()
        assert all(p.grad is not None for p in restored.parameters())
    finally:
        restored.close()
    assert not os.path.exists(snapshot)
