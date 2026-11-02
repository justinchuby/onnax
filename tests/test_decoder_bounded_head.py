# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""One bounded output-head path across decoder families and portable exports."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional training dependencies.

import copy
import json

import numpy as np
import pytest

pytest.importorskip("mobius")
pytest.importorskip("trl")
torch = pytest.importorskip("torch")

from examples.huggingface.decoder_sft import OnnxDecoder, batch, prepare
from examples.huggingface.sft import token_loss
from examples.huggingface.validation import add_reference_parameters
from examples.lora.train import session


@pytest.mark.parametrize("family", ["phi3", "qwen3", "qwen35"])
def test_bounded_family_gradients_updates_and_portable_export(tmp_path, family):
    torch.set_num_threads(1)
    model, reference = prepare(family, head_chunk_size=3)
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
        inference = session(str(tmp_path / "export/execution" / task["inference"]["model"]))
        values = {name: value.numpy() for name, value in data.items() if name != "labels"}
        np.testing.assert_allclose(
            inference.run(None, values)[0], predicted.numpy(), rtol=1e-4, atol=1e-6
        )
        restored = OnnxDecoder.load(tmp_path / "export")
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
