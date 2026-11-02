# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Same-primal local derivatives across native decoder families."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional model dependencies.

import dataclasses

import pytest

pytest.importorskip("mobius")
pytest.importorskip("trl")
torch = pytest.importorskip("torch")

from examples.huggingface.decoder_sft import batch, prepare
from examples.huggingface.shared_vjp import shared_activation_audit


@pytest.mark.parametrize("family", ["phi3", "qwen3", "qwen35"])
@pytest.mark.parametrize("ignored", [False, True])
def test_shared_activation_gradients_and_caller_state(family, ignored):
    torch.set_num_threads(1)
    model, reference = prepare(family)
    executions = model.onnx.executions
    original_modules = {
        row["module"]: reference.get_submodule(row["module"]) for row in model.mapping
    }
    reference.train()
    flags = [parameter.requires_grad for parameter in reference.parameters()]
    try:
        with torch.no_grad():
            for row in model.mapping:
                model.onnx.parameters_by_name()[row["B"]].fill_(0.01)
        parameters = list(model.parameters())
        for parameter in parameters:
            parameter.grad = torch.ones_like(parameter)
        original = model.onnx.arrays()
        data = batch()
        if ignored:
            data["labels"].fill_(-100)
        report = shared_activation_audit(model, reference, data)
        assert report["decoder_gradients_passed"]
        assert len(report["layers"]) == model.config.num_hidden_layers
        assert report["shifted_targets"] == (0 if ignored else 6)
        assert report["frozen_suffix"]["pairwise"]["pointwise_violations"] == 0
        assert all(
            row["forward_valid_tokens"]["pointwise_violations"] == 0
            for row in report["layers"]
        )
        assert model.onnx.executions is executions
        for name, value in model.onnx.arrays().items():
            assert value.tobytes() == original[name].tobytes()
        for parameter in parameters:
            assert torch.equal(parameter.grad, torch.ones_like(parameter))
        assert all(not layer._forward_hooks for layer in reference.model.layers)
        assert all(not layer._forward_pre_hooks for layer in reference.model.layers)
        assert reference.training
        assert flags == [parameter.requires_grad for parameter in reference.parameters()]
        assert all(
            reference.get_submodule(name) is module
            for name, module in original_modules.items()
        )
        if family == "qwen35" and not ignored:
            assert shared_activation_audit(model, reference, data) == report
    finally:
        model.close()


def test_mismatched_local_derivative_is_reported():
    model, reference = prepare("qwen35")
    original = model.onnx.executions[1]

    class Perturbed:
        def run(self, outputs, feeds):
            values = original.backward.run(outputs, feeds)
            values[0] = values[0] + 0.01
            return values

    model.onnx.executions[1] = dataclasses.replace(original, backward=Perturbed())
    try:
        report = shared_activation_audit(model, reference, batch())
        assert not report["decoder_gradients_passed"]
        assert any(
            value["pointwise_violations"] > 0
            for value in report["layers"][0]["gradients"].values()
        )
    finally:
        model.close()


def test_exception_removes_reference_hooks_and_restores_sessions(monkeypatch):
    import examples.huggingface.shared_vjp as audit_module

    model, reference = prepare("qwen35")
    executions = model.onnx.executions
    original_modules = {
        row["module"]: reference.get_submodule(row["module"]) for row in model.mapping
    }

    def fail(*args, **kwargs):
        raise RuntimeError("injected reference failure")

    monkeypatch.setattr(audit_module, "_compare", fail)
    try:
        with pytest.raises(RuntimeError, match="injected reference"):
            shared_activation_audit(model, reference, batch())
        assert model.onnx.executions is executions
        assert all(not layer._forward_hooks for layer in reference.model.layers)
        assert all(not layer._forward_pre_hooks for layer in reference.model.layers)
        assert all(
            reference.get_submodule(name) is module
            for name, module in original_modules.items()
        )
        model(**batch()).loss.backward()
    finally:
        model.close()


def test_grouped_checkpoints_rejected_before_recording():
    model, reference = prepare("qwen35", checkpoint_layers=2)
    executions = model.onnx.executions
    try:
        with pytest.raises(ValueError, match="one checkpoint per decoder layer"):
            shared_activation_audit(model, reference, batch())
        assert model.onnx.executions is executions
    finally:
        model.close()
