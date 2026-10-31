# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Keep diagnostic controls separate from independent training state."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional dependencies.

import json
import sys

import pytest

pytest.importorskip("mobius")
pytest.importorskip("trl")
pytest.importorskip("peft")
torch = pytest.importorskip("torch")

from examples.huggingface import mlx_validation


@pytest.mark.parametrize("fail", [False, True])
def test_same_parameter_control_preserves_training_state(monkeypatch, fail):
    model = torch.nn.Linear(2, 1)
    reference = torch.nn.Linear(2, 1)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    gradients = [p.grad.clone() for p in model.parameters()]
    reference_state = {k: v.clone() for k, v in reference.state_dict().items()}

    def audit(actual, control, batch, *, sync, diagnose_rounding):
        assert control is not reference
        assert sync and diagnose_rounding
        for parameter in actual.parameters():
            parameter.grad = torch.zeros_like(parameter)
        with torch.no_grad():
            for parameter in control.parameters():
                parameter.zero_()
        if fail:
            raise RuntimeError("control failed")
        return {"loss": 1.0}

    monkeypatch.setattr(mlx_validation, "audit", audit)
    if fail:
        with pytest.raises(RuntimeError, match="control failed"):
            mlx_validation._same_parameter_control(model, reference, {}, 7)
    else:
        assert mlx_validation._same_parameter_control(model, reference, {}, 7) == {
            "after_updates": 7,
            "loss": 1.0,
        }
    for parameter, gradient in zip(model.parameters(), gradients, strict=True):
        torch.testing.assert_close(parameter.grad, gradient)
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(value, reference_state[name])


def test_cli_saves_failed_report_and_exits_nonzero(monkeypatch, tmp_path):
    report = {"strict_checks_passed": False, "checks": [{"loss": 1.0}]}

    def run(**kwargs):
        raise mlx_validation.ValidationError(report)

    output = tmp_path / "nested" / "report.json"
    monkeypatch.setattr(mlx_validation, "run", run)
    monkeypatch.setattr(sys, "argv", ["mlx_validation", "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        mlx_validation.main()
    assert error.value.code == 1
    assert json.loads(output.read_text()) == report
