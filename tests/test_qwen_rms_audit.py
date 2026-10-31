# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Independent reference controls must leave the training model intact."""

import pytest

pytest.importorskip("mobius")
pytest.importorskip("trl")
torch = pytest.importorskip("torch")


def test_control_restores_reference_on_failure():
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

    from examples.huggingface.rms_audit import division_control

    norm = Qwen3RMSNorm(16)
    x = torch.randn(2, 3, 16)
    original = norm.forward
    expected = norm(x).detach().clone()
    # Preserve an existing instance override as well as class-provided forwards.
    norm.forward = original
    with (
        pytest.raises(RuntimeError, match="probe failure"),
        division_control(norm) as count,
    ):
        assert count == 1
        with division_control(norm):
            torch.testing.assert_close(norm(x), expected, rtol=1e-4, atol=1e-6)
        raise RuntimeError("probe failure")
    assert norm.forward is original
    torch.testing.assert_close(norm(x), expected, rtol=0, atol=0)


def test_tiny_nf4_reference_control():
    from examples.huggingface.rms_audit import run

    report = run(nf4=True)
    assert report["normalizations"] == 9
    assert report["adapter_updates"] == 0
    assert report["ort_memory_reuse"] is False
    for audit in report["audits"].values():
        assert audit["strict_gradient_check_passed"]
        assert audit["strict_logit_check_passed"]


@pytest.mark.parametrize("kind", ["rms", "trajectory"])
def test_diagnostic_runtime_options_reach_both_sessions(monkeypatch, kind):
    from examples.huggingface import reference_trajectory, rms_audit

    module = rms_audit if kind == "rms" else reference_trajectory
    original = module.diagnostic_session
    observed = []

    def factory(source):
        executor = original(source)
        options = executor.get_session_options()
        assert options.enable_mem_reuse is False
        assert options.intra_op_num_threads == options.inter_op_num_threads == 1
        observed.append(executor)
        return executor

    monkeypatch.setattr(module, "diagnostic_session", factory)
    if kind == "rms":
        report = module.run(nf4=True)
    else:
        report = module.run(nf4=True, steps=2, accumulation=2)
    assert report["ort_memory_reuse"] is False
    assert len(observed) == 2
