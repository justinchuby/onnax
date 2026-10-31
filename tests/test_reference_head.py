# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Check FP32 head accumulation against scalar sums and autograd contracts."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_pairwise_cancellation_and_tail():
    from examples.huggingface.reference_head import pairwise_matmul

    left = torch.ones(1, 4097, dtype=torch.float32)
    left[:, :128] = 0
    left[0, 0], left[0, -1] = 1e8, -1e8
    right = torch.ones(4097, 1, dtype=torch.float32)
    expected = math.fsum(float(value) for value in left[0])
    actual = pairwise_matmul(left, right)
    assert actual.dtype == left.dtype
    assert actual.device == left.device
    assert actual.item() == expected == 3968


def test_head_forward_backward_and_restoration():
    from examples.huggingface.reference_head import pairwise_head

    torch.manual_seed(7)
    head = torch.nn.Linear(7, 257, bias=False).requires_grad_(False)
    reference = SimpleNamespace(lm_head=head)
    original = head.forward
    head.forward = original
    hidden = torch.randn(2, 3, 7, requires_grad=True)
    cotangent = torch.randn(2, 3, 257)
    before = head.weight.detach().clone()
    prediction = head(hidden).detach().clone()
    with pytest.raises(RuntimeError, match="probe failure"), pairwise_head(reference):
        with pairwise_head(reference):
            actual = head(hidden)
            torch.testing.assert_close(actual, prediction, rtol=0, atol=0)
            actual.backward(cotangent)
        assert head.forward is not original
        raise RuntimeError("probe failure")
    assert head.forward is original
    torch.testing.assert_close(head.weight, before, rtol=0, atol=0)
    assert head.weight.grad is None
    expected = np.empty((2, 3, 7), np.float32)
    for batch in range(2):
        for token in range(3):
            for column in range(7):
                expected[batch, token, column] = math.fsum(
                    float(a) * float(b)
                    for a, b in zip(cotangent[batch, token], before[:, column], strict=True)
                )
    np.testing.assert_allclose(hidden.grad.numpy(), expected, rtol=1e-5, atol=1e-6)


def test_trainable_head_is_rejected():
    from examples.huggingface.reference_head import pairwise_head

    reference = SimpleNamespace(lm_head=torch.nn.Linear(3, 5, bias=False))
    with pytest.raises(ValueError, match="frozen"), pairwise_head(reference):
        pass


def test_independent_tiny_trajectory_detects_parameter_drift():
    pytest.importorskip("mobius")
    pytest.importorskip("trl")
    from unittest.mock import patch

    from examples.huggingface.reference_trajectory import run

    baseline = run(nf4=True, steps=2, accumulation=2)
    assert baseline["frozen_weights_unchanged"]
    assert all(step["gradients"]["passed"] for step in baseline["steps"])
    assert all(step["parameters"]["passed"] for step in baseline["steps"])
    assert all(
        state["passed"]
        for step in baseline["steps"]
        for state in step["optimizer_state"].values()
    )
    original = torch.optim.AdamW.step
    calls = 0

    def perturb_reference(optimizer, *args, **kwargs):
        nonlocal calls
        result = original(optimizer, *args, **kwargs)
        calls += 1
        if calls == 2:
            with torch.no_grad():
                optimizer.param_groups[0]["params"][0].add_(0.01)
        return result

    with patch.object(torch.optim.AdamW, "step", perturb_reference):
        perturbed = run(nf4=True, steps=2, accumulation=2)
    assert all(not step["parameters"]["passed"] for step in perturbed["steps"])


def test_failed_trajectory_cli_keeps_report(tmp_path, monkeypatch):
    pytest.importorskip("mobius")
    pytest.importorskip("trl")
    from examples.huggingface import reference_trajectory

    output = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["reference_trajectory", "--output", str(output)])
    monkeypatch.setattr(
        reference_trajectory, "run", lambda **kwargs: {"checks": {"parameters": False}}
    )
    with pytest.raises(SystemExit) as error:
        reference_trajectory.main()
    assert error.value.code == 1
    import json

    assert json.loads(output.read_text())["checks"]["parameters"] is False


def test_tiny_head_scalar_oracle():
    pytest.importorskip("mobius")
    pytest.importorskip("trl")
    from examples.huggingface.head_reduction_audit import run

    report = run(nf4=True)
    assert report["compute_dtype"] == "float32"
    assert report["seed_shape"] == [1, 64, 256]
    assert report["samples"]
    for sample in report["samples"]:
        for error in sample["absolute_errors"].values():
            assert error <= 1e-6 + 1e-4 * abs(sample["scalar_fsum"])
