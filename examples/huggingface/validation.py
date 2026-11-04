# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Independent PyTorch Qwen3 LoRA forward and derivative checks."""

import numpy as np
import torch

from examples.huggingface.sft import token_loss


def precision_comparison(actual, expected, control=None, *, rtol=1e-4, atol=1e-6):
    """Compare backend error with the measured effect of projection precision.

    The control runs FP32 arithmetic on the same frozen weight values. The L2
    backend error uses that reference precision effect plus the original FP32
    comparison floor. Maximum error and strict coordinate violations are recorded.
    """

    def array(value):
        return (
            value.detach().cpu().numpy()
            if isinstance(value, torch.Tensor)
            else np.asarray(value)
        )

    a, b = array(actual), array(expected)
    c = b if control is None else array(control)
    if a.shape != b.shape or c.shape != b.shape or a.dtype != b.dtype or c.dtype != b.dtype:
        raise AssertionError("Precision comparison shapes or dtypes differ")
    if not all(np.isfinite(value).all() for value in (a, b, c)):
        raise FloatingPointError("Nonfinite precision comparison")
    error, effect = np.abs(a - b), np.abs(b - c)
    floor = atol + rtol * np.abs(b)

    def maximum(value):
        return float(np.max(value, initial=0))

    def norm(value):
        return float(np.linalg.norm(value.reshape(-1)))

    result = {
        "max_abs_error": maximum(error),
        "l2_error": norm(error),
        "reference_precision_max_abs": maximum(effect),
        "reference_precision_l2": norm(effect),
        "max_abs_reference_scale": maximum(effect) + maximum(floor),
        "l2_limit": norm(effect) + norm(floor),
        "strict_violations": int(np.count_nonzero(error > floor)),
        "rtol": rtol,
        "atol": atol,
        "fp32_floor_l2": norm(floor),
    }
    result["max_abs_within_reference_scale"] = (
        result["max_abs_error"] <= result["max_abs_reference_scale"]
    )
    result["comparison_policy"] = (
        "fp32_allclose" if control is None else "reference_precision_effect_l2"
    )
    if control is None:
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
    elif result["l2_error"] > result["l2_limit"]:
        raise AssertionError(f"Backend error exceeds measured precision effect: {result}")
    return result


class TorchLora(torch.nn.Module):
    def __init__(self, base, a, b, scale):
        super().__init__()
        self.base = base
        self.A = torch.nn.Parameter(torch.from_numpy(a.copy()))
        self.B = torch.nn.Parameter(torch.from_numpy(b.copy()))
        self.scale = scale

    def forward(self, x):
        return self.base(x) + self.scale * ((x @ self.A) @ self.B)


def add_reference_adapters(reference, model):
    return add_reference_parameters(reference, model.onnx.arrays(), model.mapping)


def add_reference_parameters(reference, arrays, mapping):
    """Install adapters from the source parameter arrays and module mapping."""
    for p in reference.parameters():
        p.requires_grad_(False)
    for row in mapping:
        parent, name = row["module"].rsplit(".", 1)
        module = reference.get_submodule(parent)
        setattr(
            module,
            name,
            TorchLora(
                getattr(module, name),
                arrays[row["A"]],
                arrays[row["B"]],
                row["alpha"] / row["rank"],
            ),
        )
    reference.eval()
    return reference


def _finite_pair(actual, expected, name):
    if actual.shape != expected.shape:
        raise AssertionError(
            f"Reference shape mismatch for {name}: {actual.shape} != {expected.shape}"
        )
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError(f"Nonfinite {name}")


def audit(model, reference, batch, *, sync=True, diagnose_rounding=False, reference_loss=None):
    arrays = model.onnx.arrays()
    for row in model.mapping:
        layer = reference.get_submodule(row["module"])
        with torch.no_grad():
            if not sync:
                continue
            layer.A.copy_(torch.from_numpy(arrays[row["A"]]))
            layer.B.copy_(torch.from_numpy(arrays[row["B"]]))
    model.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    inputs = {n: v for n, v in batch.items() if n != "labels"}
    actual = model(**batch)
    if actual.logits is None:
        with torch.no_grad():
            actual.logits = model(**inputs).logits
    expected = reference(**inputs, use_cache=False).logits
    expected_loss = (reference_loss or token_loss)(expected, batch["labels"])
    valid = batch["attention_mask"].bool()
    actual_values = actual.logits.detach()[valid].numpy()
    expected_values = expected.detach()[valid].numpy()
    _finite_pair(actual_values, expected_values, "logits")
    _finite_pair(actual.loss.detach().numpy(), expected_loss.detach().numpy(), "loss")
    difference = np.abs(actual_values - expected_values)
    violations = difference > 1e-6 + 1e-4 * np.abs(expected_values)
    if not diagnose_rounding:
        np.testing.assert_allclose(actual_values, expected_values, rtol=1e-4, atol=1e-6)
    loss_passed = bool(
        np.isclose(
            actual.loss.detach().numpy(), expected_loss.detach().numpy(), rtol=1e-4, atol=1e-6
        )
    )
    if not diagnose_rounding:
        np.testing.assert_allclose(
            actual.loss.detach(), expected_loss.detach(), rtol=1e-4, atol=1e-6
        )
    actual.loss.backward()
    expected_loss.backward()
    maximum = 0.0
    gradient_failures = {}
    for row in model.mapping:
        layer = reference.get_submodule(row["module"])
        for key in ("A", "B"):
            index = model.onnx.names.index(row[key])
            gradient = model.onnx.adapters[index].grad.numpy()
            expected_gradient = getattr(layer, key).grad.numpy()
            _finite_pair(gradient, expected_gradient, f"gradient: {row[key]}")
            maximum = max(maximum, float(np.max(np.abs(gradient - expected_gradient))))
            failed = np.abs(gradient - expected_gradient) > 1e-6 + 1e-4 * np.abs(
                expected_gradient
            )
            if np.any(failed):
                gradient_failures[row[key]] = {
                    "violations": int(np.count_nonzero(failed)),
                    "max_abs": float(np.max(np.abs(gradient - expected_gradient))),
                }
            if not diagnose_rounding:
                np.testing.assert_allclose(
                    gradient, expected_gradient, rtol=1e-4, atol=1e-6, err_msg=row[key]
                )

    return {
        "strict_loss_check_passed": loss_passed,
        "reference_loss": float(expected_loss.detach()),
        "strict_logit_check_passed": bool(
            np.isfinite(actual_values).all()
            and np.isfinite(expected_values).all()
            and not np.any(violations)
        ),
        "logit_violations": int(np.count_nonzero(violations)),
        "logits_relative_l2": float(
            np.linalg.norm(actual_values - expected_values) / np.linalg.norm(expected_values)
        ),
        "logits_max_abs": float((actual.logits[valid] - expected[valid]).detach().abs().max()),
        "gradients_max_abs": maximum,
        "strict_gradient_check_passed": not bool(gradient_failures),
        "gradient_failures": gradient_failures,
        "loss": float(actual.loss.detach()),
    }
