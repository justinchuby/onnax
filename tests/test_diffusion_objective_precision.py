# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Objective metadata and checkpoint recovery preserve the selected compute policy."""

import json

import onnx_ir as ir
import pytest

pytest.importorskip("torch")
from examples.diffusion.objective import read_objective, validate_checkpoint_objective


@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp16"])
def test_objective_policy_matches_saved_source(tmp_path, precision):
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[1])
    source = ir.Model(
        ir.Graph([x], [x], nodes=[], opset_imports={"": 23}, name="precision_manifest"),
        ir_version=10,
    )
    if precision != "fp32":
        source.metadata_props["onnax.example.precision"] = json.dumps(
            {
                "version": 1,
                "projection_compute": "bfloat16" if precision == "bf16" else "float16",
            }
        )
    ir.save(source, tmp_path / "model.onnx")
    interface = {"identity": "base", "files": {"model.onnx": "base-sha"}}
    record = dict(
        version=1,
        base_identity="base",
        base_model_sha256="base-sha",
        objective_identity="loss-identity",
        reduction="mean_squared_error",
        projection_precision=precision,
    )
    (tmp_path / "objective.json").write_text(json.dumps(record))
    assert read_objective(tmp_path, interface) == record
    record["projection_precision"] = "fp16" if precision != "fp16" else "bf16"
    (tmp_path / "objective.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="projection precision"):
        read_objective(tmp_path, interface)


@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp16"])
def test_checkpoint_objective_identity_cannot_cross_policies(precision):
    interface = {"identity": "base"}
    objective = {"objective_identity": "loss-" + precision, "projection_precision": precision}
    validate_checkpoint_objective(
        {"objective_identity": objective["objective_identity"]}, interface, objective
    )
    with pytest.raises(ValueError, match="different training objective"):
        validate_checkpoint_objective(
            {"objective_identity": "another-loss"}, interface, objective
        )
    if precision == "fp32":
        validate_checkpoint_objective({}, interface, objective)
    else:
        with pytest.raises(ValueError, match="requires a checkpoint objective identity"):
            validate_checkpoint_objective({}, interface, objective)


def test_original_prediction_workload_keeps_legacy_recovery():
    interface = {"identity": "base"}
    validate_checkpoint_objective({}, interface)
    validate_checkpoint_objective({"objective_identity": "base"}, interface)
    with pytest.raises(ValueError, match="different training objective"):
        validate_checkpoint_objective({"objective_identity": "loss"}, interface)


def _benchmark_artifacts(tmp_path, precision):
    import numpy as np

    common = dict(
        model="demo",
        revision="pinned",
        export_identity="base",
        adapter_state={"adapter.A": {"shape": [2], "dtype": "float32"}},
        data={},
        dtype="float32",
        latent_shape=[1, 4, 8, 8],
        conditioning_shape=[1, 3, 24],
        learning_rate=0.0,
        optimizer="AdamW",
        optimizer_epsilon=1e-5,
        warmup=0,
        samples=1,
        full_step_seconds={"median": 1.0, "samples": 1},
        physical_footprint={},
        peak_process_rss_bytes=0,
        provider="test",
        projection_precision=precision,
    )
    paths = [str(tmp_path / name) for name in ("baseline.json", "candidate.json")]
    for path in paths:
        with open(path, "w") as stream:
            json.dump(common, stream)
        np.savez(
            path + ".npz",
            prediction=np.ones(1, np.float32),
            loss=np.asarray(1, np.float32),
            **{"adapter.A": np.ones(2, np.float32)},
        )
    return paths, common


@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp16"])
def test_benchmark_uses_selected_precision_without_crossing_policies(tmp_path, precision):
    import numpy as np

    from examples.diffusion.compare_unet import compare

    paths, report = _benchmark_artifacts(tmp_path, precision)
    np.savez(
        paths[1] + ".npz",
        prediction=np.ones(1, np.float32),
        loss=np.asarray(1, np.float32),
        **{"adapter.A": np.full(2, 1.0004, np.float32)},
    )
    if precision == "fp32":
        with pytest.raises(AssertionError, match="Tensor norm mismatch"):
            compare(*paths)
    else:
        result = compare(*paths)
        assert result["projection_precision"] == precision
        assert result["checks"]["adapter.A"]["pointwise_tolerance_failures"] == 2
        np.savez(
            paths[1] + ".npz",
            prediction=np.ones(1, np.float32),
            loss=np.asarray(1, np.float32),
            **{"adapter.A": np.full(2, 1.2, np.float32)},
        )
        with pytest.raises(AssertionError, match="Combined gradient norm"):
            compare(*paths)
    report["projection_precision"] = "fp16" if precision != "fp16" else "bf16"
    with open(paths[1], "w") as stream:
        json.dump(report, stream)
    with pytest.raises(ValueError, match="projection_precision"):
        compare(*paths)


@pytest.mark.parametrize("precision", ["bf16", "fp16"])
@pytest.mark.parametrize("needs", [(False, False), (True, False), (False, True), (True, True)])
def test_reference_retains_only_operands_needed_for_selected_gradients(precision, needs):
    import torch

    from examples.torch_precision import projection_product

    torch.manual_seed(19)
    a = torch.randn(5, 7, requires_grad=needs[0])
    b = torch.randn(7, 3, requires_grad=needs[1])
    g = torch.randn(5, 3)
    saved = []

    def pack(value):
        saved.append((tuple(value.shape), value.dtype))
        return value

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        output = projection_product(precision).apply(a, b)
    expected_saved = []
    if needs[1]:
        expected_saved.append(((5, 7), torch.float32))
    if needs[0]:
        expected_saved.append(((7, 3), torch.float32))
    assert saved == expected_saved
    if any(needs):
        (output * g).sum().backward()
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        if needs[0]:
            torch.testing.assert_close(
                a.grad, g @ b.detach().to(dtype).float().T, rtol=1e-4, atol=1e-6
            )
        if needs[1]:
            torch.testing.assert_close(
                b.grad, a.detach().to(dtype).float().T @ g, rtol=1e-4, atol=1e-6
            )
