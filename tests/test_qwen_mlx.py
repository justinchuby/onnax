# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Opt-in native MLX Qwen forward, gradients and independent updates."""

import os

import pytest


@pytest.mark.skipif(
    os.environ.get("ONNAX_TEST_MLX_EP") != "1", reason="Native MLX Qwen opt-in"
)
@pytest.mark.parametrize("mode", ["full", "chunked", "joint"])
def test_native_sft_checkpoint_resume(tmp_path, monkeypatch, mode):
    import subprocess
    import sys

    pytest.importorskip("mobius")
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    pytest.importorskip("onnxruntime_ep_mlx")
    from examples.huggingface import sft
    from examples.huggingface.replay import verify

    original = sft.make_trainer

    def trainer(*args, **kwargs):
        result = original(*args, **kwargs)
        result.args.save_steps = 2
        return result

    monkeypatch.setattr(sft, "make_trainer", trainer)
    first, resumed = str(tmp_path / "first"), str(tmp_path / "resumed")
    library = os.environ.get("ONNAX_MLX_EP_LIBRARY")
    report = sft.run(
        first,
        steps=4,
        mode=mode,
        nf4=True,
        save_activations=True,
        provider="mlx",
        library=library,
        segment_nodes=128,
        stable_reference=True,
    )
    assert report["cpu_fallback_disabled"]
    assert report["frozen_buffers_unchanged"]
    assert report["validation_loss"][1] < report["validation_loss"][0]
    assert all(row["strict_gradient_check_passed"] for row in report["audits"])
    assert all(
        row["strict_gradient_check_passed"]
        for row in report["stable_reference_audits"]["checks"]
    )
    command = [
        sys.executable,
        "-m",
        "examples.huggingface.sft",
        "--output-dir",
        resumed,
        "--steps",
        "4",
        "--mode",
        mode,
        "--nf4",
        "--save-activations",
        "--provider",
        "mlx",
        "--segment-nodes",
        "128",
        "--resume",
        os.path.join(first, "checkpoint-2"),
    ]
    if library:
        command.extend(["--library", library])
    subprocess.run(command, check=True, capture_output=True, text=True)
    replay = verify(first, resumed, step=4)
    assert replay["adapter_bitwise_equal"] and replay["optimizer_bitwise_equal"]
    if mode == "joint":
        assert replay["export_load_loss_gradients_bitwise_equal"]
        assert replay["exported_gradient_tensors"] == replay["adapter_tensors"]


@pytest.mark.skipif(
    os.environ.get("ONNAX_TEST_MLX_EP") != "1", reason="Native MLX Qwen opt-in"
)
@pytest.mark.parametrize("segmented", [False, True])
@pytest.mark.parametrize("mode", ["saved", "joint"])
def test_nf4_qwen_native_training(segmented, mode):
    pytest.importorskip("mobius")
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    pytest.importorskip("onnxruntime_ep_mlx")
    from examples.huggingface.mlx_validation import run

    run(
        library=os.environ.get("ONNAX_MLX_EP_LIBRARY"),
        segmented=segmented,
        mode=mode,
        stable_reference=mode == "joint",
    )


@pytest.mark.skipif(
    os.environ.get("ONNAX_TEST_MLX_EP") != "1", reason="Native MLX Qwen opt-in"
)
def test_native_same_input_layer_audit():
    pytest.importorskip("mobius")
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    pytest.importorskip("onnxruntime_ep_mlx")
    from examples.huggingface.layer_audit import run

    report = run(mlx=True, library=os.environ.get("ONNAX_MLX_EP_LIBRARY"))
    assert report["execution_provider"] == "MLXExecutionProvider"
    assert report["cpu_fallback_disabled"]
    assert report["logits"]["relative_l2"] < 1e-4
    for op in ("RMSNormalization", "Swish"):
        assert report["operator_summary"][op]["violations"] == 0
    dots = [row["worst_scaled_dot"] for row in report["operators"] if row["op"] == "MatMul"]
    assert dots and all("ort_mlx" in dot and "ort_cpu" not in dot for dot in dots)


@pytest.mark.skipif(
    os.environ.get("ONNAX_TEST_MLX_EP") != "1", reason="Native MLX Qwen opt-in"
)
def test_paired_primal_vjp_audit():
    pytest.importorskip("mobius")
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    pytest.importorskip("onnxruntime_ep_mlx")
    from examples.huggingface.paired_audit import run

    report = run(library=os.environ.get("ONNAX_MLX_EP_LIBRARY"), target_nodes=128)
    assert all(value["failed_tensors"] == 0 for value in report["same_primal_vjp"].values())
    assert report["independent_transformers"]["strict_gradient_check_passed"]
    assert report["independent_transformers_cpu"]["strict_gradient_check_passed"]
    assert report["primal_swap_cpu_vjp"]["failed_tensors"] == 0
    assert report["frozen_buffers_unchanged"]
    assert report["segments"]["vjp"] > 1
    assert report["logical_vjp_boundary_bytes"] > 0


@pytest.mark.skipif(
    os.environ.get("ONNAX_TEST_MLX_EP") != "1", reason="Native MLX Qwen opt-in"
)
@pytest.mark.parametrize("forward,backward", [("cpu", "mlx"), ("mlx", "cpu")])
def test_hybrid_stable_reference_updates(forward, backward):
    pytest.importorskip("mobius")
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    pytest.importorskip("onnxruntime_ep_mlx")
    from examples.huggingface.mlx_validation import run

    report = run(
        library=os.environ.get("ONNAX_MLX_EP_LIBRARY"),
        provider=forward,
        backward_provider=backward,
        stable_reference=True,
        steps=4,
    )
    assert report["strict_checks_passed"]
    assert report["provider"] == "hybrid"
    assert report["forward_provider"] == forward
    assert report["backward_provider"] == backward
    assert report["cpu_fallback_scope"] == "MLX sessions"
