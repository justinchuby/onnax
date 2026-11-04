# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Exercise the saved-boundary CLI offline against an independent analytic VJP."""

import json

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("mobius")

from examples.diffusion import saved_shapes
from examples.huggingface.block_runtime import digest


def test_saved_workload_exports_and_executes_without_reference_weights(tmp_path, monkeypatch):
    import huggingface_hub
    import torch

    export = tmp_path / "source" / "export"
    export.mkdir(parents=True)
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=["batch", 3])
    w = ir.val("w", dtype=ir.DataType.FLOAT, shape=[3, 2])
    tape = ir.tape.Tape()
    projection = tape.op("MatMul", [x, w])
    projection.name = "projection"
    projection.dtype, projection.shape = ir.DataType.FLOAT, ir.Shape(["batch", 2])
    y = tape.op("Tanh", [projection])
    y.name, y.dtype, y.shape = "prediction", ir.DataType.FLOAT, ir.Shape(["batch", 2])
    model = ir.Model(
        ir.Graph(
            [x, w],
            [y],
            nodes=tape.nodes,
            opset_imports={"": 18},
            name="offline_saved_workload",
        ),
        ir_version=10,
    )
    ir.save(model, export / "model.onnx")
    initial = np.arange(6, dtype=np.float32).reshape(3, 2) / 20
    interface = {
        "identity": "offline-workload",
        "files": {"model.onnx": digest(export / "model.onnx")},
    }
    monkeypatch.setattr(saved_shapes, "load_state", lambda _: ({}, interface, {"w": initial}))
    scheduler = tmp_path / "scheduler.json"
    scheduler.write_text("{}")

    def artifact(model_id, filename, *, revision):
        assert model_id == saved_shapes.MODEL_ID and revision == saved_shapes.REVISION
        assert filename == "scheduler/scheduler_config.json"
        return str(scheduler)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", artifact)
    monkeypatch.setattr(saved_shapes, "hf_hub_download", artifact)
    monkeypatch.setattr(saved_shapes, "register_library", lambda _: None)

    def session(path):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 1
        return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    monkeypatch.setattr(saved_shapes, "session", session)

    class Memory:
        def __init__(self, _):
            pass

        def reset_peak(self):
            pass

        def snapshot(self):
            return {"active_bytes": 0, "cache_bytes": 0, "peak_bytes": 0}

    monkeypatch.setattr(saved_shapes, "MlxMemory", Memory)
    monkeypatch.setattr(
        saved_shapes,
        "training_batch",
        lambda clean, conditioning, noise, timesteps, scheduler: ({"x": clean}, noise),
    )
    inputs = np.arange(6, dtype=np.float32).reshape(2, 3) / 10
    target = np.full((2, 2), 0.2, np.float32)
    batch = tmp_path / "batch.pt"
    torch.save(
        {
            "clean": torch.from_numpy(inputs),
            "noise": torch.from_numpy(target),
            "conditioning": torch.zeros(1),
            "timesteps": torch.zeros(1),
        },
        batch,
    )
    output = tmp_path / "measured"
    report = saved_shapes.measure(
        str(export.parent),
        str(batch),
        str(output),
        library=str(scheduler),
        fused_pullbacks=True,
        save_shapes=True,
        warmup=1,
        samples=2,
    )
    assert report["gradients_checked"] == 1
    assert report["timing_seconds"]["paired"]["samples"] == 2
    assert report["shape_vector_adjoints"] is True
    assert json.loads((output / "report.json").read_text())["inputs"] == report["inputs"]
    forward, backward = session(output / "forward.onnx"), session(output / "backward.onnx")
    feeds = {"x": inputs, "w": initial}
    values = dict(
        zip(
            (value.name for value in forward.get_outputs()),
            forward.run(None, feeds),
            strict=True,
        )
    )
    prediction = np.tanh(inputs @ initial)
    np.testing.assert_allclose(values["prediction"], prediction, rtol=1e-6, atol=1e-6)
    (seed,) = json.loads(
        ir.load(output / "joint.onnx").graph.metadata_props["onnax.autodiff.interface"]
    )["seeds"]
    cotangent = (prediction - target) * np.float32(2 / target.size)
    available = {**feeds, **values, seed: cotangent}
    actual = backward.run(
        None, {value.name: available[value.name] for value in backward.get_inputs()}
    )[0]
    wanted = inputs.T @ (cotangent * (1 - prediction**2))
    np.testing.assert_allclose(actual, wanted, rtol=1e-4, atol=1e-6)
