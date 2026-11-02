# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Replay real checkpoint arguments without reconstructing unrelated sessions."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional training dependencies.

import json

import numpy as np
import pytest

pytest.importorskip("mobius")
pytest.importorskip("trl")
torch = pytest.importorskip("torch")

from examples.huggingface.block_liveness import report as logical_report
from examples.huggingface.block_runtime import block_rows, capture, measure
from examples.huggingface.decoder_sft import batch, prepare
from examples.lora.train import session


@pytest.fixture(params=["phi3", "qwen35"])
def recorded(tmp_path, request):
    torch.set_num_threads(1)
    model, _ = prepare(request.param, session_factory=session)
    export = tmp_path / "export"
    model.export(export)
    model.close()
    output = tmp_path / "capture"
    result = capture(
        export,
        names=["decoder_0_1", "decoder_1_2"],
        batch=batch(),
        output=output,
        factory=session,
    )
    return export / "execution", output, result


def test_reused_block_session_replays_actual_gradient_inputs(recorded):
    bundle, captures, captured = recorded
    created = []

    def factory(source):
        created.append(source)
        return session(source)

    for name in captured["blocks"]:
        report = measure(bundle, captures, name, factory=factory, warmup=1, samples=3)
        assert len(report["samples"]) == 3
        assert report["allocator_after"] is None
        assert report["incremental_allocator_peak_bytes"] is None
        assert set(report["outputs"]) == set(captured["blocks"][name]["output_names"])
        assert all(row["max_abs"] == 0 for row in report["outputs"].values())
        assert report["final_outputs"] == report["outputs"]
        assert report["input_arrays"] == captured["blocks"][name]["input_arrays"]
    assert len(created) == 2


def test_changed_capture_is_rejected_before_session_creation(recorded):
    bundle, captures, captured = recorded
    path = captures / captured["blocks"]["decoder_0_1"]["feeds"]
    with np.load(path) as arrays:
        values = {key: arrays[key] for key in arrays.files}
    key = next(key for key, value in values.items() if value.dtype == np.float32)
    values[key] = values[key] + np.float32(1)
    np.savez(path, **values)

    def forbidden(source):
        pytest.fail("Invalid captures must not create a session")

    with pytest.raises(ValueError, match="recorded hashes"):
        measure(bundle, captures, "decoder_0_1", factory=forbidden, samples=1)


def test_wrong_block_model_is_rejected(recorded):
    bundle, captures, _ = recorded
    path = captures / "capture.json"
    data = json.loads(path.read_text())
    data["blocks"]["decoder_0_1"]["model"] = "block_2_backward.onnx"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="block interface"):
        measure(bundle, captures, "decoder_0_1", factory=session, samples=1)


def test_logical_analysis_uses_the_executed_graph_and_captured_shapes(recorded):
    bundle, captures, captured = recorded
    for name in captured["blocks"]:
        result = logical_report(bundle, captures, name)
        assert result["input_arrays"] == captured["blocks"][name]["input_arrays"]
        assert result["liveness"]["status"] == "complete"
        assert result["dimensions"]["batch"] == batch()["input_ids"].shape[0]
        assert result["liveness"]["peak_total_tensor_bytes"] > 0


def test_no_new_allocation_keeps_the_initial_live_memory_in_the_window(recorded):
    class Counter:
        def reset_peak(self):
            pass

        def snapshot(self):
            return {"active_bytes": 64, "cache_bytes": 0, "peak_bytes": 0}

    bundle, captures, _ = recorded
    result = measure(
        bundle, captures, "decoder_0_1", factory=session, memory=Counter(), warmup=0, samples=1
    )
    assert result["window_allocator_peak_bytes"] == 64
    assert result["incremental_allocator_peak_bytes"] == 0


@pytest.mark.parametrize("names", [[], ["missing"], ["layer", "layer"]])
def test_block_selection_requires_unique_differentiable_names(names):
    with pytest.raises(ValueError):
        block_rows({"blocks": [{"name": "layer", "backward": True}]}, names)
