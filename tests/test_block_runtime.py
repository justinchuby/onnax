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


@pytest.mark.parametrize("checkpoint", [False, True])
def test_split_replay_qualifies_captured_gradients(recorded, checkpoint):
    from examples.huggingface.block_recompute import measure as measure_split

    bundle, captures, captured = recorded
    result = measure_split(
        bundle,
        captures,
        "decoder_0_1",
        factory=session,
        warmup=0,
        samples=1,
        checkpoint=checkpoint,
    )
    assert set(result["gradient_checks"]) == set(
        captured["blocks"]["decoder_0_1"]["output_names"]
    )
    assert result["gradient_checks"] == result["final_gradient_checks"]
    assert (result["retained_activation_bytes"] == 0) == checkpoint
    assert result["plan"]["recomputed_values"] == []
    assert result["timing_seconds"]["paired"]["median"] > 0


def test_recorded_training_batch_preserves_padding_and_prompt_mask(tmp_path):
    from examples.huggingface.block_runtime import load_batch

    arrays = {
        "input_ids": np.array([[1, 2, 3, 0], [3, 1, 0, 0]], np.int64),
        "attention_mask": np.array([[1, 1, 1, 0], [1, 1, 0, 0]], np.int64),
        "position_ids": np.broadcast_to(np.arange(4, dtype=np.int64), (2, 4)),
        "labels": np.array([[-100, 2, 3, -100], [-100, 1, -100, -100]], np.int64),
    }
    path = tmp_path / "batch.npz"
    np.savez(path, **arrays)
    actual = load_batch(path, vocab_size=4, max_sequence=8)
    for key in arrays:
        np.testing.assert_array_equal(actual[key], arrays[key])


@pytest.mark.parametrize(
    "fault",
    ["keys", "dtype", "shape", "context", "ids", "labels", "mask", "padding", "positions"],
)
def test_invalid_recorded_training_batch_is_rejected(tmp_path, fault):
    from examples.huggingface.block_runtime import load_batch

    arrays = {
        key: np.ones((1, 4), np.int64)
        for key in ("input_ids", "labels", "attention_mask", "position_ids")
    }
    if fault == "keys":
        arrays.pop("labels")
    elif fault == "dtype":
        arrays["input_ids"] = arrays["input_ids"].astype(np.int32)
    elif fault == "shape":
        arrays["labels"] = arrays["labels"][:, :2]
    elif fault == "context":
        arrays = {key: np.ones((1, 9), np.int64) for key in arrays}
    elif fault == "ids":
        arrays["input_ids"][0, 0] = 4
    elif fault == "labels":
        arrays["labels"][0, 0] = -1
    elif fault == "mask":
        arrays["attention_mask"][0, 0] = 2
    elif fault == "padding":
        arrays["attention_mask"][0, 0] = 0
    elif fault == "positions":
        arrays["position_ids"][0, 0] = 8
    path = tmp_path / "batch.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        load_batch(path, vocab_size=4, max_sequence=8)


def test_shape_vector_saved_block_qualifies_captured_gradients(recorded):
    from examples.huggingface.block_recompute import measure as measure_split

    bundle, captures, _ = recorded
    result = measure_split(
        bundle, captures, "decoder_0_1", factory=session, save_shapes=True, warmup=0, samples=1
    )
    assert result["plan"]["shape_boundaries"]["compressed"]
    assert result["gradient_checks"] == result["final_gradient_checks"]
