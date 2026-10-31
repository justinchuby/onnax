# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Exercise independent training-quality comparisons and paired initialization."""

import json

import numpy as np
import pytest

pytest.importorskip("mobius")
pytest.importorskip("trl")
pytest.importorskip("torch")


def test_independent_quality_runs_share_initialization_and_data(tmp_path):
    from examples.huggingface.training_impact import run

    report = run(
        engines=("torch_cpu", "onnx_reference"),
        seeds=(17, 29),
        steps=2,
        interval=1,
        microbatch=2,
        accumulation=2,
        capture_path=tmp_path / "snapshots.npz",
    )
    assert report["frozen_unchanged"]
    assert report["capture"]["parameter_order"] == sorted(report["capture"]["parameter_order"])
    with np.load(tmp_path / "snapshots.npz", allow_pickle=False) as arrays:
        assert len(arrays.files) == 12
        assert arrays["torch_cpu__17__parameters"].dtype == np.float32
        assert (
            arrays["torch_cpu__17__gradient_1"].shape
            == arrays["onnx_reference__17__gradient_1"].shape
        )
    assert report["train_lines"] > 0 and report["heldout_lines"] > 0
    assert len(report["trials"]) == 4
    for trial in report["trials"]:
        assert len(trial["training_loss"]) == 2
        assert trial["final_training_loss"] < trial["initial_training_loss"]
        assert np.isfinite(trial["final_validation_loss"])
    for comparison in report["comparisons"]:
        assert comparison["gradient_snapshots"][0]["cosine"] > 0.9999
        assert np.isfinite(comparison["parameters"]["relative_l2"])
    assert len(report["validation_summary"]["torch_cpu"]["values"]) == 2

    from examples.huggingface.training_quality_compare import compare

    paths = []
    for engine in report["engines"]:
        component = {
            **report,
            "engines": [engine],
            "trials": [row for row in report["trials"] if row["engine"] == engine],
        }
        path = tmp_path / f"{engine}.json"
        path.write_text(json.dumps(component))
        paths.append(path)
    combined = compare(paths, [tmp_path / "snapshots.npz"] * 2)
    assert combined["validation_summary"] == {
        engine: {
            key: report["validation_summary"][engine][key] for key in ("values", "mean", "std")
        }
        for engine in report["engines"]
    }
    for actual, expected in zip(combined["comparisons"], report["comparisons"], strict=True):
        assert actual["gradient_snapshots"] == expected["gradient_snapshots"]
        assert actual["parameters"] == expected["parameters"]
    from examples.huggingface.training_quality_replay import replay

    replays = []
    for engine in report["engines"]:
        result = replay(
            paths,
            [tmp_path / "snapshots.npz"] * 2,
            engine=engine,
            capture_path=tmp_path / f"{engine}-replay.npz",
        )
        assert result["frozen_unchanged"]
        assert len(result["states"]) == 4
        for state in result["states"]:
            if state["source_engine"] == engine:
                assert state["validation_loss_gap_from_source"] == pytest.approx(0, abs=1e-6)
        (tmp_path / f"{engine}-replay.json").write_text(json.dumps(result))
        replays.append(result)
    with (
        np.load(tmp_path / "torch_cpu-replay.npz") as a,
        np.load(tmp_path / "onnx_reference-replay.npz") as b,
    ):
        from examples.huggingface.training_impact import _vector_metrics

        for left, right in zip(replays[0]["states"], replays[1]["states"], strict=True):
            assert left["parameter_sha256"] == right["parameter_sha256"]
            key = left["gradient_key"]
            assert _vector_metrics(a[key], b[key])["cosine"] > 0.9999
    from examples.huggingface.training_quality_replay_compare import compare_replays

    replay_paths = [tmp_path / f"{engine}-replay.json" for engine in report["engines"]]
    capture_paths = [tmp_path / f"{engine}-replay.npz" for engine in report["engines"]]
    shared = compare_replays(replay_paths, capture_paths)
    assert len(shared["comparisons"]) == 4
    assert all(row["gradient"]["cosine"] > 0.9999 for row in shared["comparisons"])
    replays[1]["states"][0]["parameter_sha256"] = "different parameters"
    replay_paths[1].write_text(json.dumps(replays[1]))
    with pytest.raises(ValueError, match="different parameter states"):
        compare_replays(replay_paths, capture_paths)
    curve = component["trials"][0]["validation_curve"]
    original = curve[1]["loss"]
    curve[1]["loss"] = float("nan")
    paths[-1].write_text(json.dumps(component))
    with pytest.raises(ValueError, match="Nonfinite validation curve"):
        compare(paths, [tmp_path / "snapshots.npz"] * 2)
    curve[1]["loss"] = original
    final_loss = component["trials"][0]["final_validation_loss"]
    component["trials"][0]["final_validation_loss"] = final_loss + 1
    paths[-1].write_text(json.dumps(component))
    with pytest.raises(ValueError, match="Validation endpoints"):
        compare(paths, [tmp_path / "snapshots.npz"] * 2)
    component["trials"][0]["final_validation_loss"] = final_loss
    omitted = curve.pop(1)
    paths[-1].write_text(json.dumps(component))
    with pytest.raises(ValueError, match="Incomplete or unordered"):
        compare(paths, [tmp_path / "snapshots.npz"] * 2)
    curve.insert(1, omitted)
    component["ort_optimizer"] = "enabled"
    paths[-1].write_text(json.dumps(component))
    with pytest.raises(ValueError, match="ORT optimizer disabled"):
        compare(paths, [tmp_path / "snapshots.npz"] * 2)
    component["ort_optimizer"] = "disabled"
    component["torch_float32_matmul_precision"] = "high"
    paths[-1].write_text(json.dumps(component))
    with pytest.raises(ValueError, match="highest FP32 matmul"):
        compare(paths, [tmp_path / "snapshots.npz"] * 2)
    component["torch_float32_matmul_precision"] = "highest"
    engines = component["engines"]
    component["engines"] = ["torch_cuda"]
    component["cuda_bf16_reduced_precision_reduction"] = True
    paths[-1].write_text(json.dumps(component))
    with pytest.raises(ValueError, match="CUDA BF16 comparisons require FP32"):
        compare(paths, [tmp_path / "snapshots.npz"] * 2)
    component["engines"] = engines
    component["cuda_bf16_reduced_precision_reduction"] = None
    component["trials"][0]["schedule_sha256"] = "different schedule"
    paths[-1].write_text(json.dumps(component))
    with pytest.raises(ValueError, match="schedule_sha256"):
        compare(paths, [tmp_path / "snapshots.npz"] * 2)


def test_gradient_direction_distinguishes_scale_and_sign():
    from examples.huggingface.training_impact import _vector_metrics

    x = np.array([1, 2, 3], np.float32)
    assert _vector_metrics(x, x * 2)["cosine"] == pytest.approx(1)
    assert _vector_metrics(x, -x)["cosine"] == pytest.approx(-1)
    assert _vector_metrics(x, x * 2)["relative_l2"] == pytest.approx(0.5)
    assert _vector_metrics(np.zeros_like(x), x)["cosine"] == 0


def test_custom_corpus_split_and_replay(tmp_path):
    from examples.huggingface.sft import tokenizer
    from examples.huggingface.training_impact import _batch, _tokenizer_digest, run
    from examples.huggingface.training_quality_replay import replay

    lines = [
        "Amber birds glide over quiet water.",
        "Silver fish swim beneath clear water.",
        "Quiet winds move across amber fields.",
        "Unseenword cafés glow beside silver water.",
    ]
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    tok = tokenizer(training_texts=lines[:3])
    assert "Unseenword" not in tok.get_vocab()
    digest = _tokenizer_digest(tok)
    _batch(tok, lines)
    assert _tokenizer_digest(tok) == digest
    capture = tmp_path / "custom.npz"
    report = run(
        engines=("torch_cpu", "onnx_reference"),
        seeds=(17,),
        steps=1,
        microbatch=1,
        accumulation=2,
        interval=1,
        corpus_path=corpus,
        train_lines=3,
        capture_path=capture,
    )
    assert report["train_lines"] == 3 and report["heldout_lines"] == 1
    assert report["tokenizer_sha256"] == digest
    paths = []
    for engine in report["engines"]:
        component = {
            **report,
            "engines": [engine],
            "trials": [r for r in report["trials"] if r["engine"] == engine],
        }
        path = tmp_path / f"{engine}.json"
        path.write_text(json.dumps(component))
        paths.append(path)
    result = replay(
        paths,
        [capture] * 2,
        engine="torch_cpu",
        corpus_path=corpus,
        capture_path=tmp_path / "replay.npz",
    )
    assert result["frozen_unchanged"]
    own = next(row for row in result["states"] if row["source_engine"] == "torch_cpu")
    assert own["validation_loss_gap_from_source"] == pytest.approx(0, abs=1e-6)
    for path in paths:
        component = json.loads(path.read_text())
        component["tokenizer_sha256"] = "different tokenizer"
        path.write_text(json.dumps(component))
    with pytest.raises(ValueError, match="Replay tokenizer differs"):
        replay(
            paths,
            [capture] * 2,
            engine="torch_cpu",
            corpus_path=corpus,
            capture_path=tmp_path / "bad-tokenizer.npz",
        )
    corpus.write_text("\n".join([*lines[:-1], "Changed held-out text."]), encoding="utf-8")
    with pytest.raises(ValueError, match="Replay corpus differs"):
        replay(
            paths,
            [capture] * 2,
            engine="torch_cpu",
            corpus_path=corpus,
            capture_path=tmp_path / "bad.npz",
        )
    for split in (0, len(lines)):
        with pytest.raises(ValueError, match="split must contain"):
            run(corpus_path=corpus, train_lines=split, engines=("torch_cpu",))
