# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""One bounded output-head path across decoder families and portable exports."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional training dependencies.

import copy
import json
import os

import numpy as np
import onnx
import pytest

pytest.importorskip("mobius")
pytest.importorskip("trl")
torch = pytest.importorskip("torch")

from examples.huggingface.decoder_sft import OnnxDecoder, batch, prepare
from examples.huggingface.sft import token_loss
from examples.huggingface.validation import add_reference_parameters
from examples.lora.train import session


@pytest.fixture(params=["cpu", "mlx"])
def runtime_factory(request):
    if request.param == "cpu":
        return session
    if os.environ.get("ONNAX_TEST_MLX_EP") != "1":
        pytest.skip("Native MLX opt-in")
    from examples.lora.mlx_runtime import register_library
    from examples.lora.mlx_runtime import session as mlx_session

    if path := os.environ.get("ONNAX_MLX_EP_LIBRARY"):
        register_library(path)
    return mlx_session


@pytest.mark.parametrize("family", ["phi3", "qwen3", "qwen35"])
def test_bounded_family_gradients_updates_and_portable_export(
    tmp_path, family, runtime_factory
):
    torch.set_num_threads(1)
    model, reference = prepare(family, head_chunk_size=3, session_factory=runtime_factory)
    try:
        add_reference_parameters(reference, model.onnx.arrays(), model.mapping)
        data = batch()
        actual_optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
        reference_optimizer = torch.optim.SGD(
            [p for p in reference.parameters() if p.requires_grad], lr=0.03
        )
        for _ in range(3):
            actual_optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            actual = model(**data, num_items_in_batch=17)
            expected = reference(**data)
            loss = token_loss(expected.logits, data["labels"], num_items=17)
            torch.testing.assert_close(actual.loss, loss, rtol=1e-4, atol=1e-6)
            actual.loss.backward()
            loss.backward()
            for row in model.mapping:
                target = reference.get_submodule(row["module"])
                for key in ("A", "B"):
                    parameter = model.onnx.parameters_by_name()[row[key]]
                    torch.testing.assert_close(
                        parameter.grad, getattr(target, key).grad, rtol=1e-4, atol=1e-6
                    )
            actual_optimizer.step()
            reference_optimizer.step()
        with torch.no_grad():
            predicted = model(**{k: v for k, v in data.items() if k != "labels"}).logits
            torch.testing.assert_close(
                predicted, reference(**data).logits, rtol=1e-4, atol=1e-6
            )
        model.export(tmp_path / "export")
        task = json.loads((tmp_path / "export/task.json").read_text())
        assert task["version"] == 2
        inference = runtime_factory(
            str(tmp_path / "export/execution" / task["inference"]["model"])
        )
        values = {name: value.numpy() for name, value in data.items() if name != "labels"}
        np.testing.assert_allclose(
            inference.run(None, values)[0], predicted.numpy(), rtol=1e-4, atol=1e-6
        )
        restored = OnnxDecoder.load(tmp_path / "export", session_factory=runtime_factory)
        try:
            torch.testing.assert_close(
                restored(**data).loss, model(**data).loss, rtol=0, atol=0
            )
            restored.export(tmp_path / "again")
            state = copy.deepcopy(model.state_dict())
            before = {
                name: p.detach().clone()
                for name, p in restored.onnx.parameters_by_name().items()
            }
            state["_head_identity"][0] ^= 1
            with pytest.raises(ValueError, match="frozen-head identity"):
                restored.load_state_dict(state)
            for name, parameter in restored.onnx.parameters_by_name().items():
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        finally:
            restored.close()
    finally:
        model.close()


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5])
def test_invalid_head_size_rejected_before_building(monkeypatch, chunk_size):
    def unexpected_build(*args, **kwargs):
        pytest.fail("Invalid chunk size reached model construction")

    monkeypatch.setattr("examples.huggingface.decoder_sft.prepare_graph", unexpected_build)
    with pytest.raises(ValueError, match="head_chunk_size"):
        prepare("qwen3", head_chunk_size=chunk_size)


@pytest.mark.parametrize("change", ["identity", "vocab", "hidden", "weight_bytes", "path"])
def test_bounded_task_rejects_inconsistent_head(tmp_path, change):
    model, _ = prepare("phi3", head_chunk_size=3)
    try:
        model.export(tmp_path / "export")
    finally:
        model.close()
    directory = tmp_path / "export"
    if change in ("identity", "vocab", "hidden"):
        path = directory / "config.json"
        record = json.loads(path.read_text())
        if change == "identity":
            record["onnax_head"]["identity"] = "00" * 32
        else:
            record["vocab_size" if change == "vocab" else "hidden_size"] += 1
        path.write_text(json.dumps(record))
    elif change == "weight_bytes":
        record = json.loads((directory / "task.json").read_text())
        name = next(name for name in record["head"]["files"] if name.endswith(".data"))
        path = directory / "execution" / name
        data = bytearray(path.read_bytes())
        data[0] ^= 1
        path.write_bytes(data)
    else:
        path = directory / "task.json"
        record = json.loads(path.read_text())
        record["head"]["files"]["../outside.data"] = "00" * 32
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=r"head|Head|shape/rank"):
        OnnxDecoder.load(directory)


def test_loaded_head_owns_artifact_snapshot(tmp_path):
    import shutil

    model, _ = prepare("phi3", head_chunk_size=3)
    try:
        expected = model(**batch()).loss.detach()
        model.export(tmp_path / "export")
    finally:
        model.close()
    restored = OnnxDecoder.load(tmp_path / "export")
    snapshot = restored.head.directory.name
    try:
        shutil.rmtree(tmp_path / "export")
        torch.testing.assert_close(restored(**batch()).loss, expected, rtol=0, atol=0)
        restored(**batch()).loss.backward()
        assert all(p.grad is not None for p in restored.parameters())
    finally:
        restored.close()
    assert not os.path.exists(snapshot)


@pytest.mark.parametrize("change", ["downgrade", "missing_config", "unexpected_config"])
def test_task_head_mode_rejected_before_loading_sessions(tmp_path, monkeypatch, change):
    from examples.huggingface import decoder_sft

    model, _ = prepare("phi3", head_chunk_size=None if change == "unexpected_config" else 3)
    try:
        model.export(tmp_path / "export")
    finally:
        model.close()
    directory = tmp_path / "export"
    if change == "downgrade":
        path = directory / "task.json"
        record = json.loads(path.read_text())
        record["version"] = 1
    else:
        path = directory / "config.json"
        record = json.loads(path.read_text())
        if change == "missing_config":
            del record["onnax_head"]
        else:
            record["onnax_head"] = {"identity": "00" * 32}
    path.write_text(json.dumps(record))

    def unexpected_load(*args, **kwargs):
        pytest.fail("Invalid head mode reached session loading")

    monkeypatch.setattr(decoder_sft.CheckpointedModule, "load", unexpected_load)
    with pytest.raises(ValueError, match="head execution mode"):
        OnnxDecoder.load(directory)


@pytest.mark.parametrize("family", ["phi3", "qwen3", "qwen35"])
def test_separate_head_session_factory_survives_export(tmp_path, family):
    calls = {"body": 0, "head": 0}

    def body_factory(source):
        calls["body"] += 1
        return session(source)

    def head_factory(source):
        calls["head"] += 1
        return session(source)

    model, _ = prepare(
        family,
        nf4=True,
        head_chunk_size=3,
        session_factory=body_factory,
        head_session_factory=head_factory,
    )
    try:
        data = batch()
        expected = model(**data).loss
        expected.backward()
        gradients = [p.grad.clone() for p in model.parameters()]
        # The lazy logits session must also use the selected head backend.
        expected_logits = model(**{k: v for k, v in data.items() if k != "labels"}).logits
        assert calls["body"] > 0 and calls["head"] == 2
        model.export(tmp_path / "export")
    finally:
        model.close()
    calls.update(body=0, head=0)
    restored = OnnxDecoder.load(
        tmp_path / "export",
        session_factory=body_factory,
        head_session_factory=head_factory,
    )
    try:
        actual = restored(**data).loss
        actual.backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for parameter, gradient in zip(restored.parameters(), gradients, strict=True):
            torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
        logits = restored(**{k: v for k, v in data.items() if k != "labels"}).logits
        torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
        assert calls["body"] > 0 and calls["head"] == 2
    finally:
        restored.close()


def test_head_session_factory_requires_a_head():
    with pytest.raises(ValueError, match="requires head_chunk_size"):
        prepare("phi3", head_session_factory=session)


@pytest.mark.parametrize(
    "options,global_disable,message",
    [
        ([], "0", "requires MLX and --head-chunk-size"),
        (["--provider", "mlx"], "0", "requires MLX and --head-chunk-size"),
        (
            ["--provider", "mlx", "--head-chunk-size", "3"],
            "1",
            "requires ONNXRUNTIME_EP_MLX_NO_COMPILE=0 or unset",
        ),
    ],
)
def test_compile_head_cli_rejects_incompatible_options(
    tmp_path, options, global_disable, message
):
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.huggingface.decoder_sft",
            "--output-dir",
            str(tmp_path),
            "--compile-head",
            *options,
        ],
        env={**os.environ, "ONNXRUNTIME_EP_MLX_NO_COMPILE": global_disable},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and message in result.stderr


@pytest.mark.parametrize("family", ["phi3", "qwen3", "qwen35"])
def test_native_compiled_head_with_eager_decoder(tmp_path, family, monkeypatch):
    if os.environ.get("ONNAX_TEST_MLX_EP") != "1":
        pytest.skip("Native MLX opt-in")
    import functools

    from examples.huggingface.decoder_sft import audit
    from examples.lora.mlx_runtime import register_library
    from examples.lora.mlx_runtime import session as mlx_session

    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_NO_COMPILE", "0")
    if path := os.environ.get("ONNAX_MLX_EP_LIBRARY"):
        register_library(path)
    calls = {False: 0, True: 0}

    def factory(source, *, compiled):
        executor = mlx_session(source, enable_compile=compiled)
        assert executor.get_session_options().get_session_config_entry(
            "ep.mlx.enable_compile"
        ) == str(int(compiled))
        calls[compiled] += 1
        return executor

    report = audit(
        family,
        str(tmp_path),
        nf4=True,
        head_chunk_size=3,
        steps=3,
        session_factory=functools.partial(factory, compiled=False),
        head_session_factory=functools.partial(factory, compiled=True),
        fused_ops=True,
    )
    assert calls[False] > 0 and calls[True] > 0
    assert report["reload"] and report["peft"]["gradient_check_passed"]
    assert report["peft"]["update_check_passed"]


def _assert_projection_close(actual, expected, precision):
    if precision == "bf16":
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
        return
    # Match the existing Qwen FP16 qualification budget: one FP16 epsilon
    # on vector norm and maximum magnitude, with FP32 absolute floor.
    delta = (actual.detach() - expected.detach()).float()
    scale = expected.detach().float()
    epsilon = torch.finfo(torch.float16).eps
    assert torch.linalg.vector_norm(delta) <= epsilon * torch.linalg.vector_norm(scale) + 1e-6
    assert delta.abs().max() <= epsilon * scale.abs().max() + 1e-6


@pytest.mark.parametrize("provider", ["reference", "mlx"])
@pytest.mark.parametrize("family", ["phi3", "qwen3", "qwen35"])
@pytest.mark.parametrize("precision", ["bf16", "fp16"])
@pytest.mark.parametrize("nf4", [False, True])
def test_half_family_updates_and_portable_export(tmp_path, family, precision, nf4, provider):
    from onnx.reference import ReferenceEvaluator

    from examples.huggingface.mixed_precision import reference_projections

    def factory(path):
        return ReferenceEvaluator(str(path))

    if provider == "reference" and family == "qwen35":
        pytest.skip("ONNX ReferenceEvaluator does not execute reverse Scan")
    if provider == "mlx":
        if os.environ.get("ONNAX_TEST_MLX_EP") != "1":
            pytest.skip("Native MLX opt-in")
        from examples.lora.mlx_runtime import register_library, session

        if path := os.environ.get("ONNAX_MLX_EP_LIBRARY"):
            register_library(path)
        factory = session

    torch.set_num_threads(1)
    model, reference = prepare(
        family,
        head_chunk_size=3,
        session_factory=factory,
        projection_precision=precision,
        nf4=nf4,
    )
    try:
        with torch.no_grad():
            for name, p in model.onnx.parameters_by_name().items():
                if name.endswith(".B"):
                    p.fill_(0.01)
        reference = reference_projections(
            add_reference_parameters(reference, model.onnx.arrays(), model.mapping),
            precision=precision,
        )
        optimizers = [
            torch.optim.AdamW(
                [p for p in m.parameters() if p.requires_grad], lr=0.003, eps=1e-5
            )
            for m in (model, reference)
        ]
        frozen = {
            n: p.detach().clone()
            for n, p in reference.named_parameters()
            if not p.requires_grad
        }
        data = batch()
        for _ in range(3):
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            actual = model(**data)
            expected = reference(
                **{k: v for k, v in data.items() if k != "labels"}, use_cache=False
            ).logits
            expected_loss = token_loss(expected, data["labels"])
            torch.testing.assert_close(actual.loss, expected_loss, rtol=1e-4, atol=1e-6)
            actual.loss.backward()
            expected_loss.backward()
            for row in model.mapping:
                target = reference.get_submodule(row["module"])
                for key in ("A", "B"):
                    p = model.onnx.parameters_by_name()[row[key]]
                    assert p.dtype == p.grad.dtype == torch.float32
                    _assert_projection_close(p.grad, getattr(target, key).grad, precision)
            for optimizer in optimizers:
                optimizer.step()
            for row in model.mapping:
                target = reference.get_submodule(row["module"])
                for key in ("A", "B"):
                    _assert_projection_close(
                        model.onnx.parameters_by_name()[row[key]],
                        getattr(target, key),
                        precision,
                    )
        assert all(
            torch.equal(p, frozen[n])
            for n, p in reference.named_parameters()
            if not p.requires_grad
        )
        model.export(tmp_path / "export")
        restored = OnnxDecoder.load(tmp_path / "export", session_factory=factory)
        try:
            assert restored.projection_precision == precision
            for m in (model, restored):
                m.zero_grad()
                m(**data).loss.backward()
            for name, p in model.onnx.parameters_by_name().items():
                torch.testing.assert_close(
                    p.grad, restored.onnx.parameters_by_name()[name].grad, rtol=0, atol=0
                )
            inputs = {k: v for k, v in data.items() if k != "labels"}
            torch.testing.assert_close(
                restored(**inputs).logits, model(**inputs).logits, rtol=0, atol=0
            )
            onnx.checker.check_model(
                str(tmp_path / "export" / "execution" / "inference.onnx"), full_check=True
            )
        finally:
            restored.close()
    finally:
        model.close()
