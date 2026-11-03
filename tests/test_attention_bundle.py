# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Deployed pullbacks train, load and re-export through the common task bridge."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional training dependencies.

import json
import os

import pytest

pytest.importorskip("mobius")
pytest.importorskip("trl")
torch = pytest.importorskip("torch")

from examples.huggingface.attention_bundle import deploy, digest
from examples.huggingface.decoder_replay import tensor_manifest
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


@pytest.mark.parametrize("family", ["qwen3", "qwen35", "phi3"])
@pytest.mark.parametrize("broadcast_adjoints", [False, True])
def test_deployed_decoder_gradients_updates_and_reexport(
    tmp_path, family, runtime_factory, broadcast_adjoints
):
    torch.set_num_threads(1)
    original, reference = prepare(
        family, head_chunk_size=3, fused_ops=True, session_factory=session
    )
    source = tmp_path / "source"
    original.export(source)
    before = {
        str(path.relative_to(source)): digest(path)
        for path in source.rglob("*")
        if path.is_file()
    }
    report = deploy(source, tmp_path / "deployed", broadcast_adjoints=broadcast_adjoints)
    if broadcast_adjoints:
        assert sum(row["broadcast_adjoint_calls"] for row in report["blocks"]) > 0
    assert report["blocks"] and report["models_checked"] == len(report["blocks"])
    assert before == {
        str(path.relative_to(source)): digest(path)
        for path in source.rglob("*")
        if path.is_file()
    }
    deployed = OnnxDecoder.load(tmp_path / "deployed", session_factory=runtime_factory)
    try:
        add_reference_parameters(reference, deployed.onnx.arrays(), deployed.mapping)
        # Use the decoder task's existing native Adam epsilon policy.
        actual = torch.optim.AdamW(deployed.parameters(), lr=0.003, eps=1e-5)
        control = torch.optim.AdamW(original.parameters(), lr=0.003, eps=1e-5)
        expected = torch.optim.AdamW(
            [p for p in reference.parameters() if p.requires_grad], lr=0.003, eps=1e-5
        )
        data = batch()
        for _ in range(3):
            for optimizer in (actual, control, expected):
                optimizer.zero_grad()
            left = deployed(**data).loss
            right = original(**data).loss
            loss = token_loss(reference(**data).logits, data["labels"])
            torch.testing.assert_close(left, loss, rtol=1e-4, atol=1e-6)
            torch.testing.assert_close(left, right, rtol=1e-4, atol=1e-6)
            left.backward()
            right.backward()
            loss.backward()
            for row in deployed.mapping:
                target = reference.get_submodule(row["module"])
                for key in ("A", "B"):
                    parameter = deployed.onnx.parameters_by_name()[row[key]]
                    torch.testing.assert_close(
                        parameter.grad, getattr(target, key).grad, rtol=1e-4, atol=1e-6
                    )
                    torch.testing.assert_close(
                        parameter.grad,
                        original.onnx.parameters_by_name()[row[key]].grad,
                        rtol=1e-4,
                        atol=1e-6,
                    )
            for optimizer in (actual, control, expected):
                optimizer.step()
            for row in deployed.mapping:
                target = reference.get_submodule(row["module"])
                for key in ("A", "B"):
                    torch.testing.assert_close(
                        deployed.onnx.parameters_by_name()[row[key]],
                        getattr(target, key),
                        rtol=1e-4,
                        atol=1e-6,
                    )
        deployed.export(tmp_path / "again")
        names = [row[key] for row in deployed.mapping for key in ("A", "B")]
        assert tensor_manifest(
            tmp_path / "deployed/execution", exclude_initializers=names
        ) == tensor_manifest(tmp_path / "again/execution", exclude_initializers=names)
        restored = OnnxDecoder.load(tmp_path / "again", session_factory=runtime_factory)
        try:
            torch.testing.assert_close(
                restored(**data).loss, deployed(**data).loss, rtol=0, atol=0
            )
            assert any(
                node.domain == "pkg.onnax" and node.op_type.startswith("AttentionVJP_v1_")
                for execution in restored.onnx.executions
                if execution.derivative is not None
                for node in execution.derivative.graph
            )
        finally:
            restored.close()
    finally:
        deployed.close()
        original.close()


def test_changed_source_bundle_is_rejected_before_copy(tmp_path):
    model, _ = prepare("qwen3", session_factory=session)
    try:
        model.export(tmp_path / "source")
    finally:
        model.close()
    root = tmp_path / "source/execution"
    record = json.loads((root / "interface.json").read_text())
    filename = next(name for name in record["files"] if name.endswith(".onnx"))
    with open(root / filename, "ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        deploy(tmp_path / "source", tmp_path / "deployed")
    assert not (tmp_path / "deployed").exists()


def test_frozen_audit_preserves_shadowed_subgraph_initializers(tmp_path):
    import numpy as np
    import onnx_ir as ir

    def branch(name, value):
        weight = ir.val(
            "parameter",
            const_value=ir.tensor(np.array([value], np.float32)),
            dtype=ir.DataType.FLOAT,
            shape=[1],
        )
        tape = ir.tape.Tape()
        result = tape.op("Identity", [weight])
        result.name = name
        result.dtype, result.shape = ir.DataType.FLOAT, ir.Shape([1])
        return ir.Graph([], [result], nodes=tape.nodes, initializers=[weight], name=name)

    weight = ir.val(
        "parameter",
        const_value=ir.tensor(np.array([1], np.float32)),
        dtype=ir.DataType.FLOAT,
        shape=[1],
    )
    cond = ir.val("cond", dtype=ir.DataType.BOOL, shape=[])
    tape = ir.tape.Tape()
    result = tape.op(
        "If", [cond], {"then_branch": branch("yes", 2), "else_branch": branch("no", 3)}
    )
    result.name = "result"
    result.dtype, result.shape = ir.DataType.FLOAT, ir.Shape([1])
    model = ir.Model(
        ir.Graph(
            [cond],
            [result],
            nodes=tape.nodes,
            initializers=[weight],
            opset_imports={"": 18},
            name="audit",
        ),
        ir_version=10,
    )
    ir.save(model, tmp_path / "model.onnx")
    complete = tensor_manifest(tmp_path)
    frozen = tensor_manifest(tmp_path, exclude_initializers=["parameter"])
    assert len(complete) == 3 and len(frozen) == 2
    assert "model.onnx/initializer/parameter" not in frozen
    assert all("/then_branch/" in key or "/else_branch/" in key for key in frozen)
