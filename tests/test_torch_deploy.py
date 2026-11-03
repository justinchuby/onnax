# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Deployment headers preserve ordinary ONNX execution and external parameter updates."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional integration dependency.

import copy
import json

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnxscript import optimizer

torch = pytest.importorskip("torch")

from examples.torch_deploy import install_header, prepare_header, read_header
from onnax.torch import OnnxModule
from onnax.torch._serialization import file_digest


def _source():
    rng = np.random.default_rng(7)
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=["batch", 3])
    w = ir.val("w", dtype=ir.DataType.FLOAT, shape=[3, 128])
    fixed = rng.normal(0, 0.1, (128, 128)).astype(np.float32)
    c = ir.val("c", dtype=ir.DataType.FLOAT, shape=fixed.shape, const_value=ir.tensor(fixed))
    tape = ir.tape.Tape()
    y = tape.op("MatMul", [tape.op("Tanh", [tape.op("MatMul", [x, w])]), c])
    y.name = "y"
    return ir.Model(
        ir.Graph([x, w], [y], nodes=tape.nodes, initializers=[c], opset_imports={"": 18}),
        ir_version=10,
    ), fixed


def test_header_reuses_payloads_and_preserves_updates(tmp_path):
    source, fixed = _source()
    initial = np.full((3, 128), 0.1, np.float32)
    bundle = tmp_path / "bundle"
    with OnnxModule(
        source, {"w": initial}, input_gradients=[], optimize=optimizer.optimize
    ) as original:
        original.export(bundle)
    header = tmp_path / "deployment" / "optimized.onnx"
    record = prepare_header(bundle, header)
    interface = json.loads((bundle / "interface.json").read_text())
    assert record["payloads"]
    assert record["reused_initializers"] > 0
    assert {p.name for p in header.parent.iterdir()} == {header.name, header.name + ".json"}
    deployment = read_header(header, interface)

    def factory(path):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        return ort.InferenceSession(
            install_header(path, deployment), options, providers=["CPUExecutionProvider"]
        )

    reference = torch.nn.Parameter(torch.from_numpy(initial.copy()))
    with OnnxModule.load(bundle, session_factory=factory) as model:
        actual_update = torch.optim.SGD(model.parameters(), lr=0.02)
        expected_update = torch.optim.SGD([reference], lr=0.02)
        for batch in (2, 1, 3):
            x = torch.arange(batch * 3, dtype=torch.float32).reshape(batch, 3) / 10
            actual = model(x=x)
            expected = torch.tanh(x @ reference) @ torch.from_numpy(fixed)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
            actual_update.zero_grad()
            expected_update.zero_grad()
            actual.square().mean().backward()
            expected.square().mean().backward()
            torch.testing.assert_close(
                model.trainable[0].grad, reference.grad, rtol=1e-4, atol=1e-6
            )
            actual_update.step()
            expected_update.step()
            torch.testing.assert_close(model.trainable[0], reference, rtol=1e-4, atol=1e-6)
    for name, digest in interface["files"].items():
        assert file_digest(bundle / name) == digest
    with pytest.raises(FileExistsError):
        prepare_header(bundle, header)
    wrong = copy.deepcopy(interface)
    wrong["identity"] = "0" * 64
    with pytest.raises(ValueError, match="source/gradient"):
        read_header(header, wrong)
    header.write_bytes(header.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        read_header(header, interface)


def test_recomputed_attention_header(tmp_path):
    from test_fused_autodiff import _model_for

    rng = np.random.default_rng(191)
    feeds = {
        name: rng.normal(0, 0.2, (1, 2, 5, 40)).astype(np.float32) for name in ("q", "k", "v")
    }
    source = _model_for("Attention", feeds)
    bundle = tmp_path / "bundle"
    with OnnxModule(source, feeds, input_gradients=[]) as module:
        module.export(bundle)
    header = tmp_path / "recomputed.onnx"
    record = prepare_header(bundle, header, recompute_attention=True)
    assert record["recomputed_attention_calls"] == 1
    assert record["fused_attention_calls"] == 1
    interface = json.loads((bundle / "interface.json").read_text())
    deployment = read_header(header, interface)

    def factory(path):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        return ort.InferenceSession(
            install_header(path, deployment), options, providers=["CPUExecutionProvider"]
        )

    with (
        OnnxModule.load(bundle) as reference,
        OnnxModule.load(bundle, session_factory=factory) as actual,
    ):
        reference().square().mean().backward()
        actual().square().mean().backward()
        for name, value in actual.parameters_by_name().items():
            torch.testing.assert_close(
                value.grad, reference.parameters_by_name()[name].grad, rtol=1e-4, atol=1e-6
            )
