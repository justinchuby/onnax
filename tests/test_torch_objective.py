# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Scalar objective execution preserves external autograd and optimizer semantics."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional integration dependency.

import json
import os

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnxscript import optimizer

torch = pytest.importorskip("torch")

import onnax
from examples.torch_deploy import install_header, prepare_header, read_header
from onnax.torch import OnnxModule


def objective():
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=["batch", 3])
    w = ir.val("w", dtype=ir.DataType.FLOAT, shape=[3, 2])
    target = ir.val("target", dtype=ir.DataType.FLOAT, shape=["batch", 2])
    t = ir.tape.Tape()
    prediction = t.op("Tanh", [t.op("MatMul", [x, w])])
    diff = t.op("Sub", [prediction, target])
    loss = t.op("ReduceMean", [t.op("Mul", [diff, diff])], {"keepdims": 0})
    loss.name = "loss"
    return ir.Model(
        ir.Graph([x, w, target], [loss], nodes=t.nodes, opset_imports={"": 18}), ir_version=10
    )


def factory(calls, deployment=None, *, provider="CPUExecutionProvider"):
    def create(path):
        if deployment is not None:
            path = install_header(path, deployment)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 1
        if provider == "MLXExecutionProvider":
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        runtime = ort.InferenceSession(path, options, providers=[provider])
        runtime.disable_fallback()

        class Session:
            def __init__(self, inner):
                self.runtime = inner

            def run(self, names, feeds):
                calls.append(tuple(names))
                return self.runtime.run(names, feeds)

            def close(self):
                self.runtime = None

        return Session(runtime)

    return create


@pytest.fixture(params=["CPUExecutionProvider", "MLXExecutionProvider"])
def provider(request):
    if request.param == "MLXExecutionProvider":
        if os.environ.get("ONNAX_TEST_MLX_EP") != "1":
            pytest.skip("Native MLX opt-in")
        from examples.lora.mlx_runtime import register_library

        register_library(os.environ["ONNAX_MLX_EP_LIBRARY"])
    return request.param


@pytest.mark.parametrize("eager,saved", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("shape_inputs", [False, True])
def test_weighted_accumulation_and_dynamic_batch_updates(eager, saved, provider, shape_inputs):
    calls = []
    initial = np.arange(6, dtype=np.float32).reshape(3, 2) / 20
    reference = torch.nn.Parameter(torch.from_numpy(initial.copy()))
    with OnnxModule(
        objective(),
        {"w": initial},
        input_gradients=["x"],
        eager_gradients=eager,
        save_activations=saved,
        session_factory=factory(calls, provider=provider),
        optimize=lambda model: (
            onnax.FuseBroadcastAdjointsPass(shape_inputs=True)(model).model
            if shape_inputs
            else model
        ),
    ) as model:
        if shape_inputs:
            derivative = ir.load(os.path.join(model._directory.name, "derivative.onnx"))
            assert any(node.op_type == "SumToShape_v2" for node in derivative.graph)
        actual = torch.optim.AdamW(model.parameters(), lr=0.01, eps=1e-5, weight_decay=0)
        expected = torch.optim.AdamW([reference], lr=0.01, eps=1e-5, weight_decay=0)
        for step in range(4):
            actual.zero_grad()
            expected.zero_grad()
            for size, weight in ((step % 3 + 1, 1 / 3), (2, 2 / 3)):
                x = (torch.arange(size * 3).reshape(size, 3).float() / 10).requires_grad_()
                control = x.detach().clone().requires_grad_()
                target = torch.full((size, 2), 0.2)
                loss = model(x=x, target=target)
                wanted = (torch.tanh(control @ reference) - target).square().mean()
                torch.testing.assert_close(loss, wanted, rtol=1e-4, atol=1e-6)
                (weight * loss).backward()
                (weight * wanted).backward()
                torch.testing.assert_close(x.grad, control.grad, rtol=1e-4, atol=1e-6)
                torch.testing.assert_close(
                    model.trainable[0].grad, reference.grad, rtol=1e-4, atol=1e-6
                )
            actual.step()
            expected.step()
            torch.testing.assert_close(model.trainable[0], reference, rtol=1e-4, atol=1e-6)
        assert len(calls) == (8 if eager else 16)
        if eager:
            assert all(row[0] == "loss" and len(row) == 3 for row in calls)
        with torch.no_grad():
            value = model(x=torch.ones(1, 3), target=torch.zeros(1, 2))
        assert not value.requires_grad and calls[-1] == ("loss",)


def test_outstanding_forwards_and_retained_backward_do_not_share_cached_gradients():
    with OnnxModule(
        objective(),
        {"w": np.full((3, 2), 0.1, np.float32)},
        input_gradients=[],
        eager_gradients=True,
    ) as model:
        x = torch.ones(2, 3)
        target = torch.zeros(2, 2)
        a = model(x=x, target=target)
        b = model(x=2 * x, target=target)
        target.fill_(99)
        reference = model.trainable[0].detach().clone().requires_grad_()
        expected_a = torch.tanh(x @ reference).square().mean()
        expected_b = torch.tanh((2 * x) @ reference).square().mean()
        b.backward()
        expected_b.backward()
        a.backward(retain_graph=True)
        expected_a.backward(retain_graph=True)
        a.backward()
        expected_a.backward()
        torch.testing.assert_close(
            model.trainable[0].grad, reference.grad, rtol=1e-4, atol=1e-6
        )


def test_parameter_versions_are_checked_before_using_cached_gradients():
    with OnnxModule(
        objective(),
        {"w": np.ones((3, 2), np.float32)},
        input_gradients=[],
        eager_gradients=True,
    ) as model:
        loss = model(x=torch.ones(1, 3), target=torch.zeros(1, 2))
        with torch.no_grad():
            model.trainable[0].add_(1)
        with pytest.raises(RuntimeError, match="modified by an inplace operation"):
            loss.backward()


def test_export_load_and_deployment_preserve_objective_mode(tmp_path):
    bundle = tmp_path / "bundle"
    with OnnxModule(
        objective(),
        {"w": np.full((3, 2), 0.1, np.float32)},
        input_gradients=[],
        eager_gradients=True,
        optimize=optimizer.optimize,
    ) as model:
        model.export(bundle)
    header = tmp_path / "joint.onnx"
    prepare_header(bundle, header, broadcast_adjoints=True)
    interface = json.loads((bundle / "interface.json").read_text())
    assert interface["eager_gradients"] is True
    deployment = read_header(header, interface)
    calls = []
    with OnnxModule.load(bundle, session_factory=factory(calls, deployment)) as loaded:
        loss = loaded(x=torch.ones(2, 3), target=torch.zeros(2, 2))
        (2 * loss).backward()
        assert len(calls) == 1 and calls[0][0] == "loss"
        expected = loaded.trainable[0].detach().clone().requires_grad_()
        (2 * torch.tanh(torch.ones(2, 3) @ expected).square().mean()).backward()
        torch.testing.assert_close(
            loaded.trainable[0].grad, expected.grad, rtol=1e-4, atol=1e-6
        )


def test_eager_requires_a_scalar_objective():
    source = objective()
    source.graph.outputs.clear()
    source.graph.outputs.append(source.graph.inputs[0])
    with pytest.raises(ValueError, match="scalar"):
        OnnxModule(source, {"w": np.ones((3, 2), np.float32)}, eager_gradients=True)
    with pytest.raises(TypeError, match="bool"):
        OnnxModule(objective(), eager_gradients="yes")


def test_execution_policy_preserves_parameter_checkpoint_identity():
    initial = {"w": np.full((3, 2), 0.1, np.float32)}
    with (
        OnnxModule(objective(), initial, input_gradients=[]) as regular,
        OnnxModule(objective(), initial, input_gradients=[], eager_gradients=True) as eager,
    ):
        eager.load_state_dict(regular.state_dict(), strict=True)
        assert regular._identity == eager._identity


def test_scalar_parameter_dtype_and_output_alias():
    weight = ir.val("w", dtype=ir.DataType.DOUBLE, shape=[])
    source = ir.Model(
        ir.Graph([weight], [weight], nodes=[], opset_imports={"": 18}), ir_version=10
    )
    with OnnxModule(source, {"w": np.array(0.25, np.float64)}, eager_gradients=True) as model:
        loss = model()
        loss.add_(4)
        (3 * loss).backward()
        assert model.trainable[0].item() == pytest.approx(0.25)
        assert model.trainable[0].grad.dtype == torch.float64
        assert model.trainable[0].grad.item() == 3


def test_task_objective_export_is_independently_loadable(tmp_path, monkeypatch):
    pytest.importorskip("diffusers")
    pytest.importorskip("mobius")
    from examples.diffusion import objective as task

    x = ir.val("features", dtype=ir.DataType.FLOAT, shape=["batch", 3])
    w = ir.val("w", dtype=ir.DataType.FLOAT, shape=[3, 2])
    tape = ir.tape.Tape()
    prediction = tape.op("Tanh", [tape.op("MatMul", [x, w])])
    prediction.name = "prediction"
    source = ir.Model(
        ir.Graph([x, w], [prediction], nodes=tape.nodes, opset_imports={"": 18}), ir_version=10
    )
    initial = {"w": np.full((3, 2), 0.1, np.float32)}
    run = tmp_path / "run"
    with OnnxModule(source, initial, input_gradients=[]) as model:
        model.export(run / "export")
    interface = json.loads((run / "export/interface.json").read_text())
    from examples.diffusion import benchmark_unet

    monkeypatch.setattr(benchmark_unet, "load_state", lambda _: ({}, interface, initial))
    destination = tmp_path / "objective"
    record = task.prepare(run, destination)
    assert record["base_identity"] == interface["identity"]
    assert json.loads((destination / "objective.json").read_text()) == record
    diagnostics = task.DiagnosticSessions(factory([]), record)
    with OnnxModule.load(destination, session_factory=diagnostics) as model:
        x = torch.ones(2, 3)
        loss = model(features=x, **{record["target"]: torch.zeros(2, 2)})
        loss.backward()
        expected = torch.from_numpy(initial["w"].copy()).requires_grad_()
        wanted = torch.tanh(x @ expected).square().mean()
        wanted.backward()
        torch.testing.assert_close(loss, wanted, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(
            model.trainable[0].grad, expected.grad, rtol=1e-4, atol=1e-6
        )
        bound = model.export(tmp_path / "reexport")
        prediction_path = tmp_path / "prediction.onnx"
        task.export_prediction(
            bound,
            prediction_path,
            record["prediction"],
            execution=ir.load(tmp_path / "reexport/execution.onnx"),
        )
        prediction = factory([])(str(prediction_path)).run(
            [record["prediction"]], {"features": x.numpy()}
        )[0]
        np.testing.assert_allclose(
            prediction, torch.tanh(x @ expected).detach().numpy(), rtol=1e-4, atol=1e-6
        )
    assert {path.name for path in (tmp_path / "reexport").iterdir()} == {
        path.name for path in destination.iterdir()
    } - {"objective.json"}
    with (
        OnnxModule.load(tmp_path / "reexport", session_factory=diagnostics) as restored,
        torch.no_grad(),
    ):
        torch.testing.assert_close(
            restored(features=x, **{record["target"]: torch.zeros(2, 2)}),
            wanted.detach(),
            rtol=1e-4,
            atol=1e-6,
        )
    with pytest.raises(FileExistsError):
        task.prepare(run, destination)
