# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Installed integration contracts, independent updates and resource ownership."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional integration dependencies.

import json
import os

import numpy as np
import onnx_ir as ir
import pytest
from onnx.reference import ReferenceEvaluator

torch = pytest.importorskip("torch")

from onnax.torch import OnnxModule


def graph(*, initializer=False, activation="Tanh", large=False):
    width = 16384 if large else 3
    x = ir.val("features", dtype=ir.DataType.FLOAT, shape=["batch", width])
    w = ir.val("block.weight", dtype=ir.DataType.FLOAT, shape=[width, 2])
    tape = ir.tape.Tape()
    hidden = tape.op(activation, [tape.op("MatMul", [x, w])])
    hidden.name = "hidden"
    total = tape.op("ReduceSum", [hidden], {"keepdims": 0})
    total.name = "sum"
    initializers = []
    inputs = [x, w]
    if initializer:
        inputs.remove(w)
        w.const_value = ir.tensor(np.full((width, 2), 0.1, np.float32), name=w.name)
        initializers.append(w)
    return ir.Model(
        ir.Graph(
            inputs,
            [hidden, total],
            nodes=tape.nodes,
            initializers=initializers,
            opset_imports={"": 18},
        ),
        ir_version=10,
    )


def weights(width=3):
    return {
        "block.weight": torch.arange(width * 2, dtype=torch.float32).reshape(width, 2) / 20
    }


def test_multio_output_gradients_and_external_updates(tmp_path):
    source = graph()
    before = ir.to_proto(source).SerializeToString()
    reference = torch.nn.Parameter(weights()["block.weight"].clone())
    with OnnxModule(source, weights()) as model:
        actual_optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0)
        expected_optimizer = torch.optim.AdamW([reference], lr=0.01, weight_decay=0)
        for step in range(6):
            x = (
                torch.arange((step % 3 + 1) * 3, dtype=torch.float32).reshape(-1, 3) / 10
            ).requires_grad_()
            target = x.detach().clone().requires_grad_()
            y, total = model(features=x)
            expected = torch.tanh(target @ reference)
            # Both outputs contribute to a caller-defined loss with different seeds.
            loss = 0.3 * y.square().mean() + 0.2 * total
            expected_loss = 0.3 * expected.square().mean() + 0.2 * expected.sum()
            actual_optimizer.zero_grad()
            expected_optimizer.zero_grad()
            loss.backward()
            expected_loss.backward()
            torch.testing.assert_close(x.grad, target.grad, rtol=1e-4, atol=1e-6)
            torch.testing.assert_close(
                model.trainable[0].grad, reference.grad, rtol=1e-4, atol=1e-6
            )
            actual_optimizer.step()
            expected_optimizer.step()
            torch.testing.assert_close(model.trainable[0], reference, rtol=1e-4, atol=1e-6)
        model.export(tmp_path / "bundle")
        with OnnxModule.load(tmp_path / "bundle") as restored:
            for a, b in zip(model.parameters(), restored.parameters(), strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            a = model(features=x.detach())
            b = restored(features=x.detach())
            for left, right in zip(a, b, strict=True):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert ir.to_proto(source).SerializeToString() == before


def test_frozen_onnx_block_propagates_into_pytorch_layer():
    upstream = torch.nn.Linear(4, 3)
    expected = torch.nn.Linear(4, 3)
    expected.load_state_dict(upstream.state_dict())
    weight = weights()["block.weight"]
    with OnnxModule(graph(initializer=True), {}, outputs=["hidden"]) as frozen:
        x = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 10
        frozen(features=upstream(x)).square().sum().backward()
        # The graph's frozen initializer has a different value from weights().
        torch.tanh(expected(x) @ torch.full_like(weight, 0.1)).square().sum().backward()
        torch.testing.assert_close(
            upstream.weight.grad, expected.weight.grad, rtol=1e-4, atol=1e-6
        )
        assert not list(frozen.parameters())


def test_unused_output_seed_and_gradient_accumulation():
    with OnnxModule(graph(), weights()) as model:
        x = torch.ones(2, 3)
        y, _ = model(features=x)
        y.sum().backward()
        first = model.trainable[0].grad.clone()
        y, _ = model(features=x)
        (2 * y.sum()).backward()
        torch.testing.assert_close(model.trainable[0].grad, 3 * first, rtol=1e-6, atol=1e-6)


def test_snapshot_and_version_checks():
    with OnnxModule(graph(), weights(), outputs=["hidden"]) as model:
        x = torch.ones(2, 3)
        y = model(features=x)
        x.zero_()
        y.sum().backward()
        expected = torch.ones_like(x).T @ (
            1 - torch.tanh(torch.ones_like(x) @ model.trainable[0]).square()
        )
        torch.testing.assert_close(model.trainable[0].grad, expected)
        y = model(features=x)
        with torch.no_grad():
            model.trainable[0].add_(0.01)
        with pytest.raises(RuntimeError, match="modified by an inplace operation"):
            y.sum().backward()
        x = torch.ones(2, 3, requires_grad=True)
        y = model(features=x)
        with torch.no_grad():
            x.add_(1)
        with pytest.raises(RuntimeError, match="modified by an inplace operation"):
            y.sum().backward()


def test_identity_rejects_different_graph_before_loading_parameters():
    with (
        OnnxModule(graph(), weights()) as first,
        OnnxModule(graph(activation="Sigmoid"), weights()) as second,
    ):
        before = second.trainable[0].detach().clone()
        with pytest.raises(ValueError, match="graph, frozen weights"):
            second.load_state_dict(first.state_dict())
        torch.testing.assert_close(before, second.trainable[0], rtol=0, atol=0)


def test_session_reuse_close_failure_and_saved_graph_loading(tmp_path, monkeypatch):
    sessions = []

    class Tracked:
        def __init__(self, path):
            self.path = path
            self.executor = ReferenceEvaluator(path)
            self.closed = 0
            sessions.append(self)

        def run(self, *args):
            return self.executor.run(*args)

        def close(self):
            self.closed += 1

    module = OnnxModule(
        graph(initializer=True), weights(), outputs=["hidden"], session_factory=Tracked
    )
    for _ in range(2):
        module(features=torch.ones(2, 3)).sum().backward()
    assert len(sessions) == 2
    module.export(tmp_path / "bundle")
    import onnax

    def forbidden(*args, **kwargs):
        raise AssertionError("Reuse exported derivative")

    monkeypatch.setattr(onnax, "vjp", forbidden)
    restored = OnnxModule.load(tmp_path / "bundle", session_factory=Tracked)
    restored(features=torch.ones(1, 3)).sum().backward()
    module.close()
    module.close()
    restored.close()
    assert all(s.closed == 1 and not os.path.exists(s.path) for s in sessions)
    with pytest.raises(RuntimeError, match="closed"):
        module(features=torch.ones(1, 3))


def test_external_weights_and_bundle_integrity(tmp_path):
    # A genuinely external frozen matrix, shared by execution and derivative graphs.
    source = graph(initializer=True, large=True)
    with OnnxModule(source, {}, outputs=["hidden"]) as model:
        model.export(tmp_path / "external")
    with OnnxModule.load(tmp_path / "external") as restored:
        x = torch.full((2, 16384), 1e-4, requires_grad=True)
        restored(features=x).sum().backward()
        reference = x.detach().clone().requires_grad_()
        torch.tanh(reference @ torch.full((16384, 2), 0.1)).sum().backward()
        torch.testing.assert_close(x.grad, reference.grad, rtol=1e-4, atol=1e-6)
    interface = json.loads((tmp_path / "external/interface.json").read_text())
    path = tmp_path / "external/execution.onnx"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="integrity mismatch"):
        OnnxModule.load(tmp_path / "external")
    assert interface["version"] == 1


def test_invalid_boundaries_and_explicit_input_gradient_optout():
    with OnnxModule(graph(), weights(), outputs=["hidden"], input_gradients=[]) as model:
        with pytest.raises(ValueError, match="Enable input_gradients"):
            model(features=torch.ones(2, 3, requires_grad=True))
        with pytest.raises(ValueError, match="expects"):
            model(features=torch.ones(2, 4))
        with pytest.raises(TypeError, match="dtype"):
            model(features=torch.ones(2, 3, dtype=torch.float64))
        with pytest.raises(ValueError, match="dtype and CPU"):
            model.double()
        with pytest.raises(ValueError, match="Expected data inputs"):
            model(unknown=torch.ones(2, 3))


def test_integer_inputs_and_nondifferentiable_outputs():
    indices = ir.val("tokens", dtype=ir.DataType.INT64, shape=["batch", 3])
    w = ir.val("embedding.weight", dtype=ir.DataType.FLOAT, shape=[5, 2])
    tape = ir.tape.Tape()
    y = tape.op("Gather", [w, indices], {"axis": 0})
    y.name = "embeddings"
    shape = tape.op("Shape", [y])
    shape.name = "shape"
    source = ir.Model(
        ir.Graph([indices, w], [y, shape], nodes=tape.nodes, opset_imports={"": 18}),
        ir_version=10,
    )
    params = torch.arange(10, dtype=torch.float32).reshape(5, 2) / 10
    with OnnxModule(source, {w.name: params}) as model:
        tokens = torch.tensor([[1, 1, 4], [0, 2, 1]])
        values, actual_shape = model(tokens=tokens)
        assert not actual_shape.requires_grad
        torch.testing.assert_close(actual_shape, torch.tensor(values.shape))
        values.square().sum().backward()
        expected = params.clone().requires_grad_()
        expected[tokens].square().sum().backward()
        torch.testing.assert_close(model.trainable[0].grad, expected.grad)


def test_floating_data_versions_and_constructor_failure_release(tmp_path):
    sessions = []

    class First:
        def __init__(self, path):
            self.path = path
            self.closed = False

        def close(self):
            self.closed = True

    def factory(path):
        if sessions:
            raise RuntimeError("factory failed")
        first = First(path)
        sessions.append(first)
        return first

    with pytest.raises(RuntimeError, match="factory failed"):
        OnnxModule(graph(), weights(), session_factory=factory)
    assert sessions[0].closed and not os.path.exists(sessions[0].path)


def test_container_inputs_and_initializer_parameter_promotion(tmp_path):
    from onnx_ir.passes.common import CheckerPass

    source = graph(initializer=True)
    ir.save(source, tmp_path / "source.onnx")
    for container in [
        source,
        source.graph,
        ir.GraphView(
            source.graph.inputs,
            source.graph.outputs,
            nodes=list(source.graph),
            initializers=list(source.graph.initializers.values()),
            opset_imports=source.graph.opset_imports,
        ),
        ir.to_proto(source),
        str(tmp_path / "source.onnx"),
    ]:
        with OnnxModule(container, weights(), outputs=["hidden"]) as model:
            actual = model(features=torch.ones(2, 3))
            expected = torch.tanh(torch.ones(2, 3) @ weights()["block.weight"])
            torch.testing.assert_close(actual, expected)
            exported = model.export(
                tmp_path / f"export-{len(list(tmp_path.glob('export-*')))}"
            )
            CheckerPass(full_check=True)(exported)
            for path in tmp_path.glob("export-*/*.onnx"):
                CheckerPass(full_check=True)(ir.load(path))


def test_double_and_name_collision():
    source = graph()
    for value in source.graph.inputs:
        value.dtype = ir.DataType.DOUBLE
    # A source input may already use the derivative suffix.
    source.graph.inputs[1].name = "features_cotangent"
    params = {"features_cotangent": weights()["block.weight"].double()}
    with OnnxModule(source, params, outputs=["hidden"]) as model:
        x = torch.ones(2, 3, dtype=torch.float64, requires_grad=True)
        model(features=x).sum().backward()
        assert x.grad.dtype == torch.float64
        assert model.gradient_names["features"] != "features_cotangent"


def test_output_views_cannot_change_saved_inputs_or_parameters():
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[2, 3])
    w = ir.val("w", dtype=ir.DataType.FLOAT, shape=[3])
    tape = ir.tape.Tape()
    alias = tape.op("Identity", [x])
    alias.name = "alias"
    scaled = tape.op("Mul", [x, w])
    scaled.name = "scaled"
    source = ir.Model(
        ir.Graph([x, w], [alias, scaled], nodes=tape.nodes, opset_imports={"": 18}),
        ir_version=10,
    )
    with OnnxModule(
        source, {"w": np.ones(3, np.float32)}, session_factory=ReferenceEvaluator
    ) as model:
        features = torch.arange(6, dtype=torch.float32).reshape(2, 3).requires_grad_()
        a, b = model(x=features)
        a.add_(0.5)
        (a * b).sum().backward()
        torch.testing.assert_close(
            model.trainable[0].grad, (features.detach() * (features.detach() + 0.5)).sum(0)
        )
