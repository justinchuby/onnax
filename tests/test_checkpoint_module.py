# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Architecture-independent checkpoint execution and bundle contracts."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional integration dependencies.

import copy
import json

import numpy as np
import onnx_ir as ir
import pytest

torch = pytest.importorskip("torch")

from onnax.torch import CheckpointedModule, OnnxModule


def shared_graph():
    x = ir.val("x", dtype=ir.DataType.DOUBLE, shape=["batch", 3])
    w = ir.val("shared.weight", dtype=ir.DataType.DOUBLE, shape=[3, 3])
    frozen = ir.val("frozen.weight", const_value=ir.tensor(np.eye(3, dtype=np.float64) * 0.7))
    tape = ir.tape.Tape()
    first = tape.op("Tanh", [tape.op("MatMul", [x, w])])
    first.name = "first"
    middle = tape.op("MatMul", [first, frozen])
    middle.name = "middle"
    second = tape.op("Tanh", [tape.op("MatMul", [middle, w])])
    second.name = "second"
    output = tape.op("Add", [second, first])
    output.name = "prediction"
    total = tape.op("ReduceSum", [output], {"keepdims": 0})
    total.name = "total"
    model = ir.Model(
        ir.Graph(
            [x, w],
            [output, total],
            nodes=tape.nodes,
            initializers=[frozen],
            opset_imports={"": 18},
        ),
        ir_version=10,
    )
    return model, {"shared.weight": np.arange(9, dtype=np.float64).reshape(3, 3) / 20}


def module(**kwargs):
    model, params = shared_graph()
    return CheckpointedModule(
        model, params, boundaries={"first": "first", "frozen": "middle"}, **kwargs
    )


def test_shared_parameter_frozen_middle_and_multistep_updates():
    source, initial = shared_graph()
    before = ir.to_proto(source).SerializeToString()
    reference = torch.nn.Parameter(torch.from_numpy(initial["shared.weight"].copy()))
    with CheckpointedModule(
        source, initial, boundaries={"first": "first", "frozen": "middle"}
    ) as model:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0)
        expected_optimizer = torch.optim.AdamW([reference], lr=0.01, weight_decay=0)
        assert len(model.gradient_names) == 3
        for step in range(4):
            x = (
                torch.arange((step + 1) * 3, dtype=torch.float64)
                .reshape(-1, 3)
                .div(10)
                .requires_grad_()
            )
            target = x.detach().clone().requires_grad_()
            optimizer.zero_grad()
            expected_optimizer.zero_grad()
            actual, total = model(x=x)
            first = torch.tanh(target @ reference)
            expected = torch.tanh((first * 0.7) @ reference) + first
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
            (actual.square().mean() + 0.3 * total).backward()
            (expected.square().mean() + 0.3 * expected.sum()).backward()
            torch.testing.assert_close(x.grad, target.grad, rtol=1e-10, atol=1e-12)
            torch.testing.assert_close(
                model.trainable[0].grad, reference.grad, rtol=1e-10, atol=1e-12
            )
            optimizer.step()
            expected_optimizer.step()
            torch.testing.assert_close(model.trainable[0], reference, rtol=1e-10, atol=1e-12)
        assert model.last_residual_bytes > 0
    assert ir.to_proto(source).SerializeToString() == before


@pytest.mark.parametrize("kind", ["gqa", "mha"])
def test_tiny_decoder_training_and_reload(kind, tmp_path):
    from examples.torch_checkpoint import run

    torch.set_num_threads(1)
    report = run(tmp_path / kind, kind=kind)
    assert report["independent_updates"] and report["reload"]
    assert report["losses"][-1] < report["losses"][0]
    assert report["checkpoint_bytes"] > 0


def test_export_load_without_autodiff_and_resume_optimizer(tmp_path, monkeypatch):
    import onnax

    with module() as model:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        x = torch.ones(2, 3, dtype=torch.float64)
        model(x=x)[0].square().mean().backward()
        optimizer.step()
        model.export(tmp_path / "bundle")
        state = copy.deepcopy(optimizer.state_dict())

        def no_derivatives(*args, **kwargs):
            raise AssertionError("Load the cached derivatives")

        monkeypatch.setattr(onnax, "vjp", no_derivatives)
        with CheckpointedModule.load(tmp_path / "bundle") as restored:
            resumed = torch.optim.AdamW(restored.parameters(), lr=0.01)
            resumed.load_state_dict(state)
            for runtime, update in ((model, optimizer), (restored, resumed)):
                update.zero_grad()
                runtime(x=x)[1].backward()
                update.step()
            torch.testing.assert_close(
                model.trainable[0], restored.trainable[0], rtol=0, atol=0
            )
            restored.export(tmp_path / "again")


def test_snapshot_retained_backward_and_versions():
    with module(outputs=["prediction"]) as model:
        x = torch.ones(2, 3, dtype=torch.float64)
        y = model(x=x)
        x.zero_()
        y.sum().backward(retain_graph=True)
        gradient = model.trainable[0].grad.clone()
        y.sum().backward()
        torch.testing.assert_close(model.trainable[0].grad, 2 * gradient)
        y = model(x=x)
        with torch.no_grad():
            model.trainable[0].add_(0.01)
        with pytest.raises(RuntimeError, match="modified by an inplace operation"):
            y.sum().backward()
        x.requires_grad_()
        y = model(x=x)
        with torch.no_grad():
            x.add_(1)
        with pytest.raises(RuntimeError, match="modified by an inplace operation"):
            y.sum().backward()


def test_unused_output_and_accumulation():
    with module() as model:
        x = torch.ones(2, 3, dtype=torch.float64)
        model(x=x)[0].sum().backward()
        first = model.trainable[0].grad.clone()
        (2 * model(x=x)[0].sum()).backward()
        torch.testing.assert_close(model.trainable[0].grad, 3 * first)


@pytest.mark.parametrize(
    "boundaries,message",
    [
        ({"cut": "missing"}, "Unknown checkpoint"),
        ({"a": "middle", "b": "first"}, "preceding"),
        ({"a": "first", "b": "first"}, "unique"),
        ({"suffix": "first"}, "block names"),
    ],
)
def test_invalid_boundaries(boundaries, message):
    source, initial = shared_graph()
    with pytest.raises(ValueError, match=message):
        CheckpointedModule(source, initial, boundaries=boundaries)


def test_sessions_are_reused_and_released_on_failure(tmp_path):
    from onnax.torch._module import _ort_session

    created = []

    class Session:
        def __init__(self, path):
            self.executor = _ort_session(path)
            self.closed = False
            created.append(self)

        def run(self, *args):
            return self.executor.run(*args)

        def close(self):
            self.closed = True
            self.executor = None

    with module(session_factory=Session) as model:
        count = len(created)
        x = torch.ones(1, 3, dtype=torch.float64)
        for _ in range(2):
            model(x=x)[1].backward()
        assert len(created) == count
        directory = model._directory.name
    assert all(s.closed for s in created)
    import os

    assert not os.path.exists(directory)
    with pytest.raises(RuntimeError, match="closed"):
        model(x=x)

    def fail(path):
        if len(created) > count:
            raise RuntimeError("Session creation failed")
        return Session(path)

    with pytest.raises(RuntimeError, match="creation failed"):
        module(session_factory=fail)
    assert all(s.closed for s in created)


def test_bundle_rejects_plan_or_payload_changes(tmp_path):
    with module() as model:
        model.export(tmp_path / "bundle")
    path = tmp_path / "bundle/interface.json"
    interface = json.loads(path.read_text())
    interface["blocks"][0]["active_inputs"] = []
    path.write_text(json.dumps(interface))
    with pytest.raises(ValueError, match="plan differs"):
        CheckpointedModule.load(path.parent)


def test_empty_boundaries_and_constant_or_passthrough_outputs():
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=["batch", 2])
    constant = ir.val("constant", const_value=ir.tensor(np.array(3, np.float32)))
    source = ir.Model(
        ir.Graph(
            [x], [x, constant], nodes=[], initializers=[constant], opset_imports={"": 18}
        ),
        ir_version=10,
    )
    with CheckpointedModule(source, boundaries={}) as model:
        data = torch.ones(1, 2, requires_grad=True)
        actual, scalar = model(x=data)
        actual.sum().backward()
        torch.testing.assert_close(data.grad, torch.ones_like(data))
        assert scalar.item() == 3


def test_state_can_move_between_flat_and_checkpointed_strategies():
    source, initial = shared_graph()
    with OnnxModule(source, initial) as flat, module() as grouped:
        grouped.load_state_dict(flat.state_dict())
        data = torch.ones(2, 3, dtype=torch.float64)
        for a, b in zip(flat(x=data), grouped(x=data), strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_fresh_process_checkpoint_recovery(tmp_path):
    import subprocess
    import sys

    with module() as model:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        data = torch.ones(2, 3, dtype=torch.float64)
        model(x=data)[1].backward()
        optimizer.step()
        model.export(tmp_path / "bundle")
        torch.save(optimizer.state_dict(), tmp_path / "optimizer.pt")
        optimizer.zero_grad()
        model(x=data)[1].backward()
        optimizer.step()
        expected = model.arrays()["shared.weight"]
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys, numpy as np, torch
from onnax.torch import CheckpointedModule
with CheckpointedModule.load(sys.argv[1]) as model:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    optimizer.load_state_dict(torch.load(sys.argv[2], weights_only=True))
    model(x=torch.ones(2, 3, dtype=torch.float64))[1].backward()
    optimizer.step()
    np.save(sys.argv[3], model.arrays()['shared.weight'])
""",
            str(tmp_path / "bundle"),
            str(tmp_path / "optimizer.pt"),
            str(tmp_path / "actual.npy"),
        ],
        check=True,
    )
    np.testing.assert_array_equal(np.load(tmp_path / "actual.npy"), expected)


def test_onnxscript_optimized_blocks_preserve_interface_and_reload(tmp_path):
    optimizer = pytest.importorskip("onnxscript.optimizer")
    with module(optimize=optimizer.optimize) as optimized, module() as baseline:
        data = torch.ones(2, 3, dtype=torch.float64)
        actual = optimized(x=data)
        expected = baseline(x=data)
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)
        actual[1].backward()
        expected[1].backward()
        torch.testing.assert_close(
            optimized.trainable[0].grad, baseline.trainable[0].grad, rtol=1e-10, atol=1e-12
        )
        optimized.export(tmp_path / "optimized")
    with CheckpointedModule.load(tmp_path / "optimized") as restored:
        restored(x=data)[1].backward()
        torch.testing.assert_close(
            restored.trainable[0].grad, baseline.trainable[0].grad, rtol=1e-10, atol=1e-12
        )


@pytest.mark.parametrize("kind", ["gqa", "mha"])
def test_decoder_dynamic_batch_sequence_and_input_gradients(kind):
    from examples.torch_checkpoint import decoder, reference

    source, params, boundaries, frozen = decoder(kind)
    targets = {name: torch.tensor(value, requires_grad=True) for name, value in params.items()}
    with CheckpointedModule(source, params, boundaries=boundaries) as model:
        for batch, length in ((1, 3), (2, 7)):
            data = (
                torch.linspace(-1, 1, batch * length * 8)
                .reshape(batch, length, 8)
                .requires_grad_()
            )
            expected_data = data.detach().clone().requires_grad_()
            mask = torch.ones(batch, 1, length, length, dtype=torch.bool)
            actual = model(features=data, mask=mask)
            expected = reference(kind, expected_data, mask, targets, frozen)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
            actual.square().sum().backward()
            expected.square().sum().backward()
            torch.testing.assert_close(data.grad, expected_data.grad, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("skip", [False, True])
def test_checkpoint_releases_values_unused_by_backward(skip, monkeypatch, tmp_path):
    """A frozen linear VJP needs its cotangent, rather than its activation input."""
    import weakref

    from onnax.torch import _checkpoint_execution as execution

    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=["batch", 3])
    w = ir.val("weight", dtype=ir.DataType.FLOAT, shape=[3, 3])
    frozen = ir.val("frozen", const_value=ir.tensor(np.eye(3, dtype=np.float32) * 0.7))
    tape = ir.tape.Tape()
    first = tape.op("Tanh", [tape.op("MatMul", [x, w])])
    first.name = "first"
    middle = tape.op("MatMul", [first, frozen])
    middle.name = "middle"
    output = tape.op("Tanh", [middle])
    if skip:
        output = tape.op("Add", [output, first])
    output.name = "prediction"
    source = ir.Model(
        ir.Graph(
            [x, w], [output], nodes=tape.nodes, initializers=[frozen], opset_imports={"": 18}
        ),
        ir_version=10,
    )
    initial = {"weight": np.eye(3, dtype=np.float32)}
    original = execution._forward_arrays
    references = []

    def observed(block, feeds):
        if block.block.name == "suffix":
            assert (references[-1]() is not None) == skip
        arrays = original(block, feeds)
        if block.block.name == "first":
            references.append(weakref.ref(arrays[0]))
        return arrays

    monkeypatch.setattr(execution, "_forward_arrays", observed)
    with CheckpointedModule(
        source, initial, boundaries={"first": "first", "frozen": "middle"}
    ) as model:
        assert "first" not in {v.name for v in model.executions[1].derivative.graph.inputs}
        model.export(tmp_path / "bundle")
        with CheckpointedModule.load(tmp_path / "bundle") as restored:
            for runtime in (model, restored):
                actual = runtime(x=torch.ones(2, 3))
                # The skip connection needs first only during forward, not
                # as a saved shape reference for its statically known adjoint.
                assert runtime.last_residual_bytes == 2 * 3 * 4
                assert "first" not in {
                    value.name for value in runtime.executions[-1].derivative.graph.inputs
                }
                parameter = torch.tensor(initial["weight"], requires_grad=True)
                first_ref = torch.tanh(torch.ones(2, 3) @ parameter)
                expected = torch.tanh(first_ref * 0.7)
                if skip:
                    expected = expected + first_ref
                torch.testing.assert_close(actual, expected)
                actual.sum().backward(retain_graph=True)
                gradient = runtime.trainable[0].grad.clone()
                actual.sum().backward()
                expected.sum().backward()
                torch.testing.assert_close(gradient, parameter.grad)
                torch.testing.assert_close(runtime.trainable[0].grad, 2 * parameter.grad)


def test_checkpoint_output_ownership_and_disjoint_scan(monkeypatch):
    from types import SimpleNamespace

    from onnax.torch._checkpoint_execution import _forward_arrays

    feed = np.arange(32, dtype=np.float32)
    shared = np.arange(16, dtype=np.float32)
    readonly = np.arange(4, dtype=np.float32)
    readonly.flags.writeable = False
    owned = [np.arange(8, dtype=np.float32) for _ in range(64)]
    outputs = [feed[::2], shared[:12], shared[4:], readonly, *owned]
    names = tuple(f"output_{i}" for i in range(len(outputs)))

    class Session:
        def run(self, requested, feeds):
            assert requested == list(names)
            assert feeds["x"] is feed
            return outputs

    execution = SimpleNamespace(
        block=SimpleNamespace(inputs=("x",), outputs=names),
        forward_inputs=None,
        forward=Session(),
    )
    original = np.shares_memory
    checks = 0

    def counted(left, right):
        nonlocal checks
        checks += 1
        return original(left, right)

    monkeypatch.setattr(np, "shares_memory", counted)
    actual = _forward_arrays(execution, {"x": feed})
    assert checks < len(outputs)
    for result, source in zip(actual, outputs, strict=True):
        np.testing.assert_array_equal(result, source)
        assert result.flags.writeable
    for result, source in zip(actual[:4], outputs[:4], strict=True):
        assert not original(result, source)
    assert all(result is source for result, source in zip(actual[4:], owned, strict=True))
