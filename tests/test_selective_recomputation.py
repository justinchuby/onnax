# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Selective replay preserves split pullbacks and their shared dependencies."""

import numpy as np
import onnx_ir as ir
import pytest

import onnax
from examples.huggingface.recomputation import describe_replay, replay_region
from examples.huggingface.residuals import split_vjp
from tests.test_saved_residuals import execute


def model():
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=["batch", 4])
    a = ir.val("a", dtype=ir.DataType.FLOAT, shape=[4])
    tape = ir.tape.Tape()
    shared = tape.op("Mul", [x, a])
    left = tape.op("Tanh", [shared])
    right = tape.op("Sigmoid", [shared])
    y = tape.op("Mul", [left, right])
    for name, value in (("shared", shared), ("left", left), ("right", right), ("y", y)):
        value.name, value.dtype, value.shape = name, ir.DataType.FLOAT, ir.Shape(["batch", 4])
    return ir.Model(
        ir.Graph([x, a], [y], nodes=tape.nodes, opset_imports={"": 18}, name="replay"),
        ir_version=10,
    )


@pytest.mark.parametrize(
    "selected", [("left",), ("left", "right"), ("shared", "left", "right")]
)
def test_replay_dynamic_shared_dependencies_and_updates(selected):
    source = model()
    joint = onnax.vjp(source, inputs=["a", "x"], outputs=["y"])
    before = [ir.to_proto(value).SerializeToString() for value in (source, joint)]
    baseline_f, baseline_b, baseline_names = split_vjp(source, joint)
    forward, backward, names = split_vjp(source, joint, recompute=selected)
    assert set(names) == set(baseline_names) - set(selected)
    assert before == [ir.to_proto(value).SerializeToString() for value in (source, joint)]
    assert (
        sum(n.op_type == "Mul" and n.outputs[0].name == "shared" for n in backward.graph) <= 1
    )
    rng = np.random.default_rng(7)
    a = rng.normal(size=4).astype(np.float32)
    reference = a.copy()
    for batch in (2, 3, 1, 4):
        x = rng.normal(size=(batch, 4)).astype(np.float32)
        seed = rng.normal(size=(batch, 4)).astype(np.float32)
        expected_forward = execute(baseline_f, {"x": x, "a": reference})
        actual_forward = execute(forward, {"x": x, "a": a})
        expected = execute(
            baseline_b,
            {
                "x": x,
                "a": reference,
                "y_cotangent": seed,
                **{name: expected_forward[name] for name in baseline_names},
            },
        )
        actual = execute(
            backward,
            {
                "x": x,
                "a": a,
                "y_cotangent": seed,
                **{name: actual_forward[name] for name in names},
            },
        )
        np.testing.assert_array_equal(actual_forward["y"], expected_forward["y"])
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name])
        reference = reference - np.float32(0.01) * expected["a_cotangent"]
        a = a - np.float32(0.01) * actual["a_cotangent"]
        np.testing.assert_array_equal(a, reference)


def test_plan_retained_cut_and_shared_producer():
    source = model()
    nodes = list(source.graph)
    boundary = [node.outputs[0] for node in nodes[:3]]
    plan = describe_replay(nodes, boundary, ("left", "right"))
    assert plan["retained_values"] == ["shared"]
    assert plan["replay_inputs"] == ["shared"]
    assert [node["op_type"] for node in plan["nodes"]] == ["Tanh", "Sigmoid"]
    plan = describe_replay(nodes, boundary, ("shared", "left", "right"))
    assert plan["replay_inputs"] == ["x", "a"]
    assert len(plan["nodes"]) == 3


@pytest.mark.parametrize("selected", [("missing",), ("left", "left")])
def test_invalid_selection(selected):
    source = model()
    with pytest.raises(ValueError):
        split_vjp(source, onnax.vjp(source, inputs=["a"], outputs=["y"]), recompute=selected)


@pytest.mark.parametrize(
    "domain,op", [("", "Dropout"), ("", "RandomNormalLike"), ("custom", "Tanh")]
)
def test_stateful_or_uncontracted_region_is_rejected(domain, op):
    source = model()
    nodes = list(source.graph)
    nodes[1].domain, nodes[1].op_type = domain, op
    with pytest.raises(ValueError, match="deterministic"):
        replay_region(nodes, [nodes[1].outputs[0]], ["left"])


def test_saved_value_report_describes_executable_replay():
    from examples.huggingface.saved_values import report

    source = model()
    # Static specialization before differentiation gives a complete local byte
    # count; dynamic execution is covered separately above.
    for value in ir.convenience.create_value_mapping(source.graph).values():
        if value.shape is not None:
            value.shape = ir.Shape([3 if str(dim) == "batch" else dim for dim in value.shape])
    result = report(
        source,
        inputs=["a"],
        outputs=["y"],
        parameters=["a"],
        dimensions={"batch": 3},
        recompute=["left", "right"],
    )
    assert result["recomputation"]["recomputed_values"] == ["left", "right"]
    assert result["recomputation"]["invocations_per_backward"] == 1
    assert result["categories"]["saved_activation"]["count"] == 1
    assert result["categories"]["saved_activation"]["known_bytes"] == 48
    assert result["backward_liveness"]["status"] == "complete"


def test_autograd_replay_nonunit_upstream_and_accumulation():
    torch = pytest.importorskip("torch")
    from examples.huggingface.bridge import OnnxModule

    graph = model()
    initial = {"a": np.array([0.2, -0.3, 0.1, 0.5], np.float32)}
    baseline = OnnxModule(graph, initial, save_activations=True)
    candidate = OnnxModule(
        graph, initial, save_activations=True, recompute=["shared", "left", "right"]
    )
    assert candidate.residual_names == ()
    rng = np.random.default_rng(4)
    for _step in range(4):
        baseline.adapters[0].grad = candidate.adapters[0].grad = None
        for batch in (2, 3):
            x = torch.from_numpy(rng.normal(size=(batch, 4)).astype(np.float32))
            scale = torch.from_numpy(rng.normal(size=(batch, 4)).astype(np.float32))
            for module in (baseline, candidate):
                (module(x=x) * scale).sum().backward()
        torch.testing.assert_close(
            baseline.adapters[0].grad, candidate.adapters[0].grad, rtol=0, atol=0
        )
        with torch.no_grad():
            for module in (baseline, candidate):
                module.adapters[0].sub_(0.01 * module.adapters[0].grad)
        torch.testing.assert_close(baseline.adapters[0], candidate.adapters[0], rtol=0, atol=0)
    assert candidate.last_residual_bytes == 0
    assert baseline.last_residual_bytes > 0


def test_adam_checkpoint_resume_keeps_selective_trajectory(tmp_path):
    torch = pytest.importorskip("torch")
    from examples.huggingface.bridge import OnnxModule

    def make():
        module = OnnxModule(
            model(),
            {"a": np.array([0.2, -0.3, 0.1, 0.5], np.float32)},
            save_activations=True,
            recompute=["shared", "left", "right"],
        )
        return module, torch.optim.AdamW(module.parameters(), lr=0.01)

    continuous, continuous_optimizer = make()
    resumed, resumed_optimizer = make()
    rng = np.random.default_rng(73)
    for step in range(4):
        x = torch.from_numpy(rng.normal(size=(step + 1, 4)).astype(np.float32))
        for module, optimizer in (
            (continuous, continuous_optimizer),
            (resumed, resumed_optimizer),
        ):
            optimizer.zero_grad()
            module(x=x).square().mean().backward()
            optimizer.step()
        if step == 1:
            checkpoint = tmp_path / "checkpoint.pt"
            torch.save(
                {"model": resumed.state_dict(), "optimizer": resumed_optimizer.state_dict()},
                checkpoint,
            )
            resumed, resumed_optimizer = make()
            state = torch.load(checkpoint, weights_only=True)
            resumed.load_state_dict(state["model"])
            resumed_optimizer.load_state_dict(state["optimizer"])
        torch.testing.assert_close(continuous.adapters[0], resumed.adapters[0], rtol=0, atol=0)
    first = continuous_optimizer.state_dict()
    second = resumed_optimizer.state_dict()
    assert first["param_groups"] == second["param_groups"]
    for key, value in first["state"][0].items():
        torch.testing.assert_close(value, second["state"][0][key], rtol=0, atol=0)
