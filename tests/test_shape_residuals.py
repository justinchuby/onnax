# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Shape-only boundaries preserve runtime dimensions and graph interfaces."""

import numpy as np
import onnx_ir as ir
import pytest

from examples.huggingface.shape_residuals import compress
from tests.test_saved_residuals import execute


def pair(*, sliced=False, duplicate=False, mixed=False, returned=False, collision=False):
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=["batch", "width"])
    tape = ir.tape.Tape()
    u = tape.op("Identity", [x])
    u.name = "u"
    y = tape.op("Identity", [u])
    y.name = "y"
    for value in (u, y):
        value.dtype, value.shape = ir.DataType.FLOAT, ir.Shape(["batch", "width"])
    forward = ir.Model(
        ir.Graph([x], [y, u], nodes=tape.nodes, opset_imports={"": 18}, name="forward"),
        ir_version=10,
    )
    saved = ir.val("u", dtype=ir.DataType.FLOAT, shape=["batch", "width"])
    seed = ir.val("seed", dtype=ir.DataType.FLOAT, shape=["batch", "width"])
    tape = ir.tape.Tape()
    shape = tape.op("Shape", [saved])
    shape.name = "original_shape"
    shape.dtype = ir.DataType.INT64
    shape.shape = ir.Shape([2])
    grad = tape.op("Reshape", [seed, shape])
    grad.name = "gradient"
    grad.dtype = ir.DataType.FLOAT
    grad.shape = ir.Shape(["batch", "width"])
    outputs = [grad]
    if duplicate:
        same = tape.op("Shape", [saved])
        same.name = "duplicate_shape"
        same.dtype, same.shape = ir.DataType.INT64, ir.Shape([2])
        outputs.append(same)
    if sliced:
        tail = tape.op("Shape", [saved], {"start": 1, "end": -1})
        tail.name = "empty_shape"
        tail.dtype, tail.shape = ir.DataType.INT64, ir.Shape([0])
        outputs.append(tail)
    if mixed:
        numeric = tape.op("Add", [saved, seed])
        numeric.name = "numeric"
        numeric.dtype, numeric.shape = ir.DataType.FLOAT, ir.Shape(["batch", "width"])
        outputs.append(numeric)
    if returned:
        outputs.append(saved)
    if collision:
        tape.initializer(ir.tensor(np.array([7], np.int64), name="u.shape"))
    backward = ir.Model(
        ir.Graph(
            [saved, seed],
            outputs,
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": 18},
            name="backward",
        ),
        ir_version=10,
    )
    return forward, backward


@pytest.mark.parametrize(
    "sliced,duplicate", [(False, False), (True, False), (False, True), (True, True)]
)
def test_dynamic_shapes_duplicate_consumers_and_sliced_shape(sliced, duplicate):
    forward, backward = pair(sliced=sliced, duplicate=duplicate)
    before = [ir.to_proto(model).SerializeToString() for model in (forward, backward)]
    f, b, names, plan = compress(forward, backward, ["u"], outputs=["y"])
    assert before == [ir.to_proto(model).SerializeToString() for model in (forward, backward)]
    assert len(names) == 1 + int(sliced)
    assert plan["compressed"][0]["tensor"] == "u"
    assert "u" not in {value.name for value in b.graph.inputs}
    assert [value.name for value in b.graph.outputs] == [
        value.name for value in backward.graph.outputs
    ]
    for shape in ((2, 3), (1, 7), (4, 1), (0, 3), (2, 0)):
        x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        seed = x / np.float32(7)
        old = execute(forward, {"x": x})
        new = execute(f, {"x": x})
        np.testing.assert_array_equal(new["y"], old["y"])
        expected = execute(backward, {"u": old["u"], "seed": seed})
        actual = execute(b, {"seed": seed, **{name: new[name] for name in names}})
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
        assert sum(new[name].nbytes for name in names) == 16


@pytest.mark.parametrize("mixed,returned", [(True, False), (False, True)])
def test_value_consumers_and_returned_gradients_keep_tensor(mixed, returned):
    forward, backward = pair(mixed=mixed, returned=returned)
    _, _, names, plan = compress(forward, backward, ["u"], outputs=["y"])
    assert names == ("u",)
    assert plan["compressed"] == []


def test_original_forward_output_and_name_collision_are_preserved():
    forward, backward = pair(collision=True)
    f, b, names, _ = compress(forward, backward, ["u"], outputs=["y", "u"])
    assert names == ("u.shape_1",)
    assert [value.name for value in f.graph.outputs] == ["y", "u", "u.shape_1"]
    assert "u.shape" in b.graph.initializers
    feeds = {"x": np.ones((2, 3), np.float32)}
    values = execute(f, feeds)
    actual = execute(b, {"seed": feeds["x"], "u.shape_1": values["u.shape_1"]})
    np.testing.assert_array_equal(actual["gradient"], feeds["x"])


def test_nested_scan_capture_and_body_output_names():
    forward, _ = pair()
    u = ir.val("u", dtype=ir.DataType.FLOAT, shape=["batch", "width"])
    seed = ir.val("seed", dtype=ir.DataType.FLOAT, shape=["batch", "width"])
    row = ir.val("row", dtype=ir.DataType.FLOAT, shape=["width"])
    tape = ir.tape.Tape()
    grad = tape.op("Identity", [row])
    grad.name = "row_gradient"
    grad.dtype, grad.shape = ir.DataType.FLOAT, ir.Shape(["width"])
    shape = tape.op("Shape", [u])
    shape.name = "row_shape"
    shape.dtype, shape.shape = ir.DataType.INT64, ir.Shape([2])
    body = ir.Graph(
        [row], [grad, shape], nodes=tape.nodes, opset_imports={"": 18}, name="body"
    )
    tape = ir.tape.Tape()
    grad, shapes = tape.op_multi_out(
        "Scan", [seed], {"body": body, "num_scan_inputs": 1}, num_outputs=2
    )
    grad.name = "gradient"
    grad.dtype, grad.shape = ir.DataType.FLOAT, ir.Shape(["batch", "width"])
    shapes.name = "shapes"
    shapes.dtype, shapes.shape = ir.DataType.INT64, ir.Shape(["batch", 2])
    backward = ir.Model(
        ir.Graph(
            [u, seed],
            [grad, shapes],
            nodes=tape.nodes,
            opset_imports={"": 18},
            name="backward",
        ),
        ir_version=10,
    )
    f, b, names, plan = compress(forward, backward, ["u"], outputs=["y"])
    assert len(plan["compressed"]) == 1
    for dims in ((2, 3), (4, 7)):
        x = np.ones(dims, np.float32)
        primal = execute(f, {"x": x})
        actual = execute(b, {"seed": x, **{name: primal[name] for name in names}})
        expected = execute(backward, {"u": x, "seed": x})
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])


@pytest.mark.parametrize(
    "names,outputs", [(["missing"], ["y"]), (["u", "u"], ["y"]), (["u"], ["missing"])]
)
def test_invalid_interface_rejected(names, outputs):
    forward, backward = pair()
    with pytest.raises(ValueError):
        compress(forward, backward, names, outputs=outputs)


@pytest.mark.parametrize("optimize", [False, True])
def test_autograd_updates_reference_and_restored_adam(tmp_path, optimize):
    torch = pytest.importorskip("torch")
    from examples.huggingface.bridge import OnnxModule

    def model():
        x = ir.val("x", dtype=ir.DataType.FLOAT, shape=["batch", 4])
        a = ir.val("a", dtype=ir.DataType.FLOAT, shape=[4])
        tape = ir.tape.Tape()
        u = tape.op("Mul", [x, a])
        u.name, u.dtype, u.shape = "u", ir.DataType.FLOAT, ir.Shape(["batch", 4])
        bias = tape.initializer(ir.tensor(np.ones(4, np.float32), name="bias"))
        y = tape.op("Add", [u, bias])
        y.name, y.dtype, y.shape = "y", ir.DataType.FLOAT, ir.Shape(["batch", 4])
        return ir.Model(
            ir.Graph(
                [x, a],
                [y],
                nodes=tape.nodes,
                initializers=tape.initializers,
                opset_imports={"": 18},
                name="shape_training",
            ),
            ir_version=10,
        )

    initial = np.array([0.2, -0.3, 0.1, 0.5], np.float32)

    def make(compressed):
        module = OnnxModule(
            model(),
            {"a": initial},
            save_activations=True,
            optimize=optimize,
            save_shapes=compressed,
        )
        return module, torch.optim.AdamW(module.parameters(), lr=0.01)

    baseline, baseline_opt = make(False)
    candidate, candidate_opt = make(True)
    assert candidate.shape_plan["compressed"]
    reference = torch.nn.Parameter(torch.from_numpy(initial.copy()))
    reference_opt = torch.optim.AdamW([reference], lr=0.01)
    rng = np.random.default_rng(127)
    for step in range(4):
        for optimizer in (baseline_opt, candidate_opt, reference_opt):
            optimizer.zero_grad()
        for batch in (2, 3):
            x = torch.from_numpy(rng.normal(size=(batch, 4)).astype(np.float32))
            seed = torch.from_numpy(rng.normal(size=(batch, 4)).astype(np.float32))
            for module in (baseline, candidate):
                module(x=x).backward(seed)
            value = x * reference
            (value + 1).backward(seed)
        torch.testing.assert_close(
            baseline.adapters[0].grad, candidate.adapters[0].grad, rtol=0, atol=0
        )
        torch.testing.assert_close(
            candidate.adapters[0].grad, reference.grad, rtol=1e-4, atol=1e-6
        )
        for optimizer in (baseline_opt, candidate_opt, reference_opt):
            optimizer.step()
        torch.testing.assert_close(baseline.adapters[0], candidate.adapters[0], rtol=0, atol=0)
        torch.testing.assert_close(candidate.adapters[0], reference, rtol=1e-4, atol=1e-6)
        if step == 1:
            path = tmp_path / "checkpoint.pt"
            torch.save(
                {"model": candidate.state_dict(), "optimizer": candidate_opt.state_dict()},
                path,
            )
            candidate, candidate_opt = make(True)
            state = torch.load(path, weights_only=True)
            candidate.load_state_dict(state["model"])
            candidate_opt.load_state_dict(state["optimizer"])
    assert candidate.last_residual_bytes < baseline.last_residual_bytes
    for key, value in baseline_opt.state_dict()["state"][0].items():
        torch.testing.assert_close(
            value, candidate_opt.state_dict()["state"][0][key], rtol=0, atol=0
        )


def test_unknown_shape_result_rank_gets_vector_metadata():
    forward, backward = pair()
    for node in backward.graph:
        if node.op_type == "Shape":
            node.outputs[0].shape = None
    f, b, names, _ = compress(forward, backward, ["u"], outputs=["y"])
    output = next(value for value in f.graph.outputs if value.name == names[0])
    assert output.shape is not None and len(output.shape) == 1
    assert output.shape[0] == 2
    x = np.ones((3, 7), np.float32)
    values = execute(f, {"x": x})
    actual = execute(b, {"seed": x, **{key: values[key] for key in names}})
    np.testing.assert_array_equal(actual["gradient"], x)


def test_sliced_shape_keeps_old_forward_opset():
    forward, backward = pair(sliced=True)
    forward.graph.opset_imports[""] = 13
    f, _, names, plan = compress(forward, backward, ["u"], outputs=["y"])
    assert f.graph.opset_imports[""] == 13
    assert names == ("u",) and plan["compressed"] == []


def test_fused_broadcast_shape_vector_boundary_keeps_only_dimensions():
    import onnax
    from tests.test_broadcast_adjoint import _model

    forward, _ = pair()
    backward = onnax.FuseBroadcastAdjointsPass()(_model()).model
    backward.graph.inputs[0].name = "seed"
    backward.graph.inputs[1].name = "u"
    backward.graph.inputs[0].shape = ir.Shape([4, "batch", "width"])
    backward.graph.inputs[1].shape = ir.Shape(["batch", "width"])
    backward.graph.outputs[0].shape = ir.Shape(["batch", "width"])
    f, b, names, plan = compress(forward, backward, ["u"], outputs=["y"])
    assert len(plan["compressed"]) == 1
    assert "u" not in {value.name for value in b.graph.inputs}
    assert next(node for node in b.graph if node.domain == "pkg.onnax").op_type == "SumToShape"
    for batch, width in ((2, 1024), (3, 7), (0, 9)):
        x = np.ones((batch, width), np.float32)
        gradient = np.ones((4, batch, width), np.float32)
        values = execute(f, {"x": x})
        saved = {name: values[name] for name in names}
        assert sum(value.nbytes for value in saved.values()) == 16
        actual = execute(b, {"seed": gradient, **saved})
        np.testing.assert_array_equal(actual["result"], gradient.sum(axis=0))
