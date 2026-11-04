# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Execute installed public transforms; copy/run outside the source checkout."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path  # ruff: ignore[banned-api] - Supporting installed-distribution checks.

import numpy as np
import onnx
import onnx_ir as ir
from onnx.reference import ReferenceEvaluator

import onnax
from onnax import (
    FuseAttentionVJPPass,
    FuseBroadcastAdjointsPass,
    GradPass,
    HoistLinearCoefficientsPass,
    JvpPass,
    VjpPass,
    VmapPass,
    grad,
    jvp,
    vjp,
    vmap,
)


def main() -> None:
    distribution = importlib.metadata.distribution("onnax")
    package = Path(onnax.__file__).resolve().parent
    assert package.parent.name == "site-packages", (
        f"Expected an installed onnax in site-packages, got {package}. "
        "Install the built artifact in an isolated environment and run outside the checkout."
    )
    assert (package / "py.typed").is_file(), f"Missing typing marker in {package}."
    assert distribution.metadata["License-Expression"] == "MIT"
    assert distribution.metadata["Description-Content-Type"] == "text/markdown"
    license_files = distribution.metadata.get_all("License-File", [])
    assert set(license_files) == {"LICENSE"}, (
        f"Expected only LICENSE in installed License-File metadata, got {license_files!r}. "
        "Set project.license-files to ['LICENSE'] and rebuild the distribution."
    )
    assert distribution.read_text("licenses/LICENSE"), (
        "Missing LICENSE in installed metadata. Include LICENSE in project.license-files "
        "and rebuild the distribution."
    )
    assert set(onnax.__all__) == {
        "grad",
        "jvp",
        "vjp",
        "vmap",
        "GradPass",
        "JvpPass",
        "VjpPass",
        "VmapPass",
        "HoistLinearCoefficientsPass",
        "FuseBroadcastAdjointsPass",
        "FuseAttentionVJPPass",
        "register_jvp",
        "register_vjp",
        "register_batching",
        "JvpRule",
        "VjpRule",
        "BatchingRule",
        "AutodiffContext",
        "BatchingContext",
    }

    gradient = ir.val("gradient", dtype=ir.DataType.DOUBLE)
    reference = ir.val("reference", dtype=ir.DataType.DOUBLE)
    broadcast_graph = ir.Graph(
        [gradient, reference],
        [],
        nodes=[],
        opset_imports={"": 18},
        name="broadcast_distribution_smoke",
    )
    reduced = onnax.AutodiffContext(broadcast_graph).unbroadcast(gradient, reference)
    gradient.shape, reference.shape = ir.Shape([2, 3]), ir.Shape([3])
    reduced.type, reduced.shape = ir.TensorType(ir.DataType.DOUBLE), ir.Shape([3])
    broadcast_graph.outputs.append(reduced)
    packaged = FuseBroadcastAdjointsPass()(ir.Model(broadcast_graph, ir_version=10))
    assert packaged.modified
    portable = ir.to_proto(packaged.model)
    onnx.checker.check_model(portable, full_check=True)
    actual = ReferenceEvaluator(portable).run(
        None,
        {"gradient": np.arange(6, dtype=np.float64).reshape(2, 3), "reference": np.zeros(3)},
    )[0]
    np.testing.assert_array_equal(actual, [3.0, 5.0, 7.0])

    shape = [1, 1, 2, 32]
    q, k, v = [ir.val(name, dtype=ir.DataType.FLOAT, shape=shape) for name in ("q", "k", "v")]
    y = ir.val("y", dtype=ir.DataType.FLOAT, shape=shape)
    attention = ir.node("Attention", [q, k, v], outputs=[y])
    model = ir.Model(
        ir.Graph(
            [q, k, v],
            [y],
            nodes=[attention],
            opset_imports={"": 23},
            name="attention_distribution_smoke",
        ),
        ir_version=10,
    )
    derivative = vjp(model, inputs=["q", "k", "v"])
    packaged = FuseAttentionVJPPass(recompute=True)(derivative)
    assert packaged.modified
    portable = ir.to_proto(packaged.model)
    onnx.checker.check_model(portable, full_check=True)
    values = ReferenceEvaluator(portable).run(
        None, {value.name: np.ones(shape, np.float32) for value in packaged.model.graph.inputs}
    )
    for actual, expected in zip(values, (1, 0, 0, 1), strict=True):
        np.testing.assert_array_equal(actual, np.full(shape, expected, np.float32))

    input_x = ir.val("x", dtype=ir.DataType.DOUBLE, shape=[2])
    tape = ir.tape.Tape()
    parameter = tape.initializer(ir.tensor(np.array([2.0, 3.0]), name="weight"))
    predicted = tape.op("Mul", [input_x, parameter])
    predicted.name = "prediction"
    squared = tape.op("Mul", [predicted, predicted])
    squared.name = "squared"
    total = tape.op("ReduceSum", [squared], {"keepdims": 0})
    total.name = "loss"
    model = ir.Model(
        ir.Graph(
            [input_x],
            [predicted, total],
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": 18},
            name="distribution_smoke",
        ),
        ir_version=10,
    )
    before = ir.to_proto(model).SerializeToString()
    x = np.array([1.0, 2.0])
    direction = np.array([0.5, -0.25])
    weight = np.array([2.0, 3.0])
    prediction = x * weight
    loss = np.sum(prediction**2)

    def run(graph: ir.Model, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        serialized = ir.to_proto(graph)
        onnx.checker.check_model(serialized, full_check=True)
        return ReferenceEvaluator(serialized).run(None, feeds)

    # Saved-boundary graph utilities must ship without depending on examples.
    from onnax._residuals import split_vjp
    from onnax._saved_shapes import compress

    saved_input = ir.val("saved_x", dtype=ir.DataType.FLOAT, shape=["batch", 3])
    saved_tape = ir.tape.Tape()
    saved_output = saved_tape.op("Tanh", [saved_input])
    saved_output.name = "saved_y"
    saved_output.dtype, saved_output.shape = ir.DataType.FLOAT, ir.Shape(["batch", 3])
    saved_model = ir.Model(
        ir.Graph(
            [saved_input],
            [saved_output],
            nodes=saved_tape.nodes,
            opset_imports={"": 18},
            name="installed_saved_vjp",
        ),
        ir_version=10,
    )
    saved_joint = vjp(saved_model, inputs=["saved_x"])
    saved_forward, saved_backward, saved_names = split_vjp(saved_model, saved_joint)
    saved_forward, saved_backward, saved_names, _ = compress(
        saved_forward, saved_backward, saved_names, outputs=["saved_y"]
    )
    saved_data = np.arange(6, dtype=np.float32).reshape(2, 3) / 10
    saved_values = dict(
        zip(
            (v.name for v in saved_forward.graph.outputs),
            run(saved_forward, {"saved_x": saved_data}),
            strict=True,
        )
    )
    saved_grads = run(
        saved_backward,
        {
            "saved_x": saved_data,
            saved_joint.graph.inputs[-1].name: np.ones_like(saved_data),
            **{name: saved_values[name] for name in saved_names},
        },
    )
    np.testing.assert_allclose(
        saved_grads[0], 1 - np.tanh(saved_data) ** 2, rtol=1e-6, atol=1e-6
    )

    for transform in (grad, lambda m, **kw: GradPass(**kw)(m).model):
        graph = transform(model, inputs=["weight"], output="loss")
        values = run(graph, {"x": x})
        assert [v.name for v in graph.graph.inputs] == ["x"]
        np.testing.assert_allclose(values[0], prediction)
        np.testing.assert_allclose(values[1], loss)
        np.testing.assert_allclose(values[2], 2 * x**2 * weight)

    for transform in (jvp, lambda m, **kw: JvpPass(**kw)(m).model):
        graph = transform(model, inputs=["x"], outputs=["prediction"])
        values = run(graph, {"x": x, graph.graph.inputs[-1].name: direction})
        np.testing.assert_allclose(values[0], prediction)
        np.testing.assert_allclose(values[1], loss)
        np.testing.assert_allclose(values[2], direction * weight)

    for transform in (vjp, lambda m, **kw: VjpPass(**kw)(m).model):
        graph = transform(model, inputs=["weight"], outputs=["prediction"])
        values = run(graph, {"x": x, graph.graph.inputs[-1].name: direction})
        np.testing.assert_allclose(values[0], prediction)
        np.testing.assert_allclose(values[1], loss)
        np.testing.assert_allclose(values[2], direction * x)

    gradient = grad(model, inputs=["weight"], output="loss")
    batch = np.stack([x, x * 2, x * 3])
    for transform in (vmap, lambda m, **kw: VmapPass(**kw)(m).model):
        graph = transform(gradient, in_axes=(0,))
        values = run(graph, {"x": batch})
        np.testing.assert_allclose(values[0], batch * weight)
        np.testing.assert_allclose(values[1], np.sum((batch * weight) ** 2, axis=1))
        np.testing.assert_allclose(values[2], 2 * batch**2 * weight)

    ir.save(gradient, "gradient.onnx")
    hvp = jvp(ir.load("gradient.onnx"), inputs=["weight"])
    values = run(hvp, {"x": x, hvp.graph.inputs[-1].name: direction})
    np.testing.assert_allclose(values[2], 2 * x**2 * weight)
    np.testing.assert_allclose(values[3], 2 * x**2 * direction)
    factored = HoistLinearCoefficientsPass(inputs=[hvp.graph.inputs[-1].name])(hvp).model
    np.testing.assert_allclose(
        run(factored, {"x": x, factored.graph.inputs[-1].name: direction})[-1],
        2 * x**2 * direction,
    )
    assert ir.to_proto(model).SerializeToString() == before

    # Exercise the separately registered extrema family from the installed package.
    for operator, slope in (
        ("Max", np.array([0.0, 0.5, 1.0])),
        ("Min", np.array([1.0, 0.5, 0.0])),
    ):
        value = ir.val("x", dtype=ir.DataType.DOUBLE, shape=[3])
        extrema_tape = ir.tape.Tape()
        zero = extrema_tape.initializer(ir.tensor(np.zeros(3), name="zero"))
        selected = extrema_tape.op(operator, [value, zero])
        square = extrema_tape.op("Mul", [selected, selected])
        total = extrema_tape.op("ReduceSum", [square], {"keepdims": 0})
        total.name = "loss"
        extrema = ir.Model(
            ir.Graph(
                [value],
                [total],
                nodes=extrema_tape.nodes,
                initializers=extrema_tape.initializers,
                opset_imports={"": 18},
                name="installed_extrema",
            ),
            ir_version=10,
        )
        product = jvp(grad(extrema))
        sample = np.array([-1.0, 0.0, 1.0])
        values = run(product, {"x": sample, product.graph.inputs[-1].name: np.ones(3)})
        chosen = np.maximum(sample, 0) if operator == "Max" else np.minimum(sample, 0)
        np.testing.assert_allclose(values[-2], 2 * chosen * slope)
        np.testing.assert_allclose(values[-1], 2 * slope**2)

    readme = distribution.metadata.get_payload()
    example = readme.split("```python\n", 1)[1].split("```", 1)[0]
    namespace: dict = {}
    exec(compile(example, "README.md", "exec"), namespace)
    sample = x.astype(np.float32)
    parameter = weight.astype(np.float32)
    seed = direction.astype(np.float32)
    feeds = {"x": sample, "weight": parameter}
    np.testing.assert_allclose(
        run(namespace["gradient"], feeds)[-1], 2 * sample**2 * parameter
    )
    np.testing.assert_allclose(
        run(ir.load("per_example.onnx"), {**feeds, "x": batch.astype(np.float32)})[-1],
        2 * batch**2 * parameter,
    )
    forward = namespace["pushforward"]
    np.testing.assert_allclose(
        run(forward, {**feeds, forward.graph.inputs[-1].name: seed})[-1],
        np.sum(2 * sample * parameter**2 * seed),
    )
    reverse = namespace["pullback"]
    np.testing.assert_allclose(
        run(reverse, {**feeds, reverse.graph.inputs[-1].name: np.array(1, np.float32)})[-1],
        2 * sample**2 * parameter,
    )
    hvp = namespace["hvp"]
    np.testing.assert_allclose(
        run(hvp, {**feeds, hvp.graph.inputs[-1].name: seed})[-1], 2 * sample**2 * seed
    )
    print(
        f"Installed onnax {distribution.version}: all public exports and compositions passed."
    )


if __name__ == "__main__":
    main()
