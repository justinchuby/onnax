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
    }

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
    print(f"Installed onnax {distribution.version}: all nine exports and compositions passed.")


if __name__ == "__main__":
    main()
