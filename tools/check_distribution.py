# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Execute installed public transforms; copy/run outside the source checkout."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import numpy as np
import onnx
import onnx_ir as ir
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

import onnax
from onnax import GradPass, JvpPass, VjpPass, VmapPass, grad, jvp, vjp, vmap


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
    assert set(distribution.metadata.get_all("License-File")) == {"LICENSE", "NOTICE"}
    for name in ("LICENSE", "NOTICE"):
        assert distribution.read_text(f"licenses/{name}"), (
            f"Missing {name} in installed metadata. Include both files in license-files."
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
    }

    proto = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Mul", ["x", "weight"], ["prediction"]),
                helper.make_node("Mul", ["prediction", "prediction"], ["squared"]),
                helper.make_node("ReduceSum", ["squared"], ["loss"], keepdims=0),
            ],
            "distribution_smoke",
            [helper.make_tensor_value_info("x", TensorProto.DOUBLE, [2])],
            [
                helper.make_tensor_value_info("prediction", TensorProto.DOUBLE, [2]),
                helper.make_tensor_value_info("loss", TensorProto.DOUBLE, []),
            ],
            initializer=[helper.make_tensor("weight", TensorProto.DOUBLE, [2], [2, 3])],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    model = ir.from_proto(proto)
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
    print(
        f"Installed onnax {distribution.version}: all eight exports and compositions passed."
    )


if __name__ == "__main__":
    main()
