# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Causal-token GGN products compose on a Mobius decoder with an independent model."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional model dependencies.

import json
import os

import numpy as np
import onnx
import onnx_ir as ir
import pytest

pytest.importorskip("mobius")
pytest.importorskip("trl")
torch = pytest.importorskip("torch")

from examples.huggingface.curvature import INTERFACE, batch_directions, causal_ggn
from examples.huggingface.qwen import build_base, decompose, inject, reference_model
from examples.huggingface.validate_curvature import reference_product
from examples.huggingface.validation import add_reference_parameters
from examples.lora.train import session


@pytest.fixture(scope="module", params=["cpu", "mlx"])
def workload(request, tmp_path_factory):
    if request.param == "mlx" and os.environ.get("ONNAX_TEST_MLX_EP") != "1":
        pytest.skip("Native MLX opt-in")
    torch.set_num_threads(1)
    reference = reference_model()
    source, parameters, mapping = inject(build_base(reference))
    source = decompose(source)
    before = ir.to_proto(source).SerializeToString()
    model = causal_ggn(source, list(parameters))
    assert ir.to_proto(source).SerializeToString() == before
    directory = tmp_path_factory.mktemp("causal-qwen")
    path = directory / "ggn.onnx"
    ir.save(model, path)
    onnx.checker.check_model(path, full_check=True)
    factory = session
    if request.param == "mlx":
        from examples.lora.mlx_runtime import register_library
        from examples.lora.mlx_runtime import session as mlx_session

        register_library(os.environ["ONNAX_MLX_EP_LIBRARY"])
        factory = mlx_session
    rng = np.random.default_rng(42)
    for row in mapping:
        parameters[row["B"]] = rng.normal(0, 0.02, parameters[row["B"]].shape).astype(
            np.float32
        )
    reference = add_reference_parameters(reference, parameters, mapping)
    interface = json.loads(model.graph.metadata_props[INTERFACE])
    directions = [
        rng.normal(size=parameters[name].shape).astype(np.float32)
        for name in interface["parameters"]
    ]
    return (
        factory(path),
        model,
        interface,
        reference,
        parameters,
        mapping,
        directions,
        factory,
        directory,
    )


def batch(length=5):
    if length == 5:
        ids = np.array([[1, 4, 5, 6, 2], [1, 7, 2, 0, 0]], np.int64)
    else:
        ids = np.array([[1, 2][:length]], np.int64)
    mask = (ids != 0).astype(np.int64)
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "position_ids": np.maximum(mask.cumsum(-1) - 1, 0),
    }, np.where(mask, ids, -100).astype(np.int64)


@pytest.mark.parametrize(
    "case", ["padded", "prompt_mask", "positions", "short", "ignored", "one_token"]
)
def test_exported_product_matches_transformers(workload, case):
    runtime, _, interface, reference, parameters, mapping, directions, _, _ = workload
    inputs, labels = batch(2 if case == "short" else 1 if case == "one_token" else 5)
    if case == "prompt_mask":
        labels[:, 1:3] = -100
    if case == "positions":
        inputs["position_ids"] = (
            inputs["position_ids"] + np.arange(labels.shape[1], dtype=np.int64) % 2
        )
    if case == "ignored":
        labels.fill(-100)
    actual = runtime.run(
        None,
        {
            **inputs,
            **parameters,
            interface["labels"]: labels,
            **dict(zip(interface["directions"], directions, strict=True)),
        },
    )
    if not np.any(labels[:, 1:] != -100):
        for value in actual:
            np.testing.assert_array_equal(value, np.zeros_like(value))
        return
    expected = reference_product(reference, mapping, parameters, inputs, labels, directions)
    for value, target in zip(actual, expected, strict=True):
        assert np.isfinite(value).all()
        np.testing.assert_allclose(value, target, rtol=1e-4, atol=1e-6)
    assert float(actual[0]) > 0


def test_mapped_directions_share_the_primal(workload):
    runtime, model, interface, _, parameters, _, _, factory, directory = workload
    mapped = batch_directions(model)
    path = directory / "batched.onnx"
    ir.save(mapped, path)
    onnx.checker.check_model(path, full_check=True)
    inputs, labels = batch()
    rng = np.random.default_rng(19)
    vectors = {
        seed: rng.normal(size=(3, *parameters[name].shape)).astype(np.float32)
        for seed, name in zip(interface["directions"], interface["parameters"], strict=True)
    }
    feeds = {**parameters, **inputs, interface["labels"]: labels}
    actual = factory(path).run(None, {**feeds, **vectors})
    expected = [
        runtime.run(None, {**feeds, **{name: value[index] for name, value in vectors.items()}})
        for index in range(3)
    ]
    for index, value in enumerate(actual):
        np.testing.assert_allclose(
            value, np.stack([row[index] for row in expected]), rtol=1e-4, atol=1e-6
        )
