# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""A small dense oracle checks exported matrix-free causal-loss products."""

import json

import numpy as np
import onnx
import onnx_ir as ir
import pytest

from examples.huggingface.curvature import INTERFACE, batch_directions, causal_ggn
from examples.lora.train import session


def linear_model(*, default=False):
    x = ir.val(
        "directional_curvature", dtype=ir.DataType.FLOAT, shape=["batch", "sequence", 3]
    )
    w = ir.val("labels", dtype=ir.DataType.FLOAT, shape=[3, 4])
    if default:
        w.const_value = ir.tensor(np.full((3, 4), 0.1, np.float32), name=w.name)
    logits = ir.val("logits", dtype=ir.DataType.FLOAT, shape=["batch", "sequence", 4])
    return ir.Model(
        ir.Graph(
            [x, w],
            [logits],
            nodes=[ir.node("MatMul", [x, w], outputs=[logits])],
            initializers=[w] if default else [],
            opset_imports={"": 18},
            name="linear_tokens",
        ),
        ir_version=10,
    )


def dense_product(x, weights, labels, direction):
    """Materialize a 12x12 Hessian only inside this independent linear-model oracle."""
    logits = (x @ weights)[:, :-1]
    probability = np.exp(logits - logits.max(-1, keepdims=True))
    probability /= probability.sum(-1, keepdims=True)
    valid = labels[:, 1:] != -100
    hessian = np.zeros((weights.size, weights.size), np.float32)
    for index in zip(*np.nonzero(valid), strict=True):
        p = probability[index]
        jacobian = np.kron(x[index][None, :], np.eye(weights.shape[1], dtype=np.float32))
        hessian += jacobian.T @ (np.diag(p) - np.outer(p, p)) @ jacobian
    hessian /= np.float32(max(int(valid.sum()), 1))
    product = (hessian @ direction.ravel()).reshape(weights.shape)
    return np.vdot(direction, product), product


@pytest.fixture(params=[(False, False), (False, True), (True, False), (True, True)])
def exported(request, tmp_path):
    default, optimize = request.param
    source = linear_model(default=default)
    before = ir.to_proto(source).SerializeToString()
    model = causal_ggn(source, ["labels"], optimize=optimize)
    assert ir.to_proto(source).SerializeToString() == before
    path = tmp_path / "ggn.onnx"
    ir.save(model, path)
    onnx.checker.check_model(path, full_check=True)
    metadata = json.loads(model.graph.metadata_props[INTERFACE])
    assert metadata["labels"] != "labels"
    assert metadata["curvature"] != "directional_curvature"
    return source, model, metadata, session(path)


@pytest.mark.parametrize("batch,length", [(1, 1), (1, 5), (2, 7)])
def test_causal_product_matches_dense_oracle(exported, batch, length):
    _, _, metadata, runtime = exported
    rng = np.random.default_rng(42)
    x = rng.normal(size=(batch, length, 3)).astype(np.float32)
    weights = rng.normal(0, 0.2, size=(3, 4)).astype(np.float32)
    direction = rng.normal(size=weights.shape).astype(np.float32)
    labels = rng.integers(0, 4, size=(batch, length), dtype=np.int64)
    labels[:, 2::2] = -100
    feeds = {
        "directional_curvature": x,
        "labels": weights,
        metadata["labels"]: labels,
        metadata["directions"][0]: direction,
    }
    expected = dense_product(x, weights, labels, direction)
    for a, b in zip(runtime.run(None, feeds), expected, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)
    # The first label is outside the causal loss; all-ignored batches produce zero.
    labels[:, 0] = -100
    for a, b in zip(runtime.run(None, feeds), expected, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)
    labels.fill(-100)
    for value in runtime.run(None, feeds):
        np.testing.assert_array_equal(value, np.zeros_like(value))


def test_batched_directions_match_independent_calls(exported, tmp_path):
    _, model, metadata, runtime = exported
    mapped = batch_directions(model)
    path = tmp_path / "batched.onnx"
    ir.save(mapped, path)
    onnx.checker.check_model(path, full_check=True)
    rng = np.random.default_rng(7)
    direction = rng.normal(size=(3, 3, 4)).astype(np.float32)
    feeds = {
        "directional_curvature": rng.normal(size=(2, 5, 3)).astype(np.float32),
        "labels": rng.normal(size=(3, 4)).astype(np.float32),
        metadata["labels"]: np.array([[1, 2, 3, -100, -100], [1, 0, 2, 3, 1]], np.int64),
    }
    seed = metadata["directions"][0]
    actual = session(path).run(None, {**feeds, seed: direction})
    independent = [runtime.run(None, {**feeds, seed: value}) for value in direction]
    for index, value in enumerate(actual):
        np.testing.assert_allclose(
            value, np.stack([row[index] for row in independent]), rtol=1e-4, atol=1e-6
        )


@pytest.mark.parametrize("names", [[], ["missing"], ["labels", "labels"]])
def test_parameter_selection_is_explicit(names):
    with pytest.raises(ValueError, match="unique runtime parameter"):
        causal_ggn(linear_model(), names)


def test_valid_token_microbatch_weighting(exported):
    _, _, metadata, runtime = exported
    rng = np.random.default_rng(23)
    x = rng.normal(size=(2, 5, 3)).astype(np.float32)
    weights = rng.normal(size=(3, 4)).astype(np.float32)
    direction = rng.normal(size=weights.shape).astype(np.float32)
    labels = np.array([[1, 2, 3, -100, -100], [1, 0, 2, 3, 1]], np.int64)

    def product(x, labels):
        return runtime.run(
            None,
            {
                "directional_curvature": x,
                "labels": weights,
                metadata["labels"]: labels,
                metadata["directions"][0]: direction,
            },
        )

    full = product(x, labels)
    chunks = [product(x[index : index + 1], labels[index : index + 1]) for index in range(2)]
    counts = (labels[:, 1:] != -100).sum(axis=1)
    for index, actual in enumerate(full):
        expected = sum(
            np.float32(count / counts.sum()) * chunk[index]
            for count, chunk in zip(counts, chunks, strict=True)
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)
