# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Shared external weights preserve values and active file mappings."""

import numpy as np
import onnx
import onnx_ir as ir
from onnx.reference import ReferenceEvaluator

from examples.huggingface.serialization import SharedWeights


def model(weight):
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[1, 128])
    tape = ir.tape.Tape()
    w = tape.initializer(ir.tensor(weight, name="weight"))
    y = tape.op("MatMul", [x, w])
    y.name, y.dtype, y.shape = "y", ir.DataType.FLOAT, ir.Shape([1, 128])
    return ir.Model(
        ir.Graph(
            [x],
            [y],
            nodes=tape.nodes,
            initializers=tape.initializers,
            opset_imports={"": 18},
            name="shared_weights",
        ),
        ir_version=10,
    )


def test_shared_payload_values_deduplication_and_same_name(tmp_path):
    weight = np.arange(128 * 128, dtype=np.float32).reshape(128, 128) / 100
    models = [model(weight), model(weight.copy()), model(2 * weight)]
    with SharedWeights(tmp_path) as writer:
        paths = [writer.save(m, f"part_{i}") for i, m in enumerate(models)]
    payloads = list(tmp_path.glob("weights-*.data"))
    assert len(payloads) == 1
    assert payloads[0].stat().st_size == 2 * weight.nbytes
    x = np.ones((1, 128), np.float32)
    for i, path in enumerate(paths):
        onnx.checker.check_model(path, full_check=True)
        actual = ReferenceEvaluator(path).run(None, {"x": x})[0]
        np.testing.assert_allclose(actual, x @ (2 * weight if i == 2 else weight), rtol=1e-6)
    assert not isinstance(
        models[0].graph.initializers["weight"].const_value, ir.ExternalTensor
    )


def test_resave_preserves_existing_external_weight_mapping(tmp_path):
    weight = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
    with SharedWeights(tmp_path) as writer:
        path = writer.save(model(weight), "original")
    loaded = ir.load(path)
    mapped = loaded.graph.initializers["weight"].const_value.numpy()
    with SharedWeights(tmp_path) as writer:
        second = writer.save(loaded, "second")
    assert len(list(tmp_path.glob("weights-*.data"))) == 1
    np.testing.assert_array_equal(mapped, weight)
    np.testing.assert_array_equal(
        ir.load(second).graph.initializers["weight"].const_value.numpy(), weight
    )
