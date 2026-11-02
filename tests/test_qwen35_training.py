# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Hybrid text-model training, frozen base and fresh-process optimizer recovery."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional model dependencies.

import os
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("mobius")
torch = pytest.importorskip("torch")
from onnxscript import optimizer

from examples.huggingface.qwen35 import prepare_graph
from examples.huggingface.qwen35_train import run
from examples.huggingface.sft import token_loss
from examples.huggingface.validation import add_reference_parameters
from onnax.torch import OnnxModule


def test_training_regression(tmp_path):
    report = run(tmp_path / "train", steps=100)
    assert report["train_loss"][1] < report["train_loss"][0]
    assert report["validation_loss"][1] < report["validation_loss"][0]
    assert report["adapter_tensors"] == 10
    assert report["frozen_weights_unchanged"] and report["export_reload_bitwise"]
    assert report["bound_forward_loaded_and_executed"]


def test_dynamic_right_padding_nonzero_adapters_and_position_ids():
    torch.set_num_threads(1)
    graph, arrays, mapping, reference = prepare_graph()
    rng = np.random.default_rng(71)
    for name in arrays:
        if name.endswith(".B"):
            arrays[name] = rng.normal(0, 0.02, arrays[name].shape).astype(np.float32)
    reference = add_reference_parameters(reference, arrays, mapping)
    with OnnxModule(
        graph, arrays, outputs=["logits"], input_gradients=[], optimize=optimizer.optimize
    ) as model:
        for batch_size, length in ((1, 5), (2, 9), (1, 37)):
            ids = torch.from_numpy(rng.integers(3, 64, (batch_size, length), dtype=np.int64))
            mask = torch.ones_like(ids)
            mask[0, -2:] = 0
            ids[mask == 0] = 0
            positions = (mask.cumsum(-1) - 1).clamp(min=0)
            feeds = dict(input_ids=ids, attention_mask=mask, position_ids=positions)
            actual = model(**feeds)
            expected = reference(**feeds, use_cache=False).logits
            torch.testing.assert_close(
                actual[mask.bool()], expected[mask.bool()], rtol=1e-4, atol=1e-6
            )
            labels = ids.clone()
            labels[mask == 0] = -100
            token_loss(actual, labels).backward()
            token_loss(expected, labels).backward()
            for row in mapping:
                target = reference.get_submodule(row["module"])
                for key in ("A", "B"):
                    torch.testing.assert_close(
                        model.parameters_by_name()[row[key]].grad,
                        getattr(target, key).grad,
                        rtol=1e-4,
                        atol=1e-6,
                    )
            model.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
        ignored = token_loss(model(**feeds), torch.full_like(ids, -100))
        assert ignored.item() == 0
        ignored.backward()
        assert all(parameter.grad.count_nonzero() == 0 for parameter in model.parameters())


def test_checkpoint_recovery_in_new_process(tmp_path):
    full, first, resumed = (tmp_path / name for name in ("full", "first", "resumed"))
    run(full, steps=8)
    run(first, steps=4)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.huggingface.qwen35_train",
            "--output-dir",
            str(resumed),
            "--steps",
            "8",
            "--resume",
            str(first / "checkpoint.pt"),
        ],
        check=True,
        env=os.environ.copy(),
        stdout=subprocess.DEVNULL,
    )
    expected = torch.load(full / "checkpoint.pt", weights_only=True)
    actual = torch.load(resumed / "checkpoint.pt", weights_only=True)

    def equal(left, right):
        assert type(left) is type(right)
        if isinstance(left, torch.Tensor):
            assert left.dtype == right.dtype and left.shape == right.shape
            assert torch.equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            assert len(left) == len(right)
            for a, b in zip(left, right, strict=True):
                equal(a, b)
        else:
            assert left == right

    equal(expected, actual)
