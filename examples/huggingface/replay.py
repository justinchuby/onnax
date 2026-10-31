# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Verify checkpoint state and loaded ONNX outputs after fresh-process resume."""

import argparse
import functools
import json
import os

import numpy as np
import torch
from safetensors.numpy import load_file

from examples.lora.train import session


def _equal_array(a, b):
    if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
        raise AssertionError("Array dtype, shape or bytes differ")


def _equal_tree(a, b):
    if type(a) is not type(b):
        raise AssertionError(f"State types differ: {type(a)} versus {type(b)}")
    if isinstance(a, torch.Tensor):
        if a.dtype != b.dtype or a.shape != b.shape:
            raise AssertionError("Tensor dtype or shape differs")
        _equal_array(
            a.contiguous().reshape(-1).view(torch.uint8).numpy(),
            b.contiguous().reshape(-1).view(torch.uint8).numpy(),
        )
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError("State keys differ")
        for key in a:
            _equal_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError("State lengths differ")
        for x, y in zip(a, b, strict=True):
            _equal_tree(x, y)
    elif a != b:
        raise AssertionError(f"State values differ: {a} versus {b}")


def _training_outputs(directory, interface, feeds, session_factory):
    labels = interface["causal_loss"]["labels"]
    inputs = {**feeds, labels: feeds["input_ids"].copy()}
    with np.load(os.path.join(directory, "export/adapter.npz")) as parameters:
        for row in interface["parameters"]:
            value = parameters[row["name"]]
            if value.dtype != np.float32 or list(value.shape) != row["shape"]:
                raise AssertionError(f"Exported parameter interface differs: {row['name']}")
            inputs[row["name"]] = value
    names = [interface["output"], *(row["gradient"] for row in interface["parameters"])]
    executor = session_factory(os.path.join(directory, "export", interface["model"]))
    values = executor.run(names, inputs)
    shapes = [[], *(row["shape"] for row in interface["parameters"])]
    for name, value, shape in zip(names, values, shapes, strict=True):
        if value.dtype != np.float32 or list(value.shape) != shape:
            raise AssertionError(f"Exported training output interface differs: {name}")
        if not np.isfinite(value).all():
            raise AssertionError(f"Nonfinite exported training output: {name}")
    return values


def verify(first, replay, *, step=30, provider="cpu", library=None):
    if provider not in ("cpu", "mlx", "reference"):
        raise ValueError("Expected cpu, mlx or reference provider")
    if library is not None and provider != "mlx":
        raise ValueError("A development library requires the mlx provider")
    # Large derivative graphs encounter ORT's quadratic reuse planner (#103).
    factory = functools.partial(session, memory_reuse=False)
    if provider == "reference":
        from examples.huggingface.serialization import reference_session

        factory = reference_session
    elif provider == "mlx":
        from examples.lora.mlx_runtime import register_library
        from examples.lora.mlx_runtime import session as mlx_session

        if library:
            register_library(library)
        factory = mlx_session
    with (
        np.load(os.path.join(first, "export/adapter.npz")) as a,
        np.load(os.path.join(replay, "export/adapter.npz")) as b,
    ):
        if a.files != b.files:
            raise AssertionError("Adapter names differ")
        for name in a.files:
            _equal_array(a[name], b[name])
        count = len(a.files)
    original_state = load_file(os.path.join(first, f"checkpoint-{step}/model.safetensors"))
    resumed_state = load_file(os.path.join(replay, f"checkpoint-{step}/model.safetensors"))
    if original_state.keys() != resumed_state.keys():
        raise AssertionError("Checkpoint tensor names differ")
    for name in original_state:
        _equal_array(original_state[name], resumed_state[name])
    for name in ("optimizer.pt", "scheduler.pt"):
        _equal_tree(
            torch.load(
                os.path.join(first, f"checkpoint-{step}", name),
                map_location="cpu",
                weights_only=True,
            ),
            torch.load(
                os.path.join(replay, f"checkpoint-{step}", name),
                map_location="cpu",
                weights_only=True,
            ),
        )
    interface = []
    for directory in (first, replay):
        with open(os.path.join(directory, "export/interface.json")) as stream:
            interface.append(json.load(stream))
    _equal_tree(interface[0], interface[1])
    ids = np.array([[1, 4, 5, 6, 7, 8, 9, 2]], np.int64)
    feeds = {
        "input_ids": ids,
        "attention_mask": np.ones_like(ids),
        "position_ids": np.arange(8, dtype=np.int64)[None],
    }
    outputs = []
    for directory in (first, replay):
        executor = factory(os.path.join(directory, "export/forward.onnx"))
        outputs.append(executor.run(None, feeds))
        del executor
    for x, y in zip(*outputs, strict=True):
        _equal_array(x, y)
    result = {
        "step": step,
        "adapter_tensors": count,
        "checkpoint_tensors": len(original_state),
        "frozen_initializers": len(interface[0]["frozen"]),
        "adapter_bitwise_equal": True,
        "checkpoint_bitwise_equal": True,
        "optimizer_bitwise_equal": True,
        "scheduler_equal": True,
        "frozen_hashes_equal": True,
        "export_load_outputs_bitwise_equal": True,
        "provider": provider,
        "ort_memory_reuse": False if provider == "cpu" else None,
    }
    if interface[0].get("kind") == "loss_gradient":
        training = [
            _training_outputs(directory, interface[0], feeds, factory)
            for directory in (first, replay)
        ]
        for x, y in zip(*training, strict=True):
            _equal_array(x, y)
        result["export_load_loss_gradients_bitwise_equal"] = True
        result["exported_gradient_tensors"] = len(training[0]) - 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uninterrupted", required=True)
    parser.add_argument("--resumed", required=True)
    parser.add_argument("--step", type=int, default=30)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provider", choices=["cpu", "mlx", "reference"], default="cpu")
    parser.add_argument("--library")
    args = parser.parse_args()
    result = verify(
        args.uninterrupted,
        args.resumed,
        step=args.step,
        provider=args.provider,
        library=args.library,
    )
    with open(args.output, "w") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
