# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Verify checkpoint state and loaded ONNX outputs after fresh-process resume."""

import argparse
import functools
import hashlib
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
    if interface.get("kind") == "checkpointed_vjp":
        from examples.huggingface.checkpointing import CheckpointedModule
        from examples.huggingface.chunked_loss import ChunkedHead

        export = os.path.join(directory, "export")
        body = CheckpointedModule.load(export, session_factory=session_factory)
        head = ChunkedHead.load(export, session_factory=session_factory)
        hidden = body(**{name: torch.from_numpy(value) for name, value in feeds.items()})
        labels = feeds["input_ids"].copy()
        labels[feeds["attention_mask"] == 0] = -100
        loss = head(hidden, torch.from_numpy(labels))
        loss.backward()
        result = [loss.detach().numpy(), *(p.grad.detach().numpy() for p in body.adapters)]
        if any(not np.isfinite(value).all() for value in result):
            raise AssertionError("Nonfinite exported checkpointed training outputs")
        return result
    labels = interface["causal_loss"]["labels"]
    targets = feeds["input_ids"].copy()
    targets[feeds["attention_mask"] == 0] = -100
    inputs = {**feeds, labels: targets}
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


def verify(first, replay, *, step=30, provider="cpu", library=None, feeds=None):
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
    if feeds is None:
        ids = np.array([[1, 4, 5, 6, 7, 8, 9, 2]], np.int64)
        feeds = {
            "input_ids": ids,
            "attention_mask": np.ones_like(ids),
            "position_ids": np.arange(8, dtype=np.int64)[None],
        }
    if set(feeds) != {"input_ids", "attention_mask", "position_ids"}:
        raise ValueError("Provide input_ids, attention_mask and position_ids")
    if any(
        value.dtype != np.int64 or value.shape != feeds["input_ids"].shape or value.ndim != 2
        for value in feeds.values()
    ):
        raise ValueError("Replay inputs must be matching rank-2 int64 arrays")
    outputs = []
    for directory in (first, replay):
        executor = factory(os.path.join(directory, "export/forward.onnx"))
        arrays = executor.run(None, feeds)
        outputs.append(
            [
                {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "sha256": hashlib.sha256(
                        memoryview(np.ascontiguousarray(value)).cast("B")
                    ).hexdigest(),
                }
                for value in arrays
            ]
        )
        del arrays, executor
    _equal_tree(outputs[0], outputs[1])
    result = {
        "step": step,
        "input_shape": list(feeds["input_ids"].shape),
        "input_sha256": hashlib.sha256(feeds["input_ids"].tobytes()).hexdigest(),
        "padding_tokens": int((feeds["attention_mask"] == 0).sum()),
        "forward_output_digests": outputs[0],
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
    if interface[0].get("kind") in ("loss_gradient", "checkpointed_vjp"):
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
    parser.add_argument(
        "--input-file", help="Real Qwen text/JSONL input for long-context replay"
    )
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--padding-tokens", type=int, default=0)
    args = parser.parse_args()
    feeds = None
    if args.input_file:
        from examples.huggingface.sft import texts, tokenizer

        if not 0 <= args.padding_tokens <= args.sequence_length - 2:
            parser.error("Leave at least two nonpadding tokens")
        tok = tokenizer(real=True)
        ids = next(
            (
                value
                for text in texts(args.input_file)
                if len(value := tok(text)["input_ids"]) >= args.sequence_length
            ),
            None,
        )
        if ids is None:
            parser.error("No input reaches the requested sequence length")
        ids = np.asarray([ids[: args.sequence_length]], dtype=np.int64)
        mask = np.ones_like(ids)
        if args.padding_tokens:
            ids[:, -args.padding_tokens :] = tok.pad_token_id
            mask[:, -args.padding_tokens :] = 0
        feeds = {
            "input_ids": ids,
            "attention_mask": mask,
            "position_ids": np.maximum(mask.cumsum(-1) - 1, 0),
        }
    result = verify(
        args.uninterrupted,
        args.resumed,
        step=args.step,
        provider=args.provider,
        library=args.library,
        feeds=feeds,
    )
    with open(args.output, "w") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
