# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Exchange an exported checkpointed Qwen adapter through PyTorch PEFT."""

import argparse
import functools
import json
import os

import numpy as np
import torch

from examples.huggingface.checkpointing import CheckpointedModule
from examples.huggingface.chunked_loss import ChunkedHead
from examples.huggingface.peft_adapter import roundtrip
from examples.huggingface.qwen import MODEL_ID, reference_model
from examples.huggingface.replay import _equal_array
from examples.huggingface.sft import OnnxQwen3
from examples.lora.train import session


def verify(directory, output, *, session_factory=session):
    if os.path.exists(output):
        raise FileExistsError("Select a new PEFT audit directory")
    export = os.path.join(directory, "export")
    with open(os.path.join(directory, "adapter/adapter_mapping.json")) as stream:
        metadata = json.load(stream)
    if metadata["base_model"] not in ("tiny-qwen3", MODEL_ID):
        raise ValueError("Expected the pinned tiny or real Qwen base")
    reference = reference_model(real=metadata["base_model"] == MODEL_ID)
    if metadata["base_quantization"] is not None:
        from examples.huggingface.nf4 import quantize_reference

        quantize_reference(reference)
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    class Loaded(torch.nn.Module):
        forward = OnnxQwen3.forward

        def __init__(self):
            super().__init__()
            self.onnx = CheckpointedModule.load(export, session_factory=session_factory)
            self.head = ChunkedHead.load(export, session_factory=session_factory)
            self.mode = "chunked"
            self.config = reference.config
            self.mapping = metadata["projections"]
            self.quantization = metadata["base_quantization"]
            self.projection_precision = self.head.projection_precision

    model = Loaded().eval()
    initial = model.onnx.arrays()
    ids = torch.tensor([[1, 4, 5, 6, 2], [1, 7, 2, 0, 0]])
    mask = ids.ne(0).long()
    labels = ids.clone()
    labels[mask == 0] = -100
    batch = {"input_ids": ids, "attention_mask": mask, "labels": labels}

    def capture():
        model.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        return [
            loss.detach().numpy().copy(),
            *(p.grad.numpy().copy() for p in model.onnx.adapters),
        ]

    before = capture()
    report = roundtrip(model, reference, batch, output, diagnose_rounding=True)
    # Restore the trained checkpoint after the audit's independent one-step update.
    with torch.no_grad():
        for name, parameter in zip(model.onnx.names, model.onnx.adapters, strict=True):
            parameter.copy_(torch.from_numpy(initial[name]))
    after = capture()
    for left, right in zip(before, after, strict=True):
        _equal_array(left, right)
    with np.load(os.path.join(export, "adapter.npz")) as arrays:
        for name, value in model.onnx.arrays().items():
            _equal_array(value, arrays[name])
    report["trained_checkpoint_restored_bitwise"] = True
    report["restored_loss_and_gradients_bitwise_equal"] = True
    report["projection_precision"] = model.projection_precision
    report["source"] = os.path.abspath(directory)
    with open(os.path.join(output, "report.json"), "w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--provider", choices=("cpu", "mlx"), default="cpu")
    parser.add_argument("--library")
    args = parser.parse_args()
    torch.set_num_threads(1)
    factory = functools.partial(session, memory_reuse=False)
    if args.provider == "mlx":
        from examples.lora.mlx_runtime import register_library
        from examples.lora.mlx_runtime import session as mlx_session

        if args.library:
            register_library(args.library)
        factory = mlx_session
    elif args.library:
        parser.error("--library requires --provider mlx")
    print(json.dumps(verify(args.training_dir, args.output_dir, session_factory=factory)))


if __name__ == "__main__":
    main()
