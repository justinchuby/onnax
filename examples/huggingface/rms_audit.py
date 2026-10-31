# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Measure Qwen FP32 gradient differences with a sqrt/div RMSNorm control."""

import argparse
import importlib.metadata
import json
import platform
from contextlib import contextmanager
from types import MethodType

import numpy as np
import onnxruntime as ort
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from examples.huggingface.sft import prepare
from examples.huggingface.validation import add_reference_adapters, audit


def diagnostic_session(path):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.enable_mem_reuse = False
    return ort.InferenceSession(
        path if isinstance(path, bytes) else str(path),
        options,
        providers=["CPUExecutionProvider"],
    )


def sqrt_div(self, hidden_states):
    x = hidden_states.to(torch.float32)
    denominator = torch.sqrt(x.square().mean(-1, keepdim=True) + self.variance_epsilon)
    return self.weight * (x / denominator).to(hidden_states.dtype)


@contextmanager
def division_control(reference):
    """Temporarily replace Qwen RMSNorm arithmetic; restore instance overrides."""
    originals = []
    try:
        for module in reference.modules():
            if isinstance(module, Qwen3RMSNorm):
                originals.append((module, module.__dict__.get("forward")))
                module.forward = MethodType(sqrt_div, module)
        yield len(originals)
    finally:
        for module, forward in originals:
            if forward is None:
                del module.forward
            else:
                module.forward = forward


def run(*, real=False, nf4=False):
    torch.set_num_threads(1)
    model, reference = prepare(
        real=real, nf4=nf4, save_activations=True, session_factory=diagnostic_session
    )
    reference = add_reference_adapters(reference, model)
    initial = model.onnx.arrays()
    ids = torch.arange(3, 67).reshape(1, -1)
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}
    baseline = audit(model, reference, batch, diagnose_rounding=True)
    with division_control(reference) as count:
        control = audit(model, reference, batch, diagnose_rounding=True)
    for name, value in model.onnx.arrays().items():
        np.testing.assert_array_equal(value, initial[name])
    return {
        "model": "Qwen3-0.6B" if real else "tiny Qwen3",
        "nf4": nf4,
        "dtype": "float32",
        "input_ids": ids.tolist(),
        "adapter_updates": 0,
        "normalizations": count,
        "ort_graph_optimization": "disabled",
        "ort_memory_reuse": False,
        "threads": 1,
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "onnax",
                "onnx",
                "onnxruntime",
                "torch",
                "transformers",
                "mobius-onnx",
            )
        },
        "platform": platform.platform(),
        "audits": {"torch_rsqrt": baseline, "torch_sqrt_div": control},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--nf4", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(real=args.real, nf4=args.nf4)
    with open(args.output, "w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
