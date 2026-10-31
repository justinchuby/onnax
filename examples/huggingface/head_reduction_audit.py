# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Audit the real FP32 frozen-head input gradient against scalar dot products."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import tempfile

import numpy as np
import onnx_ir as ir
import torch

from examples.huggingface.layer_audit import compare
from examples.huggingface.nf4 import lower_decoder
from examples.huggingface.qwen import REVISION, build_base, decompose, inject, reference_model
from examples.huggingface.reference_head import pairwise_matmul
from examples.huggingface.rms_audit import diagnostic_session
from examples.huggingface.serialization import save_graph
from examples.huggingface.sft import token_loss


def run(
    *,
    real=False,
    nf4=False,
    projection_precision="fp32",
    provider="cpu",
    library=None,
    capture=None,
):
    if projection_precision not in ("fp32", "bf16"):
        raise ValueError("Expected fp32 or bf16 projection precision")
    if provider not in ("cpu", "reference", "mlx"):
        raise ValueError("Expected cpu, reference or mlx provider")
    if projection_precision == "bf16" and provider == "cpu":
        raise ValueError("BF16 forward requires reference or mlx")
    if library is not None and provider != "mlx":
        raise ValueError("A development library requires the mlx provider")
    factory = diagnostic_session
    library_hash = None
    if provider == "reference":
        from examples.huggingface.serialization import reference_session

        factory = reference_session
    elif provider == "mlx":
        from examples.lora.mlx_runtime import register_library, session

        factory = session
        if library is not None:
            register_library(library)
            with open(library, "rb") as stream:
                library_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    torch.set_num_threads(1)
    reference = reference_model(real)
    model, params, _ = inject(build_base(reference))
    model = decompose(model)
    if nf4:
        model, _ = lower_decoder(model, reference, precision=projection_precision)
    weight = reference.lm_head.weight.detach()
    rhs = model.graph.outputs[0].producer().inputs[1]
    stored = (
        rhs.const_value.numpy()
        if rhs.const_value is not None
        else rhs.producer().inputs[0].const_value.numpy().T
    )
    np.testing.assert_array_equal(stored.T, weight.numpy())
    if projection_precision == "bf16":
        from examples.huggingface.mixed_precision import bf16_projections

        model = bf16_projections(model)
        weight = weight.to(torch.bfloat16).float()
    ids = np.arange(3, 67, dtype=np.int64)[None]
    feeds = {
        "input_ids": ids,
        "attention_mask": np.ones_like(ids),
        "position_ids": np.arange(64, dtype=np.int64)[None],
        **params,
    }
    with tempfile.TemporaryDirectory(prefix="onnax-head-reduction-") as directory:
        path = os.path.join(directory, "forward.onnx")
        save_graph(model, path, data_name="forward.data")
        executor = factory(path)
        logits = torch.tensor(executor.run(["logits"], feeds)[0], requires_grad=True)
        del executor
    loss = token_loss(logits, torch.from_numpy(ids))
    loss.backward()
    seed = logits.grad
    flat = projection_precision == "bf16"
    seed_array = seed.numpy().reshape(-1, seed.shape[-1]) if flat else seed.numpy()
    weight_array = np.ascontiguousarray(weight.numpy().T) if flat else weight.numpy()
    g = ir.val("g", dtype=ir.DataType.FLOAT, shape=list(seed_array.shape))
    w = ir.val("w", dtype=ir.DataType.FLOAT, shape=list(weight_array.shape))
    tape = ir.tape.Tape()
    rhs = tape.op("Transpose", [w], {"perm": [1, 0]}) if flat else w
    y = tape.op("MatMul", [g, rhs])
    y.name, y.dtype, y.shape = (
        "head_gradient",
        ir.DataType.FLOAT,
        ir.Shape([*seed_array.shape[:-1], weight.shape[1]]),
    )
    graph = ir.Model(
        ir.Graph([g, w], [y], nodes=tape.nodes, opset_imports={"": 23}), ir_version=10
    )
    actual = (
        factory(ir.to_proto(graph).SerializeToString())
        .run(None, {"g": seed_array, "w": weight_array})[0]
        .reshape(*seed.shape[:-1], weight.shape[1])
    )
    ordinary = (seed @ weight).numpy()
    balanced = pairwise_matmul(seed, weight).numpy()
    numpy = seed.numpy() @ weight.numpy()
    for value in (actual, ordinary, balanced, numpy):
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise AssertionError("Expected finite FP32 head gradients")
    if capture is not None:
        np.savez(
            capture,
            seed=seed.numpy(),
            weight=weight.numpy(),
            actual=actual,
            torch=ordinary,
            pairwise=balanced,
        )
    difference = np.abs(actual - ordinary)
    coordinates = []
    for score in (difference, difference / (1e-6 + 1e-4 * np.abs(ordinary))):
        for index in np.argsort(score.reshape(-1))[-8:][::-1]:
            coordinate = tuple(int(i) for i in np.unravel_index(index, score.shape))
            if coordinate not in coordinates:
                coordinates.append(coordinate)
    samples = []
    seed_array, weight_array = seed.numpy(), weight.numpy()
    for batch, token, column in coordinates:
        exact = math.fsum(
            float(a) * float(b)
            for a, b in zip(seed_array[batch, token], weight_array[:, column], strict=True)
        )
        coordinate = (batch, token, column)
        values = {
            ("ort_cpu" if provider == "cpu" else provider): float(actual[coordinate]),
            "torch": float(ordinary[coordinate]),
            "numpy": float(numpy[coordinate]),
            "pairwise_fp32": float(balanced[coordinate]),
        }
        samples.append(
            {
                "coordinate": coordinate,
                "scalar_fsum": exact,
                "nearest_fp32": float(np.float32(exact)),
                "values": values,
                "absolute_errors": {
                    name: abs(value - exact) for name, value in values.items()
                },
            }
        )
    return {
        "model": "Qwen3-0.6B" if real else "tiny Qwen3",
        "revision": REVISION if real else None,
        "nf4": nf4,
        "compute_dtype": "float32",
        "projection_precision": projection_precision,
        "provider": provider,
        "library_sha256": library_hash,
        "head_weight": "BF16-rounded FP32" if projection_precision == "bf16" else "FP32",
        "runtime_product": "Rank-two g @ Transpose(contiguous weight.T)"
        if flat
        else "Batched g @ weight",
        "comparisons_to_pairwise_fp32": {
            ("ort_cpu" if provider == "cpu" else provider): compare(actual, balanced),
            "torch": compare(ordinary, balanced),
            "numpy": compare(numpy, balanced),
        },
        "oracle": "math.fsum of binary64 products of the stored FP32 operands",
        "seed_shape": list(seed.shape),
        "weight_shape": list(weight.shape),
        "input_ids": ids.tolist(),
        "pairwise_block_size": 128,
        "torch_matmul_precision": torch.get_float32_matmul_precision(),
        "numpy_blas": np.show_config(mode="dicts")["Build Dependencies"]["blas"],
        "ort_graph_optimization": "disabled",
        "threads": 1,
        "samples": samples,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("onnax", "onnxruntime", "torch", "numpy", "mobius-onnx")
        },
        "platform": platform.platform(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--nf4", action="store_true")
    parser.add_argument("--projection-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--provider", choices=["cpu", "reference", "mlx"], default="cpu")
    parser.add_argument("--library")
    parser.add_argument(
        "--capture", help="Save the exact FP32 dot-product operands and outputs as NPZ"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(
        real=args.real,
        nf4=args.nf4,
        projection_precision=args.projection_precision,
        provider=args.provider,
        library=args.library,
        capture=args.capture,
    )
    with open(args.output, "w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
