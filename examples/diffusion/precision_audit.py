# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Locate rounding-boundary differences in a deterministic tiny UNet."""

import argparse
import hashlib
import json
import os
import platform
from importlib.metadata import version
from unittest.mock import patch

import numpy as np
import onnx_ir as ir
import torch
from diffusers import UNet2DConditionModel
from onnx.reference import ReferenceEvaluator

from examples.diffusion.adapters import inject
from examples.diffusion.unet import build, tiny_config
from examples.diffusion.unet_reference import make_reference
from examples.huggingface.mixed_precision import precision_projections
from examples.lora.bf16_rules import DOMAIN
from examples.torch_precision import projection_product


def _key(value):
    return value.shape, hashlib.sha256(value.tobytes()).hexdigest()


def _kernel(precision, library):
    from examples.lora.mlx_runtime import register_library, session

    register_library(library)
    dtype = ir.DataType.BFLOAT16 if precision == "bf16" else ir.DataType.FLOAT16
    a = ir.val("a", dtype=ir.DataType.FLOAT, shape=["m", "k"])
    b = ir.val("b", dtype=ir.DataType.FLOAT, shape=["k", "n"])
    tape = ir.tape.Tape()
    qa, qb = [tape.op("Cast", [value], {"to": int(dtype)}) for value in (a, b)]
    c = tape.op("MatMul", [qa, qb])
    y = tape.op("Cast", [c], {"to": int(ir.DataType.FLOAT)})
    y.name, y.dtype, y.shape = "result", ir.DataType.FLOAT, ir.Shape(["m", "n"])
    return session(
        ir.Model(
            ir.Graph(
                [a, b],
                [y],
                nodes=tape.nodes,
                opset_imports={"": 23},
                name="shared_projection_inputs",
            ),
            ir_version=10,
        )
    )


def audit(*, precision="bf16", library=None):
    """Compare full paths and replay each projection at identical input values."""
    torch.set_num_threads(1)
    torch.manual_seed(20261103)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    product = projection_product(precision)
    config = tiny_config()
    original = UNet2DConditionModel(**config).eval()
    graph, params, mapping = inject(build(config, original.state_dict()))
    rng = np.random.default_rng(43)
    for name in params:
        if name.endswith(".B"):
            params[name] = rng.normal(0, 0.02, params[name].shape).astype(np.float32)
    reference, _ = make_reference(
        config, original.state_dict(), params, mapping, precision=precision
    )
    lowered = precision_projections(graph, precision=precision)
    batch = dict(
        sample=torch.randn(1, 4, 8, 12),
        timestep=torch.tensor([317.0]),
        encoder_hidden_states=torch.randn(1, 3, 24),
    )
    records, apply = [], product.apply

    def capture(a, b):
        c = apply(a, b)
        records.append(tuple(value.detach().numpy().copy() for value in (a, b, c)))
        return c

    with torch.no_grad(), patch.object(product, "apply", capture):
        expected = reference(**batch).sample.numpy()
    feeds = {**params, **{name: value.numpy() for name, value in batch.items()}}
    values = ReferenceEvaluator(ir.to_proto(lowered)).run(None, feeds, intermediate=True)
    lookup = {_key(b): (index, a, c) for index, (a, b, c) in enumerate(records)}
    if len(lookup) != len(records):
        raise ValueError("Projection weight values must uniquely identify this seeded trace")
    runtime = _kernel(precision, library) if library else None

    def rounded(value):
        return torch.from_numpy(value.copy()).to(dtype).float().numpy()

    def ordered_bits(value):
        bits = (
            torch.from_numpy(value.copy()).to(dtype).view(torch.int16).numpy().astype(np.int32)
        )
        return np.where(bits < 0, -(bits & 0x7FFF), bits)

    rows = []
    for node in lowered.graph:
        if node.domain != DOMAIN or node.op_type not in ("MatMulBF16", "MatMulFP16"):
            continue
        a, b, c = [values[value.name] for value in (*node.inputs, node.outputs[0])]
        index, expected_a, expected_c = lookup[_key(b)]
        expected_a, expected_c = expected_a.reshape(a.shape), expected_c.reshape(c.shape)
        with torch.no_grad():
            shared = apply(torch.from_numpy(a.copy()), torch.from_numpy(b.copy())).numpy()
        row = dict(
            matrix=node.inputs[1].name,
            reference_index=index,
            input_max_abs=float(abs(a - expected_a).max()),
            rounded_input_mismatches=int(np.count_nonzero(rounded(a) != rounded(expected_a))),
            same_input_kernel_mismatches=int(np.count_nonzero(c != shared)),
            same_input_kernel_bit_mismatches=int(
                np.count_nonzero(c.view(np.uint32) != shared.view(np.uint32))
            ),
            same_input_kernel_max_ulp=int(abs(ordered_bits(c) - ordered_bits(shared)).max()),
            same_input_kernel_max_abs=float(abs(c - shared).max()),
            path_output_max_abs=float(abs(c - expected_c).max()),
        )
        if runtime is not None:
            native = runtime.run(None, {"a": a, "b": b})[0]
            row.update(
                native_shared_input_mismatches=int(np.count_nonzero(native != shared)),
                native_shared_input_bit_mismatches=int(
                    np.count_nonzero(native.view(np.uint32) != shared.view(np.uint32))
                ),
                native_shared_input_max_ulp=int(
                    abs(ordered_bits(native) - ordered_bits(shared)).max()
                ),
                native_shared_input_max_abs=float(abs(native - shared).max()),
            )
        rows.append(row)
    actual = values[lowered.graph.outputs[0].name]
    library_sha256 = None
    if library is not None:
        with open(library, "rb") as stream:
            library_sha256 = hashlib.sha256(stream.read()).hexdigest()
    return dict(
        version=1,
        profile="seeded-tiny-unet",
        precision=precision,
        seeds={"torch": 20261103, "nonzero_adapter_B": 43},
        config=config,
        projection_count=len(rows),
        prediction_max_abs=float(abs(actual - expected).max()),
        prediction_relative_l2=float(
            np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        ),
        first_rounded_input_difference=next(
            (row for row in rows if row["rounded_input_mismatches"]), None
        ),
        first_same_input_kernel_difference=next(
            (row for row in rows if row["same_input_kernel_mismatches"]), None
        ),
        native_exact_projections=None
        if runtime is None
        else sum(row["native_shared_input_bit_mismatches"] == 0 for row in rows),
        library_sha256=library_sha256,
        cpu_fallback=None if runtime is None else False,
        ort_graph_optimizer=None if runtime is None else False,
        scope="ReferenceEvaluator/Torch forward trace and optional MLX replay at identical operands",
        platform=platform.platform(),
        versions={name: version(name) for name in ("numpy", "torch", "onnx", "onnxruntime")},
        rows=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--library")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.lexists(args.output):
        raise FileExistsError(args.output)
    record = audit(precision=args.precision, library=args.library)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({name: value for name, value in record.items() if name != "rows"}))


if __name__ == "__main__":
    main()
