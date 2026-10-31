# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Compare native BF16 and FP32 accumulation on captured projection operands."""

import argparse
import hashlib
import json

import numpy as np
import torch


def audit(path):
    """Use one captured input/weight pair for both independent products.

    The NPZ contains reference_input, weight and mlx_output as FP32 arrays.
    Operands are rounded to BF16 before either product. This isolates product
    accumulation from differences in upstream model activations.
    """
    with np.load(path, allow_pickle=False) as arrays:
        a, b, expected = [
            arrays[key].copy() for key in ("reference_input", "weight", "mlx_output")
        ]
    if any(value.dtype != np.float32 for value in (a, b, expected)):
        raise ValueError("Captured arrays must be FP32")
    if (
        a.ndim != 2
        or b.ndim != 2
        or a.shape[1] != b.shape[0]
        or expected.shape != (a.shape[0], b.shape[1])
    ):
        raise ValueError("Expected compatible rank-two projection operands and output")
    if not all(np.isfinite(value).all() for value in (a, b, expected)):
        raise ValueError("Captured arrays must be finite")
    qa, qb = [torch.from_numpy(value).to(torch.bfloat16) for value in (a, b)]
    products = {
        "torch_bf16_native": (qa @ qb).float().numpy(),
        "torch_fp32_accumulation_emulation": (
            (qa.float() @ qb.float()).to(torch.bfloat16).float().numpy()
        ),
    }
    comparisons = {}
    for name, actual in products.items():
        if not np.isfinite(actual).all():
            raise ValueError(f"{name} produced nonfinite outputs")
        delta = np.abs(actual - expected)
        comparisons[name] = {
            "mismatches": int(np.count_nonzero(actual != expected)),
            "max_abs": float(delta.max()),
            "tolerance_violations": int(
                np.count_nonzero(delta > 1e-6 + 1e-4 * np.abs(expected))
            ),
        }
    with open(path, "rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "capture_sha256": digest,
        "input_shape": list(a.shape),
        "weight_shape": list(b.shape),
        "output_elements": int(expected.size),
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "rtol": 1e-4,
        "atol": 1e-6,
        "scope": "Shared rounded operands; captured MLX output is the comparison target",
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, help="Captured projection NPZ")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = audit(args.capture)
    with open(args.output, "w") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
