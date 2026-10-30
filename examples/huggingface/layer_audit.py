# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Replay Qwen operators at identical FP32 inputs to locate local rounding errors."""

import argparse
import json
import math
import os
import platform
import tempfile

import numpy as np
import onnxruntime as ort
import torch

from examples.huggingface.bridge import graph_session
from examples.huggingface.qwen import (
    MODEL_ID,
    REVISION,
    build_base,
    decompose,
    reference_model,
)


def compare(actual, expected):
    difference = np.abs(actual - expected)
    return {
        "max_abs": float(difference.max()),
        "relative_l2": float(
            np.linalg.norm(difference.astype(np.float64))
            / max(float(np.linalg.norm(expected.astype(np.float64))), 1e-30)
        ),
        "violations": int(np.count_nonzero(difference > 1e-6 + 1e-4 * np.abs(expected))),
        "elements": int(expected.size),
    }


def summarize(rows):
    result = {}
    for op in ("MatMul", "RMSNormalization", "Swish"):
        metrics = [row["same_input_fp32"] for row in rows if row["op"] == op]
        result[op] = {
            "nodes": len(metrics),
            "violations": sum(row["violations"] for row in metrics),
            "elements": sum(row["elements"] for row in metrics),
            "max_abs": max(row["max_abs"] for row in metrics),
        }
    return result


def probes(reference):
    model = build_base(reference)
    records = []
    existing = {v.name for v in model.graph.outputs}
    for node in model.graph:
        if node.domain or node.op_type not in ("MatMul", "RMSNormalization", "Swish"):
            continue
        record = {
            "name": node.name,
            "op": node.op_type,
            "inputs": [v.name for v in node.inputs],
            "output": node.outputs[0].name,
        }
        if node.op_type == "RMSNormalization":
            record["epsilon"] = float(np.float32(node.attributes.get_float("epsilon", 1e-5)))
        if node.op_type == "Swish":
            record["alpha"] = float(np.float32(node.attributes.get_float("alpha", 1.0)))
        records.append(record)
        for value in (*node.inputs, node.outputs[0]):
            if value.const_value is None and value.name not in existing:
                model.graph.outputs.append(value)
                existing.add(value.name)
    constants = {name: v.const_value for name, v in model.graph.initializers.items()}
    return decompose(model), records, constants


def run(*, real=False):
    torch.set_num_threads(1)
    reference = reference_model(real)
    model, records, constants = probes(reference)
    # Fixed token IDs avoid tokenizer differences in the operator comparison.
    rng = np.random.default_rng(20261030)
    ids = rng.integers(3, min(reference.config.vocab_size, 128), (1, 64), dtype=np.int64)
    ids[0, 0], ids[0, -1] = 1, 2
    feeds = {
        "input_ids": ids,
        "attention_mask": np.ones_like(ids),
        "position_ids": np.arange(ids.shape[1], dtype=np.int64)[None],
    }
    with tempfile.TemporaryDirectory(prefix="onnax-layer-audit-") as directory:
        executor = graph_session(model, directory, "probes")
        observed = dict(
            zip([v.name for v in model.graph.outputs], executor.run(None, feeds), strict=True)
        )
    observed.update(feeds)

    def value(name):
        return observed[name] if name in observed else constants[name].numpy()

    rows = []
    with torch.no_grad():
        for record in records:
            inputs = [
                torch.from_numpy(np.array(value(n), copy=True)) for n in record["inputs"]
            ]
            x = inputs[0]
            if record["op"] == "MatMul":
                expected = x @ inputs[1]
            elif record["op"] == "RMSNormalization":
                variance = x.square().mean(-1, keepdim=True)
                expected = inputs[1] * (x * torch.rsqrt(variance + record["epsilon"]))
            else:
                expected = x * torch.sigmoid(x * record["alpha"])
            actual = value(record["output"])
            expected_array = expected.numpy()
            row = {**record, "same_input_fp32": compare(actual, expected_array)}
            if record["op"] == "MatMul" and inputs[1].ndim == 2:
                scaled = np.abs(actual - expected_array) / (
                    1e-6 + 1e-4 * np.abs(expected_array)
                )
                index = np.unravel_index(scaled.argmax(), scaled.shape)
                left = inputs[0].numpy()[index[:-1]]
                right = inputs[1].numpy()[:, index[-1]]
                oracle = math.fsum(
                    float(a) * float(b) for a, b in zip(left, right, strict=True)
                )
                row["worst_scaled_dot"] = {
                    "index": [int(i) for i in index],
                    "width": len(left),
                    "scalar_fsum_oracle": oracle,
                    "rounded_fp32_oracle": float(np.float32(oracle)),
                    "ort_cpu": float(actual[index]),
                    "torch_cpu": float(expected_array[index]),
                    "ort_abs_error": abs(float(actual[index]) - oracle),
                    "torch_abs_error": abs(float(expected_array[index]) - oracle),
                }
            rows.append(row)
        prediction = reference(
            input_ids=torch.from_numpy(ids),
            attention_mask=torch.from_numpy(feeds["attention_mask"]),
            position_ids=torch.from_numpy(feeds["position_ids"]),
            use_cache=False,
        ).logits.numpy()
    return {
        "model": MODEL_ID if real else "tiny-qwen3",
        "revision": REVISION if real else None,
        "dtype": "float32",
        "swish_decomposition": "bounded_exponential_sigmoid",
        "ort_graph_optimization": "disabled",
        "scope": "Initial frozen base; identical observed ONNX inputs replayed with PyTorch FP32 operators. Norms in report statistics use FP64 host arithmetic only. Probes change graph outputs; this is an accuracy diagnostic, not a timing benchmark.",
        "input_ids": ids.tolist(),
        "versions": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "onnxruntime": ort.__version__,
            "platform": platform.platform(),
        },
        "logits": compare(observed["logits"], prediction),
        "operator_summary": summarize(rows),
        "operators": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(real=args.real)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"logits": result["logits"], "operators": result["operator_summary"]}))


if __name__ == "__main__":
    main()
