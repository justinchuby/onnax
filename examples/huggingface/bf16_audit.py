# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Compare projection primals and cotangents at a shared BF16 training state."""

import argparse
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import onnx_ir as ir
import torch
from onnx.reference import ReferenceEvaluator

import onnax
from examples.huggingface.joint_loss import JointLoss, causal_loss_graph
from examples.huggingface.mixed_precision import (
    _Product,
    bf16_projections,
    reference_projections,
)
from examples.huggingface.qwen import build_base, decompose, inject, reference_model
from examples.huggingface.reference_losses import stable_token_loss
from examples.huggingface.serialization import reference_session
from examples.huggingface.validation import add_reference_adapters
from examples.lora.bf16_rules import DOMAIN, OP, matmul_vjp


def audit():
    """Audit the second gradient evaluation after one ordinary AdamW update."""
    torch.set_num_threads(1)
    reference = reference_model()
    base, params, mapping = inject(build_base(reference))
    source = bf16_projections(decompose(base))
    joint = JointLoss(causal_loss_graph(source), params, session_factory=reference_session)
    ids = torch.tensor([[1, 4, 5, 6, 2], [1, 7, 2, 0, 0]])
    mask = ids.ne(0).long()
    inputs = dict(
        input_ids=ids, attention_mask=mask, position_ids=(mask.cumsum(-1) - 1).clamp(min=0)
    )
    labels = ids.clone()
    labels[mask == 0] = -100
    count = labels[:, 1:].ne(-100).sum()
    optimizer = torch.optim.AdamW(joint.parameters(), lr=0.003, weight_decay=0)
    (joint(**inputs, labels=labels) / count).backward()
    optimizer.step()
    reference = reference_projections(
        add_reference_adapters(reference, SimpleNamespace(onnx=joint, mapping=mapping))
    )
    records, reference_gradients = [], {}
    original_apply = _Product.apply

    def record(a, b):
        result = original_apply(a, b)
        index = len(records)
        records.append([x.detach().float().numpy().copy() for x in (a, b, result)])
        if result.requires_grad:
            result.register_hook(
                lambda g: reference_gradients.update({index: g.detach().numpy().copy()})
            )
        return result

    with patch.object(_Product, "apply", record):
        stable_token_loss(reference(**inputs, use_cache=False).logits, labels).backward()
    gradient_names = {}

    def trace(ctx, node, cotangents, input_indices):
        gradient_names[node.inputs[1].name] = cotangents[0].name
        return matmul_vjp(ctx, node, cotangents, input_indices)

    # Record names during graph construction, then restore the example's rule.
    onnax.register_vjp(DOMAIN, OP, since_version=1, until_version=1)(trace)
    try:
        derivative = onnax.grad(
            causal_loss_graph(source), inputs=list(params), output="loss_sum"
        )
    finally:
        onnax.register_vjp(DOMAIN, OP, since_version=1, until_version=1)(matmul_vjp)
    feeds = {**{k: v.numpy() for k, v in inputs.items()}, **joint.arrays()}
    values = ReferenceEvaluator(ir.to_proto(source)).run(None, feeds, intermediate=True)
    gradients = ReferenceEvaluator(ir.to_proto(derivative)).run(
        None, {**feeds, "labels": labels.numpy()}, intermediate=True
    )

    def key(value):
        return hashlib.sha256(value.tobytes()).hexdigest()

    def rounded(value):
        return torch.from_numpy(value).to(torch.bfloat16).float().numpy()

    lookup = {key(b): (i, a, c) for i, (a, b, c) in enumerate(records)}
    rows = []
    for node in source.graph:
        if node.domain != DOMAIN or node.op_type != OP:
            continue
        a, b = [values[v.name] for v in node.inputs]
        c = values[node.outputs[0].name]
        index, expected_a, expected_c = lookup[key(b)]
        expected_a, expected_c = expected_a.reshape(a.shape), expected_c.reshape(c.shape)
        row = {
            "matrix": node.inputs[1].name,
            "input_max_abs": float(np.abs(a - expected_a).max()),
            "rounded_input_mismatches": int((rounded(a) != rounded(expected_a)).sum()),
            "output_max_abs": float(np.abs(c - expected_c).max()),
            "output_mismatches": int((c != expected_c).sum()),
        }
        if node.inputs[1].name in gradient_names:
            actual = gradients[gradient_names[node.inputs[1].name]] / np.float32(count)
            expected = reference_gradients[index].reshape(actual.shape)
            delta = np.abs(actual - expected)
            row.update(
                gradient_max_abs=float(delta.max()),
                gradient_violations=int((delta > 1e-6 + 1e-4 * np.abs(expected)).sum()),
            )
        rows.append(row)
    return {
        "state": "shared parameters after one ONNX AdamW update",
        "scope": "ReferenceEvaluator primals/VJP against independent PyTorch projections",
        "rtol": 1e-4,
        "atol": 1e-6,
        "projection_count": len(rows),
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit()
    with open(args.output, "w") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
