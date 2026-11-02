# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Train two tiny decoder architectures through named ONNX checkpoints."""

import argparse
import json
import os

import numpy as np
import onnx_ir as ir
import torch
import torch.nn.functional as functional

from onnax.torch import CheckpointedModule


def decoder(kind):
    """Build two-layer GQA/RMS/SiLU or MHA/LayerNorm/GELU graphs with LoRA."""
    if kind not in ("gqa", "mha"):
        raise ValueError("Choose gqa or mha")
    rng = np.random.default_rng(20261101)
    tape = ir.tape.Tape()
    x = ir.val("features", dtype=ir.DataType.FLOAT, shape=["batch", "sequence", 8])
    mask = ir.val("mask", dtype=ir.DataType.BOOL, shape=["batch", 1, "sequence", "sequence"])
    inputs, constants, weights, parameters = [x, mask], [], {}, {}

    def weight(name, shape, *, trainable=False, ones=False):
        array = (
            np.ones(shape, np.float32)
            if ones
            else rng.normal(0, 0.1, shape).astype(np.float32)
        )
        value = ir.val(name, dtype=ir.DataType.FLOAT, shape=shape)
        if trainable:
            inputs.append(value)
            parameters[name] = array
        else:
            value.const_value = ir.tensor(array, name=name)
            constants.append(value)
            weights[name] = torch.from_numpy(array.copy())
        return value

    kv_heads = 1 if kind == "gqa" else 2
    hidden = x
    boundaries = {}
    for layer in range(2):
        prefix = f"layers.{layer}"

        def norm(value, name, prefix=prefix):
            scale = weight(f"{prefix}.{name}.scale", (8,), ones=True)
            op = "RMSNormalization" if kind == "gqa" else "LayerNormalization"
            return tape.op(op, [value, scale], {"axis": -1, "epsilon": 1e-5})

        normalized = norm(hidden, "norm1")
        projections = []
        for name, width in (("q", 8), ("k", kv_heads * 4), ("v", kv_heads * 4)):
            value = tape.op(
                "MatMul", [normalized, weight(f"{prefix}.{name}.weight", (8, width))]
            )
            if name in ("q", "v"):
                a = weight(f"{prefix}.{name}.A", (8, 2), trainable=True)
                b = weight(f"{prefix}.{name}.B", (2, width), trainable=True)
                low_rank = tape.op("MatMul", [tape.op("MatMul", [normalized, a]), b])
                value = tape.op("Add", [value, low_rank])
            projections.append(value)
        attended = tape.op(
            "Attention",
            [*projections, mask],
            {"q_num_heads": 2, "kv_num_heads": kv_heads, "is_causal": 1},
        )
        projected = tape.op("MatMul", [attended, weight(f"{prefix}.o.weight", (8, 8))])
        residual = tape.op("Add", [hidden, projected])
        normalized = norm(residual, "norm2")
        up = tape.op("MatMul", [normalized, weight(f"{prefix}.up.weight", (8, 16))])
        if kind == "gqa":
            gate = tape.op("MatMul", [normalized, weight(f"{prefix}.gate.weight", (8, 16))])
            up = tape.op("Mul", [up, tape.op("Mul", [gate, tape.op("Sigmoid", [gate])])])
        else:
            up = tape.op("Gelu", [up])
        hidden = tape.op(
            "Add",
            [residual, tape.op("MatMul", [up, weight(f"{prefix}.down.weight", (16, 8))])],
        )
        hidden.name = f"{prefix}.hidden"
        boundaries[f"decoder_{layer}"] = hidden.name
    logits = tape.op("MatMul", [hidden, weight("head.weight", (8, 16))])
    logits.name = "logits"
    model = ir.Model(
        ir.Graph(
            inputs,
            [logits],
            nodes=tape.nodes,
            initializers=constants,
            opset_imports={"": 25},
            name=f"tiny_{kind}",
        ),
        ir_version=10,
    )
    return model, parameters, boundaries, weights


def reference(kind, x, mask, parameters, weights):
    """Independent PyTorch forward, including causal attention and frozen blocks."""
    kv_heads = 1 if kind == "gqa" else 2
    batch, length, _ = x.shape
    for layer in range(2):
        prefix = f"layers.{layer}"

        def norm(value, name, prefix=prefix):
            scale = weights[f"{prefix}.{name}.scale"]
            if kind == "gqa":
                return (
                    value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5) * scale
                )
            return functional.layer_norm(value, (8,), scale, eps=1e-5)

        normalized = norm(x, "norm1")
        projections = []
        for name in ("q", "k", "v"):
            value = normalized @ weights[f"{prefix}.{name}.weight"]
            if name in ("q", "v"):
                value = (
                    value
                    + (normalized @ parameters[f"{prefix}.{name}.A"])
                    @ parameters[f"{prefix}.{name}.B"]
                )
            heads = 2 if name == "q" else kv_heads
            value = value.reshape(batch, length, heads, 4).transpose(1, 2)
            projections.append(value.repeat_interleave(2 // heads, dim=1))
        q, k, v = projections
        scores = (q @ k.transpose(-1, -2)) * 0.5
        causal = torch.ones(length, length, dtype=torch.bool).tril()
        scores = scores.masked_fill(~(mask & causal), -torch.inf)
        attended = (scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, length, 8)
        residual = x + attended @ weights[f"{prefix}.o.weight"]
        normalized = norm(residual, "norm2")
        up = normalized @ weights[f"{prefix}.up.weight"]
        if kind == "gqa":
            up = up * functional.silu(normalized @ weights[f"{prefix}.gate.weight"])
        else:
            up = functional.gelu(up)
        x = residual + up @ weights[f"{prefix}.down.weight"]
    return x @ weights["head.weight"]


def run(directory, *, kind="gqa", steps=12):
    """Compare predictions, losses, every adapter gradient/update, then reload."""
    source, initial, boundaries, frozen = decoder(kind)
    target = {
        name: torch.nn.Parameter(torch.from_numpy(value.copy()))
        for name, value in initial.items()
    }
    expected_optimizer = torch.optim.AdamW(target.values(), lr=0.01, weight_decay=0)
    rng = np.random.default_rng(7)
    features = torch.from_numpy(rng.normal(size=(2, 5, 8)).astype(np.float32))
    mask = torch.ones(2, 1, 5, 5, dtype=torch.bool)
    mask[1, :, :, -2:] = False
    losses = []
    with CheckpointedModule(
        source, initial, boundaries=boundaries, input_gradients=()
    ) as model:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0)
        for _ in range(steps):
            optimizer.zero_grad()
            expected_optimizer.zero_grad()
            actual = model(features=features, mask=mask)
            expected = reference(kind, features, mask, target, frozen)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
            loss, expected_loss = actual.square().mean(), expected.square().mean()
            torch.testing.assert_close(loss, expected_loss, rtol=1e-4, atol=1e-6)
            loss.backward()
            expected_loss.backward()
            for name, value in model.parameters_by_name().items():
                torch.testing.assert_close(value.grad, target[name].grad, rtol=1e-4, atol=1e-6)
            optimizer.step()
            expected_optimizer.step()
            for name, value in model.parameters_by_name().items():
                torch.testing.assert_close(value, target[name], rtol=1e-4, atol=1e-6)
            losses.append(loss.item())
        expected = model(features=features, mask=mask).detach()
        model.export(directory)
        residual_bytes = model.last_residual_bytes
    with CheckpointedModule.load(directory) as restored:
        torch.testing.assert_close(
            restored(features=features, mask=mask), expected, rtol=0, atol=0
        )
    assert losses[-1] < losses[0]
    return {
        "architecture": kind,
        "steps": steps,
        "losses": losses,
        "independent_updates": True,
        "reload": True,
        "checkpoint_bytes": residual_bytes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kind", choices=("gqa", "mha", "both"), default="both")
    args = parser.parse_args()
    torch.set_num_threads(1)
    kinds = ("gqa", "mha") if args.kind == "both" else (args.kind,)
    reports = [run(os.path.join(args.output_dir, kind), kind=kind) for kind in kinds]
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
