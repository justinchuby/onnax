# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Compare local Qwen pullbacks at observed activations and loss cotangents."""

import argparse
import importlib.metadata
import json
import os
import platform
import tempfile
from collections import Counter
from unittest.mock import patch

import numpy as np
import torch

import onnax
from examples.huggingface.layer_audit import compare
from examples.huggingface.nf4 import lower_decoder
from examples.huggingface.qwen import REVISION, build_base, decompose, inject, reference_model
from examples.huggingface.rms_audit import diagnostic_session
from examples.huggingface.serialization import save_graph
from examples.huggingface.sft import token_loss
from examples.lora.nf4 import DOMAIN, dequantize
from onnax._autodiff._registry import _VjpRegistry

ARITHMETIC = frozenset(
    {
        "MatMul",
        "Div",
        "Sqrt",
        "Softmax",
        "Mul",
        "Add",
        "Sub",
        "Neg",
        "Exp",
        "ReduceMean",
        "Where",
    }
)
LAYOUT = frozenset(
    {"Concat", "Expand", "Reshape", "Split", "Squeeze", "Transpose", "Unsqueeze"}
)


def operation(node):
    return "NF4MatMul" if node.domain == DOMAIN else node.op_type


def trace(model, names):
    """Observe rule inputs/results without changing registrations or rule bodies."""
    probes, unchecked = [], Counter()
    lookup = _VjpRegistry.lookup

    def observe_lookup(registry, node, imports):
        rule = lookup(registry, node, imports)

        def observe(context, node, seeds, indices):
            contributions = rule(context, node, seeds, indices)
            if (not node.domain and node.op_type in ARITHMETIC | LAYOUT) or (
                node.domain == DOMAIN and node.op_type.startswith("MatMul_")
            ):
                probes.append((node, tuple(seeds), dict(contributions)))
            else:
                unchecked[f"{node.domain or 'ai.onnx'}::{node.op_type}"] += 1
            return contributions

        return observe

    with patch.object(_VjpRegistry, "lookup", observe_lookup):
        joint = onnax.vjp(model, inputs=list(names), outputs=["logits"])
    seed_name = joint.graph.inputs[-1].name
    exposed = {value.name for value in joint.graph.outputs}

    def expose(value, primal):
        # Rule intermediates need explicit interfaces when promoted to outputs.
        value.dtype, value.shape = primal.dtype, primal.shape
        if value.name not in exposed and value.name not in joint.graph.initializers:
            joint.graph.outputs.append(value)
            exposed.add(value.name)

    for node, seeds, contributions in probes:
        for value in node.inputs:
            if value is not None:
                expose(value, value)
        for seed, output in zip(seeds, node.outputs, strict=True):
            expose(seed, output)
        for index, value in contributions.items():
            expose(value, node.inputs[index])
    return joint, seed_name, probes, unchecked


def reference_output(node, operands):
    """Evaluate each observed operation using PyTorch's own autograd rules."""
    if node.domain == DOMAIN:
        _, k, n, block = node.op_type.split("_")
        weight = dequantize(
            operands[1].numpy(), operands[2].numpy(), (int(k), int(n)), block_size=int(block)
        )
        return operands[0] @ torch.from_numpy(weight)
    op = node.op_type
    x = operands[0]
    if op == "MatMul":
        return x @ operands[1]
    if op == "Div":
        return x / operands[1]
    if op == "Mul":
        return x * operands[1]
    if op == "Add":
        return x + operands[1]
    if op == "Sub":
        return x - operands[1]
    if op == "Neg":
        return -x
    if op == "Sqrt":
        return torch.sqrt(x)
    if op == "Exp":
        return torch.exp(x)
    if op == "Softmax":
        return torch.softmax(x, dim=node.attributes.get_int("axis", -1))
    if op == "Where":
        return torch.where(*operands)
    if op == "Transpose":
        return x.permute(node.attributes.get_ints("perm", tuple(reversed(range(x.ndim)))))
    if op == "Concat":
        return torch.cat(operands, dim=node.attributes.get_int("axis"))
    if op == "Reshape":
        shape = [int(v) for v in operands[1].reshape(-1)]
        if not node.attributes.get_int("allowzero", 0):
            shape = [x.shape[i] if dim == 0 else dim for i, dim in enumerate(shape)]
        return x.reshape(shape)
    if op == "Expand":
        shape = tuple(int(v) for v in operands[1].reshape(-1))
        return torch.broadcast_to(x, torch.broadcast_shapes(x.shape, shape))
    if op == "Squeeze":
        return (
            x.squeeze(tuple(int(v) for v in operands[1].reshape(-1)))
            if len(operands) > 1
            else x.squeeze()
        )
    if op == "Unsqueeze":
        axes = [int(v) for v in operands[1].reshape(-1)]
        rank = x.ndim + len(axes)
        for axis in sorted(value % rank for value in axes):
            x = x.unsqueeze(axis)
        return x
    if op == "Split":
        axis = node.attributes.get_int("axis", 0)
        if len(operands) > 1:
            sizes = [int(v) for v in operands[1].reshape(-1)]
        else:
            count = len(node.outputs)
            chunk = (x.shape[axis] + count - 1) // count
            sizes = [
                max(0, min(chunk, x.shape[axis] - index * chunk)) for index in range(count)
            ]
        return x.split(sizes, dim=axis)
    if op == "ReduceMean":
        axes = tuple(int(v) for v in operands[1].reshape(-1)) if len(operands) > 1 else ()
        if not axes:
            if node.attributes.get_int("noop_with_empty_axes", 0):
                return x
            axes = tuple(range(x.ndim))
        return x.mean(dim=axes, keepdim=bool(node.attributes.get_int("keepdims", 1)))
    raise NotImplementedError(node.op_identifier())


def check(probes, observed, constants):
    summaries, failures = {}, []

    def array(value):
        return (
            observed[value.name] if value.name in observed else constants[value.name].numpy()
        )

    for node, seeds, contributions in probes:
        operands = [
            torch.tensor(array(value), requires_grad=index in contributions)
            for index, value in enumerate(node.inputs)
            if value is not None
        ]
        output = reference_output(node, operands)
        expected = torch.autograd.grad(
            (output,) if isinstance(output, torch.Tensor) else tuple(output),
            [operands[index] for index in contributions],
            tuple(torch.tensor(array(seed)) for seed in seeds),
        )
        summary = summaries.setdefault(
            operation(node),
            {"nodes": 0, "input_partials": 0, "failed_partials": 0, "max_abs": 0.0},
        )
        summary["nodes"] += 1
        for (index, value), gradient in zip(contributions.items(), expected, strict=True):
            actual, reference = array(value), gradient.numpy()
            if actual.shape != reference.shape or actual.dtype != reference.dtype:
                raise AssertionError(
                    f"Local gradient interface mismatch: {node.name}, input {index}"
                )
            if not np.isfinite(actual).all() or not np.isfinite(reference).all():
                raise AssertionError(f"Nonfinite local gradient: {node.name}, input {index}")
            row = {
                "node": node.name,
                "input": index,
                "shape": list(actual.shape),
                **compare(actual, reference),
            }
            summary["input_partials"] += 1
            summary["failed_partials"] += int(row["violations"] > 0)
            if "worst_partial" not in summary or row["max_abs"] > summary["max_abs"]:
                summary["max_abs"], summary["worst_partial"] = row["max_abs"], row
            if row["violations"]:
                failures.append({"op": operation(node), **row})
    return {"operators": summaries, "failed_partials": len(failures), "failures": failures}


def run(*, real=False, nf4=False, b_values=(0.0, 0.01)):
    if not b_values or not all(np.isfinite(value) for value in b_values):
        raise ValueError("Expected finite adapter B values")
    torch.set_num_threads(1)
    reference = reference_model(real)
    model, params, _ = inject(build_base(reference))
    model = decompose(model)
    if nf4:
        model, _ = lower_decoder(model, reference)
    del reference
    joint, seed_name, probes, unchecked = trace(model, params)
    constants = {name: value.const_value for name, value in joint.graph.initializers.items()}
    ids = np.arange(3, 67, dtype=np.int64)[None]
    feeds = {
        "input_ids": ids,
        "attention_mask": np.ones_like(ids),
        "position_ids": np.arange(64, dtype=np.int64)[None],
    }
    states = []
    with tempfile.TemporaryDirectory(prefix="onnax-local-vjp-") as directory:
        for name, graph in (("forward", model), ("joint", joint)):
            save_graph(
                graph, os.path.join(directory, name + ".onnx"), data_name=name + ".data"
            )
        forward = diagnostic_session(os.path.join(directory, "forward.onnx"))
        backward = diagnostic_session(os.path.join(directory, "joint.onnx"))
        for b in b_values:
            arrays = {
                name: np.full_like(value, b) if name.endswith(".B") else value
                for name, value in params.items()
            }
            arguments = {**feeds, **arrays}
            logits = torch.tensor(forward.run(["logits"], arguments)[0], requires_grad=True)
            loss = token_loss(logits, torch.from_numpy(ids))
            loss.backward()
            arguments[seed_name] = logits.grad.numpy()
            values = backward.run(None, arguments)
            observed = dict(zip([v.name for v in joint.graph.outputs], values, strict=True))
            np.testing.assert_array_equal(observed["logits"], logits.detach().numpy())
            states.append(
                {
                    "adapter_b": b,
                    "loss": float(loss.detach()),
                    **check(probes, observed, constants),
                }
            )
    return {
        "model": "Qwen3-0.6B" if real else "tiny Qwen3",
        "revision": REVISION if real else None,
        "nf4": nf4,
        "dtype": "float32",
        "rtol": 1e-4,
        "atol": 1e-6,
        "input_ids": ids.tolist(),
        "trainable_tensors": len(params),
        "covered_nodes": len(probes),
        "unchecked_active_nodes": dict(sorted(unchecked.items())),
        "ort_graph_optimization": "disabled",
        "ort_memory_reuse": False,
        "threads": 1,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("onnax", "onnxruntime", "torch", "mobius-onnx")
        },
        "platform": platform.platform(),
        "states": states,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--nf4", action="store_true")
    parser.add_argument("--b-values", type=float, nargs="+", default=[0.0, 0.01])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(real=args.real, nf4=args.nf4, b_values=args.b_values)
    with open(args.output, "w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
