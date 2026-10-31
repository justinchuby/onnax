# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Compare local Qwen pullbacks at observed activations and loss cotangents."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from collections import Counter
from unittest.mock import patch

import numpy as np
import onnx_ir as ir
import torch

import onnax
from examples.huggingface.layer_audit import compare
from examples.huggingface.nf4 import lower_decoder
from examples.huggingface.qwen import REVISION, build_base, decompose, inject, reference_model
from examples.huggingface.rms_audit import diagnostic_session
from examples.huggingface.serialization import save_graph
from examples.huggingface.sft import token_loss
from examples.lora.bf16_rules import DOMAIN as BF16_DOMAIN
from examples.lora.bf16_rules import OP as BF16_OP
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
    return "NF4" + node.op_type.split("_")[0] if node.domain == DOMAIN else node.op_type


def trace(model, names):
    """Observe rule inputs/results without changing registrations or rule bodies."""
    probes, unchecked = [], Counter()
    lookup = _VjpRegistry.lookup

    def observe_lookup(registry, node, imports):
        rule = lookup(registry, node, imports)

        def observe(context, node, seeds, indices):
            contributions = rule(context, node, seeds, indices)
            if (
                (not node.domain and node.op_type in ARITHMETIC | LAYOUT)
                or (
                    node.domain == DOMAIN
                    and node.op_type.startswith(("MatMul_", "MatMulBF16_"))
                )
                or (node.domain == BF16_DOMAIN and node.op_type == BF16_OP)
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
        for output in node.outputs:
            expose(output, output)
        for value in node.inputs:
            if value is not None:
                expose(value, value)
        for seed, output in zip(seeds, node.outputs, strict=True):
            expose(seed, output)
        for index, value in contributions.items():
            expose(value, node.inputs[index])
    return joint, seed_name, probes, unchecked


def reference_output(node, operands, *, pairwise_head=False):
    """Evaluate each observed operation using PyTorch's own autograd rules."""
    if pairwise_head:
        from examples.huggingface.reference_head import _BF16FrozenHead, _FrozenHead

        product = _BF16FrozenHead if node.domain == BF16_DOMAIN else _FrozenHead
        return product.apply(operands[0], operands[1].T, 128)
    if node.domain == DOMAIN:
        _, k, n, block = node.op_type.split("_")
        weight = dequantize(
            operands[1].numpy(), operands[2].numpy(), (int(k), int(n)), block_size=int(block)
        )
        if node.op_type.startswith("MatMulBF16_"):
            from examples.huggingface.mixed_precision import _Product

            return _Product.apply(operands[0], torch.from_numpy(weight))
        return operands[0] @ torch.from_numpy(weight)
    if node.domain == BF16_DOMAIN and node.op_type == BF16_OP:
        from examples.huggingface.mixed_precision import _Product

        return _Product.apply(*operands)
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


def compose(
    model, names, observed, constants, seed, derivative_outputs, *, pairwise_head=False
):
    """Independently reverse the primal graph using captured operand values.

    Reachability and fan-in are computed from the primal graph. Generated rule
    contributions and observed intermediate cotangents are not used.
    """

    def array(value):
        return (
            observed[value.name] if value.name in observed else constants[value.name].numpy()
        )

    active = set(names)
    for node in model.graph:
        if any(value is not None and value.name in active for value in node.inputs):
            active.update(
                value.name
                for value in node.outputs
                if value.dtype in (ir.DataType.FLOAT, ir.DataType.DOUBLE)
            )
    pending = {"logits": torch.tensor(seed)}
    nodes = additions = heads = 0
    vocab = next(value for value in model.graph.outputs if value.name == "logits").shape[-1]
    for node in reversed(list(model.graph)):
        if not any(value.name in pending for value in node.outputs):
            continue
        indices = [
            i
            for i, value in enumerate(node.inputs)
            if value is not None and value.name in active
        ]
        seeds = tuple(pending.pop(value.name, None) for value in node.outputs)
        if not indices:
            continue
        seeds = tuple(
            value if value is not None else torch.zeros_like(torch.tensor(array(output)))
            for value, output in zip(seeds, node.outputs, strict=True)
        )
        operands = [
            torch.tensor(array(value), requires_grad=i in indices)
            if value is not None
            else None
            for i, value in enumerate(node.inputs)
        ]
        is_head = bool(
            pairwise_head
            and len(operands) == 2
            and operands[1] is not None
            and operands[1].ndim == 2
            and operands[1].shape[-1] == vocab
            and 1 not in indices
            and (
                (not node.domain and node.op_type == "MatMul")
                or (node.domain == BF16_DOMAIN and node.op_type == BF16_OP)
            )
        )
        heads += is_head
        output = reference_output(node, operands, pairwise_head=is_head)
        gradients = torch.autograd.grad(
            (output,) if isinstance(output, torch.Tensor) else tuple(output),
            [operands[i] for i in indices],
            seeds,
        )
        nodes += 1
        for index, gradient in zip(indices, gradients, strict=True):
            name = node.inputs[index].name
            if name in pending:
                pending[name] = pending[name] + gradient
                additions += 1
            else:
                pending[name] = gradient
    rows = []
    for name, output in zip(names, derivative_outputs, strict=True):
        actual = observed[output.name]
        expected = pending[name].numpy() if name in pending else np.zeros_like(actual)
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise AssertionError(f"Composed gradient interface mismatch: {name}")
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise AssertionError(f"Nonfinite composed gradient: {name}")
        rows.append({"parameter": name, **compare(actual, expected)})
    return {
        "active_nodes": nodes,
        "fan_in_additions": additions,
        "pairwise_head_nodes": heads,
        "parameters": len(rows),
        "failed_parameters": sum(row["violations"] > 0 for row in rows),
        "rows": rows,
    }


def run(
    *,
    real=False,
    nf4=False,
    b_values=(0.0, 0.01),
    projection_precision="fp32",
    provider="cpu",
    library=None,
    adapter=None,
    pairwise_head=False,
):
    if not b_values or not all(np.isfinite(value) for value in b_values):
        raise ValueError("Expected finite adapter B values")
    if projection_precision not in ("fp32", "bf16"):
        raise ValueError("Expected fp32 or bf16 projection precision")
    if provider not in ("cpu", "reference", "mlx"):
        raise ValueError("Expected cpu, reference or mlx provider")
    if projection_precision == "bf16" and provider == "cpu":
        raise ValueError("BF16 requires reference or mlx execution")
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
    if projection_precision == "bf16":
        from examples.huggingface.mixed_precision import bf16_projections

        model = bf16_projections(model)
    del reference
    joint, seed_name, probes, unchecked = trace(model, params)
    constants = {name: value.const_value for name, value in joint.graph.initializers.items()}
    derivative_outputs = list(joint.graph.outputs)[
        len(model.graph.outputs) : len(model.graph.outputs) + len(params)
    ]
    ids = np.arange(3, 67, dtype=np.int64)[None]
    feeds = {
        "input_ids": ids,
        "attention_mask": np.ones_like(ids),
        "position_ids": np.arange(64, dtype=np.int64)[None],
    }
    state_inputs = [
        (
            {"adapter_b": b},
            {
                name: np.full_like(value, b) if name.endswith(".B") else value
                for name, value in params.items()
            },
        )
        for b in b_values
    ]
    if adapter is not None:
        with np.load(adapter, allow_pickle=False) as archive:
            if set(archive.files) != set(params):
                raise ValueError("Adapter parameter names differ from the model")
            trained = {name: archive[name].copy() for name in params}
        for name, value in trained.items():
            if (
                value.shape != params[name].shape
                or value.dtype != np.float32
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"Invalid adapter array: {name}")
        with open(adapter, "rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        state_inputs.append(({"adapter_archive_sha256": digest}, trained))
    states = []
    with tempfile.TemporaryDirectory(prefix="onnax-local-vjp-") as directory:
        for name, graph in (("forward", model), ("joint", joint)):
            save_graph(
                graph, os.path.join(directory, name + ".onnx"), data_name=name + ".data"
            )
        forward = factory(os.path.join(directory, "forward.onnx"))
        backward = factory(os.path.join(directory, "joint.onnx"))
        for state, arrays in state_inputs:
            arguments = {**feeds, **arrays}
            logits = torch.tensor(forward.run(["logits"], arguments)[0], requires_grad=True)
            loss = token_loss(logits, torch.from_numpy(ids))
            loss.backward()
            arguments[seed_name] = logits.grad.numpy()
            values = backward.run(None, arguments)
            observed = dict(zip([v.name for v in joint.graph.outputs], values, strict=True))
            np.testing.assert_array_equal(observed["logits"], logits.detach().numpy())
            stable = {}
            if pairwise_head:
                stable["pairwise_head_composition"] = compose(
                    model,
                    params,
                    observed,
                    constants,
                    arguments[seed_name],
                    derivative_outputs,
                    pairwise_head=True,
                )
            states.append(
                {
                    **stable,
                    **state,
                    "loss": float(loss.detach()),
                    **check(probes, observed, constants),
                    "composition": compose(
                        model,
                        params,
                        observed,
                        constants,
                        arguments[seed_name],
                        derivative_outputs,
                    ),
                }
            )
    return {
        "model": "Qwen3-0.6B" if real else "tiny Qwen3",
        "revision": REVISION if real else None,
        "nf4": nf4,
        "dtype": "float32",
        "projection_precision": projection_precision,
        "provider": provider,
        "library_sha256": library_hash,
        "adjoints": "FP32 with BF16-rounded operands"
        if projection_precision == "bf16"
        else "FP32",
        "rtol": 1e-4,
        "atol": 1e-6,
        "input_ids": ids.tolist(),
        "trainable_tensors": len(params),
        "covered_nodes": len(probes),
        "unchecked_active_nodes": dict(sorted(unchecked.items())),
        "ort_graph_optimization": "disabled",
        "ort_memory_reuse": False if provider == "cpu" else None,
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
    parser.add_argument("--projection-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--provider", choices=["cpu", "reference", "mlx"], default="cpu")
    parser.add_argument("--library")
    parser.add_argument("--adapter", help="Also audit parameters from an exported adapter.npz")
    parser.add_argument(
        "--pairwise-head",
        action="store_true",
        help="Also compose with FP32 pairwise frozen-head accumulation",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(
        real=args.real,
        nf4=args.nf4,
        b_values=args.b_values,
        projection_precision=args.projection_precision,
        provider=args.provider,
        library=args.library,
        adapter=args.adapter,
        pairwise_head=args.pairwise_head,
    )
    with open(args.output, "w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
