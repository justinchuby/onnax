# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Separate classifier forward rounding from analytic backpropagation."""

import numpy as np
import onnx
import onnx_ir as ir

from examples.lora.validation import discrepancy


def instrument(model):
    """Expose rounded matrix operands, hidden preactivations and logits."""
    result = model.clone()
    products = [
        n
        for n in result.graph
        if n.op_type == "MatMul"
        and not n.domain
        and n.outputs[0].dtype == ir.DataType.BFLOAT16
    ]
    if len(products) != 2:
        raise ValueError("Expected two BF16 classifier matrix products")
    logits = [v for n in result.graph for v in n.outputs if v.name == "logits"]
    if len(logits) != 1:
        raise ValueError("Expected one named logits value")
    count = len(result.graph.outputs)
    tape = ir.tape.Tape(result.graph)
    nonlinearities = [n for n in result.graph if not n.domain and n.op_type == "Tanh"]
    if len(nonlinearities) != 1:
        raise ValueError("Expected one classifier Tanh")
    for value in (
        products[0].inputs[0],
        products[1].inputs[1],
        nonlinearities[0].inputs[0],
        products[1].inputs[0],
    ):
        observed = tape.op("Cast", [value], {"to": int(ir.DataType.FLOAT)})
        observed.dtype = ir.DataType.FLOAT
        observed.shape = ir.Shape(value.shape)
        result.graph.outputs.append(observed)
    result.graph.outputs.append(logits[0])
    onnx.checker.check_model(ir.to_proto(result), full_check=True)
    return result, count


def _score_gradient(logits, targets):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return (np.exp(logp) - targets) / np.float32(len(targets))


def head_reference(hidden, logits, targets):
    """Recompute output-layer gradients from observed forward values in NumPy."""
    dz = _score_gradient(logits, targets)
    return {"layer2.weight": hidden.T @ dz, "layer2.bias": dz.sum(axis=0)}


def reference(qx, qw2, z, hidden, logits, targets):
    """Recompute all straight-through gradients using observed forward values."""
    dz = _score_gradient(logits, targets)
    exponential = np.exp(-2 * np.abs(z))
    denominator = 1 + exponential
    slope = 4 * exponential / (denominator * denominator)
    dh = (dz @ qw2.T) * slope
    return {
        "layer1.weight": qx.T @ dh,
        "layer1.bias": dh.sum(axis=0),
        **head_reference(hidden, logits, targets),
    }


def inspect(executor, feeds, original, mapping, *, original_count):
    """Check output preservation and compare analytic gradients at native primals."""
    values = executor.run(None, feeds)
    preserved = all(
        a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
        for a, b in zip(original, values[:original_count], strict=True)
    )
    if not preserved:
        raise ValueError("Instrumentation changed original classifier outputs")
    outputs = dict(zip([v.name for v in executor.get_outputs()], values, strict=True))
    expected = reference(*values[-5:], feeds["targets"])
    return {
        "original_outputs_equal": preserved,
        "observed_primal_gradients": {
            name: discrepancy(outputs[mapping[name]], gradient)
            for name, gradient in expected.items()
        },
        "observed_primal_head_gradients": {
            name: discrepancy(outputs[mapping[name]], gradient)
            for name, gradient in expected.items()
            if name.startswith("layer2.")
        },
    }


def _replay(directory, *, step, steps):
    """Yield pointwise diagnostics while replaying one saved training trajectory."""
    import json
    import os

    from examples.lora.adam import adam_step
    from examples.lora.mlx_runtime import session
    from examples.mlp import NAMES
    from examples.mlp_bf16 import reference as independent_reference

    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("Expected a nonnegative integer step")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("Expected a positive integer steps count")
    with open(os.path.join(directory, "interface.json")) as stream:
        interface = json.load(stream)
    with np.load(os.path.join(directory, "benchmark.npz")) as arrays:
        params = {n: arrays[n].copy() for n in NAMES}
        feeds = {n: arrays[n].copy() for n in ("features", "targets")}
    model = ir.load(os.path.join(directory, "gradient.onnx"))
    executor = session(model)
    mapping = {p["input"]: p["gradient"] for p in interface["parameters"]}
    names = [v.name for v in executor.get_outputs()]
    observed, count = instrument(model)
    observer = session(observed)
    state = None
    for index in range(step + steps):
        values = executor.run(None, {**feeds, **params})
        outputs = dict(zip(names, values, strict=True))
        if index >= step:
            result = inspect(
                observer, {**feeds, **params}, values, mapping, original_count=count
            )
            expected = independent_reference(feeds["features"], feeds["targets"], params)[2]
            result.update(
                {
                    "step": index,
                    "profile": interface["profile"],
                    "rtol": 1e-4,
                    "atol": 1e-6,
                    "independent_gradients": {
                        n: discrepancy(outputs[mapping[n]], expected[n]) for n in NAMES
                    },
                    "independent_head_gradients": {
                        n: discrepancy(outputs[mapping[n]], expected[n]) for n in NAMES[2:]
                    },
                }
            )
            yield result
        if index + 1 < step + steps:
            params, state = adam_step(
                params,
                {n: outputs[mapping[n]] for n in NAMES},
                state,
                xp=np,
                **interface["adam"],
            )


def replay(directory, *, step):
    """Replay a saved classifier run to a zero-based pre-update step and inspect it."""
    return next(_replay(directory, step=step, steps=1))


def audit(directory, *, steps=150):
    """Summarize all four gradient comparisons over consecutive training states."""
    comparisons = {}
    for result in _replay(directory, step=0, steps=steps):
        for group in ("observed_primal_gradients", "independent_gradients"):
            for name, difference in result[group].items():
                entry = comparisons.setdefault(
                    group + "." + name,
                    {
                        "failed_steps": 0,
                        "max_abs": 0.0,
                    },
                )
                entry["max_abs"] = max(entry["max_abs"], difference["max_abs"])
                if difference["mismatched"]:
                    entry["failed_steps"] += 1
                    entry.setdefault("first_failure", {"step": result["step"], **difference})
    return {
        "steps": steps,
        "profile": result["profile"],
        "rtol": 1e-4,
        "atol": 1e-6,
        "original_outputs_equal": True,
        "comparisons": comparisons,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="Saved mlp_device run directory")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--step", type=int, help="Zero-based pre-update step")
    selection.add_argument(
        "--steps", type=int, help="Audit this many consecutive training states"
    )
    args = parser.parse_args()
    result = (
        replay(args.directory, step=args.step)
        if args.step is not None
        else audit(args.directory, steps=args.steps)
    )
    print(json.dumps(result, indent=2))
