# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Separate classifier forward rounding from output-layer backpropagation."""

import numpy as np
import onnx
import onnx_ir as ir

from examples.lora.validation import discrepancy


def instrument(model):
    """Expose the classifier's rounded hidden activations and logits."""
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
    hidden = products[1].inputs[0]
    observed = tape.op("Cast", [hidden], {"to": int(ir.DataType.FLOAT)})
    observed.dtype = ir.DataType.FLOAT
    observed.shape = ir.Shape(hidden.shape)
    result.graph.outputs.extend([observed, logits[0]])
    onnx.checker.check_model(ir.to_proto(result), full_check=True)
    return result, count


def head_reference(hidden, logits, targets):
    """Recompute output-layer gradients from observed forward values in NumPy."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    dz = (np.exp(logp) - targets) / np.float32(len(targets))
    return {"layer2.weight": hidden.T @ dz, "layer2.bias": dz.sum(axis=0)}


def inspect(executor, feeds, original, mapping, *, original_count):
    """Check output preservation and compare analytic head gradients at native primals."""
    values = executor.run(None, feeds)
    preserved = all(
        a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
        for a, b in zip(original, values[:original_count], strict=True)
    )
    if not preserved:
        raise ValueError("Instrumentation changed original classifier outputs")
    outputs = dict(zip([v.name for v in executor.get_outputs()], values, strict=True))
    expected = head_reference(values[-2], values[-1], feeds["targets"])
    return {
        "original_outputs_equal": preserved,
        "observed_primal_head_gradients": {
            name: discrepancy(outputs[mapping[name]], gradient)
            for name, gradient in expected.items()
        },
    }


def replay(directory, *, step):
    """Replay a saved classifier run to a zero-based pre-update step and inspect it."""
    import json
    import os

    from examples.lora.adam import adam_step
    from examples.lora.mlx_runtime import session
    from examples.mlp import NAMES
    from examples.mlp_bf16 import reference

    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("Expected a nonnegative integer step")
    with open(os.path.join(directory, "interface.json")) as stream:
        interface = json.load(stream)
    with np.load(os.path.join(directory, "benchmark.npz")) as arrays:
        params = {n: arrays[n].copy() for n in NAMES}
        feeds = {n: arrays[n].copy() for n in ("features", "targets")}
    model = ir.load(os.path.join(directory, "gradient.onnx"))
    executor = session(model)
    mapping = {p["input"]: p["gradient"] for p in interface["parameters"]}
    names = [v.name for v in executor.get_outputs()]
    state = None
    for _ in range(step):
        outputs = dict(zip(names, executor.run(None, {**feeds, **params}), strict=True))
        params, state = adam_step(
            params,
            {n: outputs[mapping[n]] for n in NAMES},
            state,
            xp=np,
            **interface["adam"],
        )
    values = executor.run(None, {**feeds, **params})
    observed, count = instrument(model)
    result = inspect(
        session(observed), {**feeds, **params}, values, mapping, original_count=count
    )
    outputs = dict(zip(names, values, strict=True))
    expected = reference(feeds["features"], feeds["targets"], params)[2]
    result.update(
        {
            "step": step,
            "profile": interface["profile"],
            "rtol": 1e-4,
            "atol": 1e-6,
            "independent_head_gradients": {
                n: discrepancy(outputs[mapping[n]], expected[n]) for n in NAMES[2:]
            },
        }
    )
    return result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="Saved mlp_device run directory")
    parser.add_argument("--step", type=int, required=True, help="Zero-based pre-update step")
    args = parser.parse_args()
    print(json.dumps(replay(args.directory, step=args.step), indent=2))
