# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Static-batch LoRA gradients for the CoreML execution provider."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
from onnxscript import optimizer

import onnax
from examples.lora.model import PARAMETERS, adapt, base_model, with_loss


def gradient_model(w, c, *, batch: int, relu_magnitude: bool = True):
    """Build this example's stable Tanh gradient with a fixed batch size.

    CoreML accepts Relu arithmetic for the Tanh slope's absolute value. The
    replacement happens after differentiation and only in this known graph.
    """
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError("batch must be a positive integer")
    source = with_loss(adapt(base_model(w, c)))
    for value in source.graph.inputs:
        if value.name == "features":
            value.shape = ir.Shape([batch, 16])
        elif value.name == "target":
            value.shape = ir.Shape([batch, 8])
    source.graph.outputs[0].shape = ir.Shape([batch, 8])
    gradient = optimizer.optimize(onnax.grad(source, inputs=PARAMETERS, output="loss"))
    if not relu_magnitude:
        return gradient
    choices = [node for node in gradient.graph if node.op_type == "Where"]
    if len(choices) != 1:
        raise ValueError("Expected one stable Tanh magnitude in the LoRA gradient")
    node = choices[0]
    condition, positive, negative = node.inputs
    assert condition is not None and positive is not None and negative is not None
    comparison, negation = condition.producer(), negative.producer()
    if (
        comparison is None
        or comparison.op_type != "GreaterOrEqual"
        or comparison.inputs[0] is not positive
        or comparison.inputs[1].const_value is None
        or np.asarray(comparison.inputs[1].const_value.numpy()).item() != 0
        or negation is None
        or negation.op_type != "Neg"
        or negation.inputs[0] is not positive
    ):
        raise ValueError("Expected Where(x >= 0, x, -x) for the stable Tanh slope")
    tape = ir.tape.Tape()
    minus = tape.initializer(ir.tensor(np.array(-1, np.float32), name="coreml.minus_one"))
    negated = tape.op("Mul", [positive, minus])
    left = tape.op("Relu", [positive])
    right = tape.op("Relu", [negated])
    for value in (negated, left, right):
        value.dtype = positive.dtype
        value.shape = ir.Shape(positive.shape)
    gradient.graph.insert_before(node, tape.nodes)
    gradient.graph.register_initializer(minus)
    node.op_type = "Add"
    node.replace_input_with(0, left)
    node.replace_input_with(1, right)
    node.resize_inputs(2)
    return optimizer.optimize(gradient)


def session(model, *, profile_prefix=None):
    """Require complete CoreML assignment; execute with FP32 accumulation."""
    if "CoreMLExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CoreMLExecutionProvider is unavailable")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if profile_prefix is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    executor = ort.InferenceSession(
        ir.to_proto(model).SerializeToString(),
        options,
        providers=[
            (
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "CPUAndGPU",
                    "RequireStaticInputShapes": "1",
                    "AllowLowPrecisionAccumulationOnGPU": "0",
                },
            )
        ],
    )
    executor.disable_fallback()
    return executor


def train(output_dir):
    """Validate 200 native CoreML updates against an independent NumPy path."""
    import hashlib
    import json
    import os

    from examples.lora.adam import adam_step
    from examples.lora.train import close, problem, reference, reference_update

    os.makedirs(output_dir, exist_ok=True)
    w, c, x, y, vx, vy, params = problem()
    frozen = (w.tobytes(), c.tobytes())
    graph = gradient_model(w, c, batch=len(x))
    import onnx

    onnx.checker.check_model(ir.to_proto(graph), full_check=True)
    path = os.path.join(output_dir, "gradient.onnx")
    ir.save(graph, path)
    artifact_digest = hashlib.sha256(ir.to_proto(graph).SerializeToString()).hexdigest()
    executor = session(ir.load(path), profile_prefix=os.path.join(output_dir, "profile"))
    expected_params = {name: value.copy() for name, value in params.items()}
    state = expected_state = None
    losses = []
    initial_validation = float(reference(vx, vy, params, w, c)[1])
    for step in range(200):
        actual = executor.run(None, {"features": x, "target": y, **params})
        expected = reference(x, y, expected_params, w, c)
        for a, b in zip(actual, [*expected[:2], *expected[2].values()], strict=True):
            close(a, b)
        if step == 0:
            assert np.count_nonzero(actual[2]) == 0
            assert np.count_nonzero(actual[3]) > 0
        elif step == 1:
            assert np.count_nonzero(actual[2]) > 0
        losses.append(float(actual[1]))
        params, state = adam_step(
            params, dict(zip(PARAMETERS, actual[2:], strict=True)), state, xp=np
        )
        expected_params, expected_state = reference_update(
            expected_params, expected[2], expected_state
        )
        for name in PARAMETERS:
            close(params[name], expected_params[name])
    final = executor.run(None, {"features": x, "target": y, **params})
    validation = float(reference(vx, vy, params, w, c)[1])
    assert final[1] < losses[0] * 0.5
    assert validation < initial_validation * 0.5
    assert frozen == (w.tobytes(), c.tobytes())
    assert (
        hashlib.sha256(ir.to_proto(graph).SerializeToString()).hexdigest() == artifact_digest
    )
    np.savez(os.path.join(output_dir, "adapter.npz"), **params)
    with np.load(os.path.join(output_dir, "adapter.npz")) as checkpoint:
        restored = session(ir.load(path)).run(
            None, {"features": x, "target": y, **dict(checkpoint)}
        )
    for a, b in zip(final, restored, strict=True):
        close(a, b)
    profile = executor.end_profiling()
    with open(profile) as stream:
        events = json.load(stream)
    providers = sorted(
        {
            event["args"]["provider"]
            for event in events
            if event.get("args", {}).get("provider")
        }
    )
    assert providers == ["CoreMLExecutionProvider"], providers
    report = {
        "interface_version": 1,
        "parameters": [
            {
                "input": name,
                "gradient": value.name,
                "shape": list(params[name].shape),
                "dtype": "float32",
            }
            for name, value in zip(PARAMETERS, graph.graph.outputs[2:], strict=True)
        ],
        "model_sha256": artifact_digest,
        "steps": 200,
        "batch": len(x),
        "initial_loss": losses[0],
        "final_loss": float(final[1]),
        "initial_validation_loss": initial_validation,
        "final_validation_loss": validation,
        "executed_providers": providers,
        "loss_history": losses,
        "model_bytes": os.path.getsize(path),
        "nodes": len(graph.graph),
        "ort_version": ort.__version__,
    }
    with open(os.path.join(output_dir, "report.json"), "w") as stream:
        json.dump(report, stream, indent=2)
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="coreml_models")
    args = parser.parse_args()
    result = train(args.output_dir)
    print(f"Loss: {result['initial_loss']:.6g} -> {result['final_loss']:.6g}")
