# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Train FP32 LoRA with shared runtime boundaries and MLX GPU Adam."""

from __future__ import annotations

import hashlib
import json
import os
from functools import partial
from importlib.metadata import version

import numpy as np
import onnx
import onnx_ir as ir
from onnxscript import optimizer

import onnax
from examples.lora.adam import AdamState, adam_step
from examples.lora.mlx_exchange import MlxDLPack, synchronize
from examples.lora.mlx_runtime import session
from examples.lora.model import PARAMETERS, adapt, base_model, digest, with_loss
from examples.lora.train import close, problem, reference, reference_update


def gradient_model(w, c, *, batch, include_prediction=False):
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError("batch must be a positive integer")
    source = with_loss(adapt(base_model(w, c)))
    for value in source.graph.inputs:
        if value.name == "features":
            value.shape = ir.Shape([batch, 16])
        elif value.name == "target":
            value.shape = ir.Shape([batch, 8])
    source.graph.outputs[0].shape = ir.Shape([batch, 8])
    frozen = {name: digest(source.graph.initializers[name]) for name in ("base.W", "base.C")}
    if not include_prediction:
        source.graph.outputs.remove(source.graph.outputs[0])
    graph = optimizer.optimize(onnax.grad(source, inputs=PARAMETERS, output="loss"))
    assert frozen == {name: digest(graph.graph.initializers[name]) for name in frozen}
    interface = {
        "version": 1,
        "loss": "loss",
        "prediction": "prediction" if include_prediction else None,
        "loss_reduction": "mean over batch and output coordinates",
        "parameters": [
            {
                "input": name,
                "gradient": value.name,
                "shape": list(value.shape),
                "dtype": "float32",
            }
            for name, value in zip(PARAMETERS, graph.graph.outputs[-2:], strict=True)
        ],
        "frozen": {
            name: digest(graph.graph.initializers[name]) for name in ("base.W", "base.C")
        },
    }
    return graph, interface


def host(value):
    """Read a synchronized shared view at validation/checkpoint boundaries."""
    import mlx.core as mx

    mx.eval(value)
    return np.asarray(value, copy=False)


def run(output_dir, *, compiled_adam=False):
    import mlx.core as mx

    os.makedirs(output_dir, exist_ok=True)
    w, c, x, y, vx, vy, initial = problem()
    graph, interface = gradient_model(w, c, batch=256)
    validation_graph, validation_interface = gradient_model(
        w, c, batch=128, include_prediction=True
    )
    forward_graph = adapt(base_model(w, c))
    for name, model in (
        ("forward", forward_graph),
        ("gradient", graph),
        ("validation", validation_graph),
    ):
        onnx.checker.check_model(ir.to_proto(model), full_check=True)
        ir.save(model, os.path.join(output_dir, name + ".onnx"))
    with open(os.path.join(output_dir, "interface.json"), "w") as stream:
        json.dump(interface, stream, indent=2)
    executor = session(
        ir.load(os.path.join(output_dir, "gradient.onnx")),
        profile_prefix=os.path.join(output_dir, "profile"),
    )
    bound = MlxDLPack(executor, interface)
    forward_executor = session(ir.load(os.path.join(output_dir, "forward.onnx")))
    validation = MlxDLPack(session(validation_graph), validation_interface)
    if compiled_adam:
        from examples.lora.mlx_adam import CompiledAdam

        update = CompiledAdam()
    else:
        update = partial(adam_step, xp=mx)
    history = []
    reference_params = {name: value.copy() for name, value in initial.items()}
    state = reference_state = None
    with mx.stream(mx.gpu):
        features, target = mx.array(x), mx.array(y)
        validation_x, validation_y = mx.array(vx), mx.array(vy)
        params = {name: mx.array(value) for name, value in initial.items()}

        def predict(current, feature_values=None):
            return forward_executor.run(
                None,
                {
                    "features": host(features if feature_values is None else feature_values),
                    **{name: host(value) for name, value in current.items()},
                },
            )[0]

        validation.bind(validation_x, validation_y, params)
        initial_validation = float(host(validation.run()[1]))
        for step in range(200):
            bound.bind(features, target, params)
            _, loss, grads = bound.run()
            prediction = predict(params)
            expected = reference(x, y, reference_params, w, c)
            close(prediction, expected[0])
            close(host(loss), expected[1])
            for name in PARAMETERS:
                close(host(grads[name]), expected[2][name])
            if step == 0:
                assert np.count_nonzero(host(grads[PARAMETERS[0]])) == 0
                assert np.count_nonzero(host(grads[PARAMETERS[1]])) > 0
            elif step == 1:
                assert np.count_nonzero(host(grads[PARAMETERS[0]])) > 0
            history.append(float(host(loss)))
            params, state = update(params, grads, state)
            synchronize(params, state)
            reference_params, reference_state = reference_update(
                reference_params, expected[2], reference_state
            )
            for name in PARAMETERS:
                close(host(params[name]), reference_params[name])
                assert params[name].__dlpack_device__() == (8, 0)
                assert state.first[name].__dlpack_device__() == (8, 0)
                assert state.second[name].__dlpack_device__() == (8, 0)
        bound.bind(features, target, params)
        final = bound.run()
        final_prediction = predict(params)
        validation.bind(validation_x, validation_y, params)
        validation_result = validation.run()
        final_validation = float(host(validation_result[1]))
        expected_validation = reference(vx, vy, reference_params, w, c)
        close(host(validation_result[0]), expected_validation[0])
        close(host(validation_result[1]), expected_validation[1])
        for name in PARAMETERS:
            close(host(validation_result[2][name]), expected_validation[2][name])
        assert float(host(final[1])) < history[0] * 0.5
        assert final_validation < initial_validation * 0.5
        saved = {name: host(value) for name, value in params.items()}
        saved.update(
            {"adam.first." + name: host(value) for name, value in state.first.items()}
        )
        saved.update(
            {"adam.second." + name: host(value) for name, value in state.second.items()}
        )
        saved["adam.step"] = np.array(state.step, np.int64)
        np.savez(os.path.join(output_dir, "checkpoint.npz"), **saved)
        with np.load(os.path.join(output_dir, "checkpoint.npz")) as checkpoint:
            restored = {name: mx.array(checkpoint[name]) for name in PARAMETERS}
            restored_state = AdamState(
                int(checkpoint["adam.step"]),
                {name: mx.array(checkpoint["adam.first." + name]) for name in PARAMETERS},
                {name: mx.array(checkpoint["adam.second." + name]) for name in PARAMETERS},
            )
        replay = MlxDLPack(
            session(ir.load(os.path.join(output_dir, "gradient.onnx"))), interface
        )
        replay.bind(features, target, restored)
        replayed = replay.run()
        close(final_prediction, predict(restored))
        for batch in (1, 7, 128):
            close(
                predict(restored, mx.array(x[:batch])),
                reference(x[:batch], y[:batch], reference_params, w, c)[0],
            )
        close(host(final[1]), host(replayed[1]))
        for name in PARAMETERS:
            close(host(final[2][name]), host(replayed[2][name]))
        updated, _ = update(params, final[2], state)
        restored_update = CompiledAdam() if compiled_adam else update
        restored_updated, _ = restored_update(restored, replayed[2], restored_state)
        synchronize(updated)
        synchronize(restored_updated)
        for name in PARAMETERS:
            close(host(updated[name]), host(restored_updated[name]))
    assert interface["frozen"] == {
        name: digest(graph.graph.initializers[name]) for name in interface["frozen"]
    }
    with open(executor.end_profiling()) as stream:
        events = json.load(stream)
    providers = sorted(
        {
            event["args"]["provider"]
            for event in events
            if event.get("args", {}).get("provider")
        }
    )
    assert providers == ["MLXExecutionProvider"], providers
    report = {
        "steps": 200,
        "compiled_adam": compiled_adam,
        "forward_reload_batches": [1, 7, 128, 256],
        "training_fetches": [value.name for value in graph.graph.outputs],
        "initial_loss": history[0],
        "final_loss": float(host(final[1])),
        "initial_validation_loss": initial_validation,
        "final_validation_loss": final_validation,
        "executed_providers": providers,
        "parameter_and_state_dlpack_device": [8, 0],
        "boundary": "Shared CPU views of Metal inputs and fresh runtime outputs imported with copy=False; pointer identity checked both ways. Plugin internal wrapping/copy-out is separate.",
        "loss_history": history,
        "nodes": len(graph.graph),
        "model_bytes": os.path.getsize(os.path.join(output_dir, "gradient.onnx")),
        "model_sha256": hashlib.sha256(ir.to_proto(graph).SerializeToString()).hexdigest(),
        "versions": {
            name: version(name)
            for name in ("mlx", "onnxruntime", "onnxruntime-ep-mlx", "onnxscript", "numpy")
        },
    }
    with open(os.path.join(output_dir, "report.json"), "w") as stream:
        json.dump(report, stream, indent=2)
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="mlx_training_models")
    parser.add_argument("--compiled-adam", action="store_true")
    args = parser.parse_args()
    result = run(args.output_dir, compiled_adam=args.compiled_adam)
    print(f"Loss: {result['initial_loss']:.6g} -> {result['final_loss']:.6g}")
