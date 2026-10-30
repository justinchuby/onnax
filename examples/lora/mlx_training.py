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
from examples.lora.train import close, reference, reference_update
from examples.lora.validation import discrepancy
from examples.lora.workloads import PROFILES, optimizer_epsilon, problem_for


def gradient_model(w, c, *, batch, include_prediction=False, rank=4):
    return gradient_from_base(
        base_model(w, c), batch=batch, include_prediction=include_prediction, rank=rank
    )


def gradient_from_base(base, *, batch, include_prediction=False, rank=4):
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError("batch must be a positive integer")
    source = with_loss(adapt(base, rank=rank))
    width = source.graph.initializers["base.W"].shape[0]
    outputs = source.graph.initializers["base.C"].shape[1]
    for value in source.graph.inputs:
        if value.name == "features":
            value.shape = ir.Shape([batch, width])
        elif value.name == "target":
            value.shape = ir.Shape([batch, outputs])
    source.graph.outputs[0].shape = ir.Shape([batch, outputs])
    frozen = {name: digest(source.graph.initializers[name]) for name in ("base.W", "base.C")}
    if not include_prediction:
        source.graph.outputs.remove(source.graph.outputs[0])
    graph = optimizer.optimize(onnax.grad(source, inputs=PARAMETERS, output="loss"))
    assert frozen == {name: digest(graph.graph.initializers[name]) for name in frozen}
    interface = {
        "version": 1,
        "data_shapes": {"features": [None, width], "target": [None, outputs]},
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


def run(output_dir, *, compiled_adam=False, profile="tiny"):
    import mlx.core as mx

    os.makedirs(output_dir, exist_ok=True)
    w, c, x, y, vx, vy, initial = problem_for(profile)
    rank = initial[PARAMETERS[0]].shape[1]
    epsilon = optimizer_epsilon(profile)
    base_path = os.path.join(output_dir, "base.onnx")
    ir.save(base_model(w, c), base_path)
    loaded_base = ir.load(base_path)
    base_before = ir.to_proto(loaded_base).SerializeToString()
    graph, interface = gradient_from_base(loaded_base, batch=len(x), rank=rank)
    validation_graph, validation_interface = gradient_from_base(
        loaded_base, batch=len(vx), include_prediction=True, rank=rank
    )
    forward_graph = adapt(loaded_base, rank=rank)
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

        update = CompiledAdam(eps=epsilon)
    else:
        update = partial(adam_step, xp=mx, eps=epsilon)
    comparison_errors = {}
    trajectory_errors = {}
    first_failure = {}
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

        def compare(name, actual, expected, *, step, atol=1e-6):
            metric = discrepancy(actual, expected, atol=atol)
            summary = comparison_errors.setdefault(
                name, {"comparisons": 0, "max_abs": 0.0, "max_mismatched": 0}
            )
            summary["comparisons"] += 1
            summary["max_abs"] = max(summary["max_abs"], metric["max_abs"])
            summary["max_mismatched"] = max(summary["max_mismatched"], metric["mismatched"])
            if not metric["mismatched"]:
                return
            if profile != "large":
                np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=atol)
            row = trajectory_errors.setdefault(
                name, {"first_step": step, "steps": 0, "max_abs": 0.0, "max_mismatched": 0}
            )
            row["steps"] += 1
            row["max_abs"] = max(row["max_abs"], metric["max_abs"])
            row["max_mismatched"] = max(row["max_mismatched"], metric["mismatched"])
            if not first_failure:
                first_failure.update({"comparison": name, "step": step, **metric})
                snapshot = {"native." + key: host(value) for key, value in params.items()}
                snapshot.update(
                    {"reference." + key: value for key, value in reference_params.items()}
                )
                np.savez(os.path.join(output_dir, "first-mismatch.npz"), **snapshot)

        validation.bind(validation_x, validation_y, params)
        initial_validation = float(host(validation.run()[1]))
        for step in range(200):
            bound.bind(features, target, params)
            _, loss, grads = bound.run()
            prediction = predict(params)
            expected = reference(x, y, reference_params, w, c)
            same_point = reference(
                x, y, {name: host(value) for name, value in params.items()}, w, c
            )
            compare("same_parameter.prediction", prediction, same_point[0], step=step)
            compare("same_parameter.loss", host(loss), same_point[1], step=step)
            for name in PARAMETERS:
                compare(
                    "same_parameter.gradient." + name,
                    host(grads[name]),
                    same_point[2][name],
                    step=step,
                    atol=1e-6 if profile == "tiny" else 1e-9,
                )
            compare("prediction", prediction, expected[0], step=step)
            compare("loss", host(loss), expected[1], step=step)
            for name in PARAMETERS:
                compare(
                    "gradient." + name,
                    host(grads[name]),
                    expected[2][name],
                    step=step,
                    atol=1e-6 if profile == "tiny" else 1e-9,
                )
            if step == 0:
                assert np.count_nonzero(host(grads[PARAMETERS[0]])) == 0
                assert np.count_nonzero(host(grads[PARAMETERS[1]])) > 0
            elif step == 1:
                assert np.count_nonzero(host(grads[PARAMETERS[0]])) > 0
            history.append(float(host(loss)))
            params, state = update(params, grads, state)
            synchronize(params, state)
            reference_params, reference_state = reference_update(
                reference_params, expected[2], reference_state, eps=epsilon
            )
            for name in PARAMETERS:
                compare(
                    "parameter." + name,
                    host(params[name]),
                    reference_params[name],
                    step=step + 1,
                )
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
        compare(
            "validation.prediction",
            host(validation_result[0]),
            expected_validation[0],
            step=200,
        )
        compare(
            "validation.loss", host(validation_result[1]), expected_validation[1], step=200
        )
        for name in PARAMETERS:
            compare(
                "validation.gradient." + name,
                host(validation_result[2][name]),
                expected_validation[2][name],
                step=200,
                atol=1e-6 if profile == "tiny" else 1e-9,
            )
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
        reload_batches = sorted({1, 7, len(vx), len(x)})
        for batch in reload_batches:
            close(
                predict(restored, mx.array(x[:batch])),
                reference(
                    x[:batch], y[:batch], {name: saved[name] for name in PARAMETERS}, w, c
                )[0],
            )
        close(host(final[1]), host(replayed[1]))
        for name in PARAMETERS:
            close(host(final[2][name]), host(replayed[2][name]))
        updated, _ = update(params, final[2], state)
        restored_update = CompiledAdam(eps=epsilon) if compiled_adam else update
        restored_updated, _ = restored_update(restored, replayed[2], restored_state)
        synchronize(updated)
        synchronize(restored_updated)
        for name in PARAMETERS:
            close(host(updated[name]), host(restored_updated[name]))
        if first_failure:
            with np.load(os.path.join(output_dir, "first-mismatch.npz")) as snapshot:
                native_point = {
                    name: mx.array(snapshot["native." + name]) for name in PARAMETERS
                }
                native_cpu = {name: snapshot["native." + name] for name in PARAMETERS}
                reference_point = {name: snapshot["reference." + name] for name in PARAMETERS}
            observed = predict(native_point)
            same = reference(x, y, native_cpu, w, c)[0]
            separate = reference(x, y, reference_point, w, c)[0]
            precise = reference(
                x.astype(np.float64),
                y.astype(np.float64),
                {name: value.astype(np.float64) for name, value in native_cpu.items()},
                w.astype(np.float64),
                c.astype(np.float64),
            )[0]
            first_failure["diagnostic"] = {
                "native_vs_same_parameter_fp32": discrepancy(observed, same),
                "native_vs_same_parameter_fp64": discrepancy(observed, precise),
                "same_parameter_fp32_vs_independent_trajectory": discrepancy(same, separate),
            }
    assert ir.to_proto(loaded_base).SerializeToString() == base_before
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
        "strict_numerical_acceptance": not trajectory_errors,
        "same_parameter_acceptance": not any(
            name.startswith("same_parameter.") for name in trajectory_errors
        ),
        "independent_trajectory_acceptance": not any(
            not name.startswith("same_parameter.") for name in trajectory_errors
        ),
        "comparison_rtol": 1e-4,
        "prediction_parameter_loss_atol": 1e-6,
        "comparison_errors": comparison_errors,
        "trajectory_errors": trajectory_errors,
        "first_failure": first_failure,
        "compiled_adam": compiled_adam,
        "profile": profile,
        "optimizer_epsilon": epsilon,
        "gradient_comparison_atol": 1e-6 if profile == "tiny" else 1e-9,
        "dimensions": {
            "input": w.shape[0],
            "hidden": w.shape[1],
            "output": c.shape[1],
            "rank": rank,
        },
        "train_rows": len(x),
        "validation_rows": len(vx),
        "parameter_bytes": sum(value.nbytes for value in initial.values()),
        "forward_reload_batches": reload_batches,
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
    parser.add_argument("--profile", choices=list(PROFILES), default="tiny")
    args = parser.parse_args()
    result = run(args.output_dir, compiled_adam=args.compiled_adam, profile=args.profile)
    print(f"Loss: {result['initial_loss']:.6g} -> {result['final_loss']:.6g}")
    print(f"Strict numerical acceptance: {result['strict_numerical_acceptance']}")
    raise SystemExit(0 if result["strict_numerical_acceptance"] else 1)
