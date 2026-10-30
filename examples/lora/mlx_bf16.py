# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Evaluate exported BF16 gradients on the native MLX ONNX Runtime provider."""

from __future__ import annotations

import json
import os
import platform
import resource
import sys
import time
from importlib.metadata import version

import numpy as np
import onnx
import onnx_ir as ir
from onnx_ir.passes.common import InlinePass
from onnxscript import optimizer

import onnax
from examples.lora.adam import adam_step
from examples.lora.bf16 import reference, reference_update
from examples.lora.bf16_rules import lower_matmuls
from examples.lora.mlx_runtime import session
from examples.lora.model import PARAMETERS, adapt, base_model, digest, with_loss
from examples.lora.train import close, problem
from examples.lora.validation import discrepancy


def gradient_model(w, c, *, batch):
    """Specialize this workload before inlining explicit BF16 matrix functions."""
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError("batch must be a positive integer")
    source = with_loss(adapt(base_model(w, c)))
    for value in source.graph.inputs:
        if value.name == "features":
            value.shape = ir.Shape([batch, 16])
        elif value.name == "target":
            value.shape = ir.Shape([batch, 8])
    source.graph.outputs[0].shape = ir.Shape([batch, 8])
    gradient = onnax.grad(lower_matmuls(source), inputs=PARAMETERS, output="loss")
    graph = InlinePass()(gradient).model
    # Preserve FP32 frozen masters and execute BF16 casts in the runtime.
    return optimizer.optimize(graph, input_size_limit=0, output_size_limit=0)


def run(output_dir, *, inspect_products=False, inspect_fp32=False):
    """Train and report both same-parameter and independent-trajectory comparisons.

    A mismatch remains visible in the report and CLI status; it is never hidden
    by widening the acceptance tolerance. Native convergence is checked separately.
    """
    inspect_products = inspect_products or inspect_fp32
    os.makedirs(output_dir, exist_ok=True)
    w, c, x, y, vx, vy, params = problem()
    graphs = {
        "gradient": gradient_model(w, c, batch=256),
        "validation": gradient_model(w, c, batch=128),
    }
    frozen = {
        name: digest(value)
        for name, value in graphs["gradient"].graph.initializers.items()
        if name in {"base.W", "base.C"}
    }
    for name, graph in graphs.items():
        onnx.checker.check_model(ir.to_proto(graph), full_check=True)
        ir.save(graph, os.path.join(output_dir, name + ".onnx"))
    started = time.perf_counter()
    executor = session(
        ir.load(os.path.join(output_dir, "gradient.onnx")),
        profile_prefix=os.path.join(output_dir, "profile"),
    )
    creation = time.perf_counter() - started
    validation = session(graphs["validation"])
    inspector = None
    if inspect_products:
        from examples.lora.bf16_inspector import ProductInspector

        inspector = ProductInspector(
            graphs["gradient"], session_factory=session, include_fp32=inspect_fp32
        )
    initial_validation = float(
        validation.run(None, {"features": vx, "target": vy, **params})[1]
    )
    independent = {name: value.copy() for name, value in params.items()}
    state = independent_state = None
    rows = []
    timings = []
    first_mismatch = None
    for step in range(200):
        started = time.perf_counter()
        actual = executor.run(None, {"features": x, "target": y, **params})
        timings.append((time.perf_counter() - started) * 1000)
        product_checks = None
        if inspector is not None:
            product_checks = inspector.check({"features": x, "target": y, **params}, actual)
        same = reference(x, y, params, w, c)
        separate = reference(x, y, independent, w, c)
        metrics = {
            name: discrepancy(a, b)
            for name, a, b in zip(
                ("prediction", "loss", *PARAMETERS),
                actual,
                [*same[:2], *same[2].values()],
                strict=True,
            )
        }
        if first_mismatch is None and any(row["mismatched"] for row in metrics.values()):
            first_mismatch = step
            np.savez(
                os.path.join(output_dir, "first-mismatch.npz"), features=x, target=y, **params
            )
        if step == 0:
            assert np.count_nonzero(actual[2]) == 0
            assert np.count_nonzero(actual[3]) > 0
        elif step == 1:
            assert np.count_nonzero(actual[2]) > 0
        params, state = adam_step(
            params, dict(zip(PARAMETERS, actual[2:], strict=True)), state, xp=np
        )
        independent, independent_state = reference_update(
            independent, separate[2], independent_state
        )
        rows.append(
            {
                "step": step,
                "product_bounds": product_checks,
                "native_loss": float(actual[1]),
                "independent_loss": float(separate[1]),
                "same_parameter": metrics,
                "independent_trajectory": {
                    name: discrepancy(a, b)
                    for name, a, b in zip(
                        ("prediction", "loss", *PARAMETERS),
                        actual,
                        [*separate[:2], *separate[2].values()],
                        strict=True,
                    )
                },
                "trajectory_parameters": {
                    name: discrepancy(params[name], independent[name]) for name in PARAMETERS
                },
            }
        )
    final = executor.run(None, {"features": x, "target": y, **params})
    final_validation = validation.run(None, {"features": vx, "target": vy, **params})
    assert final[1] < rows[0]["native_loss"] * 0.5
    assert final_validation[1] < initial_validation * 0.5
    assert frozen == {
        name: digest(graphs["gradient"].graph.initializers[name]) for name in frozen
    }
    checkpoint = {**params, "adam.step": np.array(state.step, np.int64)}
    checkpoint.update({"adam.first." + name: value for name, value in state.first.items()})
    checkpoint.update({"adam.second." + name: value for name, value in state.second.items()})
    np.savez(os.path.join(output_dir, "checkpoint.npz"), **checkpoint)
    from examples.lora.adam import AdamState

    with np.load(os.path.join(output_dir, "checkpoint.npz")) as saved:
        restored = {name: saved[name] for name in PARAMETERS}
        restored_state = AdamState(
            int(saved["adam.step"]),
            {name: saved["adam.first." + name] for name in PARAMETERS},
            {name: saved["adam.second." + name] for name in PARAMETERS},
        )
    replay = session(ir.load(os.path.join(output_dir, "gradient.onnx"))).run(
        None, {"features": x, "target": y, **restored}
    )
    for a, b in zip(final, replay, strict=True):
        close(a, b)
    updated, _ = adam_step(params, dict(zip(PARAMETERS, final[2:], strict=True)), state, xp=np)
    replay_updated, _ = adam_step(
        restored, dict(zip(PARAMETERS, replay[2:], strict=True)), restored_state, xp=np
    )
    for name in PARAMETERS:
        close(updated[name], replay_updated[name])
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
    trajectory_passed = all(
        not metric["mismatched"]
        for row in rows
        for group in ("trajectory_parameters", "independent_trajectory")
        for metric in row[group].values()
    )
    report = {
        "version": 1,
        "product_bounds": {
            "enabled": inspect_products,
            "include_fp32_backward": inspect_fp32,
            "arithmetic_model": {
                "unit_roundoff": 2**-24,
                "absolute_underflow_per_operation": 2**-126,
                "operation_budget": "2*k",
                "final_rounding": "nearest-even BF16 for BF16 products; outward FP32 endpoints for FP32 products",
                "reference": "exact integer products/sums and rational endpoints",
            },
            "all_observed_products_pass": all(
                row["product_bounds"]["original_outputs_equal"]
                and all(
                    not p["violations"] for p in row["product_bounds"]["products"].values()
                )
                for row in rows
            )
            if inspect_products
            else None,
            "scope": "Local matrix products at observed inputs. FP32 accumulation model, normal/zero operands; BF16 forward products and optional FP32 backward products. Original graph outputs compared byte-for-byte.",
        },
        "models": {
            name: {
                "nodes": len(graph.graph),
                "bytes": os.path.getsize(os.path.join(output_dir, name + ".onnx")),
            }
            for name, graph in graphs.items()
        },
        "parameter_bytes": sum(value.nbytes for value in params.values()),
        "threads": {"intra_op": 1, "inter_op": 1},
        "ort_graph_optimizer": "disabled",
        "cpu_ep_fallback": "disabled",
        "steps": 200,
        "comparison_tolerance": {"rtol": 1e-4, "atol": 1e-6},
        "first_same_parameter_mismatch_step": first_mismatch,
        "strict_numerical_acceptance": first_mismatch is None and trajectory_passed,
        "initial_loss": rows[0]["native_loss"],
        "final_loss": float(final[1]),
        "initial_validation_loss": initial_validation,
        "final_validation_loss": float(final_validation[1]),
        "independent_final_validation_loss": float(reference(vx, vy, independent, w, c)[1]),
        "executed_providers": providers,
        "session_creation_ms": creation * 1000,
        "execution_ms": {
            "median": float(np.median(timings[10:110])),
            "p95": float(np.percentile(timings[10:110], 95)),
        },
        "timing_scope": "Original-graph execution/exchange only, first 10 steps warmup and next 100 measured. References, validation/replay and optional product inspection execute between measurements in this process; Adam, references, export and inspection are excluded from timings. Use isolated benchmarks for provider comparisons.",
        "versions": {
            name: version(name)
            for name in ("onnxruntime", "onnxruntime-ep-mlx", "onnx", "onnxscript", "numpy")
        },
        "platform": platform.platform(),
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "rss_scope": "Isolated CLI process including imports, graph generation, training/validation/replay sessions, optional product inspection, references, checkpoint replay and allocators.",
        "frozen_master": frozen,
        "parameters": [
            {
                "input": name,
                "gradient": value.name,
                "shape": list(params[name].shape),
                "dtype": "float32",
            }
            for name, value in zip(
                PARAMETERS, graphs["gradient"].graph.outputs[2:], strict=True
            )
        ],
        "history": rows,
    }
    if first_mismatch is not None:
        report["first_mismatch_diagnostic"] = diagnose(output_dir)
    with open(os.path.join(output_dir, "report.json"), "w") as stream:
        json.dump(report, stream, indent=2)
    return report


def diagnose(output_dir):
    """Expose BF16 matrix operands and compare the first prediction mismatch."""
    from onnx.reference import ReferenceEvaluator

    from examples.lora.bf16 import round_bf16

    graph = ir.load(os.path.join(output_dir, "gradient.onnx"))
    with np.load(os.path.join(output_dir, "first-mismatch.npz")) as saved:
        feeds = dict(saved)
    # The last forward BF16 product maps hidden activations to predictions.
    products = [
        node
        for node in graph.graph
        if node.op_type == "MatMul"
        and node.outputs[0].dtype == ir.DataType.BFLOAT16
        and list(node.outputs[0].shape) == [256, 8]
    ]
    if len(products) != 1:
        raise ValueError("Expected the LoRA prediction's BF16 matrix product")
    product = products[0]
    tape = ir.tape.Tape(graph.graph)
    for value in product.inputs:
        output = tape.op("Cast", [value], {"to": int(ir.DataType.FLOAT)})
        output.dtype = ir.DataType.FLOAT
        output.shape = ir.Shape(value.shape)
        graph.graph.outputs.append(output)
    native = session(graph).run(None, feeds)
    oracle = ReferenceEvaluator(ir.to_proto(graph)).run(None, feeds)
    bad = ~np.isclose(native[0], oracle[0], rtol=1e-4, atol=1e-6)
    if not np.any(bad):
        return {"prediction_mismatch_reproduced": False}
    row, column = (int(index) for index in np.argwhere(bad)[0])
    a, b = oracle[-2:]
    precise = float(a[row].astype(np.float64) @ b[:, column].astype(np.float64))
    fp32 = float((a @ b)[row, column])
    return {
        "prediction_mismatch_reproduced": True,
        "index": [row, column],
        "bf16_operands_identical": all(
            np.array_equal(a, b) for a, b in zip(native[-2:], oracle[-2:], strict=True)
        ),
        "native_prediction": float(native[0][row, column]),
        "reference_prediction": float(oracle[0][row, column]),
        "fp64_operand_dot": precise,
        "numpy_fp32_operand_dot": fp32,
        "rounded_fp64_dot": float(round_bf16(np.array(precise, np.float32))),
        "rounded_numpy_fp32_dot": float(round_bf16(np.array(fp32, np.float32))),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="mlx_bf16_models")
    parser.add_argument("--inspect-products", action="store_true")
    parser.add_argument("--inspect-fp32", action="store_true")
    args = parser.parse_args()
    inspect_products = args.inspect_products or args.inspect_fp32
    result = run(
        args.output_dir, inspect_products=inspect_products, inspect_fp32=args.inspect_fp32
    )
    print(f"Loss: {result['initial_loss']:.6g} -> {result['final_loss']:.6g}")
    print(f"Strict numerical acceptance: {result['strict_numerical_acceptance']}")
    if inspect_products:
        print(
            f"Local matrix product bounds: {result['product_bounds']['all_observed_products_pass']}"
        )
    accepted = result["strict_numerical_acceptance"] and (
        not inspect_products or result["product_bounds"]["all_observed_products_pass"]
    )
    raise SystemExit(0 if accepted else 1)
