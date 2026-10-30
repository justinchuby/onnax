# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Isolated MLX GPU Adam timing with shared DLPack or NumPy-view boundaries."""

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from importlib.metadata import version

import numpy as np
import onnx_ir as ir

from examples.lora.adam import adam_step
from examples.lora.mlx_exchange import MlxDLPack, synchronize
from examples.lora.mlx_runtime import session
from examples.lora.mlx_training import gradient_model, host
from examples.lora.train import problem


def benchmark(mode, *, batch=256):
    import mlx.core as mx

    if mode not in {"dlpack", "numpy_view", "numpy_adam"}:
        raise ValueError("Expected dlpack, numpy_view or numpy_adam mode")
    w, c, x, y, _, _, initial = problem()
    repeats = (batch + len(x) - 1) // len(x)
    x, y = np.tile(x, (repeats, 1))[:batch], np.tile(y, (repeats, 1))[:batch]
    graph, interface = gradient_model(w, c, batch=batch)
    started = time.perf_counter()
    executor = session(graph)
    creation = time.perf_counter() - started
    bound = MlxDLPack(executor, interface) if mode == "dlpack" else None
    output_names = [value.name for value in executor.get_outputs()]
    timings = {
        key: [] for key in ("prepare", "execute_exchange", "import", "adam_sync", "step")
    }
    with mx.stream(mx.gpu):
        data = {"features": mx.array(x), "target": mx.array(y)}
        cpu_adam = mode == "numpy_adam"
        params = {
            name: value.copy() if cpu_adam else mx.array(value)
            for name, value in initial.items()
        }
        synchronize(data)
        if not cpu_adam:
            synchronize(params)
        cpu_data = {name: np.asarray(value, copy=False) for name, value in data.items()}
        state = None
        initial_active = mx.get_active_memory()
        mx.reset_peak_memory()
        for index in range(110):
            started = time.perf_counter()
            if bound is not None:
                bound.bind(data["features"], data["target"], params)
            elif cpu_adam:
                numpy_feeds = {**cpu_data, **params}
            else:
                feeds = {**data, **params}
                synchronize(feeds)
                numpy_feeds = {
                    name: np.asarray(value, copy=False) for name, value in feeds.items()
                }
            prepared = time.perf_counter()
            if bound is not None:
                bound.execute()
            else:
                values = executor.run(None, numpy_feeds)
            executed = time.perf_counter()
            if bound is not None:
                _, loss, grads = bound.import_outputs()
            elif cpu_adam:
                named = dict(zip(output_names, values, strict=True))
                loss = named[interface["loss"]]
                grads = {
                    row["input"]: named[row["gradient"]] for row in interface["parameters"]
                }
            else:
                arrays = [mx.from_dlpack(value, copy=False) for value in values]
                for array, value in zip(arrays, values, strict=True):
                    if np.asarray(array, copy=False).ctypes.data != value.ctypes.data:
                        raise RuntimeError("NumPy-view output import copied storage")
                named = dict(zip(output_names, arrays, strict=True))
                loss = named[interface["loss"]]
                grads = {
                    row["input"]: named[row["gradient"]] for row in interface["parameters"]
                }
            imported = time.perf_counter()
            params, state = adam_step(params, grads, state, xp=np if cpu_adam else mx)
            if not cpu_adam:
                synchronize(params, state)
            updated = time.perf_counter()
            if index >= 10:
                for key, elapsed in zip(
                    timings,
                    (
                        prepared - started,
                        executed - prepared,
                        imported - executed,
                        updated - imported,
                        updated - started,
                    ),
                    strict=True,
                ):
                    timings[key].append(elapsed * 1000)
        device_memory = {
            "active": mx.get_active_memory(),
            "peak": max(initial_active, mx.get_peak_memory()),
            "cache": mx.get_cache_memory(),
        }
        final_params = {
            name: (value if cpu_adam else host(value)).tolist()
            for name, value in params.items()
        }
        final_loss = float(loss if cpu_adam else host(loss))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "mode": mode,
        "fetches": output_names,
        "batch": batch,
        "session_creation_ms": creation * 1000,
        "timings_ms": {
            name: {"median": float(np.median(values)), "p95": float(np.percentile(values, 95))}
            for name, values in timings.items()
        },
        "python_mlx_allocator_bytes": device_memory,
        "python_mlx_allocator_scope": "Python MLX allocator; plugin allocator is separate. Peak includes the active initialization baseline and the allocation maximum over measured/warmup updates.",
        "peak_process_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
        "final_parameters": final_params,
        "final_loss": final_loss,
        "nodes": len(graph.graph),
        "model_bytes": len(ir.to_proto(graph).SerializeToString()),
        "model_sha256": hashlib.sha256(ir.to_proto(graph).SerializeToString()).hexdigest(),
        "parameter_bytes": sum(value.nbytes for value in initial.values()),
        "dataset_bytes": x.nbytes + y.nbytes,
        "platform": platform.platform(),
        "versions": {
            name: version(name)
            for name in (
                "mlx",
                "onnxruntime",
                "onnxruntime-ep-mlx",
                "onnx",
                "onnxscript",
                "numpy",
            )
        },
        "threads": {"intra_op": 1, "inter_op": 1},
        "adam_stream": "NumPy CPU" if cpu_adam else "MLX GPU",
        "ort_graph_optimizer": "disabled",
        "cpu_ep_fallback": "disabled",
        "plugin_trace_enabled": bool(os.environ.get("ONNXRUNTIME_EP_MLX_TRACE")),
        "scope": "Isolated process per mode/batch, sequential runs, 10 warmup/100 measured updates. All modes use the same native gradient graph and shared dataset views. dlpack and numpy_view use MLX GPU Adam and share Python-boundary storage; numpy_adam keeps parameters/state on CPU. Prepare includes input synchronization and views/binding. Execute includes plugin internal wrapping/copy-out. Adam time synchronizes parameters and both moments. RSS includes imports, graph generation, runtime and allocators.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["dlpack", "numpy_view", "numpy_adam"], required=True
    )
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = benchmark(args.mode, batch=args.batch)
    with open(args.output, "w") as stream:
        json.dump(result, stream, indent=2)
