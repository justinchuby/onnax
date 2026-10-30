# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Isolated CoreML/CPU comparison: python -m examples.lora.coreml_benchmark."""

import argparse
import json
import platform
import resource
import sys
import time
from importlib.metadata import version

import numpy as np
import onnx_ir as ir
import onnxruntime as ort

from examples.lora.adam import adam_step
from examples.lora.coreml import gradient_model, session
from examples.lora.model import PARAMETERS
from examples.lora.train import problem
from examples.lora.train import session as cpu_session


def benchmark(provider, batch):
    w, c, x, y, _, _, params = problem()
    repeats = (batch + len(x) - 1) // len(x)
    x, y = np.tile(x, (repeats, 1))[:batch], np.tile(y, (repeats, 1))[:batch]
    graph = gradient_model(w, c, batch=batch)
    encoded = ir.to_proto(graph).SerializeToString()
    started = time.perf_counter()
    executor = session(graph) if provider == "coreml" else cpu_session(encoded)
    creation = time.perf_counter() - started
    timings = {key: [] for key in ("feed", "execute_and_exchange", "adam", "step")}
    state = None
    for index in range(110):
        started = time.perf_counter()
        feeds = {"features": x, "target": y, **params}
        prepared = time.perf_counter()
        outputs = executor.run(None, feeds)
        executed = time.perf_counter()
        params, state = adam_step(
            params, dict(zip(PARAMETERS, outputs[2:], strict=True)), state, xp=np
        )
        updated = time.perf_counter()
        if index >= 10:
            for key, elapsed in zip(
                timings,
                (
                    prepared - started,
                    executed - prepared,
                    updated - executed,
                    updated - started,
                ),
                strict=True,
            ):
                timings[key].append(elapsed * 1000)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "provider": provider,
        "batch": batch,
        "session_creation_ms": creation * 1000,
        "timings_ms": {
            key: {"median": float(np.median(v)), "p95": float(np.percentile(v, 95))}
            for key, v in timings.items()
        },
        "peak_process_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
        "model_bytes": len(encoded),
        "nodes": len(graph.graph),
        "parameter_bytes": sum(value.nbytes for value in params.values()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "onnx": version("onnx"),
        "onnx_ir": version("onnx-ir"),
        "onnxscript": version("onnxscript"),
        "onnxruntime": ort.__version__,
        "threads": {"intra_op": 1, "inter_op": 1},
        "ort_graph_optimizer": "disabled",
        "coreml_options": {
            "ModelFormat": "MLProgram",
            "MLComputeUnits": "CPUAndGPU",
            "RequireStaticInputShapes": "1",
            "AllowLowPrecisionAccumulationOnGPU": "0",
            "cpu_ep_fallback": "disabled",
        },
        "scope": "One process per provider/batch; 10 warmup and 100 measured external Adam updates. Synchronous run fetches prediction, loss and both gradients; execute_and_exchange includes host/device exchange and Python outputs. RSS includes imports, graph generation, compilation, runtime and allocator. CoreML selects CPU/GPU placement within the provider.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["cpu", "coreml"], required=True)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = benchmark(args.provider, args.batch)
    with open(args.output, "w") as stream:
        json.dump(result, stream, indent=2)
