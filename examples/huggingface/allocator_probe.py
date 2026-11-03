# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Measure host and MLX memory across one complete decoder forward/backward."""

import argparse
import json
import os
import platform
import time
from importlib.metadata import version

import numpy as np
import onnxruntime as ort
import torch

from examples.huggingface.block_runtime import MlxMemory, arrays_record, digest, load_batch
from examples.huggingface.decoder_sft import OnnxDecoder
from examples.lora.mlx_runtime import register_library
from examples.process_memory import physical_footprint


def measure(export, batch_file, library, output, *, cpu_arena=False):
    """Replay a saved token batch and preserve per-session memory observations."""
    if any(os.path.lexists(output + suffix) for suffix in ("", ".npz", ".progress.json")):
        raise FileExistsError("Select a new measurement output")
    torch.set_num_threads(1)
    register_library(library)
    memory = MlxMemory(library)
    events = []

    def factory(path):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.enable_cpu_mem_arena = cpu_arena
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        runtime = ort.InferenceSession(path, options, providers=["MLXExecutionProvider"])
        runtime.disable_fallback()

        class ObservedSession:
            def __init__(self, session):
                self.runtime = session

            def run(self, outputs, feeds):
                start = time.perf_counter()
                values = self.runtime.run(outputs, feeds)
                events.append(
                    dict(
                        model=os.path.basename(path),
                        seconds=time.perf_counter() - start,
                        physical_memory=physical_footprint(),
                        allocator=memory.snapshot(),
                    )
                )
                with open(output + ".progress.json", "w", encoding="utf-8") as stream:
                    json.dump(events, stream, indent=2)
                return values

            def close(self):
                self.runtime = None

        return ObservedSession(runtime)

    model = OnnxDecoder.load(export, session_factory=factory)
    try:
        batch = load_batch(
            batch_file,
            vocab_size=model.config.vocab_size,
            max_sequence=model.config.max_position_embeddings,
        )
        before = physical_footprint()
        started = time.perf_counter()
        loss = model(**batch).loss
        forward_seconds = time.perf_counter() - started
        started = time.perf_counter()
        loss.backward()
        backward_seconds = time.perf_counter() - started
        gradients = {}
        for name, parameter in model.onnx.parameters_by_name().items():
            if parameter.grad is None:
                raise AssertionError(f"Missing adapter gradient: {name}")
            gradients[name] = parameter.grad.detach().cpu().numpy().copy()
        if not np.isfinite(float(loss.detach())) or any(
            not np.isfinite(value).all() for value in gradients.values()
        ):
            raise FloatingPointError("Nonfinite loss or adapter gradient")
        np.savez(
            output + ".npz", loss=np.asarray(float(loss.detach()), np.float32), **gradients
        )
        record = dict(
            version=1,
            source_identity=model.onnx._identity,
            source_interface_sha256=digest(
                os.path.join(export, "execution", "interface.json")
            ),
            batch=arrays_record({key: value.numpy() for key, value in batch.items()}),
            gradient_arrays=arrays_record(gradients),
            loss=float(loss.detach()),
            cpu_mem_arena=cpu_arena,
            forward_seconds=forward_seconds,
            backward_seconds=backward_seconds,
            allocator=memory.snapshot(),
            physical_before=before,
            physical_after=physical_footprint(),
            cpu_fallback=False,
            ort_graph_optimizer=False,
            library_sha256=digest(library),
            tool_source_sha256=digest(__file__),
            platform=platform.platform(),
            cpu_threads=1,
            runtime_environment={
                key: os.environ.get(key)
                for key in (
                    "ONNXRUNTIME_EP_MLX_NO_COMPILE",
                    "ONNXRUNTIME_EP_MLX_CACHE_LIMIT_BYTES",
                )
            },
            versions={key: version(key) for key in ("onnxruntime", "mlx", "torch")},
            measurement_scope="One cold forward/backward, including host exchange; process peak includes construction; no optimizer update",
        )
        with open(output, "w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
        return record
    finally:
        model.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpu-arena", action="store_true")
    args = parser.parse_args()
    result = measure(
        args.export, args.batch, args.library, args.output, cpu_arena=args.cpu_arena
    )
    print(json.dumps({key: result[key] for key in ("loss", "physical_after", "allocator")}))


if __name__ == "__main__":
    main()
