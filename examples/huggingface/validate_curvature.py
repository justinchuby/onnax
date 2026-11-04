# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Qualify exported causal-token curvature products against Transformers."""

import argparse
import hashlib
import json
import os
from importlib.metadata import version

import numpy as np
import onnx
import torch

from examples.huggingface.bridge import frozen_hashes
from examples.huggingface.curvature import INTERFACE, batch_directions, causal_ggn
from examples.huggingface.optimization import optimize_training_graph
from examples.huggingface.qwen import (
    MODEL_ID,
    REVISION,
    build_base,
    decompose,
    inject,
    reference_model,
)
from examples.huggingface.sft import texts, token_loss, tokenizer
from examples.huggingface.validation import add_reference_parameters
from examples.lora.train import session
from examples.process_memory import physical_footprint
from onnax._shared_weights import SharedWeights


def reference_product(reference, mapping, parameters, batch, labels, directions):
    """Differentiate the independent model and token loss using PyTorch FP32."""
    names = [row[key] for row in mapping for key in ("A", "B")]
    paths = [row["module"] + "." + key for row in mapping for key in ("A", "B")]
    inputs = tuple(torch.tensor(parameters[name], requires_grad=True) for name in names)
    feed = {name: torch.from_numpy(value) for name, value in batch.items()}

    def predict(*values):
        return torch.func.functional_call(
            reference, dict(zip(paths, values, strict=True)), (), {**feed, "use_cache": False}
        ).logits

    prediction, tangent = torch.autograd.functional.jvp(
        predict, inputs, tuple(torch.from_numpy(value) for value in directions)
    )
    logits = prediction.detach().requires_grad_()
    gradient = torch.autograd.grad(
        token_loss(logits, torch.from_numpy(labels)), logits, create_graph=True
    )[0]
    head_product = torch.autograd.grad(gradient, logits, grad_outputs=tangent)[0]
    products = torch.autograd.grad(predict(*inputs), inputs, grad_outputs=head_product)
    return [
        np.asarray(float((tangent * head_product).sum()), np.float32),
        *[value.detach().numpy() for value in products],
    ]


def _check(actual, expected):
    maximum = 0.0
    for index, (value, reference) in enumerate(zip(actual, expected, strict=True)):
        if (
            value.shape != reference.shape
            or value.dtype != np.float32
            or not np.isfinite(value).all()
        ):
            raise AssertionError("Invalid curvature output")
        np.testing.assert_allclose(value, reference, rtol=1e-4, atol=1e-6)
        if index:
            maximum = max(maximum, float(np.max(np.abs(value - reference))))
    return maximum


def run(directory, *, real=False, provider="cpu", library=None, target_nodes=None):
    """Export and execute a shared-primal GGN at a seeded nonzero adapter point."""
    if provider not in ("cpu", "mlx") or (provider == "cpu" and library):
        raise ValueError("Choose cpu, or mlx with an optional development library")
    if target_nodes is not None and (provider != "mlx" or target_nodes < 1):
        raise ValueError("target_nodes requires MLX and a positive node count")
    os.makedirs(directory, exist_ok=False)
    torch.set_num_threads(1)
    print(json.dumps({"phase": "load_source", "real": real}), flush=True)
    reference = reference_model(real)
    source, parameters, mapping = inject(build_base(reference))
    source = decompose(source)
    before = frozen_hashes(source)
    print(
        json.dumps({"phase": "build_product", "source_nodes": len(source.graph)}), flush=True
    )
    model = causal_ggn(source, list(parameters))
    interface = json.loads(model.graph.metadata_props[INTERFACE])
    models = {"single": model}
    if not real:
        models["batched"] = optimize_training_graph(batch_directions(model))
    print(json.dumps({"phase": "export", "nodes": len(model.graph)}), flush=True)
    with SharedWeights(directory) as writer:
        paths = {name: writer.save(graph, name) for name, graph in models.items()}
    for path in paths.values():
        onnx.checker.check_model(path, full_check=True)
    factory = session
    if provider == "mlx":
        from examples.lora.mlx_runtime import register_library
        from examples.lora.mlx_runtime import session as mlx_session

        if library:
            register_library(library)

        def factory(graph):
            return mlx_session(graph, target_nodes=target_nodes)

    print(json.dumps({"phase": "create_session", "provider": provider}), flush=True)
    runtime = factory(paths["single"])
    rng = np.random.default_rng(42)
    for row in mapping:
        parameters[row["B"]] = rng.normal(0, 0.02, parameters[row["B"]].shape).astype(
            np.float32
        )
    reference = add_reference_parameters(reference, parameters, mapping)
    names = interface["parameters"]
    directions = [rng.normal(size=parameters[name].shape).astype(np.float32) for name in names]
    if real:
        encoded = tokenizer(True)(
            texts()[:1],
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=64,
        )
        ids, mask = encoded["input_ids"], encoded["attention_mask"]
    else:
        ids = np.array([[1, 4, 5, 6, 2], [1, 7, 2, 0, 0]], np.int64)
        mask = (ids != 0).astype(np.int64)
    batch = {
        "input_ids": ids,
        "attention_mask": mask,
        "position_ids": np.maximum(mask.cumsum(-1) - 1, 0),
    }
    labels = np.where(mask, ids, -100).astype(np.int64)
    feeds = {
        **parameters,
        **batch,
        interface["labels"]: labels,
        **dict(zip(interface["directions"], directions, strict=True)),
    }
    print(json.dumps({"phase": "execute", "parameter_tensors": len(names)}), flush=True)
    actual = runtime.run(None, feeds)
    print(json.dumps({"phase": "independent_reference"}), flush=True)
    expected = reference_product(reference, mapping, parameters, batch, labels, directions)
    maximum = _check(actual, expected)
    if not float(actual[0]) > 0:
        raise AssertionError("Expected a positive directional curvature")
    np.savez(os.path.join(directory, "feeds.npz"), **feeds)
    np.savez(
        os.path.join(directory, "outputs.npz"),
        **dict(zip((value.name for value in model.graph.outputs), actual, strict=True)),
    )
    batch_checks = 0
    if not real:
        mapped = factory(paths["batched"])
        vectors = {
            seed: rng.normal(size=(3, *parameters[name].shape)).astype(np.float32)
            for seed, name in zip(interface["directions"], names, strict=True)
        }
        batched = mapped.run(None, {**feeds, **vectors})
        sequential = [
            runtime.run(
                None, {**feeds, **{name: value[index] for name, value in vectors.items()}}
            )
            for index in range(3)
        ]
        for index, value in enumerate(batched):
            np.testing.assert_allclose(
                value, np.stack([row[index] for row in sequential]), rtol=1e-4, atol=1e-6
            )
        np.savez(os.path.join(directory, "batched_feeds.npz"), **{**feeds, **vectors})
        batch_checks = 3
    if frozen_hashes(source) != before:
        raise AssertionError("Source frozen weights changed")
    with open(__file__, "rb") as stream:
        driver_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    report = dict(
        model=MODEL_ID if real else "tiny-qwen3",
        revision=REVISION if real else None,
        provider=provider,
        versions={
            name: version(name)
            for name in (
                "numpy",
                "onnx",
                "onnx-ir",
                "onnxscript",
                "onnxruntime",
                "torch",
                "transformers",
                "mobius-onnx",
            )
        },
        driver_sha256=driver_sha256,
        target_nodes=target_nodes,
        dtype="float32",
        interface=interface,
        parameter_tensors=len(names),
        parameter_elements=sum(value.size for value in parameters.values()),
        source_frozen_sha256=hashlib.sha256(
            json.dumps(before, sort_keys=True).encode()
        ).hexdigest(),
        frozen_weights_unchanged=True,
        reference="Transformers JVP and cross-entropy Hessian-vector/VJP",
        maximum_product_absolute_error=maximum,
        directional_curvature=float(actual[0]),
        reference_curvature=float(expected[0]),
        batched_directions_checked=batch_checks,
        valid_shifted_targets=int((labels[:, 1:] != -100).sum()),
        full_checker_graphs=len(paths),
        nodes={name: len(graph.graph) for name, graph in models.items()},
        rtol=1e-4,
        atol=1e-6,
        physical_footprint=physical_footprint(),
        ort_graph_optimizer=False,
        cpu_fallback=False if provider == "mlx" else None,
    )
    with open(os.path.join(directory, "report.json"), "w") as stream:
        json.dump(report, stream, indent=2)
    return report


def main():
    """Run the offline model or the pinned Qwen3-0.6B qualification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--provider", choices=("cpu", "mlx"), default="cpu")
    parser.add_argument("--library")
    parser.add_argument("--target-nodes", type=int)
    args = parser.parse_args()
    report = run(
        args.output_dir,
        real=args.real,
        provider=args.provider,
        library=args.library,
        target_nodes=args.target_nodes,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
