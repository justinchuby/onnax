# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Check decoder checkpoints and portable exports through the replay CLI."""

import gc
import hashlib
import json
import os
from pathlib import Path  # ruff: ignore[banned-api] - Artifact validation.

import onnx
import onnx_ir as ir
import torch
from safetensors.torch import load_file

from examples.huggingface.decoder_sft import OnnxDecoder
from examples.huggingface.replay import _equal_tree, _output_digest


def tensor_manifest(directory):
    directory = Path(directory)
    result, hashed = {}, {}

    def tensor_record(tensor):
        if isinstance(tensor, ir.ExternalTensor):
            path = os.path.realpath(os.path.join(tensor.base_dir, tensor.location))
            offset, length = tensor.offset or 0, tensor.nbytes
            key = path, offset, length
            if key not in hashed:
                digest = hashlib.sha256()
                remaining = length
                with open(path, "rb") as stream:
                    stream.seek(offset)
                    while remaining:
                        chunk = stream.read(min(1024**2, remaining))
                        if not chunk:
                            raise ValueError("Truncated external tensor")
                        digest.update(chunk)
                        remaining -= len(chunk)
                hashed[key] = digest.hexdigest()
            digest = hashed[key]
        else:
            digest = hashlib.sha256(tensor.tobytes()).hexdigest()
        return {"dtype": str(tensor.dtype), "shape": list(tensor.shape), "sha256": digest}

    def visit(graph, scope):
        for value in graph.initializers.values():
            result[scope + "/initializer/" + value.name] = tensor_record(value.const_value)
        for node in graph:
            for name, attribute in node.attributes.items():
                if attribute.is_ref():
                    continue
                key = (
                    scope
                    + "/node/"
                    + ",".join(v.name or "" for v in node.outputs)
                    + "/"
                    + name
                )
                if attribute.type == ir.AttributeType.TENSOR:
                    result[key] = tensor_record(attribute.as_tensor())
                elif attribute.type == ir.AttributeType.TENSORS:
                    for index, tensor in enumerate(attribute.as_tensors()):
                        result[key + "/" + str(index)] = tensor_record(tensor)
                elif attribute.type == ir.AttributeType.GRAPH:
                    visit(attribute.as_graph(), key)
                elif attribute.type == ir.AttributeType.GRAPHS:
                    for index, child in enumerate(attribute.as_graphs()):
                        visit(child, key + "/" + str(index))

    for path in sorted(directory.glob("*.onnx")):
        model = ir.load(path)
        visit(model.graph, path.name)
        for key, function in model.functions.items():
            visit(function.graph, path.name + "/function/" + str(key))
    return result


def adapter_metadata(root):
    config = json.loads((Path(root) / "adapter/adapter_config.json").read_text())
    for name in ("target_modules", "exclude_modules", "modules_to_save"):
        if isinstance(config.get(name), list):
            config[name] = sorted(config[name])
    mapping = json.loads((Path(root) / "adapter/adapter_mapping.json").read_text())
    return {"config": config, "mapping": mapping}


def verify(first, resumed, *, step, session_factory, feeds, provider):
    """Compare checkpoint state and each exported execution route independently."""
    roots = [Path(first), Path(resumed)]
    states = [load_file(root / f"checkpoint-{step}/model.safetensors") for root in roots]
    _equal_tree(*states)
    _equal_tree(*[adapter_metadata(root) for root in roots])
    for name in ("optimizer.pt", "scheduler.pt"):
        _equal_tree(
            *[
                torch.load(
                    root / f"checkpoint-{step}" / name, map_location="cpu", weights_only=True
                )
                for root in roots
            ]
        )
    _equal_tree(*[load_file(root / "adapter/adapter_model.safetensors") for root in roots])
    tasks = [json.loads((root / "export/task.json").read_text()) for root in roots]
    for key in ("version", "family", "mapping"):
        _equal_tree(tasks[0][key], tasks[1][key])
    manifests = [tensor_manifest(root / "export/execution") for root in roots]
    if not manifests[0]:
        raise ValueError("Expected exported decoder tensors")
    _equal_tree(*manifests)
    checked, results = [], []
    for root, state, task in zip(roots, states, tasks, strict=True):
        execution = root / "export/execution"
        models = sorted(execution.glob("*.onnx"))
        for path in models:
            onnx.checker.check_model(str(path), full_check=True)
        checked.append(len(models))
        model = OnnxDecoder.load(root / "export", session_factory=session_factory)
        try:
            _equal_tree(dict(model.state_dict()), state)
            inputs = {name: torch.from_numpy(value) for name, value in feeds.items()}
            labels = inputs["input_ids"].clone()
            labels[inputs["attention_mask"] == 0] = -100
            model.zero_grad(set_to_none=True)
            output = model(**inputs, labels=labels)
            output.loss.backward()
            values = {"loss": _output_digest(output.loss.detach().numpy())}
            for name, parameter in model.onnx.parameters_by_name().items():
                if parameter.grad is None:
                    raise AssertionError(f"Missing exported gradient: {name}")
                values["gradient/" + name] = _output_digest(parameter.grad.numpy())
            with torch.no_grad():
                values["logits"] = _output_digest(model(**inputs).logits.numpy())
            filename = task.get("inference", {}).get("model", "forward.onnx")
            executor = session_factory(str(execution / filename))
            values["standalone_logits"] = _output_digest(executor.run(None, feeds)[0])
            del executor
            results.append(values)
        finally:
            model.close()
        del output, model
        gc.collect()
    _equal_tree(*results)
    return {
        "step": step,
        "family": tasks[0]["family"],
        "provider": provider,
        "checkpoint_tensors": len(states[0]),
        "checkpoint_bitwise_equal": True,
        "optimizer_bitwise_equal": True,
        "scheduler_equal": True,
        "adapter_bitwise_equal": True,
        "adapter_metadata_equal": True,
        "named_tensor_records": len(manifests[0]),
        "named_tensor_manifest_sha256": hashlib.sha256(
            json.dumps(manifests[0], sort_keys=True).encode()
        ).hexdigest(),
        "onnx_models_checked": checked,
        "export_load_loss_gradients_bitwise_equal": True,
        "exported_gradient_tensors": len(results[0]) - 3,
        "export_load_outputs_bitwise_equal": True,
        "standalone_outputs_bitwise_equal": True,
        "output_digests": results[0],
        "input_shape": list(feeds["input_ids"].shape),
    }
