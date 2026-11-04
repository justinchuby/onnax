# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Compare decoder pullbacks using identical activations and loss cotangents."""

import argparse
import dataclasses
import hashlib
import json
import os
import platform
from contextlib import contextmanager, nullcontext, suppress
from importlib.metadata import version

import torch

from examples.huggingface.decoder_families import FAMILIES, reference_model
from examples.huggingface.decoder_sft import OnnxDecoder, batch, text_batch
from examples.huggingface.pretrained import PretrainedSource, load_reference, load_tokenizer
from examples.huggingface.reference_head import pairwise_head
from examples.huggingface.sft import token_loss
from examples.huggingface.validation import add_reference_parameters

RTOL, ATOL = 1e-4, 1e-6


class _RecordingSession:
    def __init__(self, original, record, *, backward):
        self.original, self.record, self.backward = original, record, backward

    def run(self, outputs, feeds):
        values = self.original.run(outputs, feeds)
        if self.backward:
            self.record["feeds"] = {name: value.copy() for name, value in feeds.items()}
            key = "gradients"
        else:
            key = "forward"
        self.record[key] = {
            name: value.copy() for name, value in zip(outputs, values, strict=True)
        }
        return values


class _RecordingHead:
    def __init__(self, original, record, handles):
        self.original, self.record, self.handles = original, record, handles

    def __call__(self, hidden, labels, num_items=None):
        self.record.update(
            hidden=hidden.detach().clone(), labels=labels.detach().clone(), num_items=num_items
        )

        def incoming(gradient):
            self.record["gradient"] = gradient.detach().clone()

        self.handles.append(hidden.register_hook(incoming))
        result = self.original(hidden, labels, num_items)
        self.record["loss"] = result.detach().clone()
        return result

    def forward(self, hidden):
        return self.original.forward(hidden)


def _head_audit(reference, captured):
    reports = {}
    for method in ("stock", "pairwise"):
        hidden = captured["hidden"].clone().requires_grad_(True)
        context = pairwise_head(reference) if method == "pairwise" else nullcontext()
        with context:
            logits = reference.lm_head(hidden)
            loss = token_loss(logits, captured["labels"], captured["num_items"])
            gradient = torch.autograd.grad(loss, hidden)[0]
        reports[method] = {
            "loss": _compare(loss, captured["loss"]),
            "hidden_gradient": _compare(gradient, captured["gradient"]),
        }
    return reports


def _compare(actual, expected):
    actual = actual.detach().float()
    expected = torch.as_tensor(expected).float()
    if actual.shape != expected.shape:
        raise ValueError("Reference comparison shapes differ")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise FloatingPointError("Nonfinite shared-activation comparison")
    delta = actual - expected
    return {
        "elements": actual.numel(),
        "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
        "relative_l2": float(
            torch.linalg.vector_norm(delta) / (torch.linalg.vector_norm(expected) + 1e-30)
        ),
        "pointwise_violations": int(
            (~torch.isclose(actual, expected, rtol=RTOL, atol=ATOL)).sum()
        ),
    }


class _FinishedDecoderError(Exception):
    pass


@contextmanager
def _reference_adapters(reference, model):
    original = {row["module"]: reference.get_submodule(row["module"]) for row in model.mapping}
    flags = [(parameter, parameter.requires_grad) for parameter in reference.parameters()]
    modes = [(module, module.training) for module in reference.modules()]
    try:
        yield add_reference_parameters(reference, model.onnx.arrays(), model.mapping)
    finally:
        for name, module in original.items():
            parent, child = name.rsplit(".", 1)
            setattr(reference.get_submodule(parent), child, module)
        for parameter, requires_grad in flags:
            parameter.requires_grad_(requires_grad)
        for module, training in modes:
            module.training = training


def _layer_hooks(reference, execution, record, index, last, parameters, valid, reports):
    block = execution.block
    candidates = [
        name
        for name in block.inputs
        if name in record["feeds"]
        and record["feeds"][name].ndim == 3
        and record["feeds"][name].shape[-1] == reference.config.hidden_size
    ]
    if len(candidates) != 1 or len(execution.seeds) != 1:
        raise ValueError("Expected one hidden activation and one cotangent per decoder layer")
    hidden_name = candidates[0]
    shared = []

    def before(module, args, kwargs):
        hidden = torch.from_numpy(record["feeds"][hidden_name]).requires_grad_(True)
        shared[:] = [hidden]
        if args:
            return (hidden, *args[1:]), kwargs
        return args, {**kwargs, "hidden_states": hidden}

    def after(module, args, output):
        value = output[0] if isinstance(output, tuple) else output
        arguments = [
            shared[0] if name == hidden_name else parameters[name]
            for name in block.active_inputs
        ]
        gradients = torch.autograd.grad(
            value,
            arguments,
            grad_outputs=torch.from_numpy(record["feeds"][execution.seeds[0]]),
        )
        expected_output = record["forward"][block.active_outputs[0]]
        reports.append(
            {
                "layer": index,
                "reference_class": type(module).__name__,
                "forward_all_tokens": _compare(value, expected_output),
                "forward_valid_tokens": _compare(value[valid], expected_output[valid.numpy()]),
                "gradients": {
                    name: _compare(gradient, record["gradients"][output_name])
                    for name, gradient, output_name in zip(
                        block.active_inputs, gradients, execution.gradients, strict=True
                    )
                },
            }
        )
        if index == last:
            raise _FinishedDecoderError()
        return value.detach()

    return before, after


def shared_activation_audit(model, reference, data):
    """Audit FP32 single-layer checkpoints against a fresh native reference.

    The reference receives the exported adapter parameters. Each native layer
    uses its ONNX activation and incoming loss cotangent, so earlier forward
    differences do not enter the local derivative comparison. Existing ONNX
    parameter gradients, execution sessions and native reference modules are preserved. Recording adds
    activation copies; run this diagnostic separately from performance timing.
    """
    executions = model.onnx.executions
    layers = reference.model.layers
    if reference.config.model_type != model.config.model_type:
        raise ValueError("Reference and ONNX decoder families differ")
    if len(executions) != len(layers) + 2 or any(
        execution.block.name != f"decoder_{index}_{index + 1}"
        for index, execution in enumerate(executions[1:-1])
    ):
        raise ValueError("Shared-activation audit requires one checkpoint per decoder layer")
    if "labels" not in data or "attention_mask" not in data:
        raise ValueError("Provide labels and an explicit attention mask")
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("The shared-activation control uses FP32 parameters")
    records = {execution.block.name: {} for execution in executions[1:]}
    model.onnx.executions = [
        executions[0],
        *(
            dataclasses.replace(
                execution,
                forward=_RecordingSession(
                    execution.forward, records[execution.block.name], backward=False
                ),
                backward=_RecordingSession(
                    execution.backward, records[execution.block.name], backward=True
                ),
            )
            for execution in executions[1:]
        ),
    ]
    handles = []
    original_head = getattr(model, "head", None)
    captured_head = {}
    previous_profile = original_head.last_profile if original_head is not None else None
    if original_head is not None:
        model.head = _RecordingHead(original_head, captured_head, handles)
    try:
        with torch.enable_grad(), _reference_adapters(reference, model):
            actual = model(**data)
            torch.autograd.grad(actual.loss, tuple(model.parameters()))
            parameters = {
                row[key]: getattr(reference.get_submodule(row["module"]), key)
                for row in model.mapping
                for key in ("A", "B")
            }
            reports = []
            valid = data["attention_mask"].bool()
            for index, layer in enumerate(layers):
                execution = executions[index + 1]
                before, after = _layer_hooks(
                    reference,
                    execution,
                    records[execution.block.name],
                    index,
                    len(layers) - 1,
                    parameters,
                    valid,
                    reports,
                )
                handles.append(layer.register_forward_pre_hook(before, with_kwargs=True))
                handles.append(layer.register_forward_hook(after))
            with suppress(_FinishedDecoderError):
                reference(
                    **{name: value for name, value in data.items() if name != "labels"},
                    use_cache=False,
                )
            if len(reports) != len(layers):
                raise RuntimeError("The native reference did not execute all decoder layers")
            for handle in handles:
                handle.remove()
            handles.clear()
            execution = executions[-1]
            record = records[execution.block.name]
            if len(execution.block.active_inputs) != 1 or len(execution.seeds) != 1:
                raise ValueError("Expected a frozen normalization/head suffix")
            hidden = torch.from_numpy(
                record["feeds"][execution.block.active_inputs[0]]
            ).requires_grad_(True)
            seed = torch.from_numpy(record["feeds"][execution.seeds[0]])
            head = {}
            output_head = None
            if original_head is not None:
                normalized = reference.model.norm(hidden)
                gradient = torch.autograd.grad(normalized, hidden, grad_outputs=seed)[0]
                head["normalization"] = _compare(
                    gradient, record["gradients"][execution.gradients[0]]
                )
                output_head = _head_audit(reference, captured_head)
            else:
                for method in ("stock", "pairwise"):
                    context = (
                        pairwise_head(reference) if method == "pairwise" else nullcontext()
                    )
                    with context:
                        logits = reference.lm_head(reference.model.norm(hidden))
                        gradient = torch.autograd.grad(logits, hidden, grad_outputs=seed)[0]
                    head[method] = _compare(
                        gradient, record["gradients"][execution.gradients[0]]
                    )
        return {
            "version": 1,
            "family": model.family,
            "source_identity": model.onnx._identity,
            "rtol": RTOL,
            "atol": ATOL,
            "input_shape": list(data["input_ids"].shape),
            "valid_tokens": int(valid.sum()),
            "shifted_targets": int(data["labels"][:, 1:].ne(-100).sum()),
            "decoder_gradients_passed": all(
                value["pointwise_violations"] == 0
                for row in reports
                for value in row["gradients"].values()
            ),
            "layers": reports,
            "frozen_suffix": head,
            "output_head": output_head,
        }
    finally:
        for handle in handles:
            handle.remove()
        model.onnx.executions = executions
        if original_head is not None:
            model.head = original_head
            original_head.last_profile = previous_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--provider", choices=("cpu", "mlx"), default="cpu")
    parser.add_argument("--library")
    args = parser.parse_args()
    torch.set_num_threads(1)
    factory = None
    if args.provider == "mlx":
        from examples.lora.mlx_runtime import register_library, session

        if args.library:
            register_library(args.library)
        factory = session
    elif args.library:
        parser.error("--library requires --provider mlx")
    model = OnnxDecoder.load(args.export, session_factory=factory)
    try:
        record = getattr(model.config, "onnax_pretrained", None)
        if record:
            source = PretrainedSource(
                record["model_id"],
                record["revision"],
                tuple(record["checkpoint_to_reference_prefix"].items()),
            )
            family = FAMILIES[model.family]
            reference = load_reference(
                family.config_type, family.reference_type, source, cache_dir=args.cache_dir
            )
            data = text_batch(load_tokenizer(source, cache_dir=args.cache_dir))
        else:
            reference, data = reference_model(model.family), batch()
        if model.quantization is not None:
            from examples.huggingface.nf4 import quantize_reference

            quantize_reference(
                reference, mapping=model.mapping, block_size=model.quantization["block_size"]
            )
        report = shared_activation_audit(model, reference, data)
        with open(__file__, "rb") as stream:
            source_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        library_sha256 = None
        if args.library:
            with open(args.library, "rb") as stream:
                library_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        report.update(
            provider=args.provider,
            pretrained_source=record,
            audit_source_sha256=source_sha256,
            library_sha256=library_sha256,
            platform=platform.platform(),
            versions={
                name: version(name)
                for name in ("onnax", "onnxruntime", "torch", "transformers", "mobius-onnx")
            },
        )
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps({key: value for key, value in report.items() if key != "layers"}))
    finally:
        model.close()


if __name__ == "__main__":
    main()
