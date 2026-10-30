# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Exchange standard single-adapter LoRA weights with PEFT."""

import json
import os

import torch
from peft import LoraConfig
from safetensors.torch import load_file, save_file


def export_adapter(model, directory):
    os.makedirs(directory, exist_ok=True)
    rows = model.mapping
    ranks = {r["rank"] for r in rows}
    alphas = {r["alpha"] for r in rows}
    if len(ranks) != 1 or len(alphas) != 1:
        raise ValueError("This adapter uses a uniform rank and alpha")
    config = LoraConfig(
        r=next(iter(ranks)),
        lora_alpha=next(iter(alphas)),
        lora_dropout=0,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        base_model_name_or_path=model.config._name_or_path or "tiny-qwen3",
        revision=getattr(model.config, "_commit_hash", None),
    )
    config.save_pretrained(directory)
    with open(os.path.join(directory, "adapter_mapping.json"), "w") as stream:
        json.dump(
            {
                "version": 1,
                "base_model": config.base_model_name_or_path,
                "revision": config.revision,
                "projections": rows,
            },
            stream,
            indent=2,
        )
    arrays = model.onnx.arrays()
    tensors = {
        f"base_model.model.{r['module']}.lora_{key}.weight": torch.from_numpy(
            arrays[r[key]].T.copy()
        )
        for r in rows
        for key in ("A", "B")
    }
    save_file(
        tensors,
        os.path.join(directory, "adapter_model.safetensors"),
        metadata={"format": "pt"},
    )


def import_adapter(model, directory):
    config = LoraConfig.from_pretrained(directory)
    if (
        config.bias != "none"
        or config.lora_dropout != 0
        or config.fan_in_fan_out
        or config.use_dora
        or config.use_rslora
        or config.modules_to_save
        or config.rank_pattern
        or config.alpha_pattern
    ):
        raise ValueError(
            "Expected standard Linear LoRA without bias, dropout or rank/alpha overrides"
        )
    expected_base = model.config._name_or_path or "tiny-qwen3"
    if config.base_model_name_or_path != expected_base or set(config.target_modules or ()) != {
        "q_proj",
        "v_proj",
    }:
        raise ValueError("PEFT base model or target configuration mismatch")
    revision = getattr(model.config, "_commit_hash", None)
    if revision is not None and config.revision is not None and revision != config.revision:
        raise ValueError("PEFT base revision mismatch")
    tensors = load_file(os.path.join(directory, "adapter_model.safetensors"))
    expected = {
        f"base_model.model.{r['module']}.lora_{key}.weight": r[key]
        for r in model.mapping
        for key in ("A", "B")
    }
    if set(tensors) != set(expected):
        raise ValueError("PEFT adapter target mapping mismatch")
    values = []
    for row in model.mapping:
        if row["rank"] != config.r or row["alpha"] != config.lora_alpha:
            raise ValueError("PEFT rank or scaling mismatch")
        for key in ("A", "B"):
            name = f"base_model.model.{row['module']}.lora_{key}.weight"
            tensor = tensors[name].T.contiguous()
            parameter = model.onnx.adapters[model.onnx.names.index(row[key])]
            if tensor.dtype != torch.float32 or tensor.shape != parameter.shape:
                raise ValueError(f"PEFT shape/dtype mismatch: {name}")
            values.append((parameter, tensor))
    with torch.no_grad():
        for parameter, value in values:
            parameter.copy_(value)


def roundtrip(model, reference, batch, directory, *, diagnose_rounding=False):
    """Exchange through PEFT, compare gradients, then exchange one AdamW update."""
    import numpy as np
    from peft import PeftModel

    export_adapter(model, directory)
    peft = PeftModel.from_pretrained(reference, directory, is_trainable=True).eval()
    initial = model.onnx.arrays()
    peft.save_pretrained(directory, save_embedding_layers=False)
    import_adapter(model, directory)
    for name, value in model.onnx.arrays().items():
        np.testing.assert_array_equal(value, initial[name])
    model.zero_grad()
    peft.zero_grad()
    actual = model(**batch)
    if actual.logits is None:
        with torch.no_grad():
            actual.logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
    expected = peft(**batch, use_cache=False)
    valid = batch["attention_mask"].bool()
    logits = actual.logits.detach()[valid].numpy()
    reference_logits = expected.logits.detach()[valid].numpy()
    logit_failures = int(
        np.count_nonzero(
            np.abs(logits - reference_logits) > 1e-6 + 1e-4 * np.abs(reference_logits)
        )
    )
    if not diagnose_rounding:
        np.testing.assert_allclose(logits, reference_logits, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(
        actual.loss.detach(), expected.loss.detach(), rtol=1e-4, atol=1e-6
    )
    actual.loss.backward()
    expected.loss.backward()
    parameters = dict(peft.named_parameters())
    gradient_failures = {}
    for row in model.mapping:
        for key in ("A", "B"):
            p = model.onnx.adapters[model.onnx.names.index(row[key])]
            q = parameters[f"base_model.model.{row['module']}.lora_{key}.default.weight"]
            actual_gradient, expected_gradient = p.grad.numpy(), q.grad.numpy().T
            delta = np.abs(actual_gradient - expected_gradient)
            failed = delta > 1e-6 + 1e-4 * np.abs(expected_gradient)
            if np.any(failed):
                gradient_failures[row[key]] = {
                    "violations": int(np.count_nonzero(failed)),
                    "max_abs": float(np.max(delta)),
                }
            if not diagnose_rounding:
                np.testing.assert_allclose(
                    actual_gradient, expected_gradient, rtol=1e-4, atol=1e-6
                )
    torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0).step()
    torch.optim.AdamW(
        [p for p in peft.parameters() if p.requires_grad], lr=0.003, weight_decay=0
    ).step()
    arrays = model.onnx.arrays()
    peft.save_pretrained(directory, save_embedding_layers=False)
    import_adapter(model, directory)
    update_failures = {}
    for name, value in model.onnx.arrays().items():
        delta = np.abs(value - arrays[name])
        failures = delta > 1e-6 + 1e-4 * np.abs(arrays[name])
        if np.any(failures):
            update_failures[name] = {
                "violations": int(np.count_nonzero(failures)),
                "max_abs": float(np.max(delta)),
            }
        if not diagnose_rounding:
            np.testing.assert_allclose(value, arrays[name], rtol=1e-4, atol=1e-6)
    return {
        "adapter_tensors": len(arrays),
        "format_roundtrip_bitwise": True,
        "gradient_check_passed": not bool(gradient_failures),
        "gradient_failures": gradient_failures,
        "update_check_passed": not bool(update_failures),
        "update_failures": update_failures,
        "logit_violations": logit_failures,
        "logits_max_abs": float(np.max(np.abs(logits - reference_logits))),
    }
