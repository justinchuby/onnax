# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Multi-layer NF4 Qwen, independent updates and fresh-process recovery."""
# ruff: file-ignore[module-import-not-at-top-of-file] - Optional dependencies.

import json
import os
import subprocess
import sys

import numpy as np
import onnx_ir as ir
import pytest

pytest.importorskip("mobius")
pytest.importorskip("trl")
pytest.importorskip("peft")
torch = pytest.importorskip("torch")

from examples.huggingface.bridge import frozen_hashes
from examples.huggingface.nf4 import quantize_reference
from examples.huggingface.qwen import reference_model
from examples.huggingface.replay import verify
from examples.huggingface.sft import prepare, run
from examples.huggingface.validation import add_reference_adapters, audit
from examples.lora.nf4 import DOMAIN


def batch():
    ids = torch.tensor([[1, 4, 5, 6, 2], [1, 7, 2, 0, 0]])
    mask = ids.ne(0).long()
    labels = ids.clone()
    labels[mask == 0] = -100
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


@pytest.mark.parametrize("mode", ["full", "chunked"])
def test_nf4_qwen_parameters_and_independent_updates(mode):
    torch.set_num_threads(1)
    model, reference = prepare(nf4=True, mode=mode, chunk_size=3, save_activations=True)
    quantization = model.quantization
    assert len(quantization["weights"]) == 14
    assert quantization["packed_decoder_bytes"] * 7 < quantization["original_decoder_bytes"]
    for name, row in quantization["weights"].items():
        assert name not in model.forward_graph.graph.initializers
        assert model.forward_graph.graph.initializers[row["packed"]].dtype == ir.DataType.UINT8
    assert any(n.domain == DOMAIN for n in model.onnx.optimized_vjp.graph)
    assert model.onnx.residual_names
    assert all(
        node.op_type.startswith("InputGrad_")
        for node in model.onnx.optimized_vjp.graph
        if node.domain == DOMAIN
    )
    independent_base = reference_model()
    quantize_reference(independent_base)
    for name in quantization["weights"]:
        module = name.removesuffix(".weight_t")
        np.testing.assert_array_equal(
            reference.get_submodule(module).weight.detach(),
            independent_base.get_submodule(module).weight.detach(),
        )
    del independent_base
    reference = add_reference_adapters(reference, model)
    data = batch()
    audit(model, reference, data)
    for name, parameter in zip(model.onnx.names, model.onnx.adapters, strict=True):
        if name.endswith(".A"):
            assert parameter.grad.count_nonzero() == 0
        else:
            assert parameter.grad.count_nonzero() > 0
    onnx_optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0)
    torch_optimizer = torch.optim.AdamW(
        [p for p in reference.parameters() if p.requires_grad], lr=0.003, weight_decay=0
    )
    for _ in range(4):
        audit(model, reference, data, sync=False)
        onnx_optimizer.step()
        torch_optimizer.step()
        for row in model.mapping:
            expected = reference.get_submodule(row["module"])
            for key in ("A", "B"):
                parameter = model.onnx.adapters[model.onnx.names.index(row[key])]
                np.testing.assert_allclose(
                    parameter.detach(), getattr(expected, key).detach(), rtol=1e-4, atol=1e-6
                )

    assert all(p.grad.count_nonzero() > 0 for p in model.onnx.adapters)
    model.zero_grad(set_to_none=True)
    ignored = {**data, "labels": torch.full_like(data["labels"], -100)}
    result = model(**ignored)
    assert result.loss.item() == 0
    result.loss.backward()
    assert all(p.grad.count_nonzero() == 0 for p in model.onnx.adapters)
    assert frozen_hashes(model.onnx.model) == model.onnx.frozen


def test_nf4_sft_training_export_and_process_recovery(tmp_path):
    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    report = run(
        str(continuous), nf4=True, steps=100, save_activations=True, peft_roundtrip=True
    )
    assert report["validation_loss"][1] < report["validation_loss"][0]
    assert all(a["strict_gradient_check_passed"] for a in report["audits"])
    assert report["peft"]["format_roundtrip_bitwise"] and report["peft"]["update_check_passed"]
    exported = json.loads((continuous / "export/quantization.json").read_text())
    assert exported == report["quantization"]
    mapping = json.loads((continuous / "adapter/adapter_mapping.json").read_text())
    assert mapping["base_quantization"] == exported
    subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.huggingface.sft",
            "--nf4",
            "--steps",
            "100",
            "--save-activations",
            "--resume",
            str(continuous / "checkpoint-50"),
            "--output-dir",
            str(resumed),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
    )
    replay = verify(str(continuous), str(resumed), step=100)
    assert replay["adapter_bitwise_equal"] and replay["optimizer_bitwise_equal"]
    assert replay["export_load_outputs_bitwise_equal"]
