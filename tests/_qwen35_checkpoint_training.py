# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Paired long-context checkpoint updates, export and fresh-process recovery."""

import argparse
import contextlib
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
import onnx_ir as ir
import torch

import onnax._autodiff._linear_attention as recurrence
from examples.huggingface import qwen35
from examples.huggingface.optimization import optimize_training_graph
from examples.huggingface.sft import token_loss
from examples.huggingface.validation import add_reference_parameters
from onnax.torch import OnnxModule


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--resume")
    p.add_argument("--length", type=int, default=4032)
    a = p.parse_args()
    root = Path(a.output)
    root.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    start = time.perf_counter()
    base = qwen35.reference_model()
    config = base.config
    config.max_position_embeddings = 4096
    torch.manual_seed(20261102)
    reference = type(base)(config).eval()
    del base
    with patch.object(qwen35, "reference_model", return_value=reference):
        graph, arrays, mapping, reference = qwen35.prepare_graph()
    rng = np.random.default_rng(310)
    for name in arrays:
        if name.endswith(".B"):
            arrays[name] = rng.normal(0, 0.01, arrays[name].shape).astype(np.float32)
    reference = add_reference_parameters(reference, arrays, mapping)
    ids = torch.from_numpy(rng.integers(3, 256, (1, a.length), dtype=np.int64))
    mask = torch.ones_like(ids)
    mask[:, -17:] = 0
    ids[mask == 0] = 0
    batch = {
        "input_ids": ids,
        "attention_mask": mask,
        "position_ids": (mask.cumsum(-1) - 1).clamp(min=0),
    }
    labels = ids.masked_fill(mask == 0, -100)
    record = {
        "status": "running",
        "recurrence_sha256": hashlib.sha256(
            Path(recurrence.__file__).read_bytes()
        ).hexdigest(),
        "physical_sequence": a.length,
        "valid_shifted_tokens": int(labels[:, 1:].ne(-100).sum()),
        "steps": [],
        "checks": [],
        "resume": a.resume,
    }

    def save():
        p = root / "report.json"
        q = p.with_suffix(".tmp")
        q.write_text(json.dumps(record, indent=2) + "\n")
        q.replace(p)

    def compare(name, x, y):
        x = x.detach().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        y = y.detach().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
        ok = np.isclose(x, y, rtol=1e-4, atol=1e-6)
        record["checks"].append(
            {
                "name": name,
                "max_abs": float(np.max(np.abs(x - y))) if x.size else 0.0,
                "violations": int(np.count_nonzero(~ok)),
                "bitwise": bool(np.array_equal(x, y)),
            }
        )
        save()
        np.testing.assert_allclose(x, y, rtol=1e-4, atol=1e-6)

    def frozen(model):
        source = ir.load(Path(model._directory.name) / "model.onnx")
        return {
            name: hashlib.sha256(v.const_value.numpy().tobytes()).hexdigest()
            for name, v in source.graph.initializers.items()
        }

    with contextlib.ExitStack() as stack:
        adaptive = stack.enter_context(
            OnnxModule(
                graph,
                arrays,
                outputs=["logits"],
                input_gradients=[],
                optimize=optimize_training_graph,
            )
        )
        with patch.object(
            recurrence, "_checkpoint_chunk", side_effect=lambda ctx, time: ctx.constant([32])
        ):
            fixed = stack.enter_context(
                OnnxModule(
                    graph,
                    arrays,
                    outputs=["logits"],
                    input_gradients=[],
                    optimize=optimize_training_graph,
                )
            )
        models = {"fixed": fixed, "adaptive": adaptive}
        optimizers = {
            k: torch.optim.AdamW(v.parameters(), lr=0.003, weight_decay=0, foreach=False)
            for k, v in models.items()
        }
        original = {k: frozen(m) for k, m in models.items()}
        params = {k: m.parameters_by_name() for k, m in models.items()}
        first = 0
        if a.resume:
            for key, model in models.items():
                checkpoint = torch.load(
                    Path(a.resume) / "fixed-checkpoint-1.pt", weights_only=True
                )
                model.load_state_dict(checkpoint["model"])
                optimizers[key].load_state_dict(checkpoint["optimizer"])
                assert checkpoint["step"] == 1
            first = 1
        save()
        for step in range(first, 3):
            losses = {}
            outputs = {}
            for key, model in models.items():
                optimizers[key].zero_grad(set_to_none=True)
                outputs[key] = model(**batch)
                losses[key] = token_loss(outputs[key], labels)
                losses[key].backward()
            compare(f"{step}/logits", outputs["adaptive"], outputs["fixed"])
            compare(f"{step}/loss", losses["adaptive"], losses["fixed"])
            for name in params["fixed"]:
                compare(
                    f"{step}/gradient/{name}",
                    params["adaptive"][name].grad,
                    params["fixed"][name].grad,
                )
            if step == 0:
                reference.zero_grad(set_to_none=True)
                expected = reference(**batch, use_cache=False).logits
                expected_loss = token_loss(expected, labels)
                expected_loss.backward()
                compare(
                    "reference/logits", outputs["adaptive"][mask.bool()], expected[mask.bool()]
                )
                compare("reference/loss", losses["adaptive"], expected_loss)
                for row in mapping:
                    layer = reference.get_submodule(row["module"])
                    for key in ("A", "B"):
                        compare(
                            "reference/gradient/" + row[key],
                            params["adaptive"][row[key]].grad,
                            getattr(layer, key).grad,
                        )
                del expected, expected_loss
            for optimizer in optimizers.values():
                optimizer.step()
            for name in params["fixed"]:
                compare(
                    f"{step}/parameter/{name}", params["adaptive"][name], params["fixed"][name]
                )
                for key in ("exp_avg", "exp_avg_sq"):
                    compare(
                        f"{step}/{key}/{name}",
                        optimizers["adaptive"].state[params["adaptive"][name]][key],
                        optimizers["fixed"].state[params["fixed"][name]][key],
                    )
            record["steps"].append(
                {
                    "step": step + 1,
                    "fixed_loss": float(losses["fixed"].detach()),
                    "adaptive_loss": float(losses["adaptive"].detach()),
                }
            )
            save()
            if step == 0:
                for key, model in models.items():
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizers[key].state_dict(),
                            "step": 1,
                        },
                        root / (key + "-checkpoint-1.pt"),
                    )
            del outputs, losses
        final = {}
        for key, model in models.items():
            assert frozen(model) == original[key]
            for name, param in params[key].items():
                final[key + "/parameter/" + name] = param.detach().numpy()
                for kind in ("exp_avg", "exp_avg_sq", "step"):
                    final[key + "/" + kind + "/" + name] = (
                        optimizers[key].state[param][kind].detach().numpy()
                    )
            directory = root / (key + "-export")
            model.export(directory)
            for path in directory.glob("*.onnx"):
                onnx.checker.check_model(str(path), full_check=True)
            loaded = stack.enter_context(OnnxModule.load(directory))
            got = loaded(**batch)
            expected = model(**batch)
            compare(key + "/loaded_logits", got, expected)
            assert torch.equal(got, expected)
            token_loss(got, labels).backward()
            model.zero_grad(set_to_none=True)
            token_loss(expected, labels).backward()
            for name, param in model.parameters_by_name().items():
                compare(
                    key + "/loaded_gradient/" + name,
                    loaded.parameters_by_name()[name].grad,
                    param.grad,
                )
                assert torch.equal(loaded.parameters_by_name()[name].grad, param.grad)
        np.savez(root / "final.npz", **final)
    record.update(
        status="complete",
        frozen_weights_unchanged=True,
        loaded_execution_bitwise=True,
        elapsed_seconds=time.perf_counter() - start,
    )
    save()
    print(json.dumps({k: v for k, v in record.items() if k != "checks"}), flush=True)


if __name__ == "__main__":
    main()
