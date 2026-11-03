# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Deploy portable attention pullbacks into an existing decoder export."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path  # ruff: ignore[banned-api] - CLI artifact paths.

import onnx
import onnx_ir as ir

import onnax
from examples.huggingface.attention_vjp import fuse_attention_vjp
from examples.huggingface.optimization import optimize_training_graph
from onnax import FuseBroadcastAdjointsPass
from onnax._file_snapshot import copy_file
from onnax._shared_weights import SharedWeights


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def deploy(source, destination, *, broadcast_adjoints=False, recompute_attention=False):
    """Create a private decoder bundle with unchanged parameter interfaces."""
    source, destination = Path(source), Path(destination)
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    task = json.loads((source / "task.json").read_text())
    if task.get("version") not in (1, 2):
        raise ValueError("Expected a versioned decoder task export")
    original = source / "execution"
    record = json.loads((original / "interface.json").read_text())
    if record.get("version") != 1 or record.get("kind") != "checkpointed_vjp":
        raise ValueError("Expected a checkpointed decoder execution bundle")
    for name, expected in record["files"].items():
        if Path(name).name != name or digest(original / name) != expected:
            raise ValueError(f"Bundle file integrity mismatch: {name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="onnax-attention-deploy-", dir=destination.parent
    ) as temporary:
        staged = Path(temporary) / "export"
        shutil.copytree(source, staged, copy_function=copy_file)
        execution = staged / "execution"
        blocks = []
        with SharedWeights(execution) as weights:
            for index, row in enumerate(record["blocks"]):
                if not row["backward"]:
                    continue
                forward = ir.load(original / f"block_{index}_forward.onnx")
                if not any(
                    node.domain == "" and node.op_type == "Attention" for node in forward.graph
                ):
                    continue
                joint = onnax.vjp(
                    forward, inputs=row["active_inputs"], outputs=row["active_outputs"]
                )
                joint = fuse_attention_vjp(joint, recompute=recompute_attention)
                if broadcast_adjoints:
                    joint = FuseBroadcastAdjointsPass()(joint).model
                calls = [
                    node.op_type
                    for node in joint.graph
                    if node.domain == "pkg.onnax"
                    and node.op_type.startswith(("AttentionVJP_v1_", "AttentionVJP_v2_"))
                ]
                if not calls:
                    continue
                interface = json.loads(joint.graph.metadata_props["onnax.autodiff.interface"])
                if (
                    interface["seeds"] != row["seeds"]
                    or interface["derivative_outputs"] != row["gradients"]
                ):
                    raise ValueError("Deployment changed the checkpoint derivative interface")
                values = {value.name: value for value in joint.graph.outputs}
                joint.graph.outputs.clear()
                joint.graph.outputs.extend(values[name] for name in row["gradients"])
                joint = optimize_training_graph(joint)
                joint.graph.metadata_props["onnax.torch.source_identity"] = record["identity"]
                weights.save(joint, f"block_{index}_backward")
                blocks.append(
                    {
                        "name": row["name"],
                        "model": f"block_{index}_backward.onnx",
                        "calls": calls,
                        "broadcast_adjoint_calls": sum(
                            n.domain == "pkg.onnax" and n.op_type == "SumToShape_v1"
                            for n in joint.graph
                        ),
                    }
                )
        if not blocks:
            raise ValueError("No eligible attention pullbacks in the bundle")
        for block in blocks:
            path = execution / block["model"]
            onnx.checker.check_model(path, full_check=True)
            record["files"][block["model"]] = digest(path)
        record["files"][weights.data_name] = digest(execution / weights.data_name)
        report = {
            "version": 1,
            "source_interface_sha256": digest(original / "interface.json"),
            "identity": record["identity"],
            "blocks": blocks,
            "models_checked": len(blocks),
        }
        (execution / "interface.json").write_text(json.dumps(record, indent=2) + "\n")
        (staged / "attention_deployment.json").write_text(json.dumps(report, indent=2) + "\n")
        os.replace(staged, destination)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--broadcast-adjoints", action="store_true")
    parser.add_argument("--recompute-attention", action="store_true")
    args = parser.parse_args()
    result = deploy(
        args.export,
        args.output,
        broadcast_adjoints=args.broadcast_adjoints,
        recompute_attention=args.recompute_attention,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
