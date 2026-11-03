# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Prepare an optimized VJP header that reuses an exported Torch bundle's weights."""

import argparse
import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path  # ruff: ignore[banned-api] - CLI artifact paths.

import numpy as np
import onnx
import onnx_ir as ir
from onnx_ir.passes.common import RemoveUnusedNodesPass
from onnxscript import optimizer

import onnax
from examples.broadcast_adjoint import fuse_broadcast_adjoints
from examples.huggingface.attention_vjp import fuse_attention_vjp
from onnax._file_snapshot import copy_file
from onnax.torch._module import source_identity
from onnax.torch._serialization import file_digest


def _tensor_key(tensor):
    array = np.ascontiguousarray(tensor.numpy())
    return (
        tensor.dtype,
        tuple(array.shape),
        hashlib.sha256(array.reshape(-1).view(np.uint8).data).digest(),
    )


def _tensors(message):
    if isinstance(message, onnx.TensorProto):
        yield message
        return
    for field, value in message.ListFields():
        if field.type == field.TYPE_MESSAGE:
            for child in value if field.is_repeated else [value]:
                yield from _tensors(child)


def prepare_header(bundle, output, *, fused_attention=False, broadcast_adjoints=False):
    """Regenerate derivatives, match immutable payloads, and write a header plus manifest."""
    started = time.perf_counter()
    bundle, output = Path(bundle), Path(output)
    sidecar = Path(str(output) + ".json")
    if output.exists() or output.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        raise FileExistsError("Select a new deployment header and manifest path")
    interface = json.loads((bundle / "interface.json").read_text())
    if interface.get("version") != 1 or interface.get("kind") == "checkpointed_vjp":
        raise ValueError("Expected a version-1 flat OnnxModule bundle")
    if not interface["gradients"]:
        raise ValueError("The bundle has no parameter or input gradients")
    for name, digest in interface["files"].items():
        if Path(name).name != name or file_digest(bundle / name) != digest:
            raise ValueError(f"Bundle file integrity mismatch: {name!r}")
    if source_identity(bundle, interface["specification"]) != interface["identity"]:
        raise ValueError("Bundle source identity mismatch")
    source = ir.load(bundle / "model.onnx")
    old = ir.load(bundle / "derivative.onnx")
    result = onnax.vjp(
        source, inputs=list(interface["gradients"]), outputs=list(interface["seeds"])
    )
    generated = json.loads(result.graph.metadata_props["onnax.autodiff.interface"])
    if (
        dict(zip(interface["gradients"], generated["derivative_outputs"], strict=True))
        != interface["gradients"]
        or dict(zip(interface["seeds"], generated["seeds"], strict=True)) != interface["seeds"]
    ):
        raise ValueError("Regenerated derivative names differ from the saved interface")
    values = {value.name: value for value in result.graph.outputs}
    result.graph.outputs.clear()
    result.graph.outputs.extend(values[name] for name in interface["gradients"].values())
    if fused_attention:
        result = fuse_attention_vjp(result)
    if broadcast_adjoints:
        result = fuse_broadcast_adjoints(result)
    result = optimizer.optimize(result, inline=False)
    RemoveUnusedNodesPass()(result)
    calls = sum(
        node.domain == "pkg.onnax" and node.op_type.startswith("AttentionVJP_v1_")
        for node in result.graph
    )
    if fused_attention and not calls:
        raise ValueError("The model has no supported Attention pullbacks to fuse")
    if ([v.name for v in result.graph.inputs], [v.name for v in result.graph.outputs]) != (
        [v.name for v in old.graph.inputs],
        [v.name for v in old.graph.outputs],
    ):
        raise ValueError("Optimization changed the derivative input/output interface")
    blobs = {
        _tensor_key(v.const_value): v.const_value
        for v in old.graph.initializers.values()
        if isinstance(v.const_value, ir.ExternalTensor)
    }
    reused = 0
    for value in result.graph.initializers.values():
        tensor = value.const_value
        if tensor.dtype == ir.DataType.STRING:
            continue
        match = blobs.get(_tensor_key(tensor))
        if match is not None:
            value.const_value = ir.ExternalTensor(
                match.location,
                match.offset,
                match.length,
                match.dtype,
                shape=ir.Shape(match.shape),
                name=value.name,
                base_dir=bundle,
            )
            reused += 1
        elif tensor.nbytes >= 64 * 1024:
            raise ValueError(f"No existing payload matches large initializer {value.name!r}")
    result.graph.metadata_props.update(old.graph.metadata_props)
    proto = ir.to_proto(result)
    payloads = {}
    for tensor in _tensors(proto):
        external = {entry.key: entry.value for entry in tensor.external_data}
        if "location" in external:
            name = external["location"]
            if Path(name).name != name or name not in interface["files"]:
                raise ValueError(f"Unexpected external payload: {name!r}")
            payloads[name] = interface["files"][name]
        elif tensor.data_type != onnx.TensorProto.STRING and tensor.ByteSize() >= 64 * 1024:
            raise ValueError("Deployment header contains a new large embedded tensor")
    data = proto.SerializeToString()
    with tempfile.TemporaryDirectory(prefix="onnax-header-check-") as temporary:
        directory = Path(temporary)
        for name in payloads:
            copy_file(bundle / name, directory / name)
        check = directory / "check.onnx"
        check.write_bytes(data)
        onnx.checker.check_model(str(check), full_check=True)
    record = dict(
        version=1,
        source_identity=interface["identity"],
        gradients=interface["gradients"],
        seeds=interface["seeds"],
        payloads=payloads,
        header_sha256=hashlib.sha256(data).hexdigest(),
        header_bytes=len(data),
        fused_attention_calls=calls,
        broadcast_adjoint_calls=sum(
            node.domain == "pkg.onnax" and node.op_type == "SumToShape_v1"
            for node in result.graph
        ),
        reused_initializers=reused,
        generation_seconds=time.perf_counter() - started,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(data)
    try:
        with sidecar.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
    except BaseException:
        output.unlink()
        raise
    return record


@dataclass(frozen=True)
class DeploymentHeader:
    data: bytes
    manifest: dict


def read_header(path, interface):
    """Bind the header to the source/gradient interface before constructing sessions."""
    path = Path(path)
    record = json.loads(Path(str(path) + ".json").read_text())
    data = path.read_bytes()
    if record.get("version") != 1 or hashlib.sha256(data).hexdigest() != record.get(
        "header_sha256"
    ):
        raise ValueError("Deployment header integrity mismatch")
    if (
        record["source_identity"] != interface["identity"]
        or record["gradients"] != interface["gradients"]
        or record["seeds"] != interface["seeds"]
    ):
        raise ValueError("Deployment source/gradient interface mismatch")
    if any(
        Path(name).name != name or interface["files"].get(name) != digest
        for name, digest in record["payloads"].items()
    ):
        raise ValueError("Deployment frozen payloads differ from the source bundle")
    model = onnx.load_model_from_string(data)
    actual_payloads = {
        entry.value
        for tensor in _tensors(model)
        for entry in tensor.external_data
        if entry.key == "location"
    }
    if actual_payloads != set(record["payloads"]):
        raise ValueError("Deployment payload references differ from its manifest")
    metadata = {entry.key: entry.value for entry in model.graph.metadata_props}
    if metadata.get("onnax.torch.source_identity") != interface["identity"] or json.loads(
        metadata.get("onnax.torch.interface", "null")
    ) != {"seeds": interface["seeds"], "gradients": interface["gradients"]}:
        raise ValueError("Deployment graph metadata differs from its source interface")
    return DeploymentHeader(data, record)


def install_header(path, deployment):
    """Install into OnnxModule.load's private snapshot, alongside its verified payloads."""
    path = Path(path)
    if path.name != "derivative.onnx":
        return os.fspath(path)
    candidate = path.with_name("derivative.deployed.onnx")
    with candidate.open("xb") as stream:
        stream.write(deployment.data)
    onnx.checker.check_model(str(candidate), full_check=True)
    return os.fspath(candidate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fused-attention", action="store_true")
    parser.add_argument("--broadcast-adjoints", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_header(
                args.bundle,
                args.output,
                fused_attention=args.fused_attention,
                broadcast_adjoints=args.broadcast_adjoints,
            )
        )
    )


if __name__ == "__main__":
    main()
