# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Mobius model adapters and source-parameter mappings for decoder training."""

import json
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import onnx_ir as ir
import torch
from mobius.models import Phi3CausalLMModel, Qwen3CausalLMModel
from mobius.models.qwen35 import Qwen35CausalLMModel
from onnx_ir.passes.common import ShapeInferencePass
from transformers import (
    Phi3Config,
    Phi3ForCausalLM,
    PreTrainedModel,
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from examples.huggingface.decoder_graph import (
    build_base,
    decompose,
    protect_decoder_boundaries,
    restore_decoder_outputs,
)
from examples.huggingface.pretrained import PretrainedSource, load_reference
from examples.huggingface.qwen35 import build_base as build_qwen35_base
from examples.huggingface.qwen35 import reference_model as qwen35_reference


@dataclass(frozen=True)
class DecoderFamily:
    """A source architecture and its Mobius builder, outside the onnax core."""

    config_type: type
    reference_type: type
    mobius_type: type
    targets: tuple[str, ...]
    fused_qkv: bool = False
    reference_factory: Callable[[], PreTrainedModel] | None = None
    graph_builder: Callable[[PreTrainedModel], ir.Model] | None = None
    target_suffixes: tuple[str, ...] = ()
    preserve_fused: bool = False
    pretrained: PretrainedSource | None = None


FAMILIES = {
    "qwen3": DecoderFamily(
        Qwen3Config,
        Qwen3ForCausalLM,
        Qwen3CausalLMModel,
        ("q_proj", "v_proj"),
        pretrained=PretrainedSource(
            "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"
        ),
    ),
    "phi3": DecoderFamily(Phi3Config, Phi3ForCausalLM, Phi3CausalLMModel, ("qkv_proj",), True),
    "qwen35": DecoderFamily(
        Qwen3_5TextConfig,
        Qwen3_5ForCausalLM,
        Qwen35CausalLMModel,
        ("in_proj_qkv", "q_proj", "v_proj"),
        reference_factory=qwen35_reference,
        graph_builder=build_qwen35_base,
        target_suffixes=(".linear_attn.in_proj_qkv", ".self_attn.q_proj", ".self_attn.v_proj"),
        preserve_fused=True,
        pretrained=PretrainedSource(
            "Qwen/Qwen3.5-0.8B",
            "2fc06364715b967f1860aea9cf38778875588b17",
            (("model.language_model.", "model."),),
        ),
    ),
}


def reference_model(family, *, real=False, cache_dir=None):
    """Create a deterministic offline model with its native source architecture."""
    entry = FAMILIES[family]
    if real:
        if entry.pretrained is None:
            raise ValueError(f"No pinned pretrained source for {family}")
        return load_reference(
            entry.config_type, entry.reference_type, entry.pretrained, cache_dir=cache_dir
        )
    if entry.reference_factory is not None:
        result = entry.reference_factory()
        result.config._name_or_path = "tiny-" + family
        return result
    torch.manual_seed(20261102)
    kwargs = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=256,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        attention_dropout=0.0,
        tie_word_embeddings=family == "qwen3",
    )
    if family == "phi3":
        kwargs.update(original_max_position_embeddings=64, resid_pdrop=0.0, embd_pdrop=0.0)
    config = entry.config_type(**kwargs)
    config._attn_implementation = "eager"
    config._name_or_path = "tiny-" + family
    return entry.reference_type(config).eval()


def _linear(node):
    if node.domain or node.op_type != "MatMul":
        return None
    rhs = node.inputs[1]
    producer = rhs.producer()
    if rhs.const_value is not None and (rhs.name or "").endswith(".weight_t"):
        return rhs.name.removesuffix(".weight_t"), tuple(rhs.shape)
    if producer is not None and producer.op_type == "Transpose":
        weight = producer.inputs[0]
        if weight.const_value is not None and (weight.name or "").endswith(".weight"):
            output, width = weight.shape
            return weight.name.removesuffix(".weight"), (width, output)
    return None


def inject(model, family, *, rank=4, alpha=1.0):
    """Preserve source Linear layouts, including one shared Phi-3 QKV adapter.

    Mobius splits Phi-3's frozen QKV weight into three projections. The adapter
    retains the source's single A/B pair: all three projections use XA, and each
    consumes the corresponding column slice of B. PEFT sees the original fused
    Linear, with the usual transposed A/B layout and alpha/rank scaling.
    """
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank < 1
        or not np.isfinite(alpha)
    ):
        raise ValueError("Expected a positive integer rank and finite alpha")
    entry = FAMILIES[family]
    result = model.clone()
    projections = {}
    for node in result.graph:
        description = _linear(node)
        if description is not None:
            name, shape = description
            if name in projections:
                raise ValueError(f"Ambiguous source projection: {name}")
            projections[name] = (node, shape)
    groups = []
    if entry.fused_qkv:
        for name in projections:
            if not name.endswith(".self_attn.q_proj"):
                continue
            prefix = name.removesuffix("q_proj")
            pieces = [prefix + name for name in ("q_proj", "k_proj", "v_proj")]
            if any(name not in projections for name in pieces):
                raise ValueError(f"Incomplete fused QKV source mapping: {prefix}")
            groups.append((prefix + "qkv_proj", pieces))
    else:
        groups = [
            (name, [name])
            for name in projections
            if name.endswith(
                entry.target_suffixes
                or tuple(".self_attn." + target for target in entry.targets)
            )
        ]
    if not groups:
        raise ValueError(f"No {family} source projections found")
    existing = set(ir.convenience.create_value_mapping(result.graph))
    params, mapping = {}, []
    for source, pieces in groups:
        width = projections[pieces[0]][1][0]
        if any(projections[name][1][0] != width for name in pieces):
            raise ValueError(f"Fused source input widths differ: {source}")
        total = sum(projections[name][1][1] for name in pieces)
        a = ir.val(source + ".lora.A", dtype=ir.DataType.FLOAT, shape=[width, rank])
        b = ir.val(source + ".lora.B", dtype=ir.DataType.FLOAT, shape=[rank, total])
        if {a.name, b.name} & existing:
            raise ValueError(f"Adapter input name collision: {source}")
        result.graph.inputs.extend([a, b])
        rng = np.random.default_rng(20261102 + len(params))
        params[a.name] = rng.normal(0, 0.02, a.shape).astype(np.float32)
        params[b.name] = np.zeros(b.shape, np.float32)
        row = {
            "module": source,
            "source_weight": source + ".weight",
            "A": a.name,
            "B": b.name,
            "rank": rank,
            "alpha": alpha,
            "pieces": [],
        }
        offset, products = 0, {}
        split_b = None
        for index, name in enumerate(pieces):
            node, (_, output_width) = projections[name]
            users = list(node.outputs[0].uses())
            tape = ir.tape.Tape()
            x = node.inputs[0]
            if x not in products:
                products[x] = tape.op("MatMul", [x, a])
            right = b
            if len(pieces) > 1:
                if split_b is None:
                    sizes = tape.op(
                        "Constant",
                        [],
                        {
                            "value": ir.tensor(
                                np.asarray(
                                    [projections[piece][1][1] for piece in pieces],
                                    dtype=np.int64,
                                )
                            )
                        },
                    )
                    split_b = tape.op_multi_out(
                        "Split", [b, sizes], {"axis": 1}, num_outputs=len(pieces)
                    )
                right = split_b[index]
            delta = tape.op("MatMul", [products[x], right])
            scale = tape.op("Constant", [], {"value": ir.tensor(np.float32(alpha / rank))})
            output = tape.op("Add", [node.outputs[0], tape.op("Mul", [delta, scale])])
            for i, operation in enumerate(tape.nodes):
                operation.name = f"{source}.adapter.{index}.{i}"
                for j, value in enumerate(operation.outputs):
                    value.name = f"{source}.adapter.{index}.{i}.{j}"
            result.graph.insert_after(node, tape.nodes)
            for use in users:
                use.node.replace_input_with(use.idx, output)
            for i, value in enumerate(result.graph.outputs):
                if value is node.outputs[0]:
                    result.graph.outputs[i] = output
            row["pieces"].append(
                {"module": name, "start": offset, "stop": offset + output_width}
            )
            offset += output_width
        mapping.append(row)
    result.graph.sort()
    ShapeInferencePass()(result)
    return result, params, mapping


def prepare_graph(
    family,
    *,
    preserve_fused=False,
    real=False,
    cache_dir=None,
    nf4=False,
    projection_precision="fp32",
):
    """Build, annotate and adapt one Mobius model without PyTorch tracing."""
    if projection_precision not in ("fp32", "bf16", "fp16"):
        raise ValueError("Expected fp32, bf16 or fp16 decoder projection precision")
    reference = reference_model(family, real=real, cache_dir=cache_dir)
    entry = FAMILIES[family]
    graph = (
        entry.graph_builder(reference)
        if entry.graph_builder is not None
        else build_base(reference, entry.mobius_type)
    )
    protect_decoder_boundaries(graph)
    graph = decompose(graph, preserve_fused=preserve_fused or entry.preserve_fused)
    graph, params, mapping = inject(graph, family)
    if real:
        assert entry.pretrained is not None
        for row in mapping:
            row["checkpoint_weight"] = entry.pretrained.checkpoint_name(row["source_weight"])
        graph.metadata_props["onnax.example.pretrained_source"] = json.dumps(
            reference.config.onnax_pretrained, sort_keys=True
        )
    if nf4:
        from examples.huggingface.nf4 import lower_decoder

        graph, quantization = lower_decoder(
            graph, reference, mapping=mapping, precision=projection_precision
        )
        reference.config.onnax_quantization = quantization
        graph.metadata_props["onnax.example.nf4"] = json.dumps(quantization, sort_keys=True)
    if projection_precision != "fp32":
        from examples.huggingface.mixed_precision import precision_projections

        graph = precision_projections(graph, precision=projection_precision)
        reference.config.onnax_precision = json.loads(
            graph.metadata_props["onnax.example.precision"]
        )
    restore_decoder_outputs(graph)
    return graph, params, mapping, reference
