# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Construct cache-free Qwen3 ONNX graphs with Mobius and inject explicit LoRA."""

import numpy as np
import onnx_ir as ir
import torch
from mobius import CausalLMConfig, ModelPackage, ep_registry
from mobius._build_context import build_context
from mobius.models import Qwen3CausalLMModel
from mobius.rewrite_rules import decompose_attention_pass
from mobius.tasks import ModelTask
from mobius.tasks._base import _make_graph, _make_model
from onnx_ir.passes.common import ShapeInferencePass
from onnxscript import optimizer
from transformers import Qwen3Config, Qwen3ForCausalLM

MODEL_ID = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def tiny_config():
    return Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=256,
        max_position_embeddings=64,
        attention_dropout=0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=True,
    )


class TrainingTask(ModelTask):
    """Full-sequence logits and hidden states from the Mobius backbone."""

    def build(self, module, config):
        graph, builder = _make_graph()
        inputs = {
            name: builder.input(name, dtype=ir.DataType.INT64, shape=["batch", "sequence"])
            for name in ("input_ids", "attention_mask", "position_ids")
        }
        hidden, _ = module.model(builder.op, **inputs, past_key_values=None)
        logits = module.lm_head(builder.op, hidden)
        builder.add_output(logits, name="logits")
        builder.add_output(hidden, name="hidden_states")
        return ModelPackage({"model": _make_model(graph)}, config)


def build_base(reference):
    config = CausalLMConfig.from_transformers(reference.config)
    module = Qwen3CausalLMModel(config)
    with build_context(ep_registry.require("default"), ir.DataType.FLOAT):
        package = TrainingTask().build(module, config)
    package.apply_weights(module.preprocess_weights(reference.state_dict()))
    return package["model"]


def reference_model(real=False):
    torch.manual_seed(20261029)
    if real:
        result = Qwen3ForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, dtype=torch.float32, attn_implementation="eager"
        ).eval()
        result.config._commit_hash = REVISION
        return result
    config = tiny_config()
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).eval()


def decompose(model):
    """Lower the Qwen3 FP32 normalization, RoPE and attention operations."""
    result = model.clone()
    ShapeInferencePass()(result)
    # The pinned Mobius decomposition accepts additive masks; ONNX Attention
    # also accepts bool masks, where False must exclude the key entirely.
    for index, node in enumerate(list(result.graph)):
        if node.domain or node.op_type != "Attention" or len(node.inputs) < 4:
            continue
        mask = node.inputs[3]
        if mask is None or mask.dtype != ir.DataType.BOOL:
            continue
        tape = ir.tape.Tape()
        zero = tape.op("Constant", [], {"value": ir.tensor(np.float32(0))})
        negative = tape.op("Constant", [], {"value": ir.tensor(np.float32(-3e38))})
        bias = tape.op("Where", [mask, zero, negative])
        bias.type, bias.shape = ir.TensorType(ir.DataType.FLOAT), mask.shape
        for i, operation in enumerate(tape.nodes):
            operation.name = f"training.mask.{index}.{i}"
            operation.outputs[0].name = f"training.mask.{index}.{i}"
        result.graph.insert_before(node, tape.nodes)
        node.replace_input_with(3, bias)
    decompose_attention_pass()(result)
    for node_index, node in enumerate(list(result.graph)):
        if node.domain or node.op_type not in ("RMSNormalization", "RotaryEmbedding", "Swish"):
            continue
        tape = ir.tape.Tape()

        def op(kind, *inputs, _tape=tape, **attrs):
            return _tape.op(kind, inputs, attrs)

        def const(value):
            return op("Constant", value=ir.tensor(np.asarray(value)))

        x = node.inputs[0]
        if node.op_type == "RMSNormalization":
            if node.attributes.get_int("axis", -1) != -1:
                raise ValueError("Qwen training decomposition expects last-axis RMSNorm")
            square = op("Mul", x, x)
            mean = op("ReduceMean", square, const(np.array([-1], np.int64)), keepdims=1)
            root = op(
                "Sqrt",
                op("Add", mean, const(np.float32(node.attributes.get_float("epsilon", 1e-5)))),
            )
            output = op("Mul", op("Div", x, root), node.inputs[1])
        elif node.op_type == "Swish":
            alpha = node.attributes.get_float("alpha", 1.0)
            scaled = op("Mul", x, const(np.float32(alpha)))
            zero, one = const(np.float32(0)), const(np.float32(1))
            positive = op("GreaterOrEqual", scaled, zero)
            magnitude = op("Where", positive, scaled, op("Neg", scaled))
            tail = op("Exp", op("Neg", magnitude))
            denominator = op("Add", one, tail)
            sigmoid = op(
                "Where", positive, op("Div", one, denominator), op("Div", tail, denominator)
            )
            output = op("Mul", x, sigmoid)
        else:
            if node.attributes.get_int("interleaved", 0) or node.attributes.get_int(
                "rotary_embedding_dim", 0
            ):
                raise ValueError("Qwen3 uses full, noninterleaved rotary embeddings")
            heads = node.attributes.get_int("num_heads")
            if x.shape is None or not isinstance(x.shape[-1], int):
                raise ValueError("Expected static Qwen projection width")
            head_dim = x.shape[-1] // heads
            shaped = op("Reshape", x, const(np.array([0, 0, heads, head_dim], np.int64)))

            def half(start, end, shaped=shaped):
                return op(
                    "Slice",
                    shaped,
                    const(np.array([start], np.int64)),
                    const(np.array([end], np.int64)),
                    const(np.array([-1], np.int64)),
                )

            rotated = op(
                "Concat",
                op("Neg", half(head_dim // 2, head_dim)),
                half(0, head_dim // 2),
                axis=-1,
            )
            cos = op(
                "Unsqueeze",
                op("Concat", node.inputs[1], node.inputs[1], axis=-1),
                const(np.array([2], np.int64)),
            )
            sin = op(
                "Unsqueeze",
                op("Concat", node.inputs[2], node.inputs[2], axis=-1),
                const(np.array([2], np.int64)),
            )
            output = op(
                "Reshape",
                op("Add", op("Mul", shaped, cos), op("Mul", rotated, sin)),
                op("Shape", x),
            )
        output.dtype, output.shape = node.outputs[0].dtype, node.outputs[0].shape
        for i, new_node in enumerate(tape.nodes):
            new_node.name = f"training.decompose.{node_index}.{i}"
            for j, value in enumerate(new_node.outputs):
                value.name = f"training.decompose.{node_index}.{i}.{j}"
        output.name = node.outputs[0].name
        result.graph.insert_before(node, tape.nodes)
        node.outputs[0].replace_all_uses_with(output, replace_graph_outputs=True)
        result.graph.remove(node)
    result = optimizer.optimize(result)
    result.graph.sort()
    ShapeInferencePass()(result)
    return result


def inject(model, *, rank=4, alpha=1.0):
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank < 1
        or not np.isfinite(alpha)
    ):
        raise ValueError("Expected a positive integer rank and finite alpha")
    result = model.clone()
    existing = {
        v.name
        for v in (
            *result.graph.inputs,
            *result.graph.initializers.values(),
            *(v for n in result.graph for v in n.outputs),
        )
    }
    params, mapping = {}, []
    for node in list(result.graph):
        if node.domain or node.op_type != "MatMul":
            continue
        rhs = node.inputs[1]
        # Mobius folds Linear's transpose into a source-named initializer.
        producer = rhs.producer()
        if rhs.const_value is not None and (rhs.name or "").endswith(".weight_t"):
            module = rhs.name.removesuffix(".weight_t")
            in_width, out_width = rhs.shape
        elif producer is not None and producer.op_type == "Transpose":
            weight = producer.inputs[0]
            module = (weight.name or "").removesuffix(".weight")
            out_width, in_width = weight.shape
        else:
            continue
        if not module.endswith((".self_attn.q_proj", ".self_attn.v_proj")):
            continue
        if {module + ".lora.A", module + ".lora.B"}.intersection(existing):
            raise ValueError(f"Adapter input name collision: {module}")
        a = ir.val(module + ".lora.A", dtype=ir.DataType.FLOAT, shape=[in_width, rank])
        b = ir.val(module + ".lora.B", dtype=ir.DataType.FLOAT, shape=[rank, out_width])
        result.graph.inputs.extend([a, b])
        rng = np.random.default_rng(20261029 + len(params))
        params[a.name] = rng.normal(0, 0.02, a.shape).astype(np.float32)
        params[b.name] = np.zeros(b.shape, np.float32)
        users = list(node.outputs[0].uses())
        tape = ir.tape.Tape()
        t = tape.op("MatMul", [node.inputs[0], a])
        d = tape.op("MatMul", [t, b])
        scale = tape.op("Constant", [], {"value": ir.tensor(np.float32(alpha / rank))})
        delta = tape.op("Mul", [d, scale])
        output = tape.op("Add", [node.outputs[0], delta])
        for i, new_node in enumerate(tape.nodes):
            new_node.name = f"{module}.adapter.{i}"
            for j, value in enumerate(new_node.outputs):
                value.name = f"{module}.adapter.{i}.{j}"
        result.graph.insert_after(node, tape.nodes)
        for use in users:
            use.node.replace_input_with(use.idx, output)
        mapping.append(
            {"module": module, "A": a.name, "B": b.name, "rank": rank, "alpha": alpha}
        )
    if not mapping:
        raise ValueError("No Qwen3 q_proj/v_proj weights found")
    ShapeInferencePass()(result)
    return result, params, mapping
