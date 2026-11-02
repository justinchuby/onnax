# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Shared cache-free Mobius decoder construction, boundaries and decomposition."""

import ast
import json

import numpy as np
import onnx_ir as ir
from mobius import CausalLMConfig, ModelPackage, ep_registry
from mobius._build_context import build_context
from mobius.rewrite_rules import decompose_attention_pass
from mobius.tasks import ModelTask
from mobius.tasks._base import _make_graph, _make_model
from onnx_ir.passes.common import ShapeInferencePass
from onnxscript import optimizer

DECODER_BOUNDARIES = "onnax.example.decoder_boundaries"


def protect_decoder_boundaries(model):
    """Expose Mobius decoder boundaries while rewriting a working model.

    The original outputs and boundary names are recorded in model metadata.
    Extra outputs keep the values available through decomposition and folding.
    """
    layers = {}
    for node in model.graph:
        scopes = ast.literal_eval(node.metadata_props.get("pkg.onnxscript.name_scopes", "[]"))
        for scope in scopes:
            if scope.startswith("layers.") and scope[7:].isdigit():
                layers.setdefault(int(scope[7:]), []).append(node)
                break
    if not layers or sorted(layers) != list(range(len(layers))):
        raise ValueError("Expected consecutive Mobius decoder layer scopes")
    normalizers = [
        node
        for node in layers[0]
        if node.op_type in ("RMSNormalization", "LayerNormalization")
        and "input_layernorm"
        in ast.literal_eval(node.metadata_props.get("pkg.onnxscript.name_scopes", "[]"))
    ]
    if len(normalizers) != 1:
        raise ValueError("Expected one decoder input normalization boundary")
    first = normalizers[0]
    names = [first.inputs[0].name]
    for index, nodes in sorted(layers.items()):
        members = set(nodes)
        exits = [
            value
            for node in nodes
            for value in node.outputs
            if any(use.node not in members for use in value.uses())
            or value in model.graph.outputs
        ]
        if len(exits) != 1:
            raise ValueError(f"Expected one hidden-state exit for decoder layer {index}")
        names.append(exits[0].name)
    original = [value.name for value in model.graph.outputs]
    values = ir.convenience.create_value_mapping(model.graph)
    for name in names:
        if name not in original:
            model.graph.outputs.append(values[name])
    model.metadata_props[DECODER_BOUNDARIES] = json.dumps(
        {"version": 1, "hidden_states": names, "original_outputs": original}
    )


def restore_decoder_outputs(model):
    """Restore the inference interface after protecting checkpoint boundaries."""
    record = json.loads(model.metadata_props[DECODER_BOUNDARIES])
    values = ir.convenience.create_value_mapping(model.graph)
    if any(name not in values for name in record["hidden_states"]):
        raise ValueError("A decoder checkpoint boundary was lost during rewriting")
    model.graph.outputs.clear()
    model.graph.outputs.extend(values[name] for name in record["original_outputs"])


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


def decompose(model, *, preserve_fused=False):
    """Lower standard decoder normalization, RoPE and attention operations."""
    result = model.clone()
    ShapeInferencePass()(result)
    if preserve_fused:
        result.graph.sort()
        return result
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
                raise ValueError("Decoder training decomposition expects last-axis RMSNorm")
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


def build_base(reference, model_type):
    """Build ONNX from Mobius modules and preserve the preprocessed weight names."""
    config = CausalLMConfig.from_transformers(reference.config)
    module = model_type(config)
    with build_context(ep_registry.require("default"), ir.DataType.FLOAT):
        package = TrainingTask().build(module, config)
    package.apply_weights(module.preprocess_weights(reference.state_dict()))
    return package["model"]
