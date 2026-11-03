# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Build source-named Diffusers UNets with Mobius and ordinary ONNX."""

import json

import onnx_ir as ir
from mobius.integrations.diffusers._configs import UNet2DConfig
from mobius.models.unet import UNet2DConditionModel
from mobius.tasks import DenoisingTask
from onnx_ir.passes.common import ShapeInferencePass

MODEL_ID = "segmind/tiny-sd"
REVISION = "3d25de3e8d5aacf94ce91176f204e6d11d120315"


def tiny_config():
    """Use the pretrained model's topology with smaller channel dimensions."""
    return dict(
        in_channels=4,
        out_channels=4,
        block_out_channels=[16, 32],
        layers_per_block=1,
        norm_num_groups=4,
        cross_attention_dim=24,
        attention_head_dim=4,
        sample_size=8,
        down_block_types=["CrossAttnDownBlock2D"] * 2,
        up_block_types=["CrossAttnUpBlock2D"] * 2,
        mid_block_type=None,
        norm_eps=1e-5,
        flip_sin_to_cos=True,
        freq_shift=0,
    )


def build(config, weights):
    """Map Diffusers weights and normalization configuration onto a Mobius graph."""
    for name, expected in (
        ("center_input_sample", False),
        ("flip_sin_to_cos", True),
        ("freq_shift", 0),
        ("resnet_time_scale_shift", "default"),
        ("resnet_out_scale_factor", 1.0),
        ("transformer_layers_per_block", 1),
        ("only_cross_attention", False),
        ("dual_cross_attention", False),
        ("addition_embed_type", None),
        ("class_embed_type", None),
        ("act_fn", "silu"),
        ("dropout", 0.0),
        ("downsample_padding", 1),
        ("mid_block_scale_factor", 1.0),
        ("resnet_skip_time_act", False),
        ("time_embedding_type", "positional"),
        ("time_embedding_dim", None),
        ("time_embedding_act_fn", None),
        ("timestep_post_act", None),
        ("time_cond_proj_dim", None),
        ("encoder_hid_dim", None),
        ("cross_attention_norm", None),
        ("attention_type", "default"),
        ("reverse_transformer_layers_per_block", None),
    ):
        if config.get(name, expected) != expected:
            raise ValueError(f"Unsupported UNet configuration: {name}={config[name]!r}")
    module = UNet2DConditionModel(UNet2DConfig.from_diffusers(config))
    model = DenoisingTask().build(module, module.config)["model"]
    renamed = module.preprocess_weights(weights)
    for name, value in model.graph.initializers.items():
        if value.const_value is None:
            array = renamed[name].detach().cpu().float().numpy()
            if tuple(value.shape) != array.shape:
                raise ValueError(f"UNet weight shape mismatch: {name}")
            value.const_value = ir.tensor(array, name=name)
    # Mobius GroupNorm defaults to 1e-6, matching Transformer2D. Diffusers
    # ResNets and the final output norm use the UNet's configurable norm_eps.
    for node in model.graph:
        if node.domain or node.op_type != "GroupNormalization":
            continue
        scale = node.inputs[1].name or ""
        if ".resnets." in scale or scale == "conv_norm_out.weight":
            node.attributes["epsilon"] = ir.AttrFloat32(
                "epsilon", config.get("norm_eps", 1e-5)
            )
    ShapeInferencePass()(model)
    return model


def pretrained():
    """Download only the pinned UNet weights/config and external scheduler config."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    def artifact(name):
        return hf_hub_download(MODEL_ID, name, revision=REVISION)

    with open(artifact("unet/config.json"), encoding="utf-8") as stream:
        config = json.load(stream)
    with open(artifact("scheduler/scheduler_config.json"), encoding="utf-8") as stream:
        scheduler = json.load(stream)
    weights = load_file(artifact("unet/diffusion_pytorch_model.safetensors"))
    return config, weights, scheduler
