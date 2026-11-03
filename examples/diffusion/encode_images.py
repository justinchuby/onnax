# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Prepare real image-caption latents with external Diffusers VAE and CLIP models."""

import argparse
import hashlib
import json
import os
from importlib.metadata import version

import numpy as np
import torch
from diffusers import AutoencoderKL
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps
from transformers import CLIPTextModel, CLIPTokenizer

from examples.diffusion.unet import MODEL_ID, REVISION
from examples.huggingface.block_runtime import arrays_record


def image_tensor(path, size):
    """Apply EXIF orientation, RGB conversion, centered square crop and [-1, 1] scaling."""
    if size < 8 or size % 8:
        raise ValueError("Image size must be a positive multiple of eight")
    with Image.open(path) as source:
        image = ImageOps.fit(
            ImageOps.exif_transpose(source).convert("RGB"),
            (size, size),
            method=Image.Resampling.LANCZOS,
        )
        pixels = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous().unsqueeze(0) / 127.5 - 1


def encode_one(pixels, caption, vae, encoder, tokenizer, *, device, seed):
    """Cache the VAE posterior mean and final CLIP hidden states in FP32 on CPU."""
    with torch.no_grad():
        posterior = vae.encode(pixels.to(device)).latent_dist
        clean = posterior.mode() * vae.config.scaling_factor
        tokens = tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=encoder.config.max_position_embeddings,
            return_tensors="pt",
        )
        mask = (
            tokens.attention_mask.to(device)
            if getattr(encoder.config, "use_attention_mask", False)
            else None
        )
        conditioning = encoder(
            tokens.input_ids.to(device), attention_mask=mask
        ).last_hidden_state
    clean, conditioning = clean.cpu().float(), conditioning.cpu().float()
    generator = torch.Generator().manual_seed(seed)
    return dict(
        clean=clean,
        conditioning=conditioning,
        noise=torch.randn(clean.shape, generator=generator),
        timesteps=torch.tensor([317]),
    )


def prepare(directory, *, manifest=None, device="mps", size=512):
    """Encode each manifest image once and retain source and tensor hashes."""
    if manifest is None:
        manifest = os.path.join(os.path.dirname(__file__), "data", "dog-photos.json")
    with open(manifest, encoding="utf-8") as stream:
        source = json.load(stream)
    if source.get("version") != 1 or not source.get("examples"):
        raise ValueError("Expected a version 1 image manifest with examples")
    os.makedirs(directory, exist_ok=False)
    torch.set_num_threads(1)
    vae = (
        AutoencoderKL.from_pretrained(
            MODEL_ID,
            subfolder="vae",
            revision=REVISION,
            use_safetensors=True,
            torch_dtype=torch.float32,
        )
        .eval()
        .to(device)
    )
    encoder = (
        CLIPTextModel.from_pretrained(
            MODEL_ID, subfolder="text_encoder", revision=REVISION, use_safetensors=True
        )
        .eval()
        .to(device=device, dtype=torch.float32)
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        MODEL_ID, subfolder="tokenizer", revision=REVISION
    )
    rows = []
    for index, row in enumerate(source["examples"]):
        path = (
            hf_hub_download(
                source["repo_id"],
                row["file"],
                repo_type="dataset",
                revision=source["revision"],
            )
            if "repo_id" in source
            else os.path.join(os.path.dirname(manifest), row["file"])
        )
        pixels = image_tensor(path, size)
        tensors = encode_one(
            pixels,
            row["caption"],
            vae,
            encoder,
            tokenizer,
            device=device,
            seed=20261103 + index,
        )
        output = f"example-{index}.pt"
        torch.save(tensors, os.path.join(directory, output))
        with open(path, "rb") as stream:
            image_digest = hashlib.file_digest(stream, "sha256").hexdigest()
        rows.append(
            dict(
                **row,
                output=output,
                image_sha256=image_digest,
                tensors=arrays_record({k: v.numpy() for k, v in tensors.items()}),
            )
        )
    report = dict(
        version=1,
        model=MODEL_ID,
        revision=REVISION,
        source=source,
        device=device,
        dtype="float32",
        image_size=size,
        preprocessing="EXIF orientation, RGB, centered square crop with Lanczos, pixels / 127.5 - 1",
        posterior="mode",
        scaling_factor=vae.config.scaling_factor,
        examples=rows,
        versions={
            name: version(name) for name in ("torch", "diffusers", "transformers", "Pillow")
        },
    )
    with open(os.path.join(directory, "manifest.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()
    report = prepare(
        args.output_dir, manifest=args.manifest, device=args.device, size=args.image_size
    )
    print(
        json.dumps(
            dict(
                examples=len(report["examples"]),
                image_size=args.image_size,
                device=args.device,
            )
        )
    )


if __name__ == "__main__":
    main()
