# Copyright (c) 2026 onnax contributors
# SPDX-License-Identifier: MIT
"""Compare FP32 pretrained UNet predictions across attention implementations."""

import argparse
import json
import os
import tempfile
from importlib.metadata import version

import numpy as np
import torch
from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor
from torch.nn.attention import SDPBackend, sdpa_kernel

from examples.diffusion.unet import MODEL_ID, REVISION, build, pretrained
from examples.diffusion.unet_train import synthetic_batch
from onnax.torch._module import _ort_session
from onnax.torch._serialization import save_model


def run():
    torch.set_num_threads(1)
    c, w, sc = pretrained()
    g = build(c, w)
    r = UNet2DConditionModel.from_config(c).eval().requires_grad_(False)
    r.load_state_dict(w)
    b, _ = synthetic_batch(c, DDPMScheduler.from_config(sc), 20261102, size=16)
    with tempfile.TemporaryDirectory(prefix="onnax-unet-numerics-") as d:
        save_model(g, d, "forward")
        session = _ort_session(os.path.join(d, "forward.onnx"))
        actual = session.run(None, {k: v.numpy() for k, v in b.items()})[0]
        with torch.no_grad():
            default = r(**b).sample.numpy()
            with sdpa_kernel(SDPBackend.MATH):
                math = r(**b).sample.numpy()
            r.set_attn_processor(AttnProcessor())
            classic = r(**b).sample.numpy()

        def diff(a, b):
            e = np.abs(a - b)
            return dict(
                max_abs=float(e.max()),
                l2=float(np.linalg.norm(e)),
                reference_l2=float(np.linalg.norm(b)),
                pointwise_failures=int((e > 1e-6 + 1e-4 * np.abs(b)).sum()),
            )

        report = {
            "onnx_default": diff(actual, default),
            "onnx_math": diff(actual, math),
            "onnx_classic": diff(actual, classic),
            "torch_default_math": diff(default, math),
            "torch_default_classic": diff(default, classic),
        }
        report.update(
            model=MODEL_ID,
            revision=REVISION,
            scope="CPU FP32 identical weights and inputs",
            latent_shape=[1, 4, 16, 16],
            torch_threads=1,
            ort_graph_optimizer=False,
            versions={
                name: version(name)
                for name in ("torch", "onnxruntime", "diffusers", "mobius-onnx")
            },
        )
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run()
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
