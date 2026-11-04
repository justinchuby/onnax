# Train a conditional diffusion denoiser

A denoiser predicts the noise added to a clean image latent. For a clean latent
$x_0$, sampled noise $\epsilon$ and schedule value $\bar\alpha_t$, training uses

\[
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad
L=\operatorname{mean}\bigl((\epsilon_\theta(x_t,t,c)-\epsilon)^2\bigr).
\]

The timestep $t$ controls the noise level. The conditioning sequence $c$ represents
text encoder features. The example generates clean latents, noise, timesteps and
conditioning outside ONNX with fixed seeds. A cosine schedule produces noisy
latents; a separate held-out seed supplies validation data.

## Build the denoising graph with Mobius

The offline model is a two-layer Mobius DiT with hidden width 64, four attention
heads, four latent channels, patch size two and 32-dimensional conditioning. It
uses patch embedding, sinusoidal timestep features, adaptive LayerNorm,
self-attention, cross-attention and a frozen output projection. Its ONNX interface
has `sample`, `timestep`, `encoder_hidden_states` inputs and a `noise_pred` output.
Batch size, spatial dimensions and conditioning length are dynamic. Image height
and width are multiples of the patch size.

Rank-four LoRA adapters modify q/v projections in both attention blocks of each
layer: eight projections and sixteen parameter tensors. An adapter computes
$XW^T + (\alpha/r)(XA)B$ with $\alpha=r=4$. A starts random and B starts at zero.
The frozen output projection still propagates gradients to these adapters.

```python
from onnxscript import optimizer
from onnax.torch import OnnxModule
from examples.diffusion.dit import build, inject

config, base, weights = build()
graph, parameters, mapping = inject(base)
model = OnnxModule(graph, parameters, input_gradients=[], optimize=optimizer.optimize)
```

`input_gradients=[]` selects adapter-only differentiation. The frozen convolution
runs in the forward pass. The task owns the noise-prediction loss and the optimizer:

```python
import torch

update = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0, eps=1e-5)
update.zero_grad(set_to_none=True)
noise_pred = model(**batch)
loss = (noise_pred - target_noise).square().mean()
loss.backward()
update.step()
```

Install the existing `training` and `huggingface` example dependency groups to use
Mobius and PyTorch. The offline workload downloads no models or datasets:

```sh
python -m examples.diffusion.train --steps 100 --output-dir /tmp/dit-training
```

The CLI writes a reloadable ONNX bundle, bound forward model, external PyTorch
checkpoint and JSON report. Restore parameters and AdamW state in a fresh process:

```sh
python -m examples.diffusion.train --steps 20 \
  --resume /tmp/dit-training/checkpoint.pt --output-dir /tmp/dit-resumed
```

The checkpoint records training-data hashes. The public module validates the
source graph, frozen weights and parameter interface when loading state.

## Independent checks

The reference implements the same denoising equations directly with PyTorch
functional operations. Each update checks forward values at shared parameters,
loss and all sixteen adapter gradients with `rtol=1e-4, atol=1e-6`. A second
optimizer maintains an independent parameter trajectory. Its loss is checked at
every update and its parameter L2 error must stay below
`1e-6 + 1e-4 * reference_parameter_L2_norm`. The report also retains the original
coordinate-wise parameter checks and independent prediction differences.

The [100-update report](https://github.com/justinchuby/onnax/blob/main/examples/diffusion/reports/dit-fp32-cpu-m1max.json)
records source and dependency versions, loss curves and numerical diagnostics.
The run reduced training loss from 1.208 to 0.498 and validation
loss from 1.222 to 0.855. Adapter gradient error peaked at `4.65e-8`. Independent
AdamW trajectories had a maximum relative parameter L2 difference of `3.68e-6`;
small coordinates account for the retained pointwise failures. At the captured
update 47, shared-parameter forward error was `5.96e-7`, all gradient checks passed,
and the two independent training losses matched at FP32 precision.

Tests cover nonzero A/B gradients, dynamic rectangular images, varying conditioning
lengths, frozen-weight hashes, full ONNX checker validation, loaded execution and
fresh-process optimizer recovery. Graph optimization uses ONNXScript; runtime
numerical checks disable ORT graph optimization. Reported elapsed time and peak RSS
cover the whole qualification, including independent reference execution and export.

## Fine-tune a pretrained Diffusers UNet

The UNet example uses the same Mobius builder for a small offline model and
[`segmind/tiny-sd`](https://huggingface.co/segmind/tiny-sd), pinned to revision
`3d25de3e8d5aacf94ce91176f204e6d11d120315`. The latter is a pretrained compressed
Stable Diffusion denoiser. Its model card describes the source checkpoint,
distillation and CreativeML Open RAIL-M license.

```sh
uv sync --group dev --group training --group diffusion
python -m examples.diffusion.unet_train --steps 100 --output-dir /tmp/unet-offline
python -m examples.diffusion.unet_train --pretrained --latent-size 16 --steps 3 \
  --output-dir /tmp/unet-pretrained
```

Both commands use fixed synthetic clean latents, noise and conditioning to check
training numerics. The offline UNet has two down/up stages, channels 16/32 and
24-dimensional conditioning. Its 24 q/v adapters contain 48 trainable tensors.
The pretrained UNet has channels 320/640/1280, three down/up stages and
768-dimensional conditioning. Both preserve dynamic batch, spatial dimensions
and conditioning length. Latent dimensions are multiples of the UNet downsampling
factor: two for the offline model and four for the pretrained model.

`build(config, weights)` loads source-layout weights through Mobius' Diffusers
name mapping. It also applies the source `norm_eps` to ResNet/output GroupNorm;
Transformer2D GroupNorm retains its own epsilon. The adapter mapping preserves
source module paths, rank, scaling and stable A/B parameter names. The independent
reference attaches the same low-rank expressions to Diffusers Linear modules.

### Connect application data and scheduling

Encode images with your VAE and prompts with your text encoder. Save CPU tensors
for the training boundary:

```python
torch.save(
    {
        "clean": scaled_image_latents,  # [batch, 4, height, width]
        "conditioning": text_hidden_states,  # [batch, tokens, cross_attention_dim]
        "noise": sampled_noise,  # same shape as clean
        "timesteps": sampled_timesteps,  # int64 [batch]
    },
    "encoded-batch.pt",
)
```

Apply the VAE's latent scaling when constructing `clean`. The application owns
image preprocessing, text encoding, data loading and noise sampling.

```sh
python -m examples.diffusion.unet_train --pretrained --batch encoded-batch.pt \
  --steps 20 --output-dir /tmp/unet-adapted
python -m examples.diffusion.unet_train --pretrained --batch encoded-batch.pt \
  --resume /tmp/unet-adapted/checkpoint.pt --steps 20 --output-dir /tmp/unet-resumed
```

The external `DDPMScheduler` constructs noisy latents. Its `prediction_type`
selects noise $\epsilon$, clean latent $x_0$, or velocity
$v_t=\sqrt{\bar\alpha_t}\epsilon-\sqrt{1-\bar\alpha_t}x_0$ as the target.
The external AdamW updates A/B with learning rate 0.003 and epsilon `1e-5`.
The loss averages over all latent elements. Validation uses a separate fixed
synthetic batch, recorded in the report.

`sample_latents` runs the exported denoiser with an external
`DPMSolverMultistepScheduler`. The scheduler owns solver state, input scaling and
timesteps; the ONNX model predicts the configured target at each step.

### Qualification and saved execution

The CLI writes `export/`, `checkpoint.pt` and `report.json`. It checks all adapter
gradients at `rtol=1e-4, atol=1e-6`, independent AdamW trajectories, frozen-weight
hashes, the full ONNX checker, bitwise loaded predictions and external-scheduler
sampling. Fresh-process tests compare model and optimizer state exactly.

The [offline 100-update report](https://github.com/justinchuby/onnax/blob/main/examples/diffusion/reports/unet-offline-fp32-cpu.json)
records training loss 1.204 → 0.746 and validation loss 1.174 → 1.196. The
[pretrained three-update report](https://github.com/justinchuby/onnax/blob/main/examples/diffusion/reports/unet-pretrained-fp32-cpu.json)
uses a 16×16 latent: training loss 0.269 → 0.117 and validation 0.343 → 0.276.
These measurements use CPU FP32 with one thread and include the independent
Diffusers reference in peak process RSS. Module construction time includes VJP
generation, ONNXScript optimization, serialization and session creation.

For long trajectories, complete-tensor prediction and parameter L2 error must
stay below `1e-6 + 1e-4 * reference_L2_norm`; reports also retain pointwise errors.
The [attention comparison](https://github.com/justinchuby/onnax/blob/main/examples/diffusion/reports/unet-attention-numerics-fp32.json)
measures identical-weight FP32 outputs: ONNX/PyTorch relative L2 error is about
`2.3e-6`, alongside `2.0e-6` between PyTorch attention implementations. Reproduce it:

```sh
python -m examples.diffusion.numerics --output /tmp/unet-numerics.json
```

The image derivative rules retain Conv and GroupNormalization primals. Conv
supports explicit padding/VALID, grouped kernels, strides and dilation; weight
products require static kernel dimensions. A ConvTranspose JVP composes these
products into forward-over-reverse HVPs. GroupNorm follows the configured
statistics dtype. Resize products cover positive integer nearest-neighbor
repetition with half-pixel/round-prefer-floor or asymmetric/floor coordinates.
The [pretrained MLX report](https://github.com/justinchuby/onnax/blob/main/examples/diffusion/reports/unet-pretrained-fp32-mlx.json)
loads the saved 16×16-latent export and runs three more updates: loss
0.1173 → 0.1081, maximum adapter-gradient error `8.03e-9`. Run the same
serialized graphs on a selected provider:

```sh
python -m examples.diffusion.validate_unet --training-run /tmp/unet-pretrained \
  --provider mlx --output /tmp/unet-mlx.json
```

Use `--mlx-library` to select a development MLX EP library.

Native MLX opt-in tests run both dynamic rectangular batches through three
updates and reload with CPU fallback disabled. The pretrained report uses the qualified MLX EP #106 runtime. For 64×64
latent resolution and fused execution, continue with the
[real-image workflow](diffusion-images.md).
