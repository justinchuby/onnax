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

The next workload uses a pretrained Diffusers denoiser and its configured
prediction target. Diffusers supplies scheduling and sampling; text encoding,
VAE execution and data loading stay with the external application.
