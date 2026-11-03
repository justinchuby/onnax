# Train on encoded image-caption batches

This Apple Silicon qualification keeps image processing, VAE/CLIP encoding, AdamW and
checkpoint storage in external Python code. onnax supplies the saved UNet forward
and VJP graphs. It uses the pinned Tiny-SD UNet from the
[Diffusers tutorial](diffusion.md).

## Encode images once

```sh
python -m examples.diffusion.encode_images --device mps --image-size 512 \
  --output-dir /tmp/unet-images
```

The default manifest selects three photographs from
[diffusers/dog-example](https://huggingface.co/datasets/diffusers/dog-example),
revision `7aac740a23542bde4c568131b815a9ed08a7e192`. The filenames retain Alvan
Nee's Unsplash identifiers. Two photos train the adapter; the third supplies
validation. All three use the caption “a photograph of a dog”.

The external VAE and CLIP come from the same pinned Tiny-SD pipeline. Images are
oriented using EXIF metadata, converted to RGB, center-cropped/resized to 512²
with Lanczos and normalized to `[-1, 1]`. The VAE posterior mean is scaled by its
configured `scaling_factor`, following the
[Diffusers VAE convention](https://huggingface.co/docs/diffusers/api/models/autoencoderkl).
CLIP supplies 77 final hidden states, including its normal padding positions.

Each `example-N.pt` contains FP32 clean latents `[1, 4, 64, 64]`, conditioning
`[1, 77, 768]`, seeded noise and an integer timestep. The manifest records image
hashes, tensor shapes/dtypes/hashes, preprocessing and encoder versions. Timestep
317 and fixed noise make the training and runtime comparisons repeatable.

Pass `--manifest` for your own images. Its version-1 JSON contains an `examples`
list with `file`, `caption` and `split` (`train` or `validation`). Local filenames
are relative to the manifest. A `repo_id` and pinned `revision` select files from
a Hugging Face dataset instead.

## Accumulate gradients and restore training

Use the pretrained export created by the first tutorial:

```sh
python -m examples.diffusion.objective --training-run /tmp/unet-pretrained \
  --output-dir /tmp/unet-objective
python -m examples.diffusion.train_images --training-run /tmp/unet-pretrained \
  --objective-export /tmp/unet-objective \
  --encoded-dir /tmp/unet-images --steps 20 --checkpoint-every 10 \
  --output-dir /tmp/unet-photo-training
```

The objective export composes MSE with the frozen source and enables
`OnnxModule(..., eager_gradients=True)`. Each differentiable forward executes loss
and gradients together; PyTorch backward applies the accumulation weight. Reuse
this export for training, timing and recovery.

The example starts from seeded A and zero B, so its first prediction uses the
pretrained base. It accumulates two microbatches of one image each, dividing each
image's mean-squared loss by two before backward. External AdamW updates the 72
adapter tensors at learning rate 0.003 and epsilon `1e-5`. Frozen convolutions and
normalizations propagate activation gradients to those adapters.

An independent Diffusers model on MPS checks predictions, loss, every adapter
gradient and each update. The recorded 20-update run reduces training loss
`0.1163 → 0.0950` and held-out loss `0.1105 → 0.1096`. The
[training report](https://github.com/justinchuby/onnax/blob/main/examples/diffusion/reports/real-images-m1max/joint-objective-training.json)
includes the input manifest and per-update diagnostics.

Resume in a new process, using the total target update count:

```sh
python -m examples.diffusion.train_images --training-run /tmp/unet-pretrained \
  --objective-export /tmp/unet-objective \
  --encoded-dir /tmp/unet-images --steps 20 \
  --resume /tmp/unet-photo-training/checkpoint-10.pt \
  --output-dir /tmp/unet-photo-resumed
python -m examples.diffusion.replay_images --first /tmp/unet-photo-training \
  --resumed /tmp/unet-photo-resumed --encoded-dir /tmp/unet-images \
  --output /tmp/unet-photo-recovery.json
```

The replay checks parameters, AdamW, step, source identity, encoded data and every
exported file. It then loads both bundles and compares prediction, loss and all
72 gradients byte for byte. Full ONNX checker validation covers both exports. The recorded replay verifies
289 state tensors, ten exported file checksums, ten checked ONNX models and
74 actual loaded prediction/loss/gradient tensors byte for byte. The final
`export/prediction.onnx` binds adapters into the cached optimized forward graph;
its standalone inference prediction also matches bitwise.

## Measure the execution boundary

Prepare the scalar objective once, then run each provider in a fresh process
on the same cached image and adapter state:

```sh
python -m examples.diffusion.objective --training-run /tmp/unet-pretrained \
  --output-dir /tmp/unet-objective
python -m examples.diffusion.benchmark_unet --training-run /tmp/unet-pretrained \
  --batch /tmp/unet-images/example-0.pt --provider mps --output /tmp/unet-mps.json
ONNXRUNTIME_EP_MLX_NO_COMPILE=0 python -m examples.diffusion.benchmark_unet \
  --training-run /tmp/unet-pretrained --batch /tmp/unet-images/example-0.pt \
  --provider mlx --objective-export /tmp/unet-objective \
  --mlx-library /path/to/libonnxruntime_mlx_ep.dylib --output /tmp/unet-mlx.json
python -m examples.diffusion.compare_unet --baseline /tmp/unet-mps.json \
  --candidate /tmp/unet-mlx.json --output /tmp/unet-comparison.json
```

`--mlx-library` selects the measured development EP. The objective manifest
identifies the base graph, scalar loss, target and execution bundle. The
`--derivative-header` route supports caller-defined external losses.
The runtime uses a private snapshot; the source export stays reusable.
The [attention deployment tutorial](attention-vjp.md) records the successive
runtime/graph variants and source hashes. These FP32 runs use batch one and
64² latents, with ORT graph optimization and CPU fallback disabled. PyTorch uses
Diffusers SDPA; `--checkpoint` selects its gradient-checkpointed route.

Each measurement performs ten warmups and one hundred natural full training
steps. AdamW uses learning rate zero to hold adapter values fixed while updating
its moments and step. Only the final boundary synchronizes a full MPS step.
Separate phase profiles synchronize after forward, backward and optimizer;
their ten samples describe phase costs. MLX ORT calls return host arrays, so each
run includes device transfer and output materialization. The CPU PyTorch bridge
owns MLX-route adapter parameters; MPS-route adapters and AdamW state stay on MPS.
Image encoding and initial input transfer are outside the timed steps.

| Route | Step median | P95 | Peak process footprint |
| --- | ---: | ---: | ---: |
| PyTorch MPS eager | 0.320 s | 0.323 s | 6.88 GiB |
| PyTorch MPS checkpointed | 0.436 s | 0.441 s | 5.42 GiB |
| MLX raw execution | 1.073 s | 1.098 s | 6.05 GiB |
| MLX fused attention + host shapes | 0.745 s | 0.770 s | 4.88 GiB |
| MLX + direct convolution padding | 0.696 s | 0.713 s | 4.95 GiB |
| MLX + broadcast adjoints, eager | 0.669 s | 0.688 s | 4.94 GiB |
| MLX + broadcast adjoints, compiled | 0.651 s | 0.661 s | 5.00 GiB |
| MLX joint scalar objective | 0.474 s | 0.482 s | 5.03 GiB |
| MLX + padded forward attention | 0.464 s | 0.472 s | 4.63 GiB |
| MLX + recomputed fused pullback | **0.372 s** | **0.382 s** | **4.76 GiB** |

Broadcast fusion packages 245 runtime reductions as portable ONNX functions.
With [MLX EP support](https://github.com/onnxruntime/onnxruntime-ep-mlx/pull/111),
the same-binary compiled comparison changes step median from 0.959 s to 0.651 s
and process peak from 6.66 GiB to 5.00 GiB. All 74 captures match bitwise.
Diagnostic subgraph executions fall from 1172 to 8, with recorded materialized
output bytes changing from 4.27 GB to 3.72 MB. The compiled two-call route spends about
0.156 s in forward and 0.495 s in loss/backward.

The [matched reports](https://github.com/justinchuby/onnax/tree/main/examples/diffusion/reports/real-images-m1max)
retain the graph/runtime variants, binary hashes and numerical comparisons.
Joint scalar-objective execution removes the separate forward replay: median/P95
becomes 0.474/0.482 s, with 4.32 GB MLX peak allocation and 5.40 GB process peak.
Prediction and all 72 gradients match the two-call route bitwise; loss differs by
7.5e-9. Phase profiles record 0.468 s for the joint run, 0.001 s for autograd
handoff and 0.002 s for AdamW. The report labels those phases explicitly.
The diagnostic prediction output leaves the public module's loss and all 72
weighted gradients bitwise unchanged.

The optional {ref}`recomputed pullback <recomputed-pullbacks>` uses
MLX's fused attention forward/backward kernel for six short-head attention calls.
Its matched run measures 0.372/0.382 s median/P95, 3.98 GB MLX peak allocation
and 5.11 GB whole-process peak. The same-binary saved-output control measures
0.465/0.472 s and 5.21 GB process peak. Prediction, loss and all 72 adapter gradients
pass the existing comparison gates. Twenty accumulated updates and a fresh-process
checkpoint recovery also pass: all 289 state tensors, ten exported file hashes
and 74 loaded execution tensors match bitwise. Export this schedule with:

```sh
python -m examples.diffusion.objective --training-run run \
  --output-dir objective --recompute-attention
```

Saved-value handoff remains the execution route for arbitrary vector cotangents.
The current UNet boundary study finds 1.75 GB of logical saved outputs, including
0.95 GB used only for shapes. Compress those boundaries before measuring an
activation-handoff plan.
Footprint and RSS cover the isolated process, including construction, warmup,
runtime and allocator caches. The raw MLX route caps its reusable cache at
512 MiB; MPS uses its default allocator. A worker guard limits physical footprint
to 24 GiB and retains 5 GiB free disk. The extra read-only replay uses verified private COW snapshots, a 12 GiB
physical-footprint cap and 3 GiB free-disk reserve. Each benchmark writes
`*.progress.json` atomically after every step, preserving completed sample counts
and timings when a guard terminates the worker. Benchmark output paths must be
new. Raw array captures remain beside each local
report for `compare_unet` to verify workload identity and numerical results.

## Keep experiment storage bounded

Reuse a qualified source export for new encoded batches and runtime measurements.
The source graph, frozen weights and cached VJP are independent of image data;
benchmarks write small JSON/NumPy captures. Training checkpoints contain adapter
and optimizer state. Retain the final trained ONNX export and the checkpoint pair
needed to demonstrate recovery, with their source revision and input manifest.

`train_images` writes a complete bundle at completion: about 2.2 GiB in this
model's logical file sizes. APFS private COW snapshots can share its frozen-weight
blocks. Select full-export milestones and retain lightweight intermediate
checkpoints/reports between them.
