# Real-image UNet with FP16 projections on M1 Max

These measurements use the same pinned UNet, 512-pixel image latent, adapter
parameters and FP32 pullback policy as the BF16 reports. Matrix projections
compute in FP16; convolution, attention, normalization, gradient products and
AdamW state remain FP32. Ten warmup steps precede 100 timed complete steps,
with fixed parameters and zero learning rate.

| Backend | Median step | P95 step | Whole-process physical peak |
|---|---:|---:|---:|
| MLX EP | 383.97 ms | 393.53 ms | 5.18 GB |
| Diffusers MPS | 348.37 ms | 356.16 ms | 8.49 GB |

MLX takes 10.2% longer for this workload. The paired comparison validates all
74 captured prediction, loss and gradient tensors. Benchmark records include
input identities, versions, thread settings, cache policy and timing scopes.
ORT graph optimization and CPU fallback are disabled.

The short runtime trace records seven invocations, including two warmups. It
shows one compilation, six cache hits and zero retraces. Synchronous `mlx.eval`
accounts for 2654.05 ms of 2802.66 ms in subgraph spans. Its spans include GPU
execution; individual ONNX build spans measure host graph construction.
The next profiling step examines Metal kernel costs.

A separate steady-state Metal capture was saved locally as
`/tmp/onnax-unet-fp16-steady.gputrace`. The capture run is separate from the
100-sample benchmark. Physical peaks include process setup and runtime caches;
the MLX cache limit is 512 MiB.
