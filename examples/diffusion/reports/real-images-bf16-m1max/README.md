# Real-image UNet with BF16 projections on M1 Max

The pinned `segmind/tiny-sd` UNet runs on cached 512-pixel image latents. All
173 matrix projections use BF16 operands and results. Convolution, attention,
normalization, loss, adapter parameters, gradient products and AdamW state use
FP32. MLX and the independent Diffusers MPS reference implement the same rounded
operand pullback.

The 100-sample measurements follow 10 warmup steps, with zero learning rate and
identical parameters. MLX median/P95 are 388.52/397.67 ms; MPS median/P95 are
358.56/362.35 ms. MLX takes 8.4% longer for this workload. Whole-process physical
footprint peaks are 5.18 GB and 7.95 GB respectively, including setup and runtime
caches. The MLX cache limit is 512 MiB. The reports record thread settings,
versions, phase timings, library identity and captured input identities.

The MPS reference saves each rounded operand only when its corresponding
input gradient requires it. Its 74 captured outputs and gradients are bitwise
identical to the earlier reference that saved both operands.

Twenty diagnostic AdamW updates reduce training loss from 0.11624 to 0.09502
and validation loss from 0.11047 to 0.10960. Frozen weights remain unchanged.
Fresh-process recovery from checkpoint 10 reproduces all 289 checkpoint state
tensors, 10 exported file checksums and 74 loaded execution tensors exactly.
Ten exported ONNX models pass the full checker.

The training report records independent optimizer trajectories. Their final
pre-update gradients differ by 5.23% in relative L2 norm. Replaying both
implementations at the same native adapter point reduces this to 1.93%, with
cosine similarity 0.999816. All 72 replayed native gradients are bitwise identical
to the captured originals. Shared-input BF16/FP16 projection and FP32 pullback
tests compare against MPS at `rtol=1e-4, atol=1e-6`.

`training-diagnostic.json` and `recovery-diagnostic.json` use
`gradient_validation: report`: the half-precision trajectory gradient differences
are recorded while prediction, MSE, parameter, frozen-weight and export checks
remain active. The default training command uses strict gradient validation.
The scalar MSE comparison uses the bound derived from the measured prediction
RMS difference; each path also checks its own MSE in FP32.

ORT graph optimization and CPU fallback are disabled. The measurements are CPU
host/Python-to-native runtime execution with MLX and MPS GPU kernels. Reported
physical footprints cover the whole process; allocator counters have their own
scopes in the benchmark reports.
