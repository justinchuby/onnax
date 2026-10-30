# Keep classifier parameters and Adam state in Metal arrays

This experiment scales the [mixed-precision classifier](mlp-bf16.md) and
compares three external update paths: NumPy CPU Adam, uncompiled Metal Adam
and compiled Metal Adam. All three execute the same ONNX gradient graph on
MLX. The Metal paths keep FP32 parameters and both Adam moments in MLX arrays
throughout training.

On the recorded Apple M1 Max run, the large FP32 model's complete-step median
falls from **4.779 ms with CPU Adam to 2.752 ms with shared bindings and compiled
Metal Adam**. After 150 updates, all three paths have byte-identical parameters
and Adam state.

```bash
uv sync --group dev --group training --group native-mlx
uv run --no-sync python -m examples.mlp_device \
  --profile large --precision fp32 --mode compiled --output-dir classifier_metal

# Same MLX graph, with NumPy CPU updates
uv run --no-sync python -m examples.mlp_device \
  --profile large --precision fp32 --mode cpu --output-dir classifier_cpu
```

`--mode eager` selects uncompiled Metal updates; `--precision bf16` selects
BF16 forward matrix operands/results. Biases, nonlinearities, loss, local
backward arithmetic and optimizer state stay in FP32. Each run saves the
strict NumPy comparisons alongside convergence and residency checks. The
CLI returns status 1 for a strict mismatch after completing training and
checkpoint recovery.

## Scale the classification task

| Profile | Input / hidden / classes | Training / validation rows | Parameters |
| --- | --- | --- | ---: |
| tiny | 2 / 16 / 3 | 384 / 192 | 99 |
| medium | 128 / 512 / 16 | 1,024 / 256 | 74,256 |
| large | 512 / 2,048 / 32 | 2,048 / 512 | 1,116,192 |

The tiny profile preserves the original dataset and initialization. The larger
profiles use fixed Gaussian class centers with independent noise of standard
deviation 0.75. Their targets use label smoothing with $\alpha=1/8$:

$$
Y_i=(1-\alpha)e_{y_i}+\frac{\alpha}{C}\mathbf1.
$$

Training minimizes the same mean cross-entropy as the smaller example.
The smoothed targets have positive entropy: the minimum possible loss is
about 0.67863 for 16 classes and 0.78493 for 32 classes. This gives the larger
workloads a finite-confidence target. Accuracy uses the largest target class.

Both precisions reach **100% validation accuracy** in all update modes. The
medium FP32 training loss falls from 2.7735 to 0.67941; the large FP32 loss
falls from 3.4725 to 0.78547. Scaled profiles use learning rate 0.01 and Adam
epsilon $10^{-6}$; the tiny profile retains 0.03 and $10^{-8}$. The interface
records parameter names/shapes, precision, target policy and optimizer settings.

## Share named buffers with the runtime

The reusable [MlxBinding adapter](https://github.com/justinchuby/onnax/blob/main/examples/mlx_binding.py)
reads static FP32 input/output metadata from a runtime session. Its feed
mapping supports arbitrary parameter and data names. The LoRA adapter now
uses the same implementation and retains its existing interface checks.

Initialize data and parameter arrays once in a Metal stream, bind them by
name and pass the returned gradients to the external updater:

```python
from examples.mlx_binding import MlxBinding, synchronize
from examples.lora.mlx_adam import CompiledAdam

bound = MlxBinding(executor)
update = CompiledAdam(**interface["adam"])
state = None
# data and params contain FP32 MLX arrays created in a Metal stream.
bound.bind({**data, **params})
outputs = bound.run()
grads = {row["input"]: outputs[row["gradient"]] for row in interface["parameters"]}
params, state = update(params, grads, state)
synchronize(params, state)
```

At the boundary, `CpuView` requests MLX's CPU DLPack view of shared Metal
storage. Every input import checks pointer identity. Each execution produces
fresh runtime output owners; MLX imports them with `copy=False` and checks
pointer identity again. Previously returned arrays retain their data across
later executions and after the binding/session is released.

The training graph exports loss and four gradients. A separate forward graph
checks logits during correctness runs. For the large model, returned training
outputs total 4,464,772 bytes per execution. Provider wrapping and output
copy-out remain part of the measured execution stage.

The large model's parameter payload is 4.26 MiB and its two FP32 moments total
8.52 MiB. Every training update checks their dtype and Metal device. Correctness
runs also take owned CPU snapshots, verify functional input preservation and
execute independent references. Timing workers execute the binding/update path
without those diagnostics.

## Verify updates and recovery

All 12 combinations of scale, precision and update mode complete 150 updates
and restore a checkpoint into a fresh session and updater. Recovery checks
loss/gradients, the next parameter update and both next moments. For each
scale/precision pair, all three modes have identical loss histories,
comparison summaries and final checkpoint arrays, including the step count.

A separate flattened NumPy Adam implementation consumes the observed gradients
and checks parameters/moments at every step. Same-parameter analytic checks
and a fully independent NumPy training trajectory are recorded separately:

| Profile / precision | Same-parameter loss | Same-parameter logits | Same-parameter gradients | Shared-gradient updates |
| --- | --- | --- | --- | --- |
| medium / FP32 | Pass | Pass | Pass | Pass |
| large / FP32 | Pass | Near-zero differences | Pass | Pass |
| medium / BF16 | Pass | Rounding differences | Output-layer differences | Pass |
| large / BF16 | Pass | Rounding differences | Output-layer differences | Pass |

The strict comparisons retain `rtol=1e-4, atol=1e-6`; independent trajectories
also show differences. In the first large FP32 logit mismatch, native and
NumPy FP32 values differ by $1.386\times10^{-6}$. Against a single-sample FP64
reference, their absolute differences are $5.72\times10^{-7}$ and
$8.13\times10^{-7}$. The report preserves the original pairwise result and
first-mismatch snapshots. BF16 output-layer gradient differences reach about
$6.5\times10^{-6}$ on the large model.

Portable tests check scaled exports against analytic gradients, explicit
parameter inputs, target normalization and the unchanged tiny dataset.
Native tests cover arbitrary output selection/order, scalar inputs, ownership,
failed-feed invalidation and complete Metal-state training/recovery.

## Measure the complete path

Each row below uses a separate process, ten warmup updates and 100 measured
updates. Within each precision the model bytes, data and initialization are
identical across modes. Timing includes feed preparation/binding, native
execution and exchange, output import, Adam and synchronization of parameters
and both moments.

| Large model | Update path | Adam/sync median (ms) | Step median (ms) | Step P95 (ms) | Peak process RSS (MiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| FP32 | NumPy CPU | 2.127 | 4.779 | 4.907 | 222.3 |
| FP32 | Metal uncompiled | 0.759 | 3.098 | 3.156 | 186.9 |
| FP32 | Metal compiled | 0.452 | 2.752 | 2.839 | 187.9 |
| BF16 | NumPy CPU | 1.980 | 4.798 | 5.078 | 223.3 |
| BF16 | Metal uncompiled | 0.764 | 3.243 | 3.712 | 188.0 |
| BF16 | Metal compiled | 0.440 | 2.889 | 2.967 | 188.0 |

The large FP32 complete path is about 42% shorter with shared bindings and
compiled Metal updates in this measurement; BF16 is about 40% shorter.
The medium model shows the crossover: compiled Metal gives 0.977 ms versus
1.071 ms for CPU Adam in FP32, while CPU Adam gives 0.797 ms versus 0.997 ms
for compiled Metal in BF16.

RSS covers the process, loaded model, data, runtime and allocation overhead.
The report also records Python MLX allocator counters; the provider has its
own runtime allocator. Session creation and first-step costs are separate
from steady-state timings. ORT uses one intra-op and one inter-op thread,
with graph optimization and CPU fallback disabled.

The [complete evaluation report](https://github.com/justinchuby/onnax/blob/main/examples/measurements/classifier-device-state-apple-arm64.json)
contains all 12 timing rows, numerical checks, checkpoint equivalence,
software versions, graph hashes and the FP64 diagnostic.
