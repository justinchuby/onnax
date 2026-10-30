# Microbatch gradients and precision

Extend the [LoRA workload](lora.md) with streaming FP32 microbatch gradients, then
compare full-batch and microbatch memory use and execution time. The [BF16 LoRA
tutorial](bf16.md) applies the dtype policy below to a mixed-precision local function on
CPU ReferenceEvaluator.

```bash
uv sync --group dev --group training
uv run python -m examples.lora.microbatch --output-dir microbatch_models
```

## Weighted gradient accumulation

For an effective batch of $N$ independent samples, split into microbatches of sizes
$n_k$. The LoRA loss is a mean over samples and output coordinates. Each exported call
returns the microbatch mean $L_k$ and parameter gradients $g_k$. At a fixed parameter
point, combine them as

$$
L=\frac{\sum_k n_k L_k}{N},\qquad
 g=\frac{\sum_k n_k g_k}{N},\qquad N=\sum_k n_k.
$$

Weighting by sample count handles a shorter final microbatch. The example uses sizes
**63, 63, 63, 63, 4** for an effective batch of 256. Every microbatch uses the same
parameters; Adam runs **once per effective batch**, with one increment of its step
counter.

`examples.lora.accumulation.accumulate_gradients(chunks, xp=...)` is external example
code. Its streaming input yields `(sample_count, mean_loss, gradients)`; it keeps only
running FP32 sums and returns the combined mean loss and gradients. It uses Array API
arithmetic and `zeros_like`, preserves input arrays and keeps computations on their
device. Names, shapes, dtypes and devices must match. Sample counts are ordinary
positive Python integers supplied by the data loader; arrays and optimizer state stay on
their original device. NumPy, `array-api-strict` and JAX CPU tests exercise weighted
reductions followed by 20 independent Adam updates.

The CPU adapter slices NumPy input views, calls the same dynamic-batch ONNX session,
then wraps its scalar loss and named gradients in the chosen namespace. It requests only
loss and gradients, avoiding copies of predictions back to Python. All original ONNX
outputs remain available in the exported model. The execution adapter uses NumPy and CPU
ORT.

The 200-step regression compares accumulated loss and every gradient with both the
full-batch ONNX graph and independent analytic NumPy derivatives at the same parameters.
A separately evolving full-batch NumPy Adam trajectory checks loss and updated
parameters using the existing `rtol=1e-4, atol=1e-6`. It checks zero initial A
gradients, nonzero B gradients and subsequent gradient traversal, frozen-weight hashes,
FP32 state and execution after parameter export/reload.

Accumulation is valid here because each sample's prediction is independent of the other
samples in its batch. Batch-dependent normalization, losses coupling samples, changing
random masks, or different reduction weights need a separate reduction scheme. FP32
summation order affects rounding, so the tests compare results with a numerical
tolerance.

## Dtype policy

| Quantity | Executed today | BF16 target policy (reference prototype linked above) |
| --- | --- | --- |
| Trainable parameters used by the optimizer | FP32 | FP32 master parameters |
| Frozen weights and matrix operands | FP32 | BF16 compute copies for selected matrix operations |
| Nonlinearities, residuals and loss reduction | FP32 | FP32 initially |
| Local backward arithmetic | FP32 | FP32 |
| Generated derivative outputs | FP32 | FP32 at the accumulation boundary |
| Cross-microbatch gradient sums | FP32 | FP32 |
| Adam first/second moments and update arithmetic | FP32 | FP32 |
| Persisted adapter/master state | FP32 | FP32, with explicit compute-copy regeneration |

A future autocast transform must preserve names and master-state mappings, insert
explicit dtype boundaries and document its derivative convention. Rounding is
discontinuous. The [BF16 example](bf16.md) explicitly registers
identity-through-rounding derivatives for its local matrix function. Direct autodiff of
raw BF16 paths is currently unsupported.

Cross-microbatch sums use FP32. Matrix kernels have their own internal accumulation
policy, which depends on the backend. The BF16 experiment checks forward, gradients and
updates against an independent reference and reports errors relative to FP32 training.

## Backend capability probe

The CLI exports `bf16-matmul.onnx`: FP32 inputs are explicitly cast to BF16, multiplied
as BF16 tensors, then cast back to FP32. It runs full ONNX checker, attempts
CPUExecutionProvider execution and independently tests onnax's active-path contract.
`report.json` records the exact outcomes and reasons. The probe tests one matrix
operation with exactly representable values.

On the recorded development machine, checker passes, CPU ORT reports a missing BF16
MatMul implementation, and onnax rejects direct autodiff of the BF16 path. The report
includes each error message.

The [BF16 reference experiment](bf16.md) now covers explicit local-function derivative
semantics, independent forward/gradient/multi-update checks and serialized master
mappings. Next steps are native BF16 execution, broader precision transforms and device
memory/transfer measurements; see the [roadmap](roadmap.md).

## Memory/time measurement

Two separate worker processes each hold identical 16,384-sample input/target arrays,
identical fitted adapters and the same optimized graph. One evaluates a single full
batch; the other accumulates 32 microbatches of 512. Both keep the entire dataset
resident during the comparison. The parent checks that the workers return matching loss
and full gradient vectors.

Each worker measures 10 warmups and 100 effective-batch gradient evaluations, reporting
median/P95 time, session creation and peak process RSS. Timings include CPU
feeds/outputs and accumulation and exclude Adam, exports and the independent reference.
Process RSS includes Python imports, data, runtime and allocator overhead. The report
also includes model bytes/node counts, data/parameter bytes, versions and thread
settings.

The [recorded development
run](https://github.com/justinchuby/onnax/blob/main/examples/measurements/microbatch-apple-arm64.json)
measured the following on CPU, with an effective batch of 16,384:

| Microbatch | Median gradient + accumulation (ms) | P95 (ms) | Peak process RSS (MB, decimal) |
| --- | ---: | ---: | ---: |
| 16,384 | 3.876 | 4.258 | 170.7 |
| 512 | 4.844 | 4.907 | 146.3 |

Microbatches of 512 reduced peak process RSS and increased execution time. Run the CLI
to choose a microbatch size for your workload.
