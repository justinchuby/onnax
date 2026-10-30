# Precision and memory boundaries

The first P3 experiment extends the [LoRA workload](lora.md) with streaming
microbatch gradients. It defines a mixed-precision policy before enabling BF16
execution. Current production derivative graphs still require FP32 or FP64;
this experiment does **not** enable BF16 differentiation.

```bash
uv sync --group dev --group training
uv run python -m examples.lora.microbatch --output-dir microbatch_models
```

## Weighted gradient accumulation

For an effective batch of $N$ independent samples, split into microbatches of
sizes $n_k$. The LoRA loss is a mean over samples and output coordinates. Each
exported call returns the microbatch mean $L_k$ and parameter gradients $g_k$.
At a fixed parameter point, combine them as

$$
L=\frac{\sum_k n_k L_k}{N},\qquad
 g=\frac{\sum_k n_k g_k}{N},\qquad N=\sum_k n_k.
$$

A plain average of microbatch gradients gives the wrong answer when the last
microbatch is shorter. The example deliberately uses sizes **63, 63, 63, 63, 4**
for an effective batch of 256. Every microbatch uses the same parameters; Adam
runs **once per effective batch**, with one increment of its step counter.
Updating Adam's moments after each microbatch is a different algorithm.

`examples.lora.accumulation.accumulate_gradients(chunks, xp=...)` is external
example code. Its streaming input yields `(sample_count, mean_loss, gradients)`;
it keeps only running FP32 sums and returns the combined mean loss and gradients.
It uses Array API arithmetic and `zeros_like`, never reads array scalars, performs
host conversions or mutates input arrays. Names, shapes, dtypes and devices must
match. Sample counts are ordinary positive Python integers supplied by the data
loader; arrays and optimizer state stay on their original device. NumPy,
`array-api-strict` and JAX CPU tests exercise weighted reductions followed by 20
independent Adam updates. This does not establish device-resident ONNX execution.

The CPU adapter slices NumPy input views, calls the same dynamic-batch ONNX
session, then wraps its scalar loss and named gradients in the chosen namespace.
It requests only loss and gradients, avoiding copies of predictions back to
Python. All original ONNX outputs remain available in the exported model.
This adapter uses NumPy/CPU; other runtime bindings require a separate adapter.

The 200-step regression compares accumulated loss and every gradient with both
the full-batch ONNX graph and independent analytic NumPy derivatives at the same
parameters. A separately evolving full-batch NumPy Adam trajectory checks loss
and updated parameters using the existing `rtol=1e-4, atol=1e-6`. It checks zero
initial A gradients, nonzero B gradients and subsequent gradient traversal,
frozen-weight hashes, FP32 state and execution after parameter export/reload.

Accumulation is valid here because each sample's prediction is independent of
the other samples in its batch. Batch-dependent normalization, losses coupling
samples, changing random masks, or different reduction weights need a separate
mathematical contract. FP32 summation order can change rounding; no bitwise
identity with a monolithic reduction is promised.

## Dtype policy

| Quantity | Executed today | BF16 target policy (not enabled) |
| --- | --- | --- |
| Trainable parameters used by the optimizer | FP32 | FP32 master parameters |
| Frozen weights and matrix operands | FP32 | BF16 compute copies for selected matrix operations |
| Nonlinearities, residuals and loss reduction | FP32 | FP32 initially |
| Local backward arithmetic | FP32 | FP32 initially; lower matrix products only after accuracy validation |
| Generated derivative outputs | FP32 | FP32 at the accumulation boundary |
| Cross-microbatch gradient sums | FP32 | FP32 |
| Adam first/second moments and update arithmetic | FP32 | FP32 |
| Persisted adapter/master state | FP32 | FP32, with explicit compute-copy regeneration |

A future autocast transform must preserve names and master-state mappings,
insert explicit dtype boundaries and document its derivative convention.
Finite-precision rounding is discontinuous; treating a cast's derivative as an
identity is a chosen real-valued surrogate, not the literal derivative of its
rounding function. Existing FP32/FP64 casts use that convention. It does not
authorize silently bypassing the current BF16 active-path rejection.

FP32 gradient accumulation between microbatches is distinct from a matrix
kernel's internal arithmetic. A BF16 tensor type alone does not establish the
backend's internal accumulation dtype or numerical accuracy. Those need backend
capability evidence and forward/gradient/update tests on the actual workload.
The initial BF16 experiment will keep loss and master state FP32 and separately
measure errors against the FP32 baseline; it must not silently fall back to FP32
matrix operations while reporting BF16 compute.

## Capability probe and remaining gates

The CLI exports `bf16-matmul.onnx`: FP32 inputs are explicitly cast to BF16,
multiplied as BF16 tensors, then cast back to FP32. It runs full ONNX checker,
attempts CPUExecutionProvider execution and independently tests onnax's
active-path contract. `report.json` records the exact outcomes and reasons.
The probe uses exactly representable values; success would demonstrate one
matrix operation, not BF16 training correctness or speed.

On the recorded development machine, checker passes, CPU ORT reports a missing
BF16 MatMul implementation, and onnax rejects the active BF16 path. This is a
measured limitation of that configuration, not a claim about every provider.
A backend failure is reported as an unsupported capability, while invalid graphs
and unrelated errors still fail the example.

Remaining P3 gates are an executing BF16-capable provider, explicit autocast and
derivative semantics, independent forward/gradient/multi-update validation,
serialized dtype/master mappings and memory/error measurements. Device-resident
bindings, DLPack, recomputation and preconditioned/batched products are subsequent
choices driven by workload measurements. BF16 training remains incomplete.

## Memory/time measurement

Two separate worker processes each hold identical 16,384-sample input/target
arrays, identical fitted adapters and the same optimized graph. One evaluates a
single full batch; the other accumulates 32 microbatches of 512. Both keep the
entire dataset resident, so the comparison isolates runtime/activation and
accumulation behavior rather than data streaming from disk. The parent checks
that the workers return matching loss and full gradient vectors.

Each worker measures 10 warmups and 100 effective-batch gradient evaluations,
reporting median/P95 time, session creation and peak process RSS. Timings include
CPU feeds/outputs and accumulation and exclude Adam, exports and the independent
reference. RSS includes Python imports, data, runtime and allocator overhead;
it is not a per-tensor peak, device-memory reading or proof of asymptotic scaling.
The report includes model bytes/node counts, data/parameter bytes, versions and
thread settings. There is no machine-dependent performance gate.

The [recorded development run](https://github.com/justinchuby/onnax/blob/main/examples/measurements/microbatch-apple-arm64.json)
measured the following on CPU, with an effective batch of 16,384:

| Microbatch | Median gradient + accumulation (ms) | P95 (ms) | Peak process RSS (MB, decimal) |
| --- | ---: | ---: | ---: |
| 16,384 | 3.876 | 4.258 | 170.7 |
| 512 | 4.844 | 4.907 | 146.3 |

This is a memory/time tradeoff: lower observed RSS took more runtime calls and
more time. Rerun on your machine before choosing a microbatch size. These process
peaks are not activation-memory measurements, and the experiment does not
establish a GPU or BF16 speedup.
