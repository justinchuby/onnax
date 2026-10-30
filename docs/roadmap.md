# Workload-driven development roadmap

onnax generates, composes and optimizes ONNX differentiation graphs for fine-tuning,
small-model training and scientific computing. External programs own optimization,
scheduling, data loading and checkpoint storage.

## P0 — Frozen-base LoRA fine-tuning (delivered)

The [LoRA tutorial](lora.md) is the first acceptance workload: load a frozen ONNX base,
inject adapters, export selected parameter gradients, and train using an external Array
API Adam update. Independent reference trajectories, reloaded artifacts, frozen-weight
checks and bounded performance measurements are required.

## P1 — Small-model training (delivered)

The [MLP classifier](mlp.md) trains all four weight/bias tensors through exported ONNX
gradients and an external Array API Adam. Stable LogSoftmax cross-entropy, independent
gradients and update trajectories, reloads and dynamic batches are validated. Further
operators must be justified by actual training workloads.

## P2 — Matrix-free scientific computing (delivered)

The [nonlinear decay fit](fitting.md) composes JVP/VJP products with external damped
Gauss–Newton CG. Exact HVPs and vmap direction batches are also validated; only the
independent small-problem oracle materializes derivatives. Solver trajectories, reloads,
failure paths and scoped measurements are covered. The [Jelly Newton-CG
example](https://www.justinchuby.com/onnax/jelly/) remains the interactive exact-HVP
workload.

## P3 — Memory, precision and device execution (in progress)

The precision policy uses FP32 master parameters, gradients, accumulation and
optimizer state, with explicit BF16 matrix computation. The [LoRA](bf16.md) and
[classifier](mlp-bf16.md) workloads validate training, independent references and
checkpoint recovery under this policy.

Delivered experiments cover four areas:

- **Runtime boundaries.** [CPU binding](iobinding.md), [JAX DLPack](dlpack.md),
  [CoreML](coreml.md) and [MLX GPU updates](mlx-training.md) execute complete
  exported training graphs. Shared storage, immutable outputs, compiled Adam and
  checkpoint resumption are tested; reports separate execution, exchange and update costs.
- **Precision diagnosis.** [Product bounds](bf16-bounds.md) and
  [composed intervals](composed-bounds.md) check LoRA arithmetic. The
  [classifier inspector](classifier-device.md) audits all four gradients at
  observed forward values across both 150-state BF16 runs. Independent NumPy
  differences retain their original tolerances and failure records.
- **Batch-size scaling.** [Metal microbatch training](metal-microbatch.md)
  combines weighted FP32 gradients on device and validates full/microbatch
  parity. A 16,384-sample workload records the memory/time curve with isolated
  process RSS and macOS physical footprint.
- **Idle residency.** The [allocator-lifecycle diagnostic](memory-lifecycle.md)
  observes provider and Python caches, samples delayed footprint changes after
  idle release, and verifies exact FP32/BF16 update resumption.

The [activation-lifetime experiment](activation-liveness.md) now reports tensor
last uses and peak live sets, validates a preactivation-recomputation candidate
through 150 updates and recovery, and compares repeated native allocations.
Next, add a deeper activation-chain workload and use the peak live set to choose
a checkpoint boundary with a measurable memory/time tradeoff.

## Engineering boundaries

- Preserve model-first transforms and global decorator registration; later
  registrations override overlapping versions. No mandatory per-call rules argument.
- Export ordinary ONNX, document backend requirements, and optimize copies with
  ONNXScript. ORT graph optimization is disabled in numerical acceptance tests.
- Preserve parameter names and record actual collision-resolved gradient names.
- Require workload evidence for new operators or local graph rewrites.
- Separate measured execution, exchange and update costs; never infer acceleration
  from executability or a smaller graph.
- FP8, QLoRA and distributed execution are outside the first milestone.
