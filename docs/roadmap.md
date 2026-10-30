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

The [precision and memory experiment](precision.md) delivers external FP32 weighted
microbatch accumulation, a defined BF16/master-state policy, runtime capability probes
and isolated full/microbatch RSS/time measurements. The [BF16 LoRA experiment](bf16.md)
now executes explicit BF16 MatMul functions on CPU ReferenceEvaluator, with declared
FP32 straight-through derivatives, an independent 200-update reference, FP32 accuracy
baseline and master-state reload. The [CPU binding experiment](iobinding.md) adds
shared output buffers, input ownership/rebinding checks, complete training parity
and a measured comparison with ordinary execution. The [JAX DLPack example](dlpack.md)
adds shared CPU parameter/gradient storage, immutable output ownership, synchronized
Array API updates and checkpoint restoration. The [CoreML experiment](coreml.md)
executes complete static-batch gradient graphs
and 200 external updates through the native provider, with CPU/CoreML timing
and process-memory comparisons. The [native MLX BF16 evaluation](mlx-bf16.md)
executes the full gradient graph and checkpoint replay and records strict
numerical comparisons, including a reproduced rounding-boundary difference.
The [BF16 accumulation bounds](bf16-bounds.md) check all 800 native forward
products with exact integer references and data-dependent rounding intervals.
The same inspector now checks all 800 FP32 backward matrix products with
outward rational endpoints and verifies original-output preservation.
The [MLX GPU training example](mlx-training.md) keeps adapter masters and Adam
moments in Metal arrays, verifies shared Python/runtime storage, and measures
provider copy-out alongside CPU/GPU update paths. Its compiled update path
reuses the same Array API arithmetic, preserves checkpoint state and records
first-update and steady-state costs. Larger adapter profiles now compare CPU/GPU
update placement and separate same-parameter accuracy from independent trajectory
drift. The [composed BF16 checks](composed-bounds.md) now carry arithmetic intervals
through nonlinearities, loss and both parameter gradients, with observed local
budgets and independent same-parameter references. The [BF16 MLP classifier](mlp-bf16.md)
now applies the same precision policy to four trainable parameters and stable
cross-entropy, with complete CPU/native training, checkpoint recovery and
isolated FP32/BF16 measurements. Its workload also simplifies reduction/broadcast
adjoints. The [larger-classifier experiment](classifier-device.md) now shares
named model buffers through a common adapter, keeps all four masters and Adam
moments on Metal, and compares CPU/uncompiled/compiled updates on 74K/1.1M
parameter models. All update modes restore byte-identical checkpoints; the
report retains the strict NumPy precision differences. An observed-forward
inspector now replays the first BF16 output-layer gradient failures and checks
independent head backpropagation at native forward values. The
[Metal microbatch example](metal-microbatch.md) now accumulates weighted FP32
gradients on device, executes complete 150-step FP32/BF16 classifier training and
checkpoint replay, and measures full/microbatch paths in isolated processes.
The classifier inspector now audits all four analytic gradients at observed
forward values across both 150-step BF16 runs, including hidden-layer
backpropagation and original-output preservation. Next evaluate memory scaling
with larger activation workloads and use those measurements to select
recomputation or buffer-lifetime work. The recorded CPU
ORT provider lacks BF16 MatMul. Device memory/transfer measurements will guide
DLPack, recomputation and structured-operator work.

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
