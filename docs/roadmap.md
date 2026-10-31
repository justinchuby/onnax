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
[Trainer integration](huggingface.md), [Mobius Qwen3 SFT](qwen-sft.md), and
[PEFT/chunked-loss experiments](qwen-adapters.md) now connect external training
tools to exported ONNX derivatives. [NF4 Qwen fine-tuning](qwen-nf4.md) extends
the same bridge to packed decoder weights, independent adapter updates and
fresh-process checkpoint recovery.

Native FP32 Qwen SFT now completes 30 updates, exchanges standard PEFT adapters
and reproduces checkpoint-15 recovery byte for byte. The stable reference
qualifies all 112 adapter gradients at the training endpoints and through
standard PEFT. [Reference accuracy checks](qwen-reference-accuracy.md) retain
the remaining pointwise-logit and independent-update differences, including
captured Adam-state histories.

The experimental MLX packed-lookup MatMul kernel executes the ordinary NF4
graphs. Isolated measurements compare fused/unfused MLX, ONNX CPU and PyTorch
CPU; [runtime reports](qwen-runtime-audit.md) record the measured source,
complete-step timing and process RSS.

The joint loss/gradient route now runs in one session and retains only adapter
gradients across the PyTorch autograd boundary. Tiny Qwen also completes 100
updates with BF16 projections and FP32 master adapters, gradients and Adam
state on ReferenceEvaluator and native MLX. Both reproduce checkpoint-50
recovery byte for byte. The [BF16 projection audit](qwen-sft.md) compares shared
parameter states; strict independent five-step update qualification is still
open.

Tiny Qwen now combines packed NF4 decoder storage with BF16 products. The
strict five-step independent updates, 100-step native training, PEFT controls
and checkpoint recovery pass for this workload.

The real NF4/BF16 transformer completes 30 updates and fresh-process
checkpoint-15 recovery, with exact adapter/state/export replay. Eager array
lifetime correction reduces its observed full-run physical footprint to about
10.5 GiB on M1 Max. Matched 100-sample measurements record a 1.873 s median
ONNX/MLX training step and 3.054 s for PyTorch CPU, with source, precision,
storage and loss policies in the [NF4 tutorial](qwen-nf4.md).

Next, resolve the real-model independent BF16 logits, loss and gradient
differences and complete independent FP32 update qualification. Shared-input
projection controls distinguish product accumulation from upstream rounding;
all comparisons retain their original tolerances. The real NF4/BF16 local
pullback audit passes all 3,522 input contributions in three adapter states.
Independent full-graph recomposition covers 580 fan-in additions per state.
A scalar-oracle audit identifies ordinary CPU reference error in the long
output-head reduction. With pairwise FP32 head accumulation, all 112 adapter
gradients pass in all three states at the original tolerance. Next connect
this check to the exported joint loss/gradient graph and resolve the remaining
independent forward and loss differences.
Measured tensor lifetimes
and the head memory curve will guide recomputation boundaries.

## Engineering boundaries

- Preserve model-first transforms and global decorator registration; later
  registrations override overlapping versions. No mandatory per-call rules argument.
- Export ordinary ONNX, document backend requirements, and optimize copies with
  ONNXScript. ORT graph optimization is disabled in numerical acceptance tests.
- Preserve parameter names and record actual collision-resolved gradient names.
- Require workload evidence for new operators or local graph rewrites.
- Separate measured execution, exchange and update costs; never infer acceleration
  from executability or a smaller graph.
- FP8 and distributed execution follow the quantized fine-tuning work.
