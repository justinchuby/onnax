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

### Qwen fine-tuning acceptance

Native FP32 Qwen3-0.6B SFT completes 30 updates, exchanges standard PEFT
adapters and reproduces checkpoint-15 recovery byte for byte. The stable
reference qualifies all 112 adapter gradients at the training endpoints and
through PEFT. [Reference accuracy checks](qwen-reference-accuracy.md) record
pointwise-logit and independent-update differences with captured Adam states.

The joint loss/gradient route retains adapter gradients across the PyTorch
autograd boundary. Tiny and real NF4/BF16 runs complete training and reproduce
checkpoint recovery, including optimizer state and loaded ONNX execution.
The real-model local pullback audit covers all 3,522 input contributions and
580 fan-in additions in each of three adapter states. Pairwise FP32 output-head
accumulation brings the saved optimized module's loss and all 112 gradients
within the original independent-reference tolerance in those states; all 532
frozen hashes remain unchanged. Scalar Adam replay explains the recorded
parameter gap from the captured gradient sequences. Cross-runtime pointwise comparisons remain diagnostic
records; workload acceptance uses gradient correctness, finite updates,
convergence, held-out loss and checkpoint recovery.

The MLX packed-lookup MatMul kernel executes ordinary NF4 graphs. Eager array
lifetime correction brings the observed real NF4/BF16 full-run physical
footprint to about 10.5 GiB on M1 Max. Matched 100-sample measurements record a
1.873 s median ONNX/MLX training step and 3.054 s for PyTorch CPU. The
[NF4 tutorial](qwen-nf4.md) and [runtime audit](qwen-runtime-audit.md) record the
source, precision, storage and loss policies for these measurements.

### Workload results and next priorities

1. **Training quality on instruction data.** The
   [training-quality experiment](training-quality.md) now uses a pinned No Robots
   split with 128 training and 32 official-test sequences. Three FP32 seeds
   complete 30 updates per runtime: CPU/MLX final held-out loss averages
   2.666917/2.666826, with a maximum paired gap of 0.000255. The matched BF16
   run reaches 2.673957/2.669161, with a maximum paired gap of 0.005285. All
   sampled validation curves decrease. Shared-parameter replay reproduces every
   recorded validation loss on its original runtime. CPU/MLX shared-state loss
   gaps peak at 4.60e-7 for FP32 and 0.000932 for BF16; BF16 gradient cosine is
   at least 0.999745. The report retains all original pointwise tolerance failures.
   The earlier 18/6 split remains available for numerical diagnosis.
2. **Evaluation cost.** Joint mode now creates a reusable loss-only session for
   no-grad evaluation. Tiny FP32/BF16 checks cover three adapter states,
   dynamic shapes, padding and ignored labels with bitwise loss agreement.
   The [evaluation qualification command](qwen-sft.md) passes the same checks
   on Qwen3-0.6B. With 100 measured calls, median evaluation time falls from
   1.375 to 0.117 s for FP32 and 1.858 to 0.359 s for BF16. Reports include
   session creation, exchange, execution and whole-process memory with both sessions.
3. **Precision acceptance.** Assess differences against the declared precision
   policy, independent local derivatives and training outcomes across seeds.
   Investigate changes that affect convergence, stability or model quality.
   Cross-backend elementwise comparisons and CUDA comparisons are optional
   diagnostics.
4. **Memory boundaries.** Use tensor lifetimes and the output-head memory curve
   to choose recomputation boundaries, then validate complete training and
   checkpoint recovery for each change.

Custom UTF-8 training/validation files now work with SFTTrainer. Checkpoints
record data and tokenizer identities and reject changed inputs before model
construction. Optimization, scheduling and checkpoint ownership stay with
Trainer and the external training code.

### Completed milestone: 4096-token Qwen fine-tuning

The [4096-token tutorial](qwen-context-training.md) records the completed
Qwen3-0.6B workload: frozen NF4 decoder weights, rank-4 q/v adapters in all
28 layers, and ordinary ONNX forward/VJP modules with decoder checkpointing.
FP32 and BF16 projection compute each completed 30 updates on complete
LongAlpaca QA, with FP32 adapters, accumulated gradients and AdamW state.
Training and held-out loss decreased in both runs. Fresh-process recovery
from update 15 reproduced update 30 exactly; PEFT adapter exchange and loaded
inference/training graphs reproduced the saved results.

The measurement report contains 22 runs at 512, 1024 and 4096 tokens, each with
10 warmups and 100 timed steps. At 4096 tokens, four-layer checkpoint groups
measured 193.9 valid tokens/s for FP32 and 197.6 for BF16 projection compute
on an M1 Max. Whole-process peak physical footprints were 19.36 and 19.54 GiB.
All successful strategies reproduced the same 110 losses within each precision
and sequence length. Timing uses one microbatch per update; quality training
uses four-microbatch accumulation.

Both 8192-token probes exceeded the configured 24 GiB process budget. The next
memory milestone is to profile construction and execution peaks, then choose
session and recomputation boundaries for an 8192-token training run. Device
binding and data exchange measurements will guide subsequent runtime work.

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
