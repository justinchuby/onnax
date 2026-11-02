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

### Next milestone: evaluator execution and reusable integration

The [matched 4k comparison](qwen-context-training.md) now includes eager and
SDPA PyTorch MPS runs for both precision policies, plus a CPU capacity probe.
Eager MPS measured 9.103/10.469 s per update for FP32/BF16 projection compute,
compared with 21.123/20.728 s for the recorded ONNX/MLX runs. The reports retain
SDPA timing variability and the declared post-cleanup repeat.

A native four-update audit records 90.63 s in MLX evaluation, 5.52 s in graph
translation and 0.43 s in output copying. Evaluation includes CPU scheduling and
GPU execution. Prioritize fused or captured evaluation and fewer materialized
intermediates; use the recorded eager path as the baseline. Qualify each candidate
with adapter gradients, updates, held-out loss, recovery and export before
extending it to 8192-token training.

Direct JVP/VJP rules now retain Attention, RMSNormalization, RotaryEmbedding,
Swish and LayerNormalization. The Qwen `--fused-ops` path selects native MLX
forward implementations and a portable ONNX backward. The optional
[attention deployment pass](attention-vjp.md) now exposes a tiled or GEMM
pullback to MLX EP, with a workspace policy and ordinary ONNX function fallback.
See [fused derivatives](fused-autodiff.md) for the transformation API. The
{ref}`100-sample comparison <m1-max-measurements>` now records native
training throughput alongside PyTorch MPS for FP32 and BF16.

The optional [`onnax.torch` module](torch.md) now owns explicit parameters,
connects forward/VJP sessions to autograd, and exports reloadable ONNX bundles.
Regression, MLP and Qwen workloads qualify caller-defined losses, data-input
gradients, accumulation, checkpoint identity, resource cleanup and installed-wheel
execution. A session factory selects the backend. Model construction, adapter
injection, tokenization and PEFT conventions stay in task adapters; Trainer and
TRL own training loops, optimizers, schedulers and checkpoint policy.

The [named checkpoint execution layer](checkpointing.md) now provides
`onnax.torch.CheckpointedModule`. Qwen and the package share graph partitioning
and block autograd execution. GQA/RMSNorm/SiLU and MHA/LayerNorm/GELU tiny
decoders validate independent gradients and multi-step updates; the execution
bundle restores saved derivatives and supports fresh-process optimizer recovery.
Model adapters supply boundaries, while Trainer/TRL retain training ownership.

The [Mobius family adapter](decoder-families.md) now exercises Qwen3 and Phi-3
through the public checkpoint module. Phi-3 preserves one source QKV adapter
across Mobius's split projections and exchanges it with PEFT. Independent
Transformers updates, TRL training, export/load and fresh-process recovery
validate the family interface. Shared graph construction and decomposition
also remain in use by the original Qwen workload.

The [8192-token workload](qwen-context-training.md) completes 30 updates with
fused operators, NF4 decoder weights and BF16 projections. Fixed training loss
fell from 3.45437 to 2.49825 and validation loss from 3.31262 to 2.52385;
whole-process physical footprint peaked at 10.24 GiB. Fresh-process recovery and
loaded forward/loss/gradient execution reproduced the final checkpoint byte for
byte. All 112 adapter tensors completed the PEFT format roundtrip; the separate
BF16 numerical reference differences are recorded in the report.

One hundred measured MLX updates at 8192 tokens yielded a 39.687-second median,
41.254-second P95 and 206.4 valid tokens/s. MPS exhausted its allocator with
one-, two- and four-layer checkpoint groups under the matched memory limits.

### Hybrid attention and diffusion

Both workloads now have an offline end-to-end example through the public
execution layer. Model construction and task conventions remain in example
adapters while the shared interface develops.

1. **[Qwen3.5 hybrid attention](qwen35-training.md).** Mobius builds a four-layer
   stack with three Gated DeltaNet layers and one full-attention layer. Direct
   JVP/VJP rules preserve `LinearAttention` and `CausalConvWithState` calls;
   the portable reverse recurrence saves chunk boundaries and recomputes one
   chunk at a time. Tests cover all six recurrence inputs, both output
   cotangents, grouped heads, both decay layouts, dynamic shapes and nonzero
   initial state. The tiny model trains ten rank-four A/B tensors for 100
   updates on CPU and MLX, with Transformers checks for logits, token loss and
   every adapter gradient. Training/validation loss reaches 4.42861 / 4.95735
   on CPU. Export/load, frozen weights and fresh-process optimizer recovery
   are verified. Right padding and the source's fused QKV/query-gate layouts
   are preserved.
2. **[Conditional diffusion fine-tuning](diffusion.md).** The two-layer Mobius
   DiT uses hidden size 64, four heads, four latent channels, patch size two,
   8-by-8 latent images and 32-dimensional text conditioning. An external
   noise schedule creates fixed clean-latent/noise/timestep triples. Sixteen
   attention A/B tensors train through ONNX VJP for 100 updates: training
   loss falls from 1.20757 to 0.49772 and validation loss from 1.22232 to
   0.85458. An independent PyTorch denoiser checks forward values and all
   adapter gradients; dynamic spatial shapes, export/load, frozen weights and
   fresh-process recovery are covered by CI.

The shared decoder task now includes Qwen3.5 alongside Qwen3 and Phi-3, using
the same Trainer/TRL/PEFT path and public checkpoint module. The pinned
Qwen3.5-0.8B FP32 run now completes 30 updates, exact checkpoint recovery and
PEFT weight exchange. The [shared-activation VJP audit](shared-activation-vjp)
checks local decoder gradients at common activations and loss cotangents.
Next are longer-sequence MLX training and a small pretrained Diffusers denoiser
with its declared prediction target. Diffusers supplies the noise schedule and sampling loop;
text encoders, VAE execution and datasets remain external. MLX measurements will
separate execution, host exchange and updates, then guide recurrent backward
fusion and state recomputation. Portable derivative functions use `pkg.onnax`
when extracted from the generated tensor graph.

## Training graph and memory priorities

The [ORT Training design handoff](ort-training-handoff.md) identifies the next
steps around supervised positions, saved values and execution lifetimes.
The gated-delta reverse kernel and shared initializer cache have completed
runtime measurement, training and checkpoint qualification.

1. **Delivered: supervised-position head scheduling.** The existing chunked head
   gathers valid shifted labels before projection, scatters hidden gradients back
   into the full context, and skips execution for all-ignored labels. CPU and MLX
   checks cover gradients, updates and supervision-density measurements.
2. **Delivered: saved-value and runtime peak reports.** Named checkpoint VJPs
   now report parameters, inputs, saved activations and nested Scan peaks. The
   [runtime study](block-runtime.md) captures actual Qwen3.5 block feeds, compares
   gradients and measures isolated exported recomputation modules. At 512 tokens,
   gated-delta reverse fusion reduces linear-block median execution from 194.92
   to 110.77 ms and active allocator peak from 1183.10 to 958.36 MiB.
3. **Delivered: shared immutable MLX weights.**
   [MLX EP #105](https://github.com/onnxruntime/onnxruntime-ep-mlx/pull/105)
   interns host buffers and reuses MLX leaves through weak indexes. FP32
   Qwen3.5-0.8B full-task peak footprint falls from 11.10 to 8.16 GiB at 512
   tokens; 100 measured losses and all 60 adapter gradients match byte for byte.
   Thirty SFT updates, fresh-process recovery, PEFT export and loaded execution
   reproduce the per-session control. The [runtime study](block-runtime.md)
   records timing scopes, initialization costs and frozen-tensor verification.
4. **Delivered: bounded heads across decoder families.** The shared
   Qwen3/Qwen3.5/Phi-3 task exports hidden states, source-layout frozen head
   modules and standalone inference ONNX. `Gemm(transB=1)` preserves the
   `[vocab, hidden]` layout for tied-weight sharing. Tests align gradients and
   updates with Transformers and cover unequal-token accumulation and ignored
   labels. Real Qwen3.5 completes 30 SFT updates and exact fresh-process recovery.
   At 512 context tokens and 128 supervised targets, median step time falls
   from 3.132 to 2.578 seconds; peak physical footprint falls from 8.57 to
   7.04 GiB, while peak RSS rises from 9.67 to 11.68 GiB. The
   [decoder tutorial](decoder-families.md) records measurement scopes and FP32
   execution-boundary comparisons. The shared replay CLI checks exports,
   optimizer state and same-route loss/gradient/inference results. Extend
   context length alongside measured block-local recomputation.
5. **Choose selective recomputation from measured peaks.** Record retained
   values, replay subgraphs, invocation counts and release points. Start with one
   decoder block, then extend to multi-layer Qwen and diffusion. Preserve shared
   edges, skip connections, dynamic shapes and deterministic replay state.
6. **Pair fusion contracts with saved values and prompt release.** Establish
   saved-value/dtype conventions for normalization, attention and recurrence.
   Check VJP/JVP and supported higher-order composition; use allocator traces to
   identify large activation holders and retain external shared-gradient accumulation.

Shape-specific MatMul/Gemm forms and shared low-rank products follow measured
costs in these workloads. Graph generation continues to prune inactive derivative
branches while propagating activation gradients through frozen weights. A small
pretrained Diffusers denoiser remains the next model-family milestone.

The [first saved-value study](vjp-memory.md) connects split VJPs with the liveness
analyzer on exported Qwen3.5 blocks. It reports complete local tensor peaks for
the full-attention block. The recurrent study also models nested Scan carry,
retained sequence outputs and per-iteration live sets, including padded chunks.
The runtime measurements now pair these estimates with native allocation peaks.
Initializer sharing addresses persistent storage, and shared bounded heads are
integrated across the decoder families. The current Qwen3.5 qualification extends
that path to complete 3.6k–4k-token QA examples with exported-graph training and
fresh-process recovery.

(block-local-recomputation)=
### Next delivery: measured block-local recomputation

Use the [ORT Training handoff](ort-training-handoff.md) to connect the existing
saved-value analysis, checkpoint executor and native memory counters:

1. Capture one linear-attention block and one full-attention block from the
   qualified 4k workload. Record forward values read by backward, their producers,
   last consumers and the allocator peak during the same invocation.
2. Select one deterministic replay region from the measured peak set. Specify its
   retained inputs, regenerated values, replay count and release points. Preserve
   cross-region edges and shared adapter contributions.
3. Execute the candidate through the existing checkpoint/autograd boundary.
   Compare every adapter gradient, multi-step updates and fresh-process recovery
   against the unchanged execution route.
4. Measure baseline and candidate in isolated processes: ten warmups, one hundred
   timed invocations, median/P95, exchange time, active allocator peak and process
   physical footprint. Use equal context, supervision, precision and thread settings.
5. Extend the selected policy to the full decoder after local qualification.
   Apply the same reporting and execution contracts to the Diffusers workload.

Keep this planner in the example layer while the saved-value and execution
contracts develop. The first delivery is an explicit, inspectable replay plan and
one executed candidate; automatic budget selection follows those measurements.

```{toctree}
:hidden:

ort-training-handoff
```

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

The [MLX precision workflow](mlx-precision.md) covers BF16/FP16 projections,
FP32 adjoints and precision-preserving lookup-4 deployment.

### Block-local replay implementation

The example VJP splitter now accepts explicit saved-value names to regenerate.
Its deterministic producer-slice planner preserves retained cut values and shared
producers. Saved-value reports expose the replay plan, and an isolated paired
forward/backward command validates candidates against actual checkpoint feeds.
Dynamic batches, accumulation, multi-step updates and Adam state restoration are
covered. The first real-model study uses the complete 4032-token Qwen3.5 QA
capture and the full-attention MLP producer slice. Compare this candidate with
both saved activations and the existing whole-block checkpoint policy before
changing decoder execution defaults.

The first 4032-token full-attention comparison now measures the whole-block,
save-all and MLP-replay routes. Whole-block replay uses the least memory and time
in this case, so it remains the default. The saved-value audit finds 15 retained
tensors read only through `Shape`. Next, represent those boundaries with shape
vectors, then repeat the three-route comparison and qualify full-decoder updates
and recovery for any selected execution policy. See the
{ref}`block-local measurements <complete-context-block-replay>`.
