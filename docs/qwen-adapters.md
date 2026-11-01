# LoRA adapters, chunked loss and saved activations

The [Mobius Qwen3 workload](qwen-sft.md) offers three execution paths: bound the output
projection by token chunks, save primal values for a separate backward graph,
or execute loss and adapter gradients together.

```bash
uv run python -m examples.huggingface.sft --mode chunked --save-activations --peft-roundtrip --fp64-audit --output-dir qwen_chunked
uv run python -m examples.huggingface.trainer --nf4 --output-dir nf4_trainer
uv run python -m examples.huggingface.benchmark --output qwen-benchmark.json
```

## Chunked output projection and loss

The transformer exports hidden states H. A separate ONNX module computes
`H_chunk @ W_head`, stable cross-entropy and the hidden-state gradient.
The wrapper processes 128 token rows at a time and feeds the assembled hidden
cotangent into the transformer's VJP. The output head remains frozen.

For N valid shifted target tokens, each chunk contributes its loss sum divided
by N. Padding labels are -100; an all-ignored batch produces zero loss and
zero gradients. Tests cover chunks of 1, 7 and 128 rows, a partial final chunk,
all adapter gradients and prediction through the full output head.

Training returns the scalar loss. Calling the wrapper without labels requests
full logits. A small SFTTrainer subclass uses Trainer's loss handling directly
for the chunked path, avoiding a full-logits pass for TRL token metrics.

The loss/gradient head session is created for training. The prediction session
is created on the first full-logits request and reused thereafter. Its serialized
weight snapshot is captured when the head is constructed, so predictions and
training use the same frozen weights. Reference audits that request full logits
also initialize this prediction session.

## Save or recompute

`prepare(save_activations=True)` cuts the flat VJP at the exact primal/derivative
boundary. Forward exports the required intermediate values; backward accepts
them as explicit ONNX inputs. Each autograd invocation owns its saved values.
Values sharing the returned prediction's storage are snapshotted, so caller
mutations cannot corrupt backward. Parameters retain PyTorch version checks.

The default recomputes those values inside the VJP. Both paths preserve the
low-rank matrix chain and share common products through CSE. In particular,
`U = G @ B.T` feeds both A's gradient and the low-rank contribution to X's
gradient. No dense A@B matrix is constructed. The runtime's optimized backward
exports only gradients, removing primal outputs unused by that execution.
Three independent updates from saved and recomputed graphs must be identical.

For each projection with T token rows, input width d, output width h and rank r,
saving `X @ A` needs `4*T*r` logical bytes in FP32 and avoids its `2*T*d*r`
recomputation FLOPs. The full cut also saves other backward dependencies;
the benchmark reports their logical extents and the process's measured RSS.

(qwen-long-context)=

## Measure longer contexts

The benchmark accepts `--sequence-length` and `--chunk-size`. For example,
measure a 256-token context with a 128-token output-head chunk:

```bash
ONNXRUNTIME_EP_MLX_NO_COMPILE=1 uv run python -m examples.huggingface.benchmark \
  --real --nf4 --mode chunked --provider mlx --library /path/to/libonnxruntime_mlx_ep.dylib \
  --batch-size 1 --sequence-length 256 --chunk-size 128 --warmup 10 --samples 100 \
  > chunked-256.json
```

Run `--mode joint` or `--mode pytorch` with the same sequence length and batch
for comparison. Each mode repeats the local corpus token prefix to fill the
requested context and records its input checksum and valid shifted-token count.
The tiny model accepts 2–64 tokens; the real model uses its configured context
limit.

For SFTTrainer, `--max-length` controls training and evaluation truncation:

```bash
ONNXRUNTIME_EP_MLX_NO_COMPILE=1 uv run python -m examples.huggingface.sft \
  --real --nf4 --mode joint --provider mlx --library /path/to/libonnxruntime_mlx_ep.dylib \
  --max-length 256 --train-file train.txt --validation-file validation.txt \
  --steps 30 --output-dir qwen-context-256
```

Each UTF-8 line supplies one training sequence. The checkpoint records context
length with data and tokenizer identities; resume uses the same `--max-length`.
A changed context is rejected before model construction. The default is 64.

`full_logits_bytes` and `chunk_logits_bytes` describe one logical FP32 logits
tensor at the selected dimensions. Process peak RSS includes construction,
runtime sessions, activations and optimizer state. Use measured memory and
step timings to choose the context and head chunk size.

### Recorded 256-token workload

Qwen3-0.6B uses the same frozen NF4 decoder values and FP32 adapters, gradients
and AdamW across four baseline processes and one matched on-demand-head rerun.
Each processes 255 shifted targets with batch one, ten warmup updates and 100
measured updates. ONNX paths use
MLX EP with compilation and ORT graph optimization disabled.

| Path | Step median / P95 (s) | Tokens/s | Peak RSS (GiB) |
| --- | ---: | ---: | ---: |
| PyTorch CPU | 1.065 / 1.818 | 239.46 | 5.31 |
| Complete ONNX VJP | 2.024 / 3.056 | 126.01 | 6.99 |
| Chunked head, eager prediction session | 2.041 / 3.199 | 124.93 | 7.98 |
| Chunked head, on-demand prediction session | 2.036 / 3.161 | 125.22 | 7.41 |
| Joint loss/gradient | 1.709 / 2.709 | 149.21 | 5.86 |

Loading the prediction session on demand removes one 593.5 MiB head-weight
instance. In the matched rerun, process peak RSS falls by 588.94 MiB, from
7.98 to 7.41 GiB. All 110 FP32 losses match the original chunked path bit for
bit. The step medians are 2.041 and 2.036 seconds. The
{download}`on-demand head report
<../examples/huggingface/reports/qwen3-nf4-context-256-lazy-head-m1max.json>`
records both runs, retained-session initializer sizes and source/runtime hashes.

The chunked head uses a 74.19 MiB logical logits tensor per full chunk; the
complete 256-token logits tensor is 148.38 MiB. Session weights, activation
lifetimes and runtime allocations determine the complete process footprint.
All paths reduce the repeated-input training loss, from about 1.742 to
0.0126–0.0187. The [No Robots experiment](training-quality.md) measures held-out
training quality on instruction data.

The {download}`256-token report
<../examples/huggingface/reports/qwen3-nf4-context-256-m1max.json>` contains all
losses, component timings, initialization, session initializer sizes, node counts,
source/library hashes and physical-footprint observations. PyTorch stores the
reconstructed FP32 base; ONNX stores packed NF4 decoder tensors. Peak RSS covers
model construction, runtime sessions, arrays and optimizer state.

## Joint loss and adapter gradients

```bash
uv run python -m examples.huggingface.sft --mode joint --nf4 --steps 100 --peft-roundtrip --output-dir qwen_joint
uv run python -m examples.huggingface.benchmark --mode joint --nf4 --warmup 10 --samples 100
```

The joint path adds causal cross-entropy to the forward graph, then asks
`onnax.grad` for every adapter parameter's gradient. ONNXScript optimizes the
combined graph. One persistent ONNX session returns the summed token loss
and all A/B gradients; intermediate activations stay inside that execution.

For valid shifted targets $\mathcal V$, the exported scalar is

$$
L_{\mathrm{sum}}=\sum_{t\in\mathcal V}-\log p(y_t\mid x_{\le t}).
$$

The PyTorch wrapper divides this sum by the effective token count. Backward
multiplies the returned gradients by the upstream scalar, so Trainer can
normalize accumulated microbatches with different lengths. Ignored labels
(`-100`) contribute zero. A sequence with no shifted targets also returns
zero loss and gradients.

Trainer owns the adapter tensors, AdamW state, scheduler and checkpoints.
Calling the wrapper without labels creates a temporary forward session for
full-logit prediction. Training continues to reuse the joint session.

The export directory contains `loss_gradient.onnx` and its external data,
`adapter.npz`, and `interface.json`. The versioned interface maps each parameter
to its gradient output, shape and dtype and records the frozen-weight hashes.
A bound `forward.onnx` restores the trained adapter values for inference.
Checkpoint loading checks the frozen base identity before restoring parameters.
The replay verifier reloads both forward and loss/gradient artifacts on ORT CPU
and compares every output byte after fresh-process checkpoint recovery. It also
checks adapter values, AdamW moments and scheduler state. Replay disables ORT
memory-reuse planning and records this allocation setting in its report. The
[reuse-planner audit](qwen-runtime-audit.md) explains the initialization hotspot.

Use the numerical audit to compare every update with the corresponding
Transformers model. It also evaluates the reference at the ONNX parameters,
separating local gradient differences from accumulated optimizer drift:

```bash
uv run python -m examples.huggingface.mlx_validation --mode joint --provider cpu --stable-reference --steps 100 --output joint-cpu-audit.json
uv run python -m examples.huggingface.mlx_validation --mode joint --library /path/to/libonnxruntime_mlx_ep.dylib --stable-reference --steps 100 --output joint-mlx-audit.json
```

Add `--real` for the pinned Qwen3-0.6B model. `--segmented --segment-nodes 1024`
bounds graph size per runtime session. Both paths use the same parameter
mapping, FP32 arithmetic and `rtol=1e-4, atol=1e-6` checks. The CLI saves all
per-step failures and exits nonzero when a check fails.

The 100-step tiny NF4 audit at the original tolerance gives:

| Check | ORT CPU | Native MLX |
| --- | --- | --- |
| Same-parameter logits, loss and all gradients | All 100 pass | All 100 pass |
| Independently updated parameters | All 100 pass | Steps 1–94 pass; 95–100 fail |
| Independent trajectory loss | All 100 pass | All 100 pass |
| First independent gradient failure | Step 87 | Step 83 |
| First independent logit failure | Step 93 | Step 92 |

Frozen buffers remain unchanged. These results separate agreement at identical
parameters from differences accumulated along two training trajectories. The
[CPU report](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/tiny-nf4-joint-cpu-100.json)
and [MLX report](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/tiny-nf4-joint-mlx-100.json)
retain every step and failed tensor. The MLX parameter divergence remains part
of the FP32 qualification work.

The benchmark reports joint loss/gradient execution, PyTorch gradient scaling,
AdamW and complete-step median/P95, plus process peak RSS and logical session
input/output bytes. It constructs the selected provider's sessions directly.
Run each path in its own process with the same provider, weights, batch and
thread settings to compare time and memory.

A native MLX run on Apple M1 Max used tiny NF4 Qwen3, batch four, sequence
length 64 and 252 valid targets per update. After 10 warmups, 100 measured
updates took 87.14 ms median and 95.62 ms P95; process peak RSS was 633.5 MiB.
The loss decreased from 5.55 to 5.09 over all 110 updates. The session accepted
15,360 logical input bytes and returned 7,172 bytes, including 7,168 bytes of
adapter gradients. The [measurement JSON](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/tiny-qwen-joint-native-m1max.json)
records the source commit, native library digest, versions and timing scope.

### Real Qwen3-0.6B training measurements

The matched execution-path comparison uses Qwen3-0.6B with NF4 decoder values,
FP32 computation and AdamW on Apple M1 Max. Every path uses batch one,
sequence length 64 and 63 valid shifted targets per update. Ten warmup updates
precede 100 measured updates. Each path runs in a separate process; all native
ONNX paths use the same MLX EP binary with compilation and ORT optimization
disabled. PyTorch CPU stores reconstructed FP32 weights; ONNX stores packed NF4.

```bash
ONNXRUNTIME_EP_MLX_NO_COMPILE=1 uv run python -m examples.huggingface.benchmark \
  --mode optimized --real --nf4 --batch-size 1 --provider mlx \
  --library /path/to/libonnxruntime_mlx_ep.dylib --warmup 10 --samples 100
```

Use `--mode pytorch`, `raw`, `chunked` or `joint` for the other paths.

| Path | Step median / P95 (s) | Valid tokens/s | Peak process RSS (GiB) |
| --- | ---: | ---: | ---: |
| PyTorch CPU, reconstructed FP32 base | 0.359 / 0.375 | 175.38 | 4.72 |
| ONNX raw VJP | 1.631 / 2.572 | 38.64 | 6.97 |
| ONNXScript-optimized VJP | 1.490 / 2.417 | 42.28 | 6.82 |
| Chunked head, recomputed body | 1.512 / 2.520 | 41.67 | 7.85 |
| Joint loss/gradients | 1.366 / 2.282 | 46.12 | 5.85 |

ONNXScript cleanup, common-expression reuse and gradient-only backward outputs
reduce the executed VJP from 46,024 nodes to 32,649. Raw and optimized paths
produce **bitwise-identical losses at all 110 updates**. Optimization lowers
median step time by 8.6% in this run. Joint execution has the lowest ONNX step
time and peak RSS; PyTorch CPU is fastest overall in this FP32 configuration.

All paths start near loss 4.562 and finish below 0.0006 on the repeated benchmark
text. PyTorch/raw/optimized use standard causal cross-entropy, while chunked and
joint use the target-excluded stable loss with the same mean normalization.
The chunked head processes 128 token rows per call; this batch fits in one chunk.

Peak RSS includes model construction, runtime, arrays and optimizer allocations.
Physical footprint is also sampled every 15 seconds. Public runtime thread
counts are one; BLAS environment variables are unset. Initialization takes
12.995 s for PyTorch and 156.192–161.122 s for the ONNX paths, including graph
construction and differentiation. Reports separate execution, input-view
preparation and AdamW timing, and record session initializer bytes and node counts.

The {download}`matched execution report
<../examples/huggingface/reports/qwen3-nf4-execution-matrix-m1max.json>` preserves
all five loss traces, component measurements, source/runtime hashes and process
journals. The earlier [saved-activation experiment
report](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/qwen3-nf4-saved-eager-m1max.json)
records the full primal/derivative cut, including 1.30 GiB of logical residual
arrays. The joint module returns 2.19 MiB of adapter gradients across that boundary.

### Real joint SFT and checkpoint recovery

```bash
export ONNXRUNTIME_EP_MLX_NO_COMPILE=1
uv run python -m examples.huggingface.sft --real --nf4 --mode joint --provider mlx \
  --library /path/to/libonnxruntime_mlx_ep.dylib --steps 30 \
  --diagnose-rounding --stable-reference --peft-roundtrip --output-dir joint-sft
uv run python -m examples.huggingface.sft --real --nf4 --mode joint --provider mlx \
  --library /path/to/libonnxruntime_mlx_ep.dylib --steps 30 \
  --diagnose-rounding --stable-reference --resume joint-sft/checkpoint-15 \
  --output-dir joint-resumed
uv run python -m examples.huggingface.replay --uninterrupted joint-sft \
  --resumed joint-resumed --step 30 --output joint-replay.json
```

The Qwen3-0.6B run uses microbatch one, four accumulated microbatches per update
and 30 Trainer updates. Validation loss falls from 7.7506 to 5.6323. All 532 frozen
initializer hashes stay unchanged. Recovery from checkpoint 15 reproduces the
112 adapters, optimizer moments, scheduler and 113 checkpoint tensors exactly.
Loaded forward outputs, summed loss and all 112 parameter gradients agree byte
for byte between the uninterrupted and resumed exports. Both exports pass the
full ONNX checker.

The stable FP32 loss/head reference passes the loss and all adapter-gradient
checks at the initial and final states. PEFT format exchange preserves all
112 adapter tensors exactly. Raw Transformers logits/gradient comparisons retain
failures at the original tolerance. With the stable reference, the PEFT
fresh-Adam update comparison exceeds parameter tolerance on 92 tensors. `--diagnose-rounding` records those comparisons
alongside the stable reference controls.

The [training report](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/qwen3-nf4-joint-sft-m1max.json)
and [recovery report](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/qwen3-nf4-joint-recovery-m1max.json)
retain commands, backend settings, source revisions, precision results and loaded
module verification. The recovery report also records disabled ORT memory-reuse
planning for CPU replay.

## PEFT files

The SFT command writes `adapter/adapter_model.safetensors`, `adapter_config.json`
and a versioned `adapter_mapping.json`. Each entry maps a source Qwen module to
its explicit ONNX A/B inputs. ONNX A `[in,rank]` and B `[rank,out]` are transposed
to PEFT Linear weight layouts. Rank, alpha and scaling remain unchanged.

`export_adapter(model, directory)` and `import_adapter(model, directory)` in
`examples.huggingface.peft_adapter` exchange standard q/v LoRA adapters. Import
validates the complete target/name/shape/dtype mapping before changing any
parameter. The example uses one adapter, uniform rank/alpha and zero dropout.

## Measurements

The benchmark starts a separate process for PyTorch, raw ONNX, optimized ONNX,
saved activations, chunked loss and joint loss/gradients. Defaults are batch four, length 64, FP32,
one CPU thread, 10 warmups and 100 measured AdamW updates. It reports valid
tokens/s, forward/loss, backward, AdamW, complete-step median/P95 and peak RSS.
RSS includes construction, runtime and optimizer allocations. CPU input-view
conversion is recorded separately; snapshots and parameter/output views are
included in the execution boundaries. ORT graph optimization is disabled.

Use `--real --batch-size 1 --samples 5 --warmup 3` for a short 0.6B measurement.
The JSON records the actual sample count and hardware/software versions.

### Recorded tiny Qwen3 CPU comparison

Apple M1 Max, batch 4, length 64, vocabulary 256, 252 valid shifted targets per
update; one CPU thread, 10 warmups and 100 measured updates:

| Path | Step median (ms) | P95 (ms) | Peak process RSS (MiB) |
| --- | ---: | ---: | ---: |
| PyTorch | 2.698 | 3.180 | 536.9 |
| Raw ONNX | 7.995 | 8.270 | 581.4 |
| Optimized ONNX | 7.941 | 8.528 | 591.7 |
| Saved activations | 6.724 | 6.809 | 596.7 |
| Chunked loss | 7.511 | 7.836 | 592.1 |

PyTorch is fastest in this CPU configuration. Saving activations reduces the
ONNX step time and retains 8.26 MiB of logical residual arrays. The small
256-entry vocabulary makes output-head storage a small part of this workload.
The JSON in `examples/huggingface/reports/tiny-qwen-benchmark.json` includes
node counts, parameter bytes, boundary timing, sample counts and versions.
