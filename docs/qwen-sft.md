# Fine-tune a Mobius Qwen3 ONNX model with TRL

Mobius constructs Qwen3 directly in ONNX and loads its Hugging Face weights.
onnax inserts LoRA inputs and generates reverse-mode graphs. TRL SFTTrainer
provides tokenization, batches, AdamW, scheduling and checkpoints.

```bash
uv sync --group dev --group training --group huggingface
uv run python -m examples.huggingface.sft --output-dir qwen_tiny
uv run python -m examples.huggingface.sft --real --diagnose-rounding --output-dir qwen_real
```

Compare held-out loss across runtimes and independent seeds with the
[training-quality experiment](training-quality.md). Its paired runs match the
frozen base, initial adapters and data schedule.

## Model and training graph

The offline model has two layers, hidden size 64, intermediate size 128,
four query heads, two KV heads, head dimension 16 and a 256-entry vocabulary.
A local WordLevel tokenizer and the checked-in corpus make this run offline.
The real workload uses `Qwen/Qwen3-0.6B` at revision
`c1899de289a04d12100db370d81485cdf75e47ca`, including its tokenizer.

Both use the same Mobius Qwen3 module and cache-free training task. The task
returns every token's logits and final hidden state. Its inputs are token IDs,
attention mask and position IDs; it has no past/present KV buffers. Mobius's
Attention decomposition and explicit RMSNorm, rotary and Swish decompositions
provide standard operations for differentiation. Boolean attention masks are
converted to zero/negative additive biases before the pinned Mobius pass.
A regression compares every fused/decomposed output with padded batches.
The source projection names
survive graph construction and are recorded in the adapter mapping.

Each decoder layer trains rank-four adapters on `q_proj` and `v_proj`:

\[
Y=XW+\tfrac14(XA)B.
\]

The base, embeddings and output head stay frozen. The example uses FP32,
right padding, complete text loss and zero dropout. Position IDs come from the
mask's cumulative token count. Logits at position t predict the label at t+1;
padding labels are -100. The loss divides by the number of valid shifted
targets, including across gradient accumulation microbatches.

## Runtime and recovery

The [autograd bridge](huggingface.md) reuses ONNX forward and VJP sessions.
Large models use external-data ONNX files to avoid protobuf size limits.
The derivative optimizer uses ONNXScript constant folding followed by IR
cleanup and common-subexpression elimination, sharing repeated low-rank
products without a full-graph pattern search for every candidate node.

The tiny run trains for 100 updates. The real run uses 30 updates, microbatch
one, four accumulation steps and a 64-token maximum length. Resume with
`--resume qwen_real/checkpoint-15`. The wrapper recreates the base and validates
its hashes and parameter mapping while loading adapter tensors.

TRL is pinned to 0.24.0 for its standard model-loss interface. The model
computes token loss outside the ONNX forward; gradients enter the ONNX VJP
through the logits cotangent. Separate token-loss ONNX modules support the
training-specific chunked execution path.

## BF16 projection experiment

Run the joint loss/gradient graph with BF16 matrix projections:

```bash
uv run python -m examples.huggingface.sft --mode joint \
  --projection-precision bf16 --provider reference --steps 100 \
  --stable-reference --diagnose-rounding --peft-roundtrip --output-dir qwen_bf16
```

Projection operands and results round to BF16. Attention, normalization, loss,
master parameters and Adam state use FP32. Registered derivatives compute FP32
straight-through adjoints using the rounded operands. The exported interface
records this policy; checkpoints validate it when loading.

The offline ReferenceEvaluator run reduced validation loss from 5.5441 to
5.4015 in 100 updates. Restoring checkpoint 50 in a new process reproduced
the final parameters, optimizer state and loaded ONNX outputs byte for byte.
The final independent loss and adapter-gradient checks passed. Final logits
had 115 tolerance violations, and the separate five-step independent update
test still fails at step five. Results are recorded in
`reports/qwen-bf16-tiny-sft-recovery.json` and
`reports/qwen-bf16-first-divergence.json`.

The native MLX run reduced validation loss from 5.5441 to 5.4012 and passed
the initial and final independent logits, loss and adapter-gradient checks.
Its checkpoint-50 resume also reproduced all final state and loaded ONNX
outputs byte for byte. Running either set of final parameters on both ONNX
executors produced identical logits on the audit sample. The remaining
five-step update discrepancy is tracked separately from checkpoint recovery.

On Qwen3-0.6B, 30 native MLX updates reduced validation loss from 7.4406 to
6.2499. Checkpoint-15 recovery reproduced all 112 adapters, Adam/scheduler
state and loaded ONNX forward/loss/gradient outputs byte for byte. Frozen
weights and the PEFT adapter format round trip also matched exactly.

Independent forward, loss and gradient comparisons exceed the original
tolerance on this model. The first frozen q projection already exposes four
BF16 output differences between PyTorch and MLX with identical operands;
ReferenceEvaluator matches MLX. [Issue #116](https://github.com/justinchuby/onnax/issues/116)
records FP32 accumulation and BF16 rounding near midpoint boundaries. The
complete training and recovery evidence is in
`reports/qwen3-bf16-joint-sft-m1max.json`.

For a captured projection, compare native BF16 multiplication with FP32
accumulation followed by BF16 rounding:

```bash
uv run python -m examples.huggingface.bf16_accumulation \
  --capture projection.npz --output accumulation.json
```

The capture contains FP32 `reference_input`, `weight` and `mlx_output` arrays.
Both products use the same BF16-rounded operands. On the first real-model
projection, the FP32-accumulation control matches all 22,528 captured MLX outputs;
the native PyTorch BF16 product differs at four entries. The command records
shapes, the capture hash, thread count and comparison errors.

Applying this accumulation control throughout the real NF4/BF16 transformer
still leaves 56 failing gradient tensors at initialization and 112 after
30 updates. Logit maximum differences remain 0.125 and 0.3125 respectively.
The control changes loss and gradient errors in both directions; a common
accumulation policy alone does not align the full model. The shared-state
results, original tolerances and experiment script are in
`reports/qwen3-nf4-bf16-accumulation-control-m1max.json`.
The [local derivative audit](qwen-runtime-audit.md) then checks identical
intermediate inputs and incoming cotangents. All 3,522 local input contributions
pass at zero B, nonzero B and the trained step-30 adapter state. Pairwise FP32 accumulation in the independent frozen-head adjoint aligns all
112 composed adapter gradients in each state at the original tolerance. A
scalar dot-product audit identifies the ordinary CPU reference's long-reduction
error. SFT and PEFT stable controls preserve BF16 forward computation while
using this pairwise adjoint. Independent full-model forward and loss comparisons
remain recorded in the reports.

A separate tiny PEFT check fills B with 0.01. Logits and loss match exactly and
all gradients pass; the stable fresh AdamW update has a maximum parameter gap
of 3.89e-6 and exceeds the original tolerance. The
{download}`nonzero-B report <../examples/huggingface/reports/tiny-nf4-bf16-peft-nonzero-b-update.json>`
records both ordinary and stable comparisons. The preservation regression checks
that enabling the stable control leaves the original report and parameter bytes
unchanged.

At the two failing coordinates, gradients around 3.4e-10 differ by 1.39e-11.
For zero initial moments, Adam's first update is
$\Delta\theta=-\eta g/(|g|+\epsilon)$. With the original learning rate 0.003
and epsilon 1e-8, these gradients produce a parameter gap of 3.88e-6. An
isolated FP32 scalar AdamW step reproduces both full-tensor updated values
exactly. The {download}`conditioning report <../examples/huggingface/reports/tiny-nf4-bf16-adam-conditioning.json>`
records the gradients, initial parameters and scalar replay.

Inspect projection inputs, outputs and incoming gradients at the same parameter
state with:

```bash
uv run python -m examples.huggingface.bf16_audit --output projection-audit.json
```

This runs one AdamW update, copies its parameters into the independent reference,
and audits the second gradient evaluation. All 23 rounded projection inputs and
outputs match exactly. Incoming gradients meet the original tolerance, with
FP32 arithmetic differences recorded for each projection.

The measurement CLI accepts the same precision policy. Run each path in a
separate process, with the same batch size and sequence length:

```bash
uv run python -m examples.huggingface.benchmark --mode pytorch \
  --projection-precision bf16 --real --batch-size 1 --warmup 10 --samples 100
ONNXRUNTIME_EP_MLX_NO_COMPILE=1 uv run python -m examples.huggingface.benchmark \
  --mode joint --provider mlx --projection-precision bf16 --real \
  --batch-size 1 --warmup 10 --samples 100
```

Reports include the projection policy, step median/P95, tokens per second,
initialization time and peak process RSS. Both paths retain FP32 base storage
and FP32 surrogate adjoints.

On an M1 Max, the real model with batch 1, sequence length 64 and 63 shifted
targets produced the following measurements (10 warmups, 100 measured steps):

| Path | Step median / P95 | Tokens/s | Peak process RSS |
| --- | ---: | ---: | ---: |
| PyTorch CPU BF16 | 3.071 / 3.135 s | 20.52 | 6.06 GiB |
| MLX joint BF16 | 1.399 / 1.466 s | 45.04 | 10.36 GiB |

Session/model initialization took 0.414 s for PyTorch and 35.432 s for the
ONNX construction, differentiation, export and MLX session path. RSS includes
construction, runtime, arrays and Adam state. PyTorch uses its standard causal
CE; ONNX uses the stable target-excluded CE, with the same valid-token mean.
The report is `reports/qwen3-bf16-matched-m1max.json`. Independent numerical
qualification remains open as described above.

For native execution select `--provider mlx`. The tested macOS ORT CPU backend
has no BF16 MatMul kernel. The default experiment uses FP32 weight storage.
Add `--nf4` to combine BF16 products with
[packed frozen decoder weights](qwen-nf4.md); embeddings and the head retain
FP32 storage.

## Validation

Offline tests compare all q/v adapter gradients against a separate PyTorch
LoRA implementation at zero and nonzero B. They check valid-token logits,
causality, ignored padding, changed position IDs, dynamic sequence lengths,
unequal valid-token counts and all-ignored targets. Padding-query outputs are
excluded from cross-model comparisons; padding keys and targets remain masked.

`report.json` records the exact model revision, update count and validation
loss. Final adapter arrays and a standalone `export/forward.onnx` with external
weight data are saved alongside the tokenizer.

## Recorded baseline Qwen3-0.6B run

The {ref}`FP32 audit <qwen-stable-recovery>` records
the current stable-Swish run and its checkpoint recovery verification.

On an Apple M1 Max with 32 GiB RAM, 30 CPU FP32 updates reduced validation
loss from 7.436 to 6.265. A separate process resumed update 15 and reproduced
all final adapter tensors byte for byte. The resumed update median was 5.48 s,
with four microbatches per update; AdamW accounted for a median 6.40 ms. The isolated process peaked at
15.51 GiB RSS, including model construction, runtime and recovery.

The original `rtol=1e-4, atol=1e-6` audit found FP32 differences: the initial
logits had relative L2 error 3.05e-6, and a subset of initial adapter gradient
elements exceeded the elementwise threshold. At the final state all adapter
gradients passed that threshold; some near-zero logits still exceeded it.
The checked-in `examples/huggingface/reports/qwen3-0.6b.json` retains those
failures, per-parameter counts and the recovery timing scope. FP64 diagnosis
separates arithmetic rounding from the graph and derivative semantics.

## Independent precision diagnosis

The real model also passes a separate FP64 functional Qwen3 oracle at both
initial and final adapter states. Maximum logit error is 1.99e-13 and maximum
adapter-gradient error is 3.79e-15. This comparison uses the same stored RoPE
and scalar constants, and evaluates the transformer independently in PyTorch.

The FP32 report retains its original elementwise failures. `--diagnose-rounding`
records logits and per-parameter gradient violations and continues training;
it keeps the loss comparison and all original tolerances. Without this flag,
the first failed comparison raises. FP64 results qualify graph/derivative
semantics; FP32 elementwise agreement remains a separate acceptance result.

The real PEFT comparison passes loss and final-gradient checks. Its first
independent AdamW update exceeds the parameter tolerance for some entries;
the report preserves the per-parameter failures. The offline tiny model passes the
full bidirectional adapter, gradient and update regression.

(qwen-operator-audit)=

## Compare operators on the same inputs

The layer audit exposes each projection, RMSNorm and Swish input and replays that
operator independently in PyTorch. This separates local arithmetic differences
from errors accumulated across decoder layers. Selected matrix outputs also use
`math.fsum` over the FP32 operands as an independent dot-product reference.

```bash
uv run python -m examples.huggingface.layer_audit --real --mlx \
  --library /path/to/libonnxruntime_mlx_ep.dylib --output mlx-layers.json
```

Omit `--mlx` and `--library` to audit ORT CPU. MLX requires complete provider
assignment, disables CPU fallback and reads the large probe model from external
data. Both paths disable ORT graph optimization. Reports record the provider,
input token IDs, local comparison errors and independent dot-product errors;
development builds also record the dylib SHA256.

The M1 Max Qwen3-0.6B FP32 audit covers 197 MatMul, 113 RMSNorm and 28 Swish
operations. RMSNorm and Swish pass `rtol=1e-4, atol=1e-6` on identical inputs.
MatMul has 694 elementwise differences against PyTorch across 31,744,000 outputs.
For each matrix, the report checks its largest tolerance-scaled difference with
an independent dot product; either runtime can have the smaller error. The
end-to-end logits retain their elementwise comparison failures in the report.
