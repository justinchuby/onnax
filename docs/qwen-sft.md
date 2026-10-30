# Fine-tune a Mobius Qwen3 ONNX model with TRL

Mobius constructs Qwen3 directly in ONNX and loads its Hugging Face weights.
onnax inserts LoRA inputs and generates reverse-mode graphs. TRL SFTTrainer
provides tokenization, batches, AdamW, scheduling and checkpoints.

```bash
uv sync --group dev --group training --group huggingface
uv run python -m examples.huggingface.sft --output-dir qwen_tiny
uv run python -m examples.huggingface.sft --real --diagnose-rounding --output-dir qwen_real
```

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

The [FP32 audit](softmax-precision.md#real-model-training-and-recovery) records
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
