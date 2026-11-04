# Fine-tune a Mobius Qwen3 ONNX model with TRL

Mobius constructs Qwen3 directly in ONNX and loads its Hugging Face weights.
onnax inserts LoRA inputs and generates reverse-mode graphs. TRL SFTTrainer
provides tokenization, batches, AdamW, scheduling and checkpoints.

```bash
uv sync --group dev --group training --group huggingface
uv run python -m examples.huggingface.sft --output-dir qwen_tiny
uv run python -m examples.huggingface.sft --real --output-dir qwen_real
```

## Train on your own text

Provide separate UTF-8 training and validation files, with one sequence per
nonempty line:

```bash
uv run python -m examples.huggingface.sft --mode joint --nf4 \
  --train-file train.txt --validation-file validation.txt \
  --output-dir qwen_custom
```

TRL handles batches, AdamW and checkpoints. Sequences use right padding;
`--max-length` sets truncation, with a default of 64 tokens. Loss covers all
shifted nonpadding text tokens. The tiny model accepts up to 64 tokens and builds
its tokenizer from the training file. `--real` uses the pinned Qwen3-0.6B model
and tokenizer and accepts longer contexts within the model's configured limit.
The [context benchmark](qwen-long-context) measures time
and memory for a selected length and head chunk size.

Resume with the same data options and `--resume qwen_custom/checkpoint-50`.
Each checkpoint includes `onnax_data.json`; the CLI checks its training,
validation and tokenizer identities and context length before rebuilding the graph.
Pass the same `--max-length` when resuming. Reports record
the same identities, and the final ONNX export includes the tokenizer.

The [pinned instruction-corpus preparation](pinned-instruction-corpus)
provides a reproducible source of training and official-test examples.

## Retain fused operators

Use `--fused-ops --provider mlx` to retain Attention, RMSNormalization,
RotaryEmbedding and Swish in the training graph. onnax differentiates these
operators directly, and MLX EP selects its SDPA, RMSNorm and RoPE implementations.
See [fused operator derivatives](fused-autodiff.md) for the equations, execution
requirements and measured results.

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

The wrapper reuses ONNX forward and derivative sessions. ONNXScript folds
constants and shares repeated low-rank products. Large graphs store tensor
payloads in external-data files.

The tiny run trains for 100 updates. The real run uses 30 updates, microbatch
one, four accumulation steps and a 64-token maximum length:

```bash
uv run python -m examples.huggingface.sft --real --output-dir qwen_real
uv run python -m examples.huggingface.sft --real --output-dir qwen_real \
  --resume qwen_real/checkpoint-15
```

Resume validates the frozen base, adapter mapping, tokenizer, data and context
length. Keep the same data and precision options. Reports and final ONNX exports
are written into the output directory.

Use `--mode joint` for a combined loss/gradient graph, or `--mode chunked` to
execute the frozen output head in bounded token chunks. Joint-mode evaluation
uses a reusable loss-only session. The [evaluation report](loss-only-evaluation)
compares both evaluation paths.

## Supervised rows in the chunked head

In `--mode chunked`, hidden row $H_{b,i}$ predicts label $t_{b,i+1}$.
The scheduler gathers rows with $t_{b,i+1}\ne -100$, then runs the frozen
projection and cross entropy in chunks of at most `--chunk-size` rows:

$$
L_{\mathrm{sum}}=\sum_{(b,i):t_{b,i+1}\ne-100}
  -\log\operatorname{softmax}(H_{b,i}W)_{t_{b,i+1}}.
$$

It scatters each hidden gradient back to its sequence position. Prompt masking
and right padding therefore reduce head work. The decoder still processes the
complete context: an ignored prompt token can receive a gradient through a
later token's attention. A batch with no valid shifted labels returns zero loss
and hidden gradient without executing the head session.

The head returns a loss sum. The wrapper divides by the effective token count,
including the count supplied by Trainer across unequal accumulation batches.
Exported heads use the same scheduling when loaded, and reuse their sessions
across steps.

`ChunkedHead.last_profile` records causal and valid row counts, executed chunks,
gather/scatter time, session execution time, boundary conversion time, and
dynamic feed/result bytes for the latest call. Session execution includes
runtime data transfers; feed/result bytes exclude frozen weights. These fields
help attribute head costs alongside complete training-step measurements.

The [head measurements](sparse-supervision-measurements)
compare dense and gathered execution at different supervision densities.

## Precision and memory

Start with FP32. For packed frozen weights and explicit BF16 projections, follow
[NF4 Qwen training](qwen-nf4.md) and the [MLX precision guide](mlx-precision.md).
[Adapter memory policies](qwen-adapters.md) cover saved activations and decoder
checkpointing; [long-context training](qwen-context-training.md) provides capacity
and recovery commands.

## Validation

Offline tests compare all q/v adapter gradients against a separate PyTorch
LoRA implementation at zero and nonzero B. They check valid-token logits,
causality, ignored padding, changed position IDs, dynamic sequence lengths,
unequal valid-token counts and all-ignored targets. Padding-query outputs are
excluded from cross-model comparisons; padding keys and targets remain masked.

`report.json` records the exact model revision, update count and validation
loss. Final adapter arrays and a standalone `export/forward.onnx` with external
weight data are saved alongside the tokenizer.

Detailed reference audits, rounding investigations and recorded runs are in
[Qwen3 qualification and measurements](qwen-sft-qualification.md). Cross-runtime
loss and task metrics are collected in [training quality](training-quality.md).
