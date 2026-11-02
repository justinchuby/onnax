# Fine-tune Mobius decoder families

The decoder task adapter connects Mobius model construction to
[`CheckpointedModule`](checkpointing.md), TRL SFTTrainer and PEFT. The package
executes and differentiates ONNX blocks; the task adapter handles source-model
names, token inputs, loss and adapter files.

The offline workload uses native Transformers and Mobius implementations of
three architectures:

| Family | Source attention projections | Feed-forward projections | LoRA targets |
| --- | --- | --- | --- |
| Qwen3 | Separate Q/K/V, with Q/K normalization | Separate gate/up | `q_proj`, `v_proj` |
| Phi-3 | Fused QKV, split by Mobius | Fused gate/up | `qkv_proj` |
| Qwen3.5 | Gated DeltaNet plus full attention with query/output gating | Separate gate/up | `linear_attn.in_proj_qkv`, `self_attn.q_proj`, `self_attn.v_proj` |

Qwen3 and Phi-3 have two layers, hidden width 64, intermediate width 128, four query
heads, two KV heads, head width 16 and vocabulary size 256. Fixed seeds and a
local tokenizer make training reproducible offline. The Phi-3 head is untied;
Qwen3 shares its embedding and output weights. Qwen3.5 uses four layers in a
3:1 linear/full-attention pattern with the same hidden/intermediate widths and
vocabulary size. It ties its embedding/head and adds ten trainable A/B tensors.

## Preserve the fused source adapter

A Phi-3 source Linear computes a concatenated projection

\[
[Q,K,V] = XW_{qkv}^T + \frac{\alpha}{r}(XA)B.
\]

Mobius splits the frozen source weight into three ONNX MatMuls. The adapter
keeps one A of shape `[hidden, rank]` and one B of shape
`[rank, q_width + k_width + v_width]`. It computes `XA` once and splits B along
its output dimension. Each projection receives its corresponding low-rank
contribution. The full adapter product AB is never materialized.

`mapping` records the original fused `qkv_proj.weight`, ONNX projection names,
column ranges, parameter names, rank and alpha. PEFT therefore sees the same
fused Linear that the source model uses. ONNX A/B transpose directly to PEFT's
`lora_A.weight` and `lora_B.weight`. Adapter targets come from this mapping.

The shared graph helpers build full-sequence, cache-free logits and hidden
states, preserve decoder boundaries through rewriting, and decompose standard
normalization, RoPE and attention operations. `--fused-ops` retains the fused
primitives and their direct onnax VJPs. Qwen3.5 keeps these boundaries by default,
including its recurrent functions and partial RoPE.

The family descriptor selects source config/model classes, projection suffixes,
and optional reference/graph builders. The shared wrapper handles token loss,
training and export. A source config type may differ from the CLI family name:
`qwen35` maps to Transformers `qwen3_5_text`. Loading checks the registered
config type and adapter mapping.

Decoder checkpoints begin at the input normalization operand. This identifies
Qwen3.5's hidden input after the normalization weight-offset computation. A
pure full-attention block may retain registered recurrent-function definitions;
those definitions keep their call-level derivative boundary and recursive
randomness checks.

## Check the independent reference

```sh
python -m examples.huggingface.decoder_sft \
  --family phi3 --audit-only --output-dir /tmp/phi3-audit
```

Select `--family qwen3` or `--family qwen35` for the same audit on those models. The audit checks logits,
causal token loss, all adapter gradients and six AdamW updates against the native
Transformers model with independent PyTorch LoRA modules. It covers zero-initialized
B, the first effective B gradient, subsequent A gradients, dynamic shapes and
explicit padding positions. Frozen source weights retain their byte hashes.
It then exports and loads the ONNX task and exchanges adapters in both directions
through PEFT, including another gradient and update comparison.

Training and audits use Adam epsilon `1e-5` and compare values at
`rtol=1e-4, atol=1e-6`. The [Adam conditioning example](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/phi3-fused-adam-conditioning.json)
includes a scalar replay of small-gradient updates.

## Train and restore with TRL

```sh
python -m examples.huggingface.decoder_sft \
  --family phi3 --steps 30 --output-dir /tmp/phi3-training
```

SFTTrainer loads the version-controlled text corpus through datasets. It owns
AdamW, the constant scheduler, two-microbatch gradient accumulation and checkpoint
storage. The ONNX module reuses its sessions through training and evaluation. The wrapper
sets `loss_type="ForCausalLM"` so Trainer counts prediction targets after the
causal shift, including when microbatches have different lengths.
Right padding uses `-100` labels. The external loss shifts labels by one token,
sums valid token losses and divides by the effective token count supplied for
the accumulation group. An entirely ignored batch has zero loss and gradients.

The CLI records fixed training-set and validation losses before and after training.
Its elapsed time covers evaluation, training and export after construction.

For Qwen3.5, run the same text-training path for 100 updates:

```sh
python -m examples.huggingface.decoder_sft \
  --family qwen35 --steps 100 --output-dir /tmp/qwen35-sft
```

This path uses the version-controlled text corpus and local tokenizer. The
[recurrent tutorial](qwen35-training.md) explains the model update and the
standalone token-pattern qualification.

Resume into another output directory:

```sh
python -m examples.huggingface.decoder_sft \
  --family phi3 --steps 30 \
  --resume /tmp/phi3-training/checkpoint-15 \
  --output-dir /tmp/phi3-resumed
```

The regression suite compares fresh-process recovery against continuous training,
including final parameters, loss and AdamW state. A different decoder family is
rejected before graph construction; graph and parameter identities also participate
in checkpoint validation.

The export contains `config.json`, `task.json`, and an `execution/` bundle with the
source graph, bound inference graph, parameter arrays and block forward/VJP graphs.
`OnnxDecoder.load(export_directory)` executes the saved derivatives. The separate
`adapter/` directory contains standard PEFT configuration and safetensors.

## Execute on MLX

```sh
ONNXRUNTIME_EP_MLX_NO_COMPILE=1 \
python -m examples.huggingface.decoder_sft \
  --family phi3 --fused-ops --provider mlx \
  --library /path/to/libonnxruntime_mlx_ep.dylib \
  --output-dir /tmp/phi3-mlx-training
```

The session factory selects MLX execution while PyTorch retains FP32 adapters,
gradients and optimizer state at the NumPy boundary. Native tests cover reference
checks and PEFT exchange for all three families. The existing
[Qwen long-context workload](qwen-context-training.md) supplies NF4, mixed-precision
projections and chunked output-head execution. This new family workload uses FP32
weights and a full-logit loss path.

## Recorded workload results

On Apple M1 Max, both CPU ORT and MLX completed 30 SFT updates with the same
source-graph identity for each family. Fixed training and held-out losses fell:

| Family/backend | Training loss, initial → final | Validation loss, initial → final | Peak process footprint (MiB) |
| --- | --- | --- | ---: |
| Phi-3 / CPU | 5.555212 → 5.419975 | 5.575892 → 5.450570 | 452.9 |
| Phi-3 / MLX | 5.555212 → 5.419975 | 5.575893 → 5.450569 | 466.9 |
| Qwen3 / CPU | 5.543500 → 5.377007 | 5.550551 → 5.430950 | 457.4 |
| Qwen3 / MLX | 5.543500 → 5.377006 | 5.550551 → 5.430950 | 471.5 |

Each run used a separate process, two layers, FP32 computation and one host
thread. The kernel footprint peak includes imports, construction, runtime,
training and export. Independent six-update audits passed on both backends,
with maximum adapter-gradient absolute differences at most `2.33e-8`.

The [evidence report](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/decoder-families-m1max.json)
contains source and binary identities, all audit results, raw losses, RSS and
footprint counters, and the software versions. Its single-run elapsed times
cover initial/final evaluation, training and export after construction.

### Qwen3.5 text training

The same wrapper completed 100 SFT updates on the version-controlled text corpus
with CPU and MLX execution:

| Backend | Training loss, initial → final | Validation loss, initial → final |
| --- | --- | --- |
| CPU | 5.582322 → 5.136364 | 5.621585 → 5.467013 |
| MLX | 5.582322 → 5.136365 | 5.621585 → 5.467014 |

Both used the same source graph, local tokenizer, FP32 parameters and AdamW
epsilon `1e-5`. Six-update independent Transformer and PEFT audits passed on
each backend; the maximum gradient absolute error was `9.32e-9`.

A fresh MLX process resumed `checkpoint-50` and reproduced step 100's adapter
parameters, AdamW tensors and scheduler state exactly. Loaded logits, loss and
all ten adapter gradients also matched the continuous run byte for byte.
All thirteen exported ONNX graphs passed the full checker, and the bound
`forward.onnx` executed with CPU ORT.

The [Qwen3.5 evidence report](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/qwen35-shared-sft-m1max.json)
contains commands, source/library hashes, corpus identity, dependency versions,
per-process RSS and physical-footprint peaks, audit results and export checks.
Recorded elapsed times cover evaluation, training and export after construction.
