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

## Packed NF4 bases

Add `--nf4` to store frozen decoder Linear weights as packed NF4 while keeping
adapters, gradients and optimizer state in FP32:

```sh
python -m examples.huggingface.decoder_sft \
  --family phi3 --nf4 --head-chunk-size 128 --steps 30 \
  --output-dir /tmp/phi3-nf4
```

The same option applies to Qwen3 and Qwen3.5. The source mapping checks that
all decoder Linear weights are covered before quantization. Phi-3's Q/K/V
pieces map back to slices of its fused QKV source weight; each graph piece
keeps its own row-major NF4 blocks. Qwen3.5's linear-attention projections use
the same mapping. Embeddings, normalization, recurrent convolutions and the
output head retain FP32 weights.

The independent Transformers reference reconstructs exactly those stored
weight values. Audits compare logits, token loss, every adapter gradient and
multiple AdamW updates, then exchange adapters through PEFT. Adapter files
record the base quantization alongside the source parameter mapping.
Export/load preserves the NF4 graph and its mapping; repeat `--nf4` when
resuming a checkpoint or loading an exported task.

The [offline MLX qualification reports](https://github.com/justinchuby/onnax/tree/main/examples/huggingface/reports/generic-nf4-m1max)
cover six independent AdamW updates and PEFT exchange for all three families.
Tiny Qwen3.5 also completes 100 SFTTrainer updates: training loss falls from
5.581 to 5.127 and validation loss from 5.623 to 5.429. A fresh process resumes
from step 50 with bitwise-equal final adapters and optimizer state. Reloaded
ONNX loss, all ten adapter gradients and standalone logits match the continuous
run exactly. Its 31 frozen Linear projections occupy 93,024 packed/scale bytes,
compared with 661,504 FP32 bytes.

The same path completes 30 MLX updates for the pinned Qwen3.5-0.8B checkpoint
at sequence length 64, microbatch one and accumulation four. On the fixed
version-controlled corpus, train loss changes from 5.361 to 0.303 and validation
loss from 6.145 to 5.813. Its 186 frozen Linear projections use 279.9 MB of NF4
codes/scales instead of 1,990.5 MB of FP32 weights. Embeddings and the head are
stored separately in FP32.

A fresh process resumes from step 15 using `--load-export` to reuse the exported
ONNX graphs. Final adapters, AdamW, scheduler, loaded loss, all 60 adapter
gradients and standalone logits are bitwise equal to continuous training.
The full checker validates 57 exported models per run.
[Pretrained reports and reproduction drivers](https://github.com/justinchuby/onnax/tree/main/examples/huggingface/reports/generic-nf4-m1max/real-qwen35)
record the source revision, input batches, memory scope and runtime hashes.

### Compile the output head independently

With MLX, add `--compile-head` to compile the chunked loss/gradient and logits
sessions while keeping decoder checkpoints eager:

```sh
ONNXRUNTIME_EP_MLX_NO_COMPILE=0 python -m examples.huggingface.decoder_sft \
  --family qwen35 --nf4 --head-chunk-size 128 --compile-head \
  --provider mlx --library /path/to/libonnxruntime_mlx_ep.dylib \
  --output-dir /tmp/qwen35-compiled-head
```

The same policy applies when auditing, resuming or loading an exported task.
Python callers can supply `session_factory` for the decoder and a separate
`head_session_factory` to `prepare`, `audit`, `run` or `OnnxDecoder.load`.
Sessions are reused, including the lazy full-logits session. Exported graphs
and parameter identities stay the same across these execution choices.

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

### Load a pretrained text backbone

`--real` selects the family's pinned checkpoint and tokenizer. For Qwen3.5:

```sh
python -m examples.huggingface.decoder_sft \
  --family qwen35 --real --steps 30 \
  --cache-dir /tmp/onnax-model-cache --output-dir /tmp/qwen35-pretrained \
  --provider mlx --library /path/to/libonnxruntime_mlx_ep.dylib
```

The source is `Qwen/Qwen3.5-0.8B` at revision
`2fc06364715b967f1860aea9cf38778875588b17`. Native Transformers loads its text
backbone into `Qwen3_5ForCausalLM` in FP32. Mobius builds the ONNX graph from
that configuration and its loaded weights. The text backbone has 752,393,024
parameters, 24 decoder layers and 60 trainable LoRA tensors. The checkpoint's
vision and multi-token prediction modules are outside this text task.

The real-model setup uses sequence length 64, microbatch one and four accumulated
microbatches per update. Batches are padded to 64 tokens; attention masks and
ignored labels identify the valid targets. The report records each batch's
executed shape and target count.

Loading checks every text weight before graph construction. The exported config
records the source revision, reference class and parameter-prefix mapping;
adapter mappings also retain each original checkpoint weight name. Resume checks
the same source identity. PEFT adapters target the native text-only CausalLM's
module paths, such as `model.layers.0.linear_attn.in_proj_qkv`.

On an M1 Max, the pretrained FP32 run completed 30 updates with MLX EP:

| Measure | Initial | After 30 updates |
| --- | ---: | ---: |
| Fixed training-set loss | 5.223939 | 0.302991 |
| Validation loss | 6.019480 | 5.621177 |

The run executed 120 training microbatches and 1,388 shifted targets. Whole-worker
peak physical footprint was 10.43 GiB, measured with macOS `proc_pid_rusage`.
Evaluation, training and export took 211 seconds after construction. The
[evidence report](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/qwen35-pretrained-fp32-m1max.json)
records versions, source hashes, shapes and FP32 diagnostics.

The FP32 controls compare Transformers' chunked and sequential Gated DeltaNet
implementations with the portable ONNX graph on CPU and MLX. Relative logit L2
differences are about two to three parts per million. The report retains the
original `rtol=1e-4, atol=1e-6` elementwise checks and the per-adapter gradient
differences for zero and nonzero B.

Resuming update 15 in a fresh process reproduces update 30 byte for byte:
adapter tensors, AdamW state, scheduler, loaded logits, loss and all 60 adapter
gradients. All 53 exported ONNX graphs pass the full checker. PEFT exchanges
all 60 adapter tensors without changing their values.

At the trained adapter state, the native PyTorch reference has a maximum gradient
difference of `1.09e-4`; a fresh AdamW update differs by up to `1.58e-3` with epsilon
`1e-5`. The additional stable-loss/pairwise-head reference also retains differences.
The shared-activation audit below separates local derivatives from differences
accumulated through the forward computation.

(shared-activation-vjp)=
### Compare local VJPs at shared activations

For a decoder layer $h_{i+1}=F_i(h_i,\theta_i)$, the local reverse product is
$D F_i(h_i,\theta_i)^T\bar h_{i+1}$. This control supplies the same activation
$h_i$, adapter parameters $\theta_i$ and incoming cotangent
$\bar h_{i+1}$ to ONNX and native Transformers. Earlier forward differences
therefore stay outside the local comparison.

```sh
python -m examples.huggingface.shared_vjp \
  --export /tmp/qwen35-pretrained/export --cache-dir /tmp/onnax-model-cache \
  --provider mlx --library /path/to/libonnxruntime_mlx_ep.dylib \
  --output /tmp/qwen35-shared-vjp.json
```

For an offline Qwen3, Phi-3 or Qwen3.5 export, omit `--cache-dir`. Select
`--provider cpu` to use the CPU reference runtime. Pretrained exports select
the model and tokenizer revision saved in their provenance record.

The JSON records each layer's valid-token and all-token forward comparisons,
adapter gradients, hidden-state gradients and the frozen normalization/head
pullback. Right-padded positions can differ in the recurrent forward computation;
the valid-token comparison and explicit incoming cotangent show which values
participate in the loss. The tool preserves existing ONNX parameter gradients
and restores execution sessions after the audit. Its recording copies belong
to diagnostic memory usage; run performance timing separately.

The suffix has two independent FP32 controls: the native CPU PyTorch head
backward and a head backward that sums 128-element partial products with a
binary tree. Both use the same activation and loss cotangent. These controls
isolate accumulation over the wide vocabulary dimension.

The [pretrained Qwen3.5 shared-activation report](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/qwen35-shared-activation-vjp-m1max.json)
uses the adapter state from the 30-update run. All 83 decoder gradient tensors
pass `rtol=1e-4, atol=1e-6`, with maximum relative L2 difference `3.54e-6`.
For the suffix, ordinary CPU head accumulation gives relative gradient L2
`1.66e-4`; FP32 pairwise accumulation gives `3.63e-6` and passes the elementwise
gate. The report includes every layer's forward and gradient comparisons.

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

## Train with a bounded output head

Use `--head-chunk-size 128` with any decoder family to train through hidden-state
VJPs and a frozen, chunked vocabulary projection:

```sh
python -m examples.huggingface.decoder_sft \
  --family qwen35 --head-chunk-size 128 \
  --output-dir /tmp/qwen35-bounded-training
```

After the causal shift, the head gathers positions whose labels differ from
`-100`. Each chunk computes cross-entropy and its hidden-state gradient; the
scheduler scatters those gradients into the complete sequence before the body
VJP. Attention therefore carries supervision through prompt and context tokens.
Trainer divides loss sums by the effective target count across accumulation
microbatches. All-ignored labels produce zero loss and skip projection execution.

The frozen projection retains its source `[vocab, hidden]` layout and uses
`Gemm(transB=1)`. Task exports include reusable body/head derivative modules and
`execution/inference.onnx`, which returns full logits. Checkpoint identity covers
the frozen head, loss convention and projection precision. Resume with the same
head mode; chunk size can change without changing the loss definition.

## Choose training text and context length

Provide separate training and validation files to train on your own text. Files
can contain one example per line or JSONL objects with a `text` field. The tiny
models train a local tokenizer from the training split; pretrained models use
their pinned tokenizer.

```sh
python -m examples.huggingface.decoder_sft \
  --family qwen35 --real --cache-dir /path/to/model-cache \
  --train-file /path/to/train.jsonl --validation-file /path/to/validation.jsonl \
  --max-length 512 --pad-to-multiple-of 64 --head-chunk-size 128 \
  --provider mlx --library /path/to/libonnxruntime_mlx_ep.dylib \
  --output-dir /tmp/qwen35-context-training
```

`max_length` truncates examples to the requested context. Padding rounds each
batch to the chosen multiple. Reports record the actual sequence length and
supervised target count for each batch. Longer examples supply longer training
contexts; increasing the padding multiple alone adds padding positions.

Checkpoints record text hashes, tokenizer encoding semantics, context length and
padding settings. Resume with the same settings and files to reproduce the
training trajectory. The CLI validates this identity before building the ONNX
model. Offline tiny models accept context lengths up to 64; pretrained models use
their source configuration's limit.

### Reuse exported graphs for training

`--load-export` loads an exported task's forward and derivative modules directly.
Use it to continue training the saved adapters or combine it with `--resume` to
restore a Trainer checkpoint:

```sh
python -m examples.huggingface.decoder_sft \
  --family qwen35 --real --cache-dir /path/to/model-cache \
  --load-export /tmp/qwen35-context-training/export \
  --resume /tmp/qwen35-context-training/checkpoint-2 --steps 4 \
  --train-file /path/to/train.jsonl --validation-file /path/to/validation.jsonl \
  --max-length 4096 --pad-to-multiple-of 64 --head-chunk-size 128 \
  --provider mlx --library /path/to/libonnxruntime_mlx_ep.dylib \
  --output-dir /tmp/qwen35-context-resumed
```

The loader validates the saved family, source revision and head mode. The saved
graph retains its operator requirements. Initial evaluation uses the restored
checkpoint parameters; Trainer restores optimizer, scheduler and RNG state.
The replay CLI compares the final state with continuous training.

### Complete long-context QA qualification

The pinned Qwen3.5-0.8B trains four optimizer updates on four complete LongAlpaca
QA examples, with two document-group-held-out validation examples. The Qwen3.5
tokenizer produces 3,642–4,027 tokens per training example. All 16 training
microbatches preserve the complete text, using a 4,096-token limit and padding
to multiples of 64. Full-text causal loss covers every shifted nonpadding token.

Training loss falls from 2.3527 to 2.1173; validation loss falls from 2.6614 to
2.4325. The guarded run reaches 9.24 GiB peak physical footprint and retains at
least 7.54 GiB free disk through training and export. The
[corpus manifest](../examples/huggingface/reports/qwen35-context4k-corpus.json)
records the dataset revision, license, sample hashes and document holdout. The
[training report](../examples/huggingface/reports/qwen35-context4k-sft-mlx.json)
retains actual sequence/target counts and runtime provenance. Fresh-process
recovery is the next qualification gate.

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
weights, with full-logit and bounded-head loss paths.

To measure both routes, run `examples.huggingface.decoder_head_benchmark` once
without `--head-chunk-size` and once with it, using separate output files and the
same sequence, valid-token count and runtime settings. Each invocation owns its
runtime sessions. Add `--real --cache-dir ... --sequence 512 --valid-tokens 128`
for the pinned pretrained model. Compare the resulting loss/gradient archives:

```sh
python -m examples.huggingface.decoder_head_comparison \
  --full /tmp/full.json --bounded /tmp/bounded.json \
  --output /tmp/head-comparison.json
```

The comparison checks workload settings, runtime versions, source hashes and
finite numerical outputs before summarizing timing and process RSS.

### Bounded-head measurement

The 64-token tiny Qwen3.5 comparison uses 32 supervised targets, FP32, batch one,
10 warmups and 100 samples per independent MLX worker. Loss and all ten adapter
gradients match between routes at `rtol=1e-4, atol=1e-6`.

| Route | Step median / P95 | Process peak RSS |
|---|---:|---:|
| Full head | 99.86 / 114.39 ms | 620.95 MiB |
| Bounded head, 16 rows | 102.63 / 118.64 ms | 622.73 MiB |

At this model size, the routes take similar time and memory. Timing covers
forward, loss, VJP and runtime exchange at fixed nonzero adapter parameters.
Peak RSS includes imports, source construction and runtime sessions. The reports
retain [full-head samples](../examples/huggingface/reports/bounded-head-tiny-qwen35-mlx-full.json)
and [bounded-head samples](../examples/huggingface/reports/bounded-head-tiny-qwen35-mlx-chunked.json),
source hashes, library digest and environment versions.

### Pretrained Qwen3.5 at 512 tokens

On the same M1 Max, the pinned Qwen3.5-0.8B uses batch one, FP32 and 128
supervised targets in a 512-token context. Each route runs in an independent
process with 10 warmups and 100 measured samples. The bounded head uses chunks
of 128 rows. Loss and all 60 adapter gradients match at `rtol=1e-4, atol=1e-6`.

| Route | Step median / P95 | Peak physical footprint | Peak RSS |
|---|---:|---:|---:|
| Full head | 3.132 / 3.184 s | 8.57 GiB | 9.67 GiB |
| Bounded head | 2.578 / 2.645 s | 7.04 GiB | 11.68 GiB |

The bounded route reduces median step time by 17.7% (1.21 times the throughput).
The full route returns complete logits for CPU cross-entropy. The bounded route
executes projection, cross-entropy and hidden derivatives in ONNX on MLX, returning
chunk loss sums and hidden gradients across the runtime boundary.
These steps compute forward, loss and all adapter gradients at fixed nonzero
adapter parameters. Process peaks include source loading, session construction
and execution. The macOS process guard records physical footprint; `getrusage`
records peak RSS.

The bounded head's median runtime call takes 82.31 ms, including execution and
runtime transfers. Python gathering takes 0.070 ms and scattering takes 0.111 ms.
The [comparison report](../examples/huggingface/reports/qwen35-bounded-head-512-mlx-comparison.json)
contains per-output numerical errors and guard summaries; the
[full-head](../examples/huggingface/reports/qwen35-bounded-head-512-mlx-full.json)
and [bounded-head](../examples/huggingface/reports/qwen35-bounded-head-512-mlx-chunked.json)
reports retain all timing samples, versions, source hashes and the native library digest.

### Real-model adapter updates

The same bounded-head task completes 30 SFTTrainer updates on the pinned
Qwen3.5-0.8B. This training run uses FP32, sequence length 64, microbatch one,
gradient accumulation four and head chunks of 128 rows. Fixed training loss
falls from 5.2241 to 0.3029; held-out loss falls from 6.0197 to 5.6222.
The [training report](../examples/huggingface/reports/qwen35-bounded-head-30step-mlx.json)
records batches, effective target counts, configuration, source identity and export path.

A fresh process restores step 15 and reaches the same step-30 parameters, Adam
state and scheduler byte for byte. PEFT adapter files match byte for byte.
Loaded task logits, loss and all 60 adapter gradients match byte for byte
between the independently constructed exports, as do the standalone inference
outputs. All 114 exported ONNX modules pass full checker validation; 3736 named
tensor records agree. The [recovery report](../examples/huggingface/reports/qwen35-bounded-head-recovery-mlx.json)
records these checks and source-weight hashes. The resumed export was rebuilt
from its completed checkpoint after the disk guard interrupted its first export.

Use the shared replay CLI to check your own continuous and resumed runs:

```sh
python -m examples.huggingface.replay \
  --uninterrupted /tmp/qwen35-bounded-training \
  --resumed /tmp/qwen35-bounded-resumed --step 30 \
  --provider mlx --library /path/to/libonnxruntime_mlx_ep.dylib \
  --output /tmp/qwen35-recovery.json
```

The same command supports every decoder family and the full-head route.
It checks checkpoint tensors, Adam/scheduler state, PEFT files, named tensor
contents and full ONNX checker validation. It compares loaded loss/gradient
outputs and standalone inference outputs between the two runs independently.
The [real-model CLI report](../examples/huggingface/reports/qwen35-bounded-head-replay-mlx.json)
retains the output digests and named-tensor manifest digest.

### FP32 inference across execution boundaries

At the recovered adapter state, a fixed eight-token probe compares checkpointed
body execution, whole-body execution and standalone inference. Appending the
head preserves the whole-body result byte for byte. Whole and checkpointed body
execution differ by `1.85e-6` relative hidden-state L2 error, producing `1.79e-6`
relative logit L2 error.

Against independent Transformers FP32 execution, both ONNX routes have about
`3.9e-6` relative logit L2 error, identical argmax tokens and loss differences below
`6e-6`. The original `rtol=1e-4, atol=1e-6` elementwise check exceeds its threshold
for some logits near zero. The [inference report](../examples/huggingface/reports/qwen35-bounded-head-inference-fp32.json)
retains those counts, absolute errors and reference losses.

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

The complete-context run also resumes from update two in a fresh process using
`--load-export`. At update four, all 62 checkpoint tensors, Adam state, scheduler
and PEFT adapter weights match continuous training byte for byte. The
[replay report](../examples/huggingface/reports/qwen35-context4k-replay.json)
checks 114 ONNX models and 3,736 named tensor records; loaded loss, all 60 adapter
gradients and inference outputs match on the recorded eight-token replay input.
The optimizer updates themselves use the complete 3.6k–4k-token QA sequences.
