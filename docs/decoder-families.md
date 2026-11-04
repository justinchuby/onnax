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

(decoder-bf16-projections)=

### BF16 projection arithmetic

Select `--projection-precision bf16` for BF16 matrix projections, including
LoRA products and the frozen output head. LoRA parameters, accumulated gradients and
AdamW moments stay FP32. Attention, normalization and recurrent states use FP32.
With `--nf4`, the frozen decoder stays packed; its decoded operands are rounded
for each projection.

For a source-layout head weight $W$, the forward computes
$Y=\operatorname{fp32}(\operatorname{bf16}(X)\operatorname{bf16}(W)^T)$.
The registered straight-through pullback computes
$\bar X=\bar Y\,R(W)$ in FP32, where $R(W)$ is the BF16-rounded weight viewed
as FP32. The exported loss/gradient modules carry this policy through reloads.

Before partitioning, ONNXScript evaluates expressions that depend only on frozen
weights. Training blocks and standalone inference reuse those constants,
including Qwen3.5’s decay rates $-\exp(A_{\log})$.

```sh
python -m examples.huggingface.decoder_sft \
  --family qwen35 --nf4 --projection-precision bf16 --fused-ops \
  --head-chunk-size 16 --steps 100 --stop 50 \
  --provider mlx --library /path/to/libonnxruntime_mlx_ep.dylib \
  --output-dir /tmp/qwen35-bf16-partial

python -m examples.huggingface.decoder_sft \
  --family qwen35 --nf4 --projection-precision bf16 --fused-ops \
  --head-chunk-size 16 --steps 100 \
  --resume /tmp/qwen35-bf16-partial/checkpoint-50 \
  --load-export /tmp/qwen35-bf16-partial/export \
  --provider mlx --library /path/to/libonnxruntime_mlx_ep.dylib \
  --output-dir /tmp/qwen35-bf16-resumed
```

Repeat the projection precision and quantization options when resuming. Their
saved configuration is checked before loading parameters. MLX EP executes all
three families; offline reference-kernel training and recovery tests cover
Phi-3, including packed and unquantized bases and both output-head modes.

On M1 Max, tiny Qwen3.5 completes 100 NF4/BF16 SFTTrainer updates: training
loss falls from 5.581 to 5.127 and validation loss from 5.623 to 5.432. A matched
FP32-compute run on the same frozen weights and data reaches 5.127 and 5.429.
Step-50 recovery reproduces 42 checkpoint tensors byte for byte. All 99 frozen
initializers and the head match their source; 51 ONNX modules pass the full
checker. Reloaded adapter gradients and standalone inference outputs agree
with continuous training. The
[raw reports and reproduction scripts](https://github.com/justinchuby/onnax/tree/main/examples/huggingface/reports/decoder-bf16-m1max)
include six family/quantization audits and both checkpoint recovery paths.

For pretrained Qwen3.5-0.8B, use `--real --steps 30 --head-chunk-size 128`,
stop at `--stop 15`, and resume from `checkpoint-15`. The M1 Max run uses
160 training sequences and 4 validation sequences from the fixed corpus,
sequence length 64, microbatch 1 and four accumulation steps.

| Workflow | Training loss | Validation loss |
| --- | --- | --- |
| onnax + MLX EP | 5.355 → 0.300 | 6.141 → 5.779 |
| Transformers + PyTorch MPS | 5.360 → 0.302 | 6.138 → 5.768 |

The reference uses byte-identical initial adapters and the same NF4 values.
Step-15 recovery reproduces all 62 checkpoint tensors, optimizer state and
60 adapter gradients byte for byte. Standalone inference matches training
exactly; each export has 57 checked ONNX modules. On a separate 64-token
audit batch, the median relative gradient difference from the BF16 reference
is 2.19%; 48 of 60 tensors exceed the precision-effect gate. The
[pretrained reports and drivers](https://github.com/justinchuby/onnax/tree/main/examples/huggingface/reports/decoder-bf16-m1max/pretrained-qwen35)
retain all per-tensor comparisons, training results and recovery checks.

A controlled build of the proposed
[symbolic-data budget](https://github.com/onnx/onnx-shape-inference/pull/146)
reduces one cold shape-inference call on this graph from 125.39 s to 0.61 s.
The complete serialized inferred graphs are byte identical. The report records
the two independent processes, package hashes and memory measurements.

Tiny-model audits include a third Transformers run using FP32 arithmetic
on the same frozen weight values. The L2 backend difference is checked against
the measured reference precision effect plus the original FP32 comparison
floor. Reports retain maximum differences and strict coordinate counts.
Projection JVP/VJP checks use independent rounded-operand formulas, and a
corrupted-adjoint regression verifies that the precision control rejects
excess error.

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

For NF4 exports, the CLI reconstructs the corresponding frozen source weight
slices before comparing local derivatives. With a chunked head, it records the
normalization VJP and the external head's loss/hidden gradient separately. The
head receives the same hidden states, labels and effective target count.
Audits supply explicit position IDs and compare valid-token predictions;
padded predictions remain visible in the diagnostic report.

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

The [pretrained NF4 controls](https://github.com/justinchuby/onnax/tree/main/examples/huggingface/reports/generic-nf4-m1max/real-qwen35/reference)
start from the trained adapter state and use two diagnostic sentences. Initial
valid-token logits have relative L2 difference `3.12e-6`. The shared-input local
VJPs report three out-of-tolerance elements in layer 15's `q_proj` B gradient,
with relative L2 difference `5.21e-6`. Normalization gradients pass the original
elementwise check. For the external vocabulary head, pairwise FP32 accumulation
reduces hidden-gradient violations from 46 to one.

Three independent fresh AdamW updates give these losses:

| Update input state | ONNX / MLX | Transformers |
| --- | ---: | ---: |
| Trained adapter | 5.507835 | 5.507665 |
| After one update | 1.640609 | 1.640494 |
| After two updates | 0.339562 | 0.339462 |

After three updates, the largest parameter difference is `2.32e-3` and valid-logit
relative L2 difference is `1.77e-4`. The reports include every gradient and update
comparison, retaining `rtol=1e-4, atol=1e-6` and its violation counts.

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

### Share loss and backward statistics

The head first differentiates the stable cross-entropy function using its
analytic rule. It then expands that function in the derivative graph and lets
ONNXScript folding and CSE share the target masks, row maxima and exponential
reductions between loss and backward. The source loss retains its analytic rule.
The expanded loss/gradient computation uses 58 operators instead of 103.

On M1 Max, the FP32 head from Qwen3.5-0.8B (248,320 classes, width 1024) gives:

| Rows / execution | Before median / P95 | Shared statistics median / P95 | Peak physical before / after |
| --- | ---: | ---: | ---: |
| 17 / compiled | 10.05 / 10.32 ms | 10.01 / 10.20 ms | 1.26 / 1.26 GiB |
| 17 / eager | 12.10 / 12.29 ms | 10.69 / 10.95 ms | 1.36 / 1.23 GiB |
| 128 / compiled | 28.79 / 29.07 ms | 29.01 / 29.28 ms | 2.18 / 2.18 GiB |
| 128 / eager | 42.73 / 43.10 ms | 33.09 / 33.54 ms | 2.63 / 2.33 GiB |

Each fresh process performs ten warmups and 100 measurements. Timings cover
head Session.run and NumPy exchange; peak memory includes construction and
runtime. Both variants match independent PyTorch loss/gradient values and
produce bitwise-equal outputs. A separate five-update Qwen3-0.6B 4K replay
compares the complete training path, including all first-two-update adapter
gradients and parameters, losses and frozen-weight hashes.
[Raw measurements and drivers](https://github.com/justinchuby/onnax/tree/main/examples/huggingface/reports/head-shared-terms-m1max)
record all samples, geometry, environment and validation results.

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

### PyTorch checkpoint control

Use the corresponding Transformers model and adapter values for a checkpointed
reference. Install the adapters before wrapping its decoder layers:

```python
from examples.huggingface.torch_context import checkpoint_decoder, chunked_token_loss

reference = checkpoint_decoder(reference, 1)
loss = chunked_token_loss(reference, batch, labels, chunk_size=128)
loss.backward()
```

One-layer checkpoints preserve the decoder count and Transformers' per-layer
mask selection. This covers Qwen3.5's recurrent/full-attention layout, Qwen3 and
Phi-3. Recomputation remains active in evaluation mode when gradients are
enabled, matching the deterministic training reference. Groups larger than one
use models whose decoder layout is explicitly full attention.

### Measure the corresponding PyTorch reference

The reference worker reads the named adapter snapshot from an exported decoder
task and loads its pinned Transformers source (or the deterministic tiny model).
It supports Phi-3, Qwen3 and Qwen3.5, restoring the export’s projection precision
and NF4 policy. Supply the same input archive to both execution paths:

```python
import numpy as np

np.savez(
    "/tmp/decoder-inputs.npz",
    **{name: value.detach().cpu().numpy() for name, value in batch.items()},
)
```

The archive contains INT64 `input_ids`, `attention_mask`, `position_ids` and
`labels`, with the same batch/sequence shape. Labels use `-100` for ignored
targets; the loss shifts by one token and averages over valid shifted targets.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m examples.huggingface.torch_decoder_benchmark \
  --export /tmp/decoder-export --inputs /tmp/decoder-inputs.npz \
  --device mps --attention sdpa --checkpoint-layers 1 \
  --head-chunk-size 128 --warmup 10 --samples 100 \
  --output /tmp/torch-decoder.json
```

Run each measurement in its own process, using the same adapter snapshot,
inputs, precision, checkpoint boundaries and head chunk size. Use `--device cpu`
for an offline reference or `--attention eager` to compare attention schedules.
The JSON records per-sample execution and output-exchange times, median/P95,
peak process RSS, source hashes and effective token count. The companion
`/tmp/torch-decoder.json.npz` stores the loss and all named adapter gradients.
Compare those arrays with the ONNX results before comparing throughput.

Inputs are staged on the device before warmup. Device work is synchronized at
timing boundaries, and gradients/loss return to CPU after each iteration.
Parameters stay fixed; the timed scope is forward, loss and backward. The
PyTorch head defaults to `--head-policy eager`: it selects valid shifted tokens,
computes native cross entropy and its hidden-state gradient per chunk, then
saves that gradient for backward. This first-order schedule releases each
chunk’s logits immediately. `--head-policy checkpoint` recomputes each frozen
projection and cross entropy chunk during backward. The ONNX head uses its
exported loss/VJP schedule. The measurements below record the checkpoint policy.
The worker validates the adapter archive against the export manifest and
records source identity; the Transformers base comes from the pinned checkpoint.
For NF4 exports, the reference reconstructs each source projection’s NF4 blocks
and holds their decoded values in FP32. BF16/FP16 projections use the same
rounded-operand adjoints as the ONNX functions; adapters and gradients stay FP32.
The report records projection compute, frozen-weight storage and autocast policy.
This makes the reference’s memory representation visible alongside runtime memory.

### BF16/NF4 short-context measurement

On an M1 Max, the pinned Qwen3.5-0.8B export above gives these results at
64 tokens, batch one, checkpoint group one and head chunk 128. Each independent
process runs ten warmups and 100 fixed-parameter forward/loss/backward samples,
including output exchange.

| Backend | Median | P95 | Peak process physical memory |
| --- | ---: | ---: | ---: |
| ONNX / MLX EP | 399 ms | 437 ms | 4.58 GiB |
| PyTorch MPS / SDPA | 471 ms | 555 ms | 6.47 GiB |

ONNX throughput is 1.18× PyTorch for this configuration. Both use BF16
projections and FP32 adapters/adjoints. ONNX retains packed NF4 decoder weights;
the reference holds their decoded values in FP32. Memory includes construction
and runtime. Input and adapter hashes match across workers.

Token losses are 2.79467 and 2.79102. Across all 60 adapter gradients, the median
relative L2 difference is 1.99%; none meets the FP32 allclose threshold. The
[30-update training comparison](decoder-bf16-projections) records both learning
curves. [Raw timing, gradient diagnostics and drivers](https://github.com/justinchuby/onnax/tree/main/examples/huggingface/reports/decoder-bf16-m1max/pretrained-qwen35/benchmark64)
include per-sample timings and process-memory observations.

### BF16/NF4 at 4K context

The same pinned export completes the paired 4,032-token workload with 4,031
prediction targets per sample, batch one, checkpoint group one and head chunk
128. Each independent process runs ten warmups and 100 measured samples.

| Backend | Median | P95 | Peak process physical memory |
| --- | ---: | ---: | ---: |
| ONNX / MLX EP | 10.788 s | 12.337 s | 7.76 GiB |
| PyTorch MPS / SDPA | 12.749 s | 12.827 s | 10.99 GiB |

The measured throughput ratio is 1.18×. Timing covers forward, shifted mean
loss, all adapter gradients and output exchange; optimizer updates are external.
Memory includes model construction and runtime. The reference holds decoded
NF4 values in FP32; ONNX retains the packed base. Both use BF16 projection
compute and FP32 adapters/adjoints.

Input and adapter hashes match. Token losses are 0.0622500 and 0.0624421.
Across all 60 gradients, median relative L2 difference is 1.35% and minimum
cosine similarity is 0.999418; none meets the FP32 allclose threshold. The
native timing's first and last 25-sample medians are 11.670 s and 10.677 s;
all samples contribute to the table's median and P95.

[Raw samples, gradient diagnostics, process observations and drivers](https://github.com/justinchuby/onnax/tree/main/examples/huggingface/reports/decoder-bf16-m1max/pretrained-qwen35/benchmark4k)
retain source and artifact hashes. The next execution comparison profiles
repeated activation casts and frozen projection conversions in both runtimes.

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

(qwen35-state-scan-performance)=

### Qwen3.5 at 4K: fused recurrent replay

The FP32 24-layer Qwen3.5-0.8B executes at 4,032 tokens with per-layer checkpoints
and a 128-row frozen head. State-only and replay scans now use Metal kernels
alongside the fused reverse recurrence.

| State/replay execution | Forward/loss/all-gradient median / P95 | Shifted targets/s | Process peak physical footprint |
| --- | ---: | ---: | ---: |
| Generic scans | 21.026 / 21.173 s | 191.7 | 7.94 GiB |
| Fused Metal scans | 12.922 / 13.114 s | 312.0 | 8.07 GiB |

Each library ran in its own M1 Max process, with 10 warmups and 100 measured
calls at the same fixed adapter snapshot and repeated-text input. Timing includes
the full forward, mean causal token loss, all 60 adapter gradients and runtime
array exchange. The optimizer runs outside this measurement. The head is eager
in both runs; ORT graph optimization and CPU fallback are disabled, with one CPU
thread and a 512 MiB MLX allocator cache.

The measured throughput ratio is **1.63**. All 60 gradients match the parent at
`rtol=1e-4, atol=1e-6`, with maximum absolute difference 4.80e-9. All 100 losses
are identical, and adapter parameter hashes remain unchanged.

Checkpoint backward drops from 17.283 to 7.494 seconds. Task forward/loss rises
from 3.737 to 5.427 seconds in this pair, including a change in head/wrapper
execution. The component samples guide the next allocator and head controls.
Process memory includes model loading, initialization and the runtime.

The [raw samples, component timings, gradient comparison, guards and drivers](https://github.com/justinchuby/onnax/tree/main/examples/huggingface/reports/qwen35-state-scans-m1max)
record library and source hashes. The native implementation is
[MLX EP #121](https://github.com/onnxruntime/onnxruntime-ep-mlx/pull/121).

(qwen35-mps-comparison)=

### Qwen3.5 at 4K: PyTorch MPS and MLX EP

The same pinned Qwen3.5-0.8B runs in FP32 with a fixed adapter snapshot,
batch one, 4,032 input tokens and 4,031 shifted prediction targets. Both paths
use one-layer decoder checkpoints and 128-token output-head chunks. Each runs
in a separate process with 10 warmups and 100 measured calls on M1 Max:

| Execution | Forward/loss/all-gradient median / P95 | Shifted targets/s | Process peak physical footprint |
| --- | ---: | ---: | ---: |
| PyTorch MPS, SDPA | 10.945 / 10.973 s | 368.3 | 10.19 GiB |
| ONNX, MLX EP | 10.809 / 11.031 s | 372.9 | 8.45 GiB |

ONNX reaches similar throughput, with a **1.01×** median ratio and **17%** lower
process peak physical footprint for this workload. All 60 adapter gradients match at `rtol=1e-4, atol=1e-6`, with maximum
absolute difference 2.94e-07; loss differs by at most
5.40e-08. Input and adapter digests agree, and parameters stay fixed.

Timing includes forward, mean shifted-token loss, backward and CPU gradient
availability. MPS inputs are staged before warmup; its device execution and
output exchange are also recorded separately. ONNX exchanges arrays inside its
block session calls. The external optimizer is outside the timed scope.
PyTorch checkpoints the frozen head and recomputes logits during backward;
ONNX shares loss/VJP statistics and saves the hidden gradient. These schedules
provide the next head-control comparison.

ONNX uses fused recurrent scans and shape-selected 64-token replay, with an
eager head, ORT optimization and CPU fallback disabled, one CPU thread and a
512 MiB MLX cache. PyTorch uses Transformers' Torch recurrent implementation
and SDPA for full attention, with MPS CPU fallback disabled. Peak physical
footprint includes model loading and runtime setup; the ONNX process also loads
18 regenerated derivative modules with a separate immutable frozen payload.
GPU workers run serially. Short CPU correctness tests ran during portions of
the measurement.

The [samples, guards, array comparisons and qualified drivers](https://github.com/justinchuby/onnax/tree/main/examples/huggingface/reports/qwen35-mps-comparison-m1max)
record this pair. The [shape-selected checkpoint report](adaptive-recurrent-checkpoints)
explains its recurrent plan; the preceding section records the earlier kernel
comparison.

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
