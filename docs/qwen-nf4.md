# NF4 Qwen fine-tuning with SFTTrainer

Store Qwen's frozen decoder matrices in NF4 and train FP32 LoRA adapters through
ordinary ONNX forward/VJP modules. The example combines Mobius model construction,
onnax differentiation, TRL SFTTrainer and PEFT adapter files.

```bash
uv sync --group dev --group training --group huggingface
uv run python -m examples.huggingface.sft --nf4 --steps 100 \
  --save-activations --peft-roundtrip --output-dir tiny_nf4
```

Use `--real --steps 30 --diagnose-rounding` for the pinned Qwen3-0.6B model.
The [Qwen tutorial](qwen-sft.md) describes the corpus, tokenizer, right padding,
causal shift and effective-token normalization.

## Frozen decoder, trainable adapters

Every decoder layer quantizes seven Linear matrices: q/k/v/o projections and
MLP gate/up/down projections. For a q- or v-projection,

$$
Y=XQ(W)+\tfrac14XAB,
\qquad \bar X=\bar YQ(W)^T+\tfrac14(\bar YB^T)A^T.
$$

Only A and B are updated. Gradient propagation through the frozen quantized
matrices connects adapters across layers. Embeddings, normalization and lm_head
stay FP32. Loss, adapter gradients, AdamW parameters and moments are FP32.

Each block of 64 weights has an FP32 absmax and 64 NF4 codes; two codes occupy a
byte. ONNX stores the transposed Linear matrix in row-major blocks, high nibble
first. `export/quantization.json` records the codebook, shapes, buffer names and
storage sizes. Reconstructing that layout reproduces the ONNX base weights.

The CPU reference functions unpack each matrix and run FP32 MatMul. Exported
models contain standard ONNX function bodies. ONNXScript preserves these function
boundaries during optimization. The runtime uses ORT graph optimization disabled.
Saved-activation splitting retains function calls and saves their output boundary.

## Independent validation and adapter exchange

The Transformers reference uses exactly the stored NF4 values, independently
decoded in NumPy and copied into its Linear weights. Tests compare predictions,
token loss, every adapter gradient and multiple independent AdamW updates using
`rtol=1e-4, atol=1e-6`. Both full-logit and chunked-loss paths are covered.

PEFT adapter tensors keep the standard Linear layout: ONNX A/B are transposed
when saved. `adapter_mapping.json` also records the base quantization. Adapter
exchange compares the same reconstructed base in ONNX and PyTorch.

Tiny Qwen runs 100 updates, exports a fully checked ONNX graph and resumes
checkpoint 50 in a fresh process. The replay verifier compares adapter bytes,
checkpoint tensors, AdamW and scheduler state, frozen-buffer hashes and outputs
from both loaded ONNX exports:

```bash
uv run python -m examples.huggingface.sft --nf4 --steps 100 \
  --save-activations --resume tiny_nf4/checkpoint-50 --output-dir resumed_nf4
uv run python -m examples.huggingface.replay --uninterrupted tiny_nf4 \
  --resumed resumed_nf4 --step 100 --output replay.json
```

The two-layer model packs 294,912 decoder-weight bytes into 41,472 bytes including
scales. This storage count measures the packed buffers. Runtime RSS includes
function execution, model construction and optimizer state.

## Measure the execution paths

```bash
uv run python -m examples.huggingface.benchmark --nf4 --samples 100 \
  --warmup 10 --output nf4-benchmark.json
```

Each execution path runs in its own process: PyTorch, raw ONNX, optimized ONNX,
saved activations and chunked loss. PyTorch uses the same reconstructed NF4
weight values stored as FP32 tensors. Reports identify weight values and storage
separately, and include step median/P95, effective tokens/s, CPU array-view time,
peak process RSS, thread counts and package versions.

The next performance work is decoding lifetime and backend fusion, guided by
these measurements.

## Recorded CPU runs

On Apple M1 Max with 32 GiB RAM, Qwen3-0.6B completed 30 NF4 adapter updates.
Validation loss decreased from **7.750500 to 5.731173**. Its 196 decoder matrices
use 247,726,080 packed/scales bytes, compared with 1,761,607,680 FP32 bytes.
The exported weight data is 893,583,360 bytes, including the FP32 embedding and
other frozen buffers. The full ONNX checker validates the export. A fresh process resumed checkpoint
15 and reproduced all 112 adapters, all 113 checkpoint tensors, optimizer and
scheduler state byte for byte. Both loaded exports returned byte-identical
logits and hidden states; all 532 frozen initializer hashes matched.

`qwen3-nf4.json` retains the original elementwise comparisons. At the final
state, three gradient elements in layer 16's q-projection B exceed the original
threshold; independent AdamW update differences are also recorded. Tiny Qwen
passes the strict gradient, update and PEFT comparisons.

The isolated tiny-model benchmark uses batch four, sequence length 64 and 252
valid shifted tokens per update. Each path warms up for 10 updates and measures
100 updates on the same NF4-reconstructed base values:

| Execution | Step median (ms) | P95 (ms) | Effective tokens/s | Peak process RSS (MiB) |
| --- | ---: | ---: | ---: | ---: |
| PyTorch, reconstructed FP32 storage | 3.29 | 16.66 | 76,683 | 541.2 |
| Raw ONNX NF4 | 13.61 | 45.73 | 18,516 | 597.5 |
| Optimized ONNX NF4 | 10.56 | 12.18 | 23,856 | 604.4 |
| Saved activations | 9.10 | 9.19 | 27,700 | 611.5 |
| Chunked loss | 11.51 | 12.14 | 21,898 | 605.8 |

The JSON report records per-phase times, CPU array-view costs and versions.
Process RSS includes imports, model construction and runtime. A sample of the
real CPU run identifies codebook Gather as a decoding hotspot; backend fusion
and decoding lifetime are the next performance targets.

## Run forward and VJP on Apple GPUs

SFTTrainer can create MLX execution sessions directly for the body and the
chunked output head. Trainer continues to own CPU PyTorch adapters, AdamW,
gradient accumulation and checkpoints; NumPy carries values across runtime
boundaries.

```bash
uv run python -m examples.huggingface.sft --nf4 --save-activations \
  --provider mlx --library /path/to/libonnxruntime_mlx_ep.dylib \
  --segment-nodes 1024 --mode chunked --output-dir native-sft
```

Use `--resume native-sft/checkpoint-50` to restore a tiny training run. The
report records the selected provider and development library hash. Every
native session disables ORT graph optimization and CPU fallback. In Python,
`prepare(session_factory=...)` accepts a runtime constructor receiving ONNX
bytes or an external-data model path; the same factory handles forward, VJP,
head prediction and head loss sessions.

On Apple M1 Max, a real-model run with microbatch 1 and gradient accumulation 4
completed 30 native chunked-loss updates with analytic CE derivatives.
Validation loss fell from **7.75064 to 5.63277**. A fresh process resumed checkpoint 15 and reproduced all 112
adapters, 113 checkpoint tensors, optimizer/scheduler state and loaded ONNX
outputs byte for byte. Frozen source weights remained unchanged. PEFT adapter
format exchange also preserved all parameter bytes.
All 112 adapter gradients pass the stable-reference checks at both endpoints.
With `--peft-roundtrip --stable-reference`, the PEFT report also includes
`stable_reference_audit`, using the same loss and pairwise frozen-head
reference. This additional control preserves the gradients used by the
original PEFT update comparison.
At the 30-step real-model checkpoint, this standard PEFT control passes all
112 adapter gradients, with a maximum absolute difference of 1.65e-6.
Independent fresh Adam updates exceed tolerance for 92 tensors. The
[PEFT reference report](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/qwen3-nf4-peft-stable-reference-m1max.json)
contains both the original and stable-reference comparisons.

The initial and final comparisons against Transformers exceed the FP32
tolerance for 47 and 92 gradient tensors, respectively. PEFT gradient and
independent-update comparisons also fail. The
[native SFT report](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/qwen3-nf4-native-analytic-ce-sft-m1max.json)
contains these failures alongside the training and recovery results.

For loss formulas, reference controls, independent update trajectories and
Adam-state analysis, see [FP32 reference checks](qwen-reference-accuracy.md).
The [paired runtime audit](qwen-runtime-audit.md) holds forward residuals fixed
to compare backward execution directly.

## Reuse frozen weights in saved backward graphs

A frozen projection often stores its weight as $W^T$ for forward execution.
Its pullback needs $(W^T)^T=W$. With `OnnxModule(..., save_activations=True, reuse_frozen_weights=True)`,
onnax composes these layouts before splitting so backward can read the original weight
initializer. This removes the transposed weight from the saved activation
boundary. Computed values are reused only when they are already available to
backward, keeping the saved boundary from growing.

Measure the change on the real model with:

```bash
uv run python -m examples.huggingface.residual_audit --real --nf4 \
  --warmup 10 --samples 100 --output residual-layout.json
```

The audit compares logits, token loss and every adapter gradient at zero and
nonzero B. Each variant runs in a separate CPU process with ORT graph
optimization disabled. The report includes saved tensor sizes, initializer
bytes, exported file sizes, session creation, VJP median/P95 and process peak
RSS. VJP timing uses fixed saved activations and a fixed cotangent. Process RSS
includes imports, sessions and correctness checks. Reusing an initializer also
changes what is stored in the backward model; the exported sizes and RSS show
that storage cost alongside the smaller activation boundary.

The Qwen3-0.6B audit checks all 112 adapter gradients at B=0 and B=0.01.
Logits are bitwise equal across the two layouts; the largest gradient difference
is 1.27e-7, within the original `rtol=1e-4, atol=1e-6` tolerance. The rewrite
removes `v_lm_head.Transpose_824` from the saved boundary and preserves the
forward nodes and local functions.


On an M1 Max (32 GiB), CPU FP32 with one runtime thread and sequence length 64,
10 warmups followed by 100 VJP executions give:

| Metric | Original split | Frozen weight reuse |
| --- | ---: | ---: |
| Logical saved tensors (MiB) | 1336.5 | 743.0 |
| Process peak RSS (GiB) | 8.69 | 8.97 |
| Forward + backward exports (MiB) | 1087.4 | 1680.9 |
| Session creation (s) | 1.85 | 2.02 |
| VJP median (s) | 4.584 | 4.501 |
| VJP P95 (s) | 4.646 | 4.536 |

The backward session holds the original frozen weight initializer, increasing
its model size and process RSS. Frozen weight reuse is therefore opt-in; the
bridge keeps the original split by default. Sharing frozen storage between
sessions is the next step for this layout. The [raw report](
https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/qwen3-nf4-residual-layout.json)
includes all gradient comparisons, exported graph hashes and identical frozen
forward-buffer hashes. All four exports pass the full ONNX checker.
