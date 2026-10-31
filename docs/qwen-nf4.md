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
completed 30 native chunked-loss updates. Validation loss fell from **7.75064
to 5.63150**. A fresh process resumed checkpoint 15 and reproduced all 112
adapters, 113 checkpoint tensors, optimizer/scheduler state and loaded ONNX
outputs byte for byte. Frozen source weights remained unchanged. PEFT adapter
format exchange also preserved all parameter bytes.

The initial and final comparisons against Transformers exceed the FP32
tolerance for 47 and 90 gradient tensors, respectively. PEFT gradient and
independent-update comparisons also fail. The
[native SFT report](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/qwen3-nf4-native-sft-m1max.json)
contains these failures alongside the training and recovery results.

### Check the FP32 reference

Long-vocabulary CE reductions affect reference accuracy. In a scalar-oracle
probe with 151,936 classes, ORT CPU and PyTorch CPU return CE 0.11903559 for
an analytic value of 0.11897194. The minimal reproduction is tracked in
[issue #111](https://github.com/justinchuby/onnax/issues/111).

The independent PyTorch reference computes
$\operatorname{softplus}(\log\sum_{j\ne y}\exp(z_j)-z_y)$ and accumulates
frozen-head products in short FP32 blocks. This preserves the small target
gradient and improves the long reduction. Add `--stable-reference` together
with `--diagnose-rounding` to record these extra audits at the initial and
final states. The original Transformers comparisons remain in `audits`.

```python
from examples.huggingface.reference_head import pairwise_head
from examples.huggingface.reference_losses import stable_token_loss
from examples.huggingface.validation import audit

with pairwise_head(reference):
    result = audit(
        model,
        reference,
        batch,
        reference_loss=stable_token_loss,
        diagnose_rounding=True,
    )
```

At the real model's initial and trained states, all 112 gradients pass the
original tolerance against this stable FP32 reference. On fixed FP32 logits,
the CE gradient error against an analytic oracle falls from 9.52e-6 to
9.12e-9 initially, and from 5.72e-6 to 9.19e-9 after training. Moving the head
normalization before its matrix product gives little change.

The integrated 30-step SFT CLI run reproduces the validation loss decrease
from 7.75064 to 5.63150 with these additional audits enabled. All 112 adapter
gradients pass at both endpoints; maximum absolute differences are 2.35e-7
and 9.69e-7. Frozen weights remain byte-identical. Pointwise logits still
exceed the original tolerance at both endpoints. The
[integrated run report](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/qwen3-nf4-native-stable-sft-m1max.json)
records both sets of comparisons and the runtime library hash.

Independent fresh AdamW updates still exceed tolerance for 48 and 87 parameter
tensors at these states with epsilon 1e-8. The
[reference-control report](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/qwen3-nf4-reference-ce-m1max.json)
retains these update failures and every original-reference comparison.

For example, two gradients of 9.12e-9 and 3.73e-9 produce first updates
-0.001431 and -0.000815. The first-step Adam formula
$-\eta g/(|g|+\epsilon)$ reproduces their difference to about 1e-10.
Replaying every captured component matches this formula within FP32 rounding
bounds. Near-zero gradient sensitivity remains visible in the parameter
comparison.

The analytic hard-CE derivative also passes all 112 adapter gradient checks
at the initial and trained states, with maximum errors of 2.38e-7 and 9.91e-7.
Independent fresh Adam checks still exceed tolerance for 48 and 87 tensors.
The [analytic CE control report](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/qwen3-nf4-analytic-ce-controls-m1max.json)
records the exact gradients and first-step formulas at the largest parameter
differences. This loss fix removes the non-target pivot cancellation described
in [FP32 loss accuracy](softmax-precision.md); the remaining update drift
requires further localization.

The native MLX validation uses the same two-layer NF4 Qwen and independently
updates its ONNX and Transformers adapters for four AdamW steps. It compares
logits, token loss, all adapter gradients and parameters at every step. ONNX
forward and VJP sessions require complete MLX assignment and disable ORT graph
optimization. PyTorch computes loss and AdamW on the CPU; NumPy arrays cross the
session boundary.

Use an MLX EP build containing [dynamic Range and ScatterElements support](
https://github.com/onnxruntime/onnxruntime-ep-mlx/pull/90):

```bash
uv run python -m examples.huggingface.mlx_validation \
  --library /path/to/libonnxruntime_mlx_ep.dylib --output mlx-qwen.json
```

Omit `--library` to use the installed plugin. The report records the library,
package versions, execution boundary and numerical tolerances. The checked-in
M1 Max report has loss decreasing from 5.50393 to 5.49281 across four steps.
Run the native regression with `ONNAX_TEST_MLX_EP=1` and set
`ONNAX_MLX_EP_LIBRARY` for a development dylib:

```bash
ONNAX_TEST_MLX_EP=1 ONNAX_MLX_EP_LIBRARY=/path/to/libonnxruntime_mlx_ep.dylib \
  uv run pytest tests/test_qwen_mlx.py -q
```

Use `--steps 100` to extend the comparison. Each step also evaluates a copied
Transformers reference at the ONNX adapter parameters. This separates derivative
accuracy at one point from the accumulated difference between two independently
updated parameter trajectories. The control preserves both training trajectories
and restores the ONNX gradients before AdamW runs.

On the M1 Max 100-step run, loss decreases from 5.50393 to 4.85798. Both the
fused and unfused paths pass every same-parameter forward, loss and gradient
comparison at `rtol=1e-4, atol=1e-6`. The independent trajectories first exceed
gradient tolerance at steps 92 and 87, respectively. All fused-path parameter
comparisons pass; unfused parameter comparisons exceed tolerance at steps 96
and 99. Both complete reports retain `strict_checks_passed=false`; the CLI
writes the report and exits with status 1. The
[100-step comparison report](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/tiny-nf4-mlx-100-comparison.json)
records the per-step checks and runtime library hash.

Use `--stable-reference` in the same validator to run an independent trajectory
with the target-excluded CE and pairwise frozen-head backward:

```bash
uv run python -m examples.huggingface.mlx_validation \
  --library /path/to/libonnxruntime_mlx_ep.dylib \
  --steps 100 --stable-reference --output stable-reference-100.json
```

The report identifies the reference computation. Learning rate, Adam epsilon
and comparison tolerances match the default run. Each step also compares
gradients at shared parameters while preserving the independent optimizer
states. A failed comparison saves the complete report and exits with status 1.

In the 100-step stable-reference run, all shared-parameter forward, loss and
gradient checks pass. Independent parameter checks exceed tolerance at steps
95–100; independent gradient checks first exceed it at step 87. Loss decreases
from 5.50393 to 4.85798. The
[stable-reference trajectory report](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/tiny-nf4-mlx-stable-reference-100.json)
retains all step comparisons and `strict_checks_passed=false`.

Run the matched CPU control with `--provider cpu` and omit `--library`. In
its 100-step stable-reference run, all independent parameter and loss checks
pass, as do all shared-parameter checks. Independent gradients exceed
tolerance at steps 92–100 and logits at steps 94–100. At step 100, the largest
parameter difference is 8.90e-7 on CPU and 2.56e-6 on MLX. The
[CPU control report](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/tiny-nf4-cpu-stable-reference-100.json)
preserves these results. Both runtimes therefore show late independent
gradient drift; the additional MLX parameter drift is the next comparison
to localize.

Use `--backward-provider` to choose the VJP runtime separately. In matched
100-step controls, changing either forward or backward to CPU reduces parameter
failures to steps 96 and 99. All shared-parameter checks pass in all four
combinations:

| Forward | VJP | Parameter failure steps | Step 100 maximum parameter difference |
| --- | --- | --- | ---: |
| CPU | CPU | none | 8.90e-7 |
| CPU | MLX | 96, 99 | 1.87e-6 |
| MLX | CPU | 96, 99 | 1.70e-6 |
| MLX | MLX | 95–100 | 2.56e-6 |

The comparison uses elementwise relative and absolute tolerances; maximum
absolute difference alone does not determine a failure. Both forward residuals
and backward arithmetic contribute to the independent update drift. The
[runtime trajectory matrix](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/tiny-nf4-runtime-trajectory-matrix.json)
includes full hybrid results and commands to reproduce each combination.

The [paired runtime audit](qwen-runtime-audit.md) compares CPU and MLX gradients
with fixed forward residuals and provides reusable segmented ONNX execution.

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
