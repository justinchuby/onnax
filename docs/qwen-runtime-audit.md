# Separate forward and backward rounding in Qwen

A gradient depends on the forward values saved for its backward pass. To locate
an end-to-end discrepancy, evaluate the same VJP with controlled forward residuals
and loss cotangents, then compare an independent Transformers implementation.

Use an MLX EP development build containing the [dynamic operator and partition
updates](https://github.com/onnxruntime/onnxruntime-ep-mlx/pull/92).

```bash
uv sync --group dev --group training --group huggingface
uv run python -m examples.huggingface.paired_audit --real \
  --library /path/to/libonnxruntime_mlx_ep.dylib \
  --segment-nodes 1024 --output paired-qwen.json
```

Omit `--real` for the offline two-layer model. The real probe uses the pinned
Qwen3-0.6B revision, NF4 decoder weights, initial rank-four adapters and 64 fixed
token IDs. Parameters, loss and gradients use FP32. The tiny probe also includes
right padding and ignored labels.

## Hold the forward values fixed

Let $R_b$ be the residuals produced by forward backend $b$, and let
$u_b=\partial L/\partial\mathrm{logits}_b$ be its PyTorch token-loss cotangent.
The audit evaluates

$$
g_{a,b}=\operatorname{VJP}_a(R_b,u_b),
\qquad a,b\in\{\mathrm{CPU},\mathrm{MLX}\}.
$$

For each forward backend, compare the CPU and MLX VJPs on identical arrays.
Next, keep the CPU VJP fixed and swap the forward residuals and cotangent.
Finally, compare both complete ONNX paths against Transformers, which computes
its own forward values and gradients. Every comparison retains
`rtol=1e-4, atol=1e-6`; the report records each parameter's failures and errors.
Nonfinite gradients fail the audit.

The {download}`M1 Max JSON report <../examples/huggingface/reports/qwen3-nf4-paired-mlx.json>`
covers all 112 adapter tensors:

| Comparison | Maximum absolute gradient difference | Tensors outside tolerance |
| --- | ---: | ---: |
| CPU vs MLX VJP, CPU forward values | 4.57e-7 | 0 |
| CPU vs MLX VJP, MLX forward values | 4.36e-7 | 0 |
| CPU VJP, swapped forward values | 6.50e-7 | 0 |
| Complete ONNX CPU vs Transformers | 1.09e-5 | 39 |
| Complete ONNX MLX vs Transformers | 1.02e-5 | 39 |

The same 39 tensors fail the two independent Transformers comparisons. These
controls place the remaining discrepancy in the shared ONNX/reference comparison;
the same-input VJP backend checks pass. The {ref}`operator audit <qwen-operator-audit>`
compares projection, normalization and Swish arithmetic at identical inputs, with
independent scalar dot-product references.

## Control RMSNorm arithmetic

Transformers computes RMSNorm with `x * rsqrt(mean(x²) + epsilon)`. The ONNX
training decomposition uses `x / sqrt(mean(x²) + epsilon)`. The following control
temporarily replaces the reference model's RMSNorm forward methods with the
sqrt/div expression, then repeats the forward and all-adapter gradient audit:

```bash
uv run python -m examples.huggingface.rms_audit --real --nf4 \
  --output rms-control.json
```

The {download}`real-model report <../examples/huggingface/reports/qwen3-nf4-rms-control.json>`
uses the same fixed 64-token sequence, initial adapters and packed weights as the
paired audit. Across all 113 normalizations, the control leaves 39 adapter tensors
outside the original tolerance. The maximum gradient difference changes from
`1.087e-5` to `1.106e-5`. This arithmetic change does not resolve the gradient gap.
The control restores the reference methods and verifies that adapter parameters
are unchanged.

This diagnostic disables ORT tensor memory reuse during session construction.
An [initialization investigation](https://github.com/justinchuby/onnax/issues/103)
reproduces expensive reusable-tensor searches on graphs with many distinct
symbolic shapes. Session construction, execution time and execution memory are
separate measurements.

## Replay local pullbacks

For an operation $y=f(x)$ on the training path, the local audit captures its
actual ONNX input $x$ and its incoming loss cotangent $\bar y$. It compares the
generated input contribution with PyTorch's independent calculation of
$J_f(x)^T\bar y$. Multi-input operations are checked one input at a time; Split
uses the cotangents of all its outputs.

```bash
uv run python -m examples.huggingface.local_vjp_audit --real --nf4 \
  --b-values 0 0.01 --output local-vjp.json
```

The command builds one derivative graph and reuses its forward and reverse
sessions for both adapter states. B starts at zero in the first state; the
second fills B with 0.01, activating gradients for A as well. The upstream
cotangent comes from shifted token cross-entropy on the fixed 64-token sequence.
Packed weights are decoded independently with NumPy before the PyTorch matrix
product. The audit also covers elementwise operations, reductions and layouts.

The {download}`real-model local report <../examples/huggingface/reports/qwen3-nf4-local-vjp.json>`
records each operator's node count, checked input contributions and worst error,
plus every failing contribution. `unchecked_active_nodes` lists active rule
calls with no local oracle. Nonfinite gradients and interface mismatches raise
an error. The offline test injects a deliberately incorrect pullback to verify
that the oracle detects it.

Both real-model states cover all 2,552 active operations, including 193 NF4
matrix products. Each state checks 3,298 input contributions with zero failures
at `rtol=1e-4, atol=1e-6`; `unchecked_active_nodes` is empty. The largest local
absolute difference is `9.71e-7` for zero B and `9.55e-7` for nonzero B.

This report checks operation pullbacks at observed activations. The paired audit
above tracks the complete-model gradient comparison with Transformers.

## Calibrate the wide output-head reduction

The frozen output head maps 1,024 hidden features to 151,936 vocabulary entries.
Its input gradient reduces across the vocabulary. On the measured M1 Max, the
ordinary PyTorch and NumPy products return the same values at the sampled
coordinates. Scalar `math.fsum` references reveal substantially smaller errors
in ORT's product and in a short-block pairwise FP32 reduction:

| Head input-gradient calculation | Largest error at sampled coordinates |
| --- | ---: |
| ORT CPU | 2.21e-9 |
| PyTorch CPU, `highest` precision | 9.69e-7 |
| NumPy | 9.69e-7 |
| PyTorch, pairwise FP32 blocks of 128 | 3.72e-10 |

The {download}`head-reduction report <../examples/huggingface/reports/qwen3-nf4-head-reduction.json>`
records the actual operands' dimensions, coordinates, scalar references,
backend values and software versions. Reproduce it with:

```bash
uv run python -m examples.huggingface.head_reduction_audit --real --nf4 \
  --output head-reduction.json
```

`examples.huggingface.reference_head.pairwise_head` keeps the reference model's
ordinary Linear forward and changes its frozen-head pullback. Short vocabulary
blocks produce FP32 matrix products, which are accumulated with a binary carry
tree. The reference keeps its parameter names and state dict. The context restores
the original forward method on exit. Scalar higher-precision arithmetic is used
to inspect selected dot products; model computation and optimizer state remain FP32.

## Follow independent parameter trajectories

The trajectory CLI initializes two adapter copies, then executes separate AdamW
updates. It reads the version-controlled corpus, uses right-padded sequences of
length 64 and normalizes each group of four microbatches by its total valid token
count. It compares gradients, parameters and both optimizer moments after each
update. Failed final checks give a nonzero exit status after saving the report.

```bash
uv run python -m examples.huggingface.reference_trajectory --real --nf4 \
  --steps 30 --output independent.json
```

The {download}`default-epsilon run <../examples/huggingface/reports/qwen3-nf4-independent-pairwise.json>`
and {download}`epsilon 1e-4 run <../examples/huggingface/reports/qwen3-nf4-independent-pairwise-eps1e-4.json>`
both completed 30 steps. The calibrated initial gradient check passes for all 112
adapters. The initial forward comparison and independent update trajectories
still have failures at `rtol=1e-4, atol=1e-6`; each report retains them.

| AdamW epsilon | First update's maximum parameter difference | Final maximum parameter difference |
| --- | ---: | ---: |
| 1e-8 | 4.46e-4 | 3.42e-3 |
| 1e-4 | 8.50e-7 | 2.51e-4 |

The first default-epsilon update illustrates the sensitivity near zero: gradients
`9.52e-9` and `5.13e-9` produce a parameter difference of `4.46e-4`. Adam's first
step, $-\eta g/(|g|+\epsilon)$, reproduces this difference. The larger epsilon
reduces this effect, while later trajectory checks continue to fail. Both runs
use the same tolerances and verify unchanged frozen ONNX initializer hashes.

## Reuse smaller ONNX sessions

`examples.huggingface.segmented.SegmentedSession` cuts a flat tensor graph into
ordinary ONNX subgraphs and reuses one runtime session per subgraph. Tensor names
preserve the original input/output interface. Local functions stay in the exported
subgraphs, and type inference supplies the new boundary interfaces.

```python
from examples.huggingface.segmented import SegmentedSession
from examples.lora.mlx_runtime import session

executor = SegmentedSession(model, target_nodes=1024, session_factory=session)
values = executor.run(output_names, feeds)
```

`model` is an ONNX IR model. `target_nodes` is a target: a cut extends through the
last use of an unknown-rank intermediate, keeping each subgraph interface valid.
Use an unsplit session for top-level control-flow graphs. Named output selection,
initializer defaults and input overrides retain their original behavior.

The real audit creates four forward sessions and 32 VJP sessions. Construction
includes serialization, type inference, splitting and runtime initialization.
The report records these times separately from the individual execution calls.
MLX sessions disable CPU fallback; both runtime paths disable ORT graph optimization.

NumPy arrays carry values across session boundaries. Intermediate arrays are
released after their last consumer, while requested outputs remain available.
`logical_vjp_boundary_bytes` sums the input and output array sizes at these
boundaries; the real probe records about 1.54 GB per VJP call. The report also
records the development library's SHA256, software versions and frozen-buffer
verification.

The existing native training validator also accepts segmented sessions. It checks
all adapter gradients and four independent AdamW updates against Transformers:

```bash
uv run python -m examples.huggingface.mlx_validation --segmented \
  --library /path/to/libonnxruntime_mlx_ep.dylib --output segmented-updates.json
```

## Measure complete native training steps

The training benchmark accepts `--provider mlx`. It executes ONNX forward
and VJP graphs on MLX, with CPU PyTorch AdamW connected through NumPy arrays.
The joint path also computes token loss inside ONNX. `--segment-nodes` uses
the reusable subgraph sessions above.

```bash
uv run python -m examples.huggingface.benchmark \
  --provider mlx --library /path/to/libonnxruntime_mlx_ep.dylib \
  --mode saved --segment-nodes 1024 --real --nf4 --batch-size 1 \
  --warmup 10 --samples 100 > native-saved.json
```

Each measured step includes forward/loss, backward and AdamW. The report records
median/P95, valid shifted tokens per second, input-view exchange time, loss
sequence, saved activation bytes and whole-process peak RSS. Construction creates
the selected provider's sessions directly; native factory times are recorded
separately. RSS includes model construction, runtime and optimizer allocations.
MLX execution disables CPU fallback and ORT graph optimization.

Older reports below include an initial CPU session setup before MLX replacement.
Their recorded source commits identify that construction path. Repeat both
variants with the current benchmark when comparing construction time or peak RSS.

Omit `--mode` to run PyTorch CPU, raw ONNX, optimized ONNX, saved-activation ONNX,
chunked loss and joint loss/gradients in separate processes and write the matrix to `--output`.
For a fusion comparison, repeat the same command with
`ONNXRUNTIME_EP_MLX_NO_LUT4_FUSION=1`. Keep batch size, sample count, segmentation
and library identical; the report records the library hash and fusion switch.

The [M1 Max runtime matrix](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/qwen3-nf4-native-runtime-m1max.json)
uses Qwen3-0.6B NF4, FP32 computation, batch 1 and sequence length 64. Each path
runs 10 warmups and 100 measured updates on the same 63 shifted tokens.

| Path | Step median (s) | Step P95 (s) | Tokens/s | Peak RSS (GiB) |
| --- | ---: | ---: | ---: | ---: |
| MLX, saved activations, fused | 1.925 | 2.042 | 32.72 | 10.22 |
| MLX, saved activations, unfused | 7.248 | 7.603 | 8.69 | 10.69 |
| ONNX CPU, saved activations | 8.709 | 8.778 | 7.23 | 12.95 |
| PyTorch CPU, reconstructed FP32 weights | 0.361 | 0.375 | 174.52 | 4.72 |

Fusion improves the matched MLX complete step by **3.76 times**. PyTorch CPU
has the shortest steps in this matrix. The saved ONNX path retains about
1.40 GB of logical residuals; graph execution and runtime boundaries remain
the next performance targets.

## Joint graph initialization

A whole-graph Qwen3 NF4 joint loss/gradient diagnostic captured both ORT
initialization hotspots: partition `GraphViewer` construction, followed by
`SequentialPlanner::ComputeReusePlan`. The process was stopped after 21.4 minutes
of active process time, including model construction and differentiation.
The [diagnostic record](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/qwen3-joint-default-initialization.json)
contains the final native frames, source and library hashes, and pause accounting.

The [joint tutorial](qwen-adapters.md) uses coalesced eager regions from MLX EP
PR #96. Its measured session factory takes 3.78 seconds for this graph. The ORT
investigations remain in [#101](https://github.com/justinchuby/onnax/issues/101)
and [#103](https://github.com/justinchuby/onnax/issues/103).
