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
