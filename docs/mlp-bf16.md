# Train a classifier with BF16 matrices and FP32 loss

This example extends the [MLP classifier](mlp.md) to BF16 matrix computation.
Its 99 parameters and Adam moments stay in FP32. Bias addition, tanh,
cross-entropy and the local backward arithmetic also use FP32.

The same exported model runs with ONNX ReferenceEvaluator on CPU and the
native MLX provider on Apple Silicon. Each path trains for 150 updates on
384 samples and evaluates on 192 held-out samples from the fixed-seed,
three-class dataset.

```bash
uv sync --group dev --group training
uv run --no-sync python -m examples.mlp_bf16 --output-dir mlp_bf16_models

# FP32 baseline
uv run --no-sync python -m examples.mlp_bf16 \
  --precision fp32 --output-dir mlp_fp32_models
```

For native MLX execution:

```bash
uv sync --group dev --group training --group native-mlx
uv run --no-sync python -m examples.mlp_bf16 \
  --backend mlx --output-dir mlp_mlx_models
```

The CLI prints loss, validation accuracy and strict numerical acceptance.
It saves the numerical comparisons and first-mismatch snapshots, and returns
status 1 when strict acceptance fails. Both BF16 paths in the recorded run
finish training successfully and report a difference between independent
parameter trajectories.

## Model and derivative policy

With $Q$ denoting nearest-even BF16 rounding, the forward model is

$$
Z=Q(Q(X)Q(W_1))+b_1,\qquad H=\tanh(Z),
$$

$$
S=Q(Q(H)Q(W_2))+b_2,\qquad
L=-\frac1N\sum_{i,c}Y_{ic}\log\operatorname{softmax}(S_i)_c.
$$

Training changes both weight matrices and both bias vectors. The gradient
of the loss with respect to logits is $(\operatorname{softmax}(S)-Y)/N$.
Backpropagation carries it through the output weights and tanh to learn
the hidden features.

The [BF16 rules](bf16.md) define identity derivatives at rounding boundaries
and FP32 local arithmetic. For example,

$$
\nabla_{W_2}L=Q(H)^\mathsf{T}\nabla_S L,\qquad
\nabla_{b_2}L=\sum_i(\nabla_S L)_i.
$$

The first-layer gradients use $Q(X)$ and the same stable tanh-slope expression
as the LoRA example. Tests compare the analytic gradients with central
differences of a smooth local surrogate that holds rounding offsets fixed.

## Keep cross-entropy stable

After differentiation, the example expands the primal `LogSoftmax` into
standard ONNX operators:

$$
M=\max_c S_c,\qquad T=S-M,\qquad
\log p=T-\log\sum_c\exp(T_c).
$$

This keeps the existing LogSoftmax derivative rule and gives
ReferenceEvaluator a stable forward expression. Tests use logits
`[-1000, 0, 1000]`, verify a finite loss of 2000 for a class-zero target,
and check invariance after adding 10,000 to every logit. The same test runs
on native MLX when enabled.

## Connect an external update loop

The exported gradient model accepts `features`, `targets` and four explicit
FP32 parameter inputs:

| Parameter | Shape | Gradient dtype |
| --- | --- | --- |
| `layer1.weight` | `[2, 16]` | FP32 |
| `layer1.bias` | `[16]` | FP32 |
| `layer2.weight` | `[16, 3]` | FP32 |
| `layer2.bias` | `[3]` | FP32 |

`interface.json` records the actual gradient output names. The training loop
reuses one session, fetches logits/loss/gradients and calls the existing
external Array API Adam with `xp=np`. MLX execution returns FP32 NumPy arrays
at the runtime boundary; this example updates parameters on CPU.

Parameter storage is 396 bytes and the two Adam moments total 792 bytes.
`checkpoint.npz` stores all four parameters, both moments and the step count.
Loading it into a fresh session reproduces outputs and the next update.

Artifacts include `loss.onnx`, raw and optimized gradient graphs, a validation
loss graph and `gradient.dynamic.onnx`. The training and validation graphs
specialize batches to 384 and 192 for both runtimes. The dynamic graph is
reloaded with ReferenceEvaluator at batch sizes 1, 7 and 192. Exported graphs
pass full ONNX checker and contain standard-domain operators. ONNXScript
optimizes copies; ORT graph optimization is disabled.

## Simplify the backward shape graph

The classifier exposed two unnecessary shape constructions in the derivative
generator. A reduction with removed axes now restores them directly with
`Unsqueeze`, including runtime negative and empty axes. A known leading
broadcast reduction removes its leading singleton dimensions with `Squeeze`.

This removes the shape scatter and reshapes that prevented the native MLX
provider from accepting the complete gradient graph. Tests cover runtime
axes, scalar biases, singleton dimensions and empty tensors. These simpler
adjoints apply to FP32 and BF16 workloads alike. Integer loss-axis constants
retain their INT64 dtype during BF16 matrix lowering.

## Check the results

All four recorded paths reach **100% validation accuracy**. BF16 validation
loss falls from about 1.07635 to 0.0021004; the FP32 baseline reaches about
0.0020997.

| Runtime | Precision | Same-parameter forward/loss/gradients | Shared-gradient Adam updates | Independent trajectory |
| --- | --- | --- | --- | --- |
| ReferenceEvaluator | FP32 | Pass | Pass | Pass |
| ReferenceEvaluator | BF16 | Pass | Pass | Difference from step 14 |
| MLX | FP32 | Pass | Pass | Pass |
| MLX | BF16 | Pass | Pass | Difference from step 6 |

Comparisons use `rtol=1e-4, atol=1e-6`. Shared-gradient checks drive a separate
flattened NumPy Adam implementation with the observed gradients, and compare
parameters and moments at every step. The independent trajectory computes
its own analytic gradients and updates.

The first BF16 trajectory differences occur while the rounded weights are
still identical. Parameter differences of roughly $10^{-7}$ move one or two
hidden values across BF16 rounding boundaries. The saved snapshots record
both parameter states and the changed rounded hidden values.

## Measure complete updates

These Apple M1 Max measurements use sequential, independent worker processes,
10 warmups and 100 measured updates. For each precision, both runtimes execute
byte-identical serialized training graphs. Each step fetches logits, loss
and all gradients and performs CPU Adam.

| Runtime | Precision | Execute/exchange median (ms) | Adam median (ms) | Step median (ms) | Step P95 (ms) | Peak process RSS (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ReferenceEvaluator | FP32 | 0.358 | 0.028 | 0.387 | 0.397 | 119.8 |
| ReferenceEvaluator | BF16 | 0.432 | 0.029 | 0.461 | 0.476 | 120.0 |
| MLX | FP32 | 0.480 | 0.030 | 0.511 | 0.772 | 146.4 |
| MLX | BF16 | 0.529 | 0.030 | 0.559 | 0.776 | 147.1 |

The FP32 reference path has the shortest recorded step for this 99-parameter
model. Execution/exchange includes synchronous host transfers, provider
copy-out and output allocation. Peak RSS includes Python, backend libraries,
the loaded model and runtime allocators. The report records model-read,
session-creation and first-step times separately; native ORT sessions use
one intra-op and one inter-op thread.

The [measurement and correctness report](https://github.com/justinchuby/onnax/blob/main/examples/measurements/mlp-mixed-precision-apple-arm64.json)
contains software versions, graph hashes, numerical comparisons and the
first BF16 trajectory diagnosis.
