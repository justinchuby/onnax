# FP32 reference and optimizer trajectory checks

Compare gradients at shared parameters, then follow independent updates to
measure accumulated numerical differences. These checks use the
[NF4 Qwen training example](qwen-nf4.md), with FP32 parameters, gradients and
optimizer state.

## Check the FP32 reference

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

The earlier integrated 30-step run, using the expanded CE derivative, records a decrease
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

## Compare independent optimizer trajectories

The native validator independently updates ONNX and Transformers adapters.
It compares logits, loss, gradients and parameters at every step. ONNX sessions
use MLX; CPU fallback and ORT graph optimization are disabled. PyTorch computes
loss and AdamW on CPU; NumPy arrays cross the session boundary.

```bash
uv run python -m examples.huggingface.mlx_validation \
  --library /path/to/libonnxruntime_mlx_ep.dylib --output mlx-qwen.json
```

Omit `--library` to use the installed plugin. The report records the library,
package versions, execution boundary and numerical tolerances. Run the native
regression with `ONNAX_TEST_MLX_EP=1` and set
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

A full optimizer-state trace locates the first native parameter failure at
step 95, `model.layers.0.self_attn.q_proj.lora.B[2,17]`. Its parameter gap
grows from -8.61e-7 to -2.02e-6 during that step. Swapping the current gradient
and moment history in the Adam formula attributes -1.71e-6 of the update gap
to history and +5.58e-7 to the current gradient. The current gradients are
about 0.00537; this trajectory crosses tolerance through accumulated history.

Reconstruction from the stored FP32 moments matches every captured update
within 1.85e-9. Capturing the states preserves all original per-step checks
and loss values exactly. The
[first-divergence report](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/tiny-nf4-adam-first-divergence.json)
contains the coordinate's complete 100-step history and the reconstruction
scripts. Scalar reconstruction uses host arithmetic; training remains FP32.

The [paired runtime audit](qwen-runtime-audit.md) compares CPU and MLX gradients
with fixed forward residuals and provides reusable segmented ONNX execution.

