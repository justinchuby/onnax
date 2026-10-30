# Check BF16 matrix-accumulation error

The [native MLX training evaluation](mlx-bf16.md) can check every forward BF16
matrix product against a derived accumulation-error interval. On the recorded
200-update LoRA run, all **800 products and 3,072,000 output elements** fit
those intervals. Instrumenting the graph preserved all original outputs bit for
bit.

```bash
uv sync --group dev --group training --group native-mlx
uv run --no-sync python -m examples.lora.mlx_bf16 \
  --inspect-products --output-dir mlx_bf16_bounds
```

The CLI prints the local product-bound result and the existing strict numerical
comparison separately. The recorded run passes the product bounds and returns
exit status 1 for the strict comparison. `report.json` retains both results.

## Derive the interval

For one output, write the exact dot product and absolute-product sum as

$$
s=\sum_{i=1}^{k}a_i b_i,\qquad S=\sum_{i=1}^{k}|a_i b_i|.
$$

Assume FP32 elementary operations obey
$|\operatorname{fl}(t)-t|\le u|t|+\tau$, with
$u=2^{-24}$ and $\tau=2^{-126}$. The absolute term covers gradual underflow
and flushing arithmetic results to zero. Operands are finite normal BF16
values or zero; the implementation rejects subnormal operands and overflow.

Budget $m=2k$ elementary operations for multiplication and addition.
The deterministic product-of-rounding-errors bound gives
$\gamma_m=mu/(1-mu)$ for $mu<1$; see the introduction and Lemma 2.2 of
[Higham and Mary](https://eprints.maths.manchester.ac.uk/2673/1/paper.pdf).
Adding the absolute underflow terms gives this experiment's interval:

$$
E=\frac{muS+m\tau}{1-mu},\qquad
\widehat{s}\in[s-E,s+E].
$$

This conservative operation budget covers different summation trees and fused
multiply-add evaluation under the stated elementary-operation model. Each
rounding perturbation contributes a factor bounded by $1+u$; their products
are bounded by $1/(1-mu)$. At most $m$ absolute underflow terms receive those
same factors.

Let $Q$ round directly to nearest-even BF16. Since $Q$ is monotone, the final
BF16 output must lie in

$$
[Q(s-E),Q(s+E)].
$$

For values far from a BF16 midpoint, both endpoints round to the same value.
Near a midpoint, the interval can contain adjacent BF16 values. The resulting
acceptance interval depends on the input data.

## Compute the reference exactly

Every BF16 value is an integer multiple of $2^{-133}$, so products and sums
are integer multiples of $2^{-266}$. The helper uses Python integer matrix
products for both $s$ and $S$, forms rational interval endpoints, and makes
nearest-even decisions by integer division and remainder. Floating-point
arithmetic enters only when storing the final representable BF16 endpoints
as FP32 arrays.

```python
import numpy as np
from examples.lora.bf16_bounds import product_bounds

# This exact dot lies just above the midpoint between 1 and 1.0078125.
a = np.array([[1, 2**-8, 2**-26]], dtype=np.float32)
b = np.ones((3, 1), dtype=np.float32)
bounds = product_bounds(a, b)
print(bounds.lower, bounds.upper)  # [[1.]] [[1.0078125]]
print(bounds.check(np.array([[1.015625]], dtype=np.float32))["violations"])  # 1
```

Tests also compare direct rational rounding with an independent nearest-point
search over the finite BF16 lattice. They exercise signed midpoints,
perturbations smaller than FP64 resolution, subnormal results, cancellation,
all summation orders of a small dot product, and rejection of wrong outputs.

## Inspect the actual training graph

An instrumented copy exposes each BF16 MatMul's two operands and result as
FP32 graph outputs. It executes at the same parameters as the original graph.
The report checks original-output equality and evaluates each product at
its observed inputs.

The [recorded summary](https://github.com/justinchuby/onnax/blob/main/examples/measurements/mlx-bf16-bounds-apple-arm64.json)
contains per-product totals and the arithmetic model. Of 3,072,000 output
intervals, 31,129 contain multiple BF16 values; no observed output falls
outside its interval.

These checks cover local forward matrix arithmetic. The existing same-parameter
and independent-training-trajectory comparisons continue to cover predictions,
loss, gradients and parameter updates. Composing local error bounds through
nonlinearities and backward arithmetic is the next numerical-contract step.
