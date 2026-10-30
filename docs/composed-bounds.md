# Propagate BF16 error through the gradient graph

The [matrix-product checks](bf16-bounds.md) validate each dot product at its
observed inputs. This example propagates intervals from the original inputs
and frozen weights through the complete exported LoRA graph. The resulting
intervals cover predictions, mean squared loss and the FP32 straight-through
gradients of A and B.

```bash
uv sync --group dev --group training --group native-mlx
uv run --no-sync python -m examples.lora.mlx_bf16 \
  --inspect-graph --output-dir mlx_composed_bounds
```

The CLI records composed intervals and strict NumPy comparisons separately.
The recorded run passes the intervals and returns status 1 for the existing
strict comparison at step 68. Both results remain in `report.json`.

## Follow the actual computation

Let $Q$ round to BF16. The forward graph computes

$$
U=Q(Q(X)Q(A)),\quad D=Q(UQ(B)),\quad
Z=Q(Q(X)Q(W))+\tfrac14D,
$$

$$
H=\tanh(Z),\qquad P=Q(Q(H)Q(C)),\qquad
L=\operatorname{mean}((P-Y)^2).
$$

The derivative convention treats rounding as an identity and evaluates the
local chain rule in FP32. Its tanh slope uses

$$
e=\exp(-2|Z|),\qquad \tanh'(Z)=\frac{4e}{(1+e)^2}.
$$

The checker evaluates each exported node in graph order. It carries input
uncertainty through the residual, reduction, slope and matrix products that
produce the A/B gradients. Parameters are fixed during each check; the
200-step loop repeats the analysis at each native parameter state.

## Compose the arithmetic bounds

Each value has lower and upper FP64 endpoints. Elementary interval operations
round their endpoints outward with `nextafter`. FP32 arithmetic then expands
the exact-result interval using the rounding/underflow model from the matrix
checks: $u=2^{-24}$, $\tau=2^{-126}$ and $\gamma_m=mu/(1-mu)$.
Matrix products use $m=2k$; the mean uses $m=2N+2$ and the sum of absolute
input values divided by $N$. FP32 overflow stops the analysis.

For uncertain matrices $A=A_c+\Delta A$ and $B=B_c+\Delta B$, with
$|\Delta A|\le A_r$ and $|\Delta B|\le B_r$, operand uncertainty contributes

$$
|\Delta(AB)|\le |A_c|B_r+A_r|B_c|+A_rB_r.
$$

The reference matrix products also include an FP64 accumulation-error bound.
BF16 conversion rounds the interval endpoints directly from FP64 to BF16,
resolving midpoint decisions before an FP32 conversion can change them.

### Enclose tanh and exp

The reference uses a positive Taylor series for $q=\exp(t)-1$. For
$0\le t\le80/256$, it sums terms through degree 24 and bounds the remainder by

$$
R\le\frac{t^{25}/25!}{1-t/26}.
$$

Eight applications of $q\leftarrow q(q+2)$ reconstruct
$\exp(256t)-1$. Reciprocal intervals handle negative exponential arguments;
$\tanh(x)=\operatorname{sign}(x)q/(q+2)$ uses $q=\exp(2|x|)-1$.
This form retains small arguments without subtracting two nearly equal numbers.
The reference supports exp arguments in $[-80,80]$ and tanh arguments in
$[-40,40]$; larger arguments stop the analysis.

The FP32 tanh and exp implementations use a declared four-ULP budget plus
the absolute underflow term. ULP spacing is the largest FP32 spacing
in the real-function enclosure, with subnormal spacing $2^{-149}$.
The inspector checks that budget at every
observed nonlinear input. It also checks local matrix, reduction and
elementary-operation budgets, and verifies that observed floating operands
are normal values or zero. These checks accompany the composed enclosures.

## Inspect the training run

On Apple M1 Max, all 200 steps pass the composed checks for both native and
independent same-parameter NumPy results. The instrumented graph preserves
all original outputs byte for byte. The run checks 10,000 intermediate
enclosures, 4,800 local arithmetic budgets and 400 nonlinear budgets.

| Output | Elements checked per implementation | Largest interval width |
| --- | ---: | ---: |
| Prediction | 409,600 | 0.0078125 |
| Loss | 200 | 0.00001548 |
| A gradient | 12,800 | 0.00003196 |
| B gradient | 19,200 | 0.00001124 |

At step 68, the previously observed native and NumPy predictions round to
adjacent BF16 values, -0.1865234375 and -0.1875. The propagated interval at that coordinate is exactly
`[-0.1875, -0.1865234375]`. The loss and both parameter gradients
also fit their composed intervals at that step.

The [composed-bound report](https://github.com/justinchuby/onnax/blob/main/examples/measurements/mlx-composed-bounds-apple-arm64.json)
records output widths, intermediate and local-budget checks, the midpoint
case, runtime versions and the existing strict-comparison result. This is
a numerical inspection run; the [isolated workload benchmarks](mlx-training.md)
measure training-step costs.

Tests compare arithmetic endpoints with exact fractions and nonlinear
endpoints with an independent 120-digit Decimal reference. They exercise
BF16 midpoints, uncertain matrix operands, corrupted outputs and nonlinear
observations, subnormal-operand rejection, and the complete exported graph.
