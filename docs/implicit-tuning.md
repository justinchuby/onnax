# Differentiate through a fitted model

Choose a model's regularization strength using validation loss. This example
exports ONNX derivative products, solves the model parameters with conjugate
gradients (CG), then differentiates the fitted solution to update the penalty.

```sh
uv sync --group dev
uv run python -m examples.implicit_tuning --output-dir implicit_tuning_models
```

The offline FP32 example uses 64 training observations, 96 validation observations
and eight features. In 30 updates, validation loss falls from about **1.094 to
0.0182**, while the ridge penalty moves from **1.5 to 0.0509**.

## Two levels of learning

The inner problem fits a linear predictor. Writing the positive penalty as
$\lambda=\exp(\rho)$, its objective is

$$
T(\theta,\rho)=\frac{1}{2n}\|X\theta-y\|^2
+\frac{\exp(\rho)}{2}\|\theta\|^2.
$$

The outer problem chooses $\rho$ to reduce validation loss:

$$
F(\rho)=V(\theta^*(\rho)),\qquad
V(\theta)=\frac{1}{2m}\|X_v\theta-y_v\|^2.
$$

At the inner optimum, $g(\theta^*,\rho)=\nabla_\theta T=0$.
Differentiating this equation gives

$$
H\frac{d\theta^*}{d\rho}+c=0,\qquad
H=\frac{\partial g}{\partial\theta},\quad
c=\frac{\partial g}{\partial\rho}.
$$

Solve $Hu=\nabla_\theta V$, then compute the scalar hypergradient

$$
\frac{dF}{d\rho}=-u^\mathsf{T}c.
$$

The outer update is simply $\rho\leftarrow\rho-\eta\,dF/d\rho$.
It differentiates the optimum using a linear solve; it does not retain a tape
of the inner solver's iterations.

## Compose the derivative graphs

```python
gradient = onnax.grad(train, inputs=["parameters"], output="loss")
name = gradient.graph.outputs[-1].name
hvp = onnax.jvp(gradient, inputs=["parameters"], outputs=[name])
mixed = onnax.jvp(gradient, inputs=["log_penalty"], outputs=[name])
```

The HVP graph takes a parameter-space direction $v$ and returns $Hv$. The mixed
JVP graph takes the scalar direction `1` and returns $c$. A separate `grad` graph
provides $\nabla_\theta V$. The external NumPy loop runs CG and updates $\rho$;
onnax supplies the derivative graphs.

For ridge regression, $g$ is affine in $\theta$. The inner CG solve uses
$H\theta^*=-g(0,\rho)$, and the adjoint CG solve reuses the same HVP session.
The positive ridge penalty makes both systems positive definite. CG uses FP32
arrays and an absolute residual tolerance of $2\times10^{-6}$.

## Export and validate

The example saves four ordinary ONNX modules and reloads them into reusable
CPU ONNX Runtime sessions. ONNXScript optimizes the graphs; ORT graph
optimization is disabled. `report.json` records the input/output mapping,
validation history, CG iteration counts and numerical checks.

Add `--per-feature` to learn eight positive penalties, one per feature:

```sh
uv run python -m examples.implicit_tuning --per-feature --output-dir feature_penalties
```

Here $\rho$ is a vector and $c$ becomes the mixed Jacobian $G$. The update needs
$-G^\mathsf{T}u$, which a single `vjp(gradient, inputs=["log_penalty"])`
computes directly. With the same observations and 30 updates, validation loss
falls from about **1.094 to 0.0566**. Each fitted penalty stays positive through
the exponential parameterization.

An independent dense ridge solution checks the fitted parameters, HVPs, mixed
products and hypergradients at every outer step. A centered finite difference
perturbs each coordinate of $\rho$ and independently solves the inner problem
again, checking the derivative of the complete outer objective. This finite-
difference reference uses FP64 to subtract nearby losses; the ONNX graphs, CG
solves and outer updates use FP32. The dense matrix belongs to this reference;
the ONNX solver uses vector products.

This is a starting point for learning regularization strengths in small models.
For nonlinear inner objectives, the same stationarity equation uses an HVP
at the fitted parameter point. The [matrix-free fitting tutorial](fitting.md)
shows how to connect derivative products to an external nonlinear solver.
