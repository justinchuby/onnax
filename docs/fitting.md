# Matrix-free nonlinear parameter fitting

Fit a decay curve from 128 noisy observations using ordinary ONNX derivative graphs and
an external damped Gauss–Newton solver. The offline example recovers an amplitude, decay
rate and background offset.

```bash
uv sync --group dev --group training
uv run python -m examples.fitting --output-dir fitting_models
```

## Model and products

For observations $(t_i,y_i)$, parameters $p=(a,b,c)$ predict $f_i(p)=a\exp(-bt_i)+c$.
Define residuals $r_i=f_i-y_i$ and $L(p)=\tfrac12\sum_i r_i^2$. The Jacobian $J$
differentiates the residual vector with respect to the three parameters. Then

$$
g=J^\mathsf{T}r,\qquad
(J^\mathsf{T}J+\lambda I)\Delta=-g.
$$

The example exports `onnax.jvp(model, inputs=["parameters"], outputs=["residual"])` and
the corresponding `onnax.vjp`. A CG iteration passes its direction $v$ to the JVP graph,
then passes the resulting $Jv$ to the VJP graph. Adding $\lambda v$ in external code
supplies the damped operator. The solver never builds or stores $J$ or $J^\mathsf{T}J$.
Each VJP reuses the current parameter point; its cotangent seed is an independent
runtime input. CG solves to an absolute residual tolerance of $10^{-11}$, with bounded
work and explicit failures for invalid curvature or nonconvergence. Armijo backtracking
bounds the nonlinear update; damping is $\lambda=10^{-3}$ in this example.

This is a Gauss–Newton approximation. The exact Hessian also contains residual
curvature:

$$
Hv=J^\mathsf{T}(Jv)+\sum_i r_i\nabla^2r_i\,v.
$$

The exported `hvp` graph uses `onnax.jvp(onnax.grad(...))`. Its values are checked
against this full expression at a point where residuals are nonzero; they differ from
the Gauss–Newton product. Damping makes the Gauss–Newton operator positive definite for
CG. An exact nonlinear Hessian can be indefinite. The [Jelly
playground](https://www.justinchuby.com/onnax/jelly/) illustrates safeguarded Newton-CG
with exact HVPs in a different application.

`onnax.vmap(jvp_graph, in_axes=(None, None, None, 0))` additionally exports a batch of
independent directions at one shared parameter point. The example checks five directions
against the independent reference. CG uses one direction per iteration.

## Verification and artifacts

Fixed seed 42 generates data near $(a,b,c)=(2,1.3,0.4)$ with Gaussian noise of standard
deviation 0.005. Starting at $(1.4,0.8,0.1)$, the reference development run reduces $L$
from 6.787 to about 0.000997 and recovers approximately $(2.00036,1.29710,0.39869)$. The
added noise accounts for the difference from the data-generating parameters.

The CLI verifies:

- Raw and ONNXScript-optimized JVP, VJP and exact HVP results against an independent
  analytic NumPy reference, for observation counts 1, 7 and 128.
- The adjoint identity $\langle Jv,w\rangle=\langle v,J^\mathsf{T}w\rangle$,
  the Gauss–Newton product, batched directions and the full solver trajectory.
- Final stationarity, a loss decrease of at least 99%, parameter recovery and
  execution after saving/loading both graph and fitted parameters.

Tests independently check the analytic Jacobian and Hessian with central finite
differences. A dense Jacobian and Hessian exist only in this small validation oracle;
production solver callbacks use ONNX products. Solver unit tests compare CG with a dense
linear solve and exercise failure paths.

The directory contains raw and optimized `residual`, `jvp`, `vjp`, `hvp` and
`batched_jvp` graphs, checked with the full ONNX checker. `interface.json` records
versioned graph input/output names and parameter order. Use these names when binding
seeds and outputs. `fit.npz` stores fitted parameters and observations, and
`report.json` contains the loss trajectory, graph sizes, node counts and measured
runtime information.

## Runtime and measurements

All arrays and graphs use FP64 and CPU ONNX Runtime, with graph optimization disabled
and one intra/inter-op thread. ONNXScript optimizes independent graph copies; the source
graph is verified unchanged. The external solver owns damping, CG, line search and
saving parameters.

A separate process measures session creation, a complete two-session Gauss–Newton
product and a complete solve, with 10 warmups and 100 measured repetitions (median/P95).
Product time includes CPU feeds, returned arrays and the intermediate host exchange;
solve time also includes Python CG and line search. Export and the dense reference are
outside measurement. Peak RSS includes Python imports, tensors, the runtime and
allocator overhead. Hardware, package versions, thread settings, model bytes and node
counts accompany the results.

Mapped graphs also return a copy of the primal outputs for each direction. For larger
inverse problems, the next extensions are structured residuals, cached linearizations,
preconditioners and device bindings.

A [recorded development-machine
report](https://github.com/justinchuby/onnax/blob/main/examples/measurements/decay-fit-apple-arm64.json)
is included for inspection. Rerun the CLI for measurements on your own machine.

(adapter-curvature)=

## Inspect LoRA adapter curvature

The same products can diagnose a fine-tuning model. For the
[LoRA workload](lora.md), write its prediction as $f(A,B)$ and its mean squared
loss as $L=\|f(A,B)-y\|^2/N$, where $N$ counts prediction elements.

```sh
uv sync --group dev --group training --extra torch
uv run python -m examples.lora.curvature --output-dir adapter-curvature
```

The example exports two matrix-free operators:

$$
Hv=\operatorname{JVP}(\nabla L,v),\qquad
Gv=\frac{2}{N}J_f^T(J_fv).
$$

`jvp(grad(...))` computes the exact Hessian product. Composed prediction JVP
and VJP graphs compute the generalized Gauss–Newton product. Its directional
curvature is $v^TGv=2\|J_fv\|^2/N\ge0$.

At the usual $B=0$ initialization, the prediction is independent of $A$, so
$\nabla_A L=0$. The mixed A/B Hessian blocks can still be nonzero. The example
uses those products to construct a unit direction with negative exact
curvature and compares it with the nonnegative Gauss–Newton curvature. This
reveals coupling that a first-order gradient inspection misses.

All products execute in FP32. Independent PyTorch products check zero and
nonzero adapters at three batch sizes, before and after ONNXScript optimization.
The command writes six checker-validated ONNX files and `report.json`; frozen
weights and the source graph remain unchanged. The exported products can be
reused by an external curvature diagnostic or solver.


## Token-loss curvature for language-model adapters

For causal-language-model logits $z(\theta)$ and mean valid-token cross-entropy,
the generalized Gauss–Newton product is

$$
Gv=J_z^T C J_zv,\qquad
Cd=\frac{m}{N}\,p\odot\left(d-\sum_{\mathrm{vocab}}p\odot d\right),
\quad p=\operatorname{softmax}(z).
$$

Here $m$ selects valid shifted targets and $N$ counts them. The product measures
how an adapter direction changes the predicted token distribution. Its scalar
curvature $v^TGv$ is nonnegative. Unlike the exact Hessian, this operator uses
the loss curvature without second derivatives of the decoder.

The example composes JVP, the softmax curvature action and VJP into one ordinary
ONNX graph. JVP and VJP share the same primal values. Labels shift one position
for causal prediction, `-100` masks ignored targets, and all-ignored batches
produce zero products.

```sh
uv sync --group dev --group training --group huggingface
uv run python -m examples.huggingface.validate_curvature \
  --output-dir token-curvature
```

The default runs an offline two-layer Qwen3 with rank-4 q/v adapters at a seeded
nonzero adapter point. It checks every FP32 adapter product against independent
Transformers JVP, cross-entropy Hessian-vector and VJP calculations. The CLI
also uses `vmap` to check three directions against sequential execution while
sharing the input batch, labels and parameters.

`single.onnx` and `batched.onnx` contain the exported products, with shared
external weight data. Their `onnax.example.causal_ggn` graph metadata records
ordered parameter names, direction inputs, labels and product outputs.
`feeds.npz`, `outputs.npz` and `batched_feeds.npz` preserve named arrays;
`report.json` records the reference, tolerances, graph sizes and runtime versions.
ORT graph optimization is disabled. The tests cover padding, position IDs,
single-token sequences and unequal valid-token microbatch weighting.

The [M1 Max report](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/causal-curvature-m1max.json)
records the offline CPU result and 14 passing CPU/MLX decoder tests. A separate
Qwen3-0.6B diagnostic at sequence length 64 measures directional curvature
3.359475 versus Transformers' 3.359814. Of 573,440 adapter-product elements,
164 exceed `rtol=1e-4, atol=1e-6`; the largest absolute difference is 7.94e-6.
The real-model strict comparison remains open. The RMSNorm sqrt/div control
produces a similar difference.
