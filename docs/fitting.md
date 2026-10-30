# Matrix-free nonlinear parameter fitting

Fit a decay curve from 128 noisy observations using ordinary ONNX derivative
graphs and an external damped Gauss–Newton solver. This is a small, offline
scientific-computing acceptance problem: recover an amplitude, decay rate and
background offset, rather than train a neural network.

```bash
uv sync --group dev --group training
uv run python -m examples.fitting --output-dir fitting_models
```

## Model and products

For observations $(t_i,y_i)$, parameters $p=(a,b,c)$ predict
$f_i(p)=a\exp(-bt_i)+c$. Define residuals $r_i=f_i-y_i$ and
$L(p)=\tfrac12\sum_i r_i^2$. The Jacobian $J$ differentiates the residual vector
with respect to the three parameters. Then

$$
g=J^\mathsf{T}r,\qquad
(J^\mathsf{T}J+\lambda I)\Delta=-g.
$$

The example exports `onnax.jvp(model, inputs=["parameters"], outputs=["residual"])`
and the corresponding `onnax.vjp`. A CG iteration passes its direction $v$ to the
JVP graph, then passes the resulting $Jv$ to the VJP graph. Adding $\lambda v$
in external code supplies the damped operator. The solver never builds or stores
$J$ or $J^\mathsf{T}J$. Each VJP reuses the current parameter point; its cotangent
seed is an independent runtime input. CG solves to an absolute residual tolerance
of $10^{-11}$, with bounded work and explicit failures for invalid curvature or
nonconvergence. Armijo backtracking bounds the nonlinear update; damping is
$\lambda=10^{-3}$ in this example.

This is a Gauss–Newton approximation. The exact Hessian also contains residual
curvature:

$$
Hv=J^\mathsf{T}(Jv)+\sum_i r_i\nabla^2r_i\,v.
$$

The exported `hvp` graph uses `onnax.jvp(onnax.grad(...))`. Its values are checked
against this full expression at a point where residuals are nonzero; they differ
from the Gauss–Newton product. CG's positive-curvature assumption is appropriate
for the damped Gauss–Newton operator here, and is not a general guarantee for an
exact nonlinear Hessian. The [Jelly playground](https://www.justinchuby.com/onnax/jelly/)
illustrates safeguarded Newton-CG with exact HVPs in a different application.

`onnax.vmap(jvp_graph, in_axes=(None, None, None, 0))` additionally exports a
batch of independent directions at one shared parameter point. The example
checks five directions against the independent reference. The scalar CG solve
uses single directions; batching is not claimed to accelerate it.

## Verification and artifacts

Fixed seed 42 generates data near $(a,b,c)=(2,1.3,0.4)$ with Gaussian noise
of standard deviation 0.005. Starting at $(1.4,0.8,0.1)$, the reference development
run reduces $L$ from 6.787 to about 0.000997 and recovers approximately
$(2.00036,1.29710,0.39869)$. Noise means the least-squares fit need not equal the
data-generating parameters exactly.

The CLI verifies:

- Raw and ONNXScript-optimized JVP, VJP and exact HVP results against an independent
  analytic NumPy reference, for observation counts 1, 7 and 128.
- The adjoint identity $\langle Jv,w\rangle=\langle v,J^\mathsf{T}w\rangle$,
  the Gauss–Newton product, batched directions and the full solver trajectory.
- Final stationarity, a loss decrease of at least 99%, parameter recovery and
  execution after saving/loading both graph and fitted parameters.

Tests independently check the analytic Jacobian and Hessian with central finite
differences. A dense Jacobian and Hessian exist only in this small validation
oracle; production solver callbacks use ONNX products. Solver unit tests compare
CG with a dense linear solve and exercise failure paths.

The directory contains raw and optimized `residual`, `jvp`, `vjp`, `hvp` and
`batched_jvp` graphs, checked with the full ONNX checker. `interface.json` records
versioned graph input/output names and parameter order; read the actual appended
seed/output names rather than guessing them. `fit.npz` stores fitted parameters
and observations, and `report.json` contains the loss trajectory, graph sizes,
node counts and measured runtime information.

## Runtime boundary and limits

All arrays and graphs use FP64 and CPU ONNX Runtime, with graph optimization
disabled and one intra/inter-op thread. ONNXScript optimizes independent graph
copies; the source graph is verified unchanged. The external solver owns damping,
CG, line search and saving parameters. No solver or training API is added to
onnax's core.

A separate process measures session creation, a complete two-session
Gauss–Newton product and a complete solve, with 10 warmups and 100 measured
repetitions (median/P95). Product time includes CPU feeds, returned arrays and
the intermediate host exchange; solve time also includes Python CG and line
search. Export and the dense reference are outside measurement. Peak RSS includes
Python imports and runtime overhead, not just tensor storage. Hardware, package
versions, thread settings, model bytes and node counts accompany the results.

This three-parameter problem establishes numerical behavior and a measurement
baseline. It does not establish acceleration, large-problem memory scaling,
GPU support, device-resident execution or mixed-precision accuracy. The graphs
retain their primal outputs, so mapped models also return repeated primal values.
Large inverse problems may need structured residuals, cached linearizations,
preconditioners or device bindings; those require separate workload evidence.

A [recorded development-machine report](https://github.com/justinchuby/onnax/blob/main/examples/measurements/decay-fit-apple-arm64.json)
is included for inspection. Rerun the CLI for measurements on your own machine.
