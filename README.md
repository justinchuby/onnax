# onnax

JAX-inspired composable transformations for ONNX graphs, built on ONNX IR.

**Private, experimental, and not published to PyPI.** ONNX IR is an upstream
dependency, not a vendored fork. Generated graphs contain ordinary ONNX
operators and run without an autodiff runtime or Python rule callbacks.

## Getting started

Requires Python 3.11 or later:

```shell
uv sync
uv run python examples/vmap.py --output-dir vmap_models
uv run python examples/autodiff.py --output-dir autodiff_models
uv run python examples/per_example_gradients.py --output-dir per_example_models
```

The examples print before/after ONNX graphs, save `.onnx` and readable `.txt`
files, reload the saved models, and check their numerical meaning.
The batching example includes a convolutional classifier and embedding-based
recommendation with dynamic batches of 1, 3, and 7. The autodiff example computes
linear-regression gradients, checks a loss-reducing optimization step, and
compares directional sensitivities and reverse attribution with NumPy.
It also computes a forward-over-reverse parameter Hessian-vector product and
directional curvature, saving the original, gradient, and HVP graphs.

The per-example example differentiates a single-example two-response regressor
and maps the generated gradients with one shared weight/bias set. It reports
per-example parameter-gradient norms for dynamic batches of 1, 3, and 7, and
contrasts them with the norm of the mean-batch-loss gradient.

## Public transforms

```python
import onnx_ir as ir
from onnax import GradPass, JvpPass, VjpPass, VmapPass

model = ir.load("single_example.onnx")
batched = VmapPass(in_axes=(0, None))(model).model
ir.save(batched, "batched.onnx")
```

| Pass | Result appended to original outputs |
| --- | --- |
| `VmapPass(in_axes, *, axis_size=None, out_axes=0)` | Maps independent examples; all outputs acquire a batch axis |
| `JvpPass(inputs=None, outputs=None)` | Directional output derivatives; adds tangent seed inputs |
| `VjpPass(inputs=None, outputs=None)` | Input cotangents; adds selected output cotangent seeds |
| `GradPass(inputs=None, output=None)` | Selected input gradients of one scalar loss; no seed inputs |

These are functional ONNX IR passes: the input model is preserved, and the
returned `PassResult.model` is a clone. Ordered input/output selections determine
seed and derivative interface order. Initializers stay fixed unless explicitly
selected for differentiation. See the **Python class docstrings** for the full
operator, shape, dtype, naming, and seed contracts; these are the API
documentation source.

## Support and composition

Standard-domain opsets **18 through 24** are supported, subject to each
operator's schema and rule restrictions. Newer opsets fail explicitly rather
than inheriting unverified rules. Models use the canonical standard domain `""`.
The shared private registry also recognizes `ai.onnx` as an alias, dispatches
by explicit node version or enclosing graph imports, rejects overlapping
registrations, and never falls back across unsupported domains or version gaps.
Explicit node versions selecting a different schema than the graph imports
are rejected because ONNX serialization cannot preserve that distinction.

Batching preserves example-level reductions, dynamic shapes, and shared
parameters. Autodiff supports FLOAT/DOUBLE first derivatives, gradient
accumulation, runtime unbroadcasting (including zero extents), and existing
shape metadata through `onnx-shape-inference>=0.4.0`. Both transforms infer on
isolated clones with the engine's `refine` policy; there is no fallback engine
or separate shape implementation. Unsupported active derivative paths, random
operators, control flow, and local functions that have not been inlined fail
explicitly.

`_batching`, `_autodiff`, and `_rule_registry` are private implementation homes.
Batching, JVP, and VJP have separate typed rule contracts and instance-local
registries, not a shared untyped plugin API. There is no `higher_order` umbrella.

`VmapPass` can batch generated reverse-mode gradients, including shared runtime
unbroadcast axes. Compose `GradPass` on a **single-example scalar loss** with
`VmapPass`: map samples and targets, keep parameter inputs shared (`None`) or
stored as initializers, and retain an example axis in each parameter gradient.
The same graph can accept batches of 1, 3, and 7 without retransformation.
This is not the gradient of an already-reduced batch loss.
Runtime reduction axes must be shared; variable-length axes need `keepdims=1`.
Mapped controls, ragged outputs, and arbitrary derivative graphs outside the
documented operator/control contracts remain unsupported.
**One forward-over-reverse composition is supported**, including after
serialization: `JvpPass` over a `GradPass` or `VjpPass` graph differentiates the
ordinary emitted ONNX operations. For a scalar loss this computes Hessian-vector
products without constructing a Hessian:

```python
gradient = GradPass(inputs=["weight"], output="loss")(model).model
hvp = JvpPass(inputs=["weight"])(gradient).model
# Original inputs/outputs and gradients remain, then a weight tangent input
# and its Hessian-vector product output are appended. Feed the direction seed
# with exactly the weight's shape/dtype; stored initializers are unchanged.
```

On a generated reverse graph, omitted `inputs` selects original non-initializer
FLOAT/DOUBLE inputs, not cotangent seeds; omitted `outputs` selects generated
gradients/cotangents, not original outputs. Explicit selections use exact graph
names and may include original outputs or cotangent seeds. Parameters and seeds
are fixed unless selected. Versioned JSON in graph metadata
`onnax.autodiff.interface` records the ordered source input/output names, seed
names, derivative output names, latest selections, and mode history, including
collision-resolved names. See `JvpPass`'s docstring for the schema.

Covered smooth paths include arithmetic (including active Pow exponents),
Exp/Log/Sqrt/Reciprocal, trigonometric/hyperbolic functions, Sigmoid/Softplus/Erf,
Softmax/LogSoftmax, MatMul/Gemm, runtime broadcasting and reductions, and
layout/indexing paths including Gather's additive ScatterND pullback.
Dimensions and reduction axes may be dynamic; shape/index computations remain
structural, while floating-point dependencies are differentiated. Unsupported
active generated paths fail precisely; for example Abs's Sign pullback is not
supported. Piecewise rules retain their first-order fixed-predicate conventions
away from boundaries; no higher derivative at a nonsmooth point is claimed.
The usual real-domain restrictions still apply. Reverse-over-forward,
reverse-over-reverse, forward-over-forward, third derivatives, and legacy
marker-only derivative models remain explicitly rejected.
General nested batching or arbitrary transform commutation is not promised.

## Development

```shell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv build
```

CI runs the same checks on Python 3.11 with minimum direct dependencies and
Python 3.12 with current dependencies. ONNX 1.23.2 or newer is required for
reference-evaluator fixes used by the runnable examples. The development suite
requires ONNX Runtime 1.29 or newer for opset 24 and CPU kernel coverage.
Tests compare batching with
independent examples, JVP/VJP with finite differences and their adjoint identity,
and gradients with analytical references.
HVPs are checked against central differences of gradients, differences of
directional derivatives, analytic curvature, and Hessian symmetry.
Serialized graphs are checked and
executed with ONNX's reference evaluator and ONNX Runtime. Some reference
evaluator kernels have limitations (for example negative Pad and LRN); those
cases are executed with ONNX Runtime.

## License and provenance

MIT; see [LICENSE](LICENSE). The author authorized MIT licensing of their
newly authored upstream transformation work. Exact source revisions and
migration changes are recorded in [NOTICE](NOTICE). Dependency licenses remain
their own; no pre-existing ONNX IR implementation is copied into this package.
