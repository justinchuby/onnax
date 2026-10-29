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
from onnax import grad, jvp, vjp, vmap

model = ir.load("single_example.onnx")
batched = vmap(model, in_axes=(0, None))
ir.save(batched, "batched.onnx")
```

The functions transform **IR models, not Python callables or arrays**. They
return a new `onnx_ir.Model`; they do not trace, evaluate, create a runtime, or
return JAX-style value/pullback tuples. The source is unchanged, and all primal
outputs remain in their original order.

| Function | Graph interface |
| --- | --- |
| `vmap(model, in_axes, *, axis_size=None, out_axes=0)` | Maps independent examples; all outputs acquire an example axis |
| `jvp(model, *, inputs=None, outputs=None)` | Appends tangent inputs in selected-input order and directional derivatives in selected-output order |
| `vjp(model, *, inputs=None, outputs=None)` | Appends cotangent inputs in selected-output order and input cotangents in selected-input order |
| `grad(model, *, inputs=None, output=None)` | Appends selected-input gradients of one scalar loss; implicit unit seed, no extra inputs |

`inputs` and `outputs` are ordered, unique **graph names**, not Python `argnums`
or PyTrees. Defaults select non-initializer FLOAT/DOUBLE graph inputs and
FLOAT/DOUBLE graph outputs. `grad` needs one rank-zero floating output; choose
`output="loss"` when the model also returns predictions. Initializers are held
fixed unless named explicitly. Selecting a stored parameter adds its derivative
interface without changing its primal value or promoting it to a runtime input.
A separate `value_and_grad` is not provided: `grad` already retains all primals.

`in_axes` has one entry per graph input: an integer inserts an example axis;
`None` keeps an input shared. Initializers are shared, and mapped inputs cannot
have initializer defaults. Negative axes index the resulting tensor. `out_axes`
is an integer for all outputs or a sequence in graph-output order, including
appended derivatives. Omitted `axis_size` supports different positive batch
lengths without retransformation; mapped lengths must agree at runtime.

Tangent and cotangent seeds are **explicit runtime graph inputs**, not values
passed to the transform. They must match the corresponding primal's runtime
shape and dtype exactly, without broadcasting. Names use `_tangent` and
`_cotangent` suffixes with numeric suffixes on collisions. Inspect the returned
`model.graph.inputs`/`outputs` or serialized interface metadata for actual names.

```python
gradient = grad(model, inputs=["weight", "bias"], output="loss")
pushforward = jvp(model, inputs=["features"], outputs=["prediction"])
pullback = vjp(model, inputs=["features"], outputs=["prediction"])
# Feed the appended features_tangent or prediction_cotangent input at execution.
# Every result retains prediction/loss before its appended derivatives.
```

The existing `VmapPass`, `JvpPass`, `VjpPass`, and `GradPass` entry points remain
compatible for ONNX IR pass pipelines, with their original constructors and
`PassResult.model` return contract. For example,
`GradPass(inputs=["weight"], output="loss")(model).model` is equivalent to
`grad(model, inputs=["weight"], output="loss")`. See the **public function and
class docstrings** for complete semantics, examples and support restrictions;
they are the generated API documentation source.

## Support and composition

Standard-domain opsets **18 through 28** are supported, subject to each
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

`_batching` and `_autodiff` are private packages. Each separates transformation
orchestration (`_transform`), graph construction/state (`_context`), typed
registry contracts (`_registry`), and actual operator-family rules. Autodiff's
`_interface` owns selection and serialized composition metadata, with small
value/support accessors in `_common`. Families cover elementwise, linear algebra,
layout/indexing and reductions; batching additionally separates native-N kernels,
normalization/quantization and indexing rules. Registries dispatch directly to
family implementations, rather than a monolithic operator switch. Registration
is deterministic and instance-local, with bounded opset intervals.
`_rule_registry` shares only dispatch infrastructure; `_inference` is the single
isolated `onnx-shape-inference` adapter. These modules are not a public plugin API,
and there is no `higher_order` umbrella.

`vmap` can batch generated reverse-mode gradients, including shared runtime
unbroadcast axes. Compose `grad` on a **single-example scalar loss** with
`vmap`: map samples and targets, keep parameter inputs shared (`None`) or
stored as initializers, and retain an example axis in each parameter gradient.
The same graph can accept batches of 1, 3, and 7 without retransformation.
This is not the gradient of an already-reduced batch loss.
Runtime reduction axes must be shared; variable-length axes need `keepdims=1`.
Mapped controls, ragged outputs, and arbitrary derivative graphs outside the
documented operator/control contracts remain unsupported.
**One forward-over-reverse composition is supported**, including after
serialization: `jvp` over a `grad` or `vjp` graph differentiates the
ordinary emitted ONNX operations. For a scalar loss this computes Hessian-vector
products without constructing a Hessian:

```python
gradient = grad(model, inputs=["weight"], output="loss")
per_example = vmap(gradient, in_axes=(0, 0))
# Here features and target are runtime inputs; weight is a stored initializer.
# With runtime weight, use in_axes=(0, 0, None) instead.
hvp = jvp(gradient, inputs=["weight"])
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

CI runs the same checks on Python 3.11 and 3.12 with current dependencies.
ONNX 1.23.2 or newer is required for
opset-28 schemas and reference-evaluator semantics. The modern IR baseline is
`onnx-ir>=1.1.0`, and shape inference remains `onnx-shape-inference>=0.4.0`.
CI resolves current dependencies on both Python versions rather than maintaining
a historical lowest-direct dependency matrix. The development suite requires
ONNX Runtime 1.31.1 or newer.
Tests compare batching with
independent examples, JVP/VJP with finite differences and their adjoint identity,
and gradients with analytical references.
HVPs are checked against central differences of gradients, differences of
directional derivatives, analytic curvature, and Hessian symmetry.
Serialized graphs are checked and
executed with ONNX's reference evaluator and ONNX Runtime. Some reference
evaluator kernels have limitations (for example negative Pad and LRN); those
cases are executed with ONNX Runtime.

IR-only validation uses `onnx_ir.passes.common.CheckerPass(full_check=True)`,
also verified in ONNX IR 0.2.1 before adopting the modern baseline; this retains ONNX schema
and full shape checks, not just graph consistency checks. The pass owns the
necessary checker serialization boundary. Reference-evaluator and runtime
boundaries still serialize once, while cloning/inference no longer uses
`from_proto(to_proto(model))`. Dedicated file, readable-text and metadata
reload tests retain genuine persistence coverage, and byte-exact source
preservation checks remain to catch tensor-metadata mutation. Example artifact
preparation reuses the inference adapter's isolation, cache refresh and
overridable-default handling rather than using serialization as a clone.

### Opsets 25–28 and execution capabilities

The schema audit covers every changed registered operator and the emitted
`Range` control nodes, not just model opset numbers. Input/output structure and
existing attributes are unchanged for those operators. Most revisions add low-
precision tensor types; these do **not** broaden FLOAT/DOUBLE autodiff support.
Additional operators such as BitCast, CumProd, Attention, and SwiGLU do not gain active
transform rules simply because their opset is accepted.

| Revised schema | Transformation contract and regression coverage |
| --- | --- |
| SpaceToDepth-28 | Preserves `mode`, including CRD and default DCR, while merging/restoring native N independently from the example axis |
| BitShift-28 | Preserves signed arithmetic right shift, left-shift wraparound, and out-of-range shift counts |
| Mod-28 | Retains floor versus truncating quotient behavior for floating inputs, including signed-zero conventions |
| Range-27 | New `stash_type` affects low-precision accumulation; generated shape/index controls remain INT64, so are unaffected |
| ReduceLogSum/ReduceLogSumExp-28 | Integer types were removed from the schemas; float batching remains covered |
| Celu/Einsum/OneHot-28, layout/quantization-25/28 | Added dtype constraints do not change axis/control semantics; existing parameters and attributes are forwarded unchanged |

Tests execute batching against independent examples, JVP/VJP against finite
differences, HVPs against gradient differences and symmetry, and serialized
per-example parameter gradients **at each actual opset 25, 26, 27 and 28**.
The installed ONNX Runtime 1.31.1 officially accepts opsets only through **26**.
Optimized CPU runtime tests cover 25–26; full ONNX checking and ONNX reference
execution cover 27–28 without downgrading their imports. Runtime capability
tests explicitly report an unsupported-opset skip for runtimes with that ceiling,
and execute the graph when the installed runtime supports it.

## License and provenance

MIT; see [LICENSE](LICENSE). The author authorized MIT licensing of their
newly authored upstream transformation work. Exact source revisions and
migration changes are recorded in [NOTICE](NOTICE). Dependency licenses remain
their own; no pre-existing ONNX IR implementation is copied into this package.
