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
```

Both examples print before/after ONNX graphs, save `.onnx` and readable `.txt`
files, reload the saved models, and check their numerical meaning.
The batching example includes a convolutional classifier and embedding-based
recommendation with dynamic batches of 1, 3, and 7. The autodiff example computes
linear-regression gradients, checks a loss-reducing optimization step, and
compares directional sensitivities and reverse attribution with NumPy.

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
shape metadata through ONNX IR's shape inference pass. No separate shape engine
is introduced. Unsupported active derivative paths, random operators, control
flow, and local functions that have not been inlined fail explicitly.

`_batching`, `_autodiff`, and `_rule_registry` are private implementation homes.
Batching, JVP, and VJP have separate typed rule contracts and instance-local
registries, not a shared untyped plugin API. There is no `higher_order` umbrella.

Batching a first-order derivative is covered for a smooth unary subset after
running ONNX IR's `ShapeInferencePass` to annotate the generated graph.
Arbitrary derivative graphs are not guaranteed to batch: for example,
runtime-computed reduction axes used by unbroadcasting remain unsupported
by the batching pass.
**Higher-order autodiff remains explicitly rejected**, including after model
serialization or batching. Forward-over-reverse Hessian-vector products belong
in a follow-up: derivative-rule closure, selectable generated interfaces, and
fixed seeds need deliberate support before removing the first-order guard.
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
and gradients with analytical references. Serialized graphs are checked and
executed with ONNX's reference evaluator and ONNX Runtime. Some reference
evaluator kernels have limitations (for example negative Pad and LRN); those
cases are executed with ONNX Runtime.

## License and provenance

MIT; see [LICENSE](LICENSE). The author authorized MIT licensing of their
newly authored upstream transformation work. Exact source revisions and
migration changes are recorded in [NOTICE](NOTICE). Dependency licenses remain
their own; no pre-existing ONNX IR implementation is copied into this package.
