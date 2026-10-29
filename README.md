# onnax

**JAX-inspired transformations for ONNX graphs.** Batch independent examples
with `vmap`, differentiate with `grad`, `jvp`, and `vjp`, and save the result as
ordinary ONNX. Built on [ONNX IR](https://github.com/onnx/ir-py).

Experimental, currently private, and not published to PyPI. These functions
transform `onnx_ir.Model` objects, not Python callables or arrays: there is no
Python tracing or execution runtime, and no claim of full JAX compatibility.

## Install from source

Requires Python 3.11+ and access to this private repository:

```shell
git clone https://github.com/justinchuby/onnax.git
cd onnax
python -m pip install .
```

`onnax` is the intended PyPI distribution name, but `pip install onnax` is
**not currently an installation method**. A future release requires a separate,
explicit publication decision.

## Transform a model

This complete example builds a scalar loss, then creates new gradient, batched,
and Hessian-vector-product graphs. It does not execute them.

```python
import onnx_ir as ir
from onnx import TensorProto, helper
from onnax import grad, jvp, vjp, vmap

proto = helper.make_model(
    helper.make_graph(
        [
            helper.make_node("Mul", ["x", "weight"], ["prediction"]),
            helper.make_node("Mul", ["prediction", "prediction"], ["squared"]),
            helper.make_node("ReduceSum", ["squared"], ["loss"], keepdims=0),
        ],
        "linear_loss",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("weight", TensorProto.FLOAT, [2]),
        ],
        [helper.make_tensor_value_info("loss", TensorProto.FLOAT, [])],
    ),
    opset_imports=[helper.make_opsetid("", 18)],
    ir_version=10,
)
model = ir.from_proto(proto)
gradient = grad(model, inputs=["weight"], output="loss")
per_example = vmap(gradient, in_axes=(0, None))
pushforward = jvp(model, inputs=["x"], outputs=["loss"])
pullback = vjp(model, inputs=["weight"], outputs=["loss"])
hvp = jvp(gradient, inputs=["weight"])
ir.save(per_example, "per_example.onnx")
```

Every transform returns a **new graph**, leaves its source unchanged, and
retains all primal outputs before appended derivatives. `grad` selects one
rank-zero loss and uses an implicit unit seed. `jvp` and `vjp` instead append
explicit runtime tangent/cotangent inputs; feed seeds with exactly the selected
primal's shape and dtype. Inspect the returned graph's inputs and outputs for
collision-resolved names.

Selections are ordered graph names, not `argnums`. By default, derivatives
select FLOAT/DOUBLE runtime inputs, excluding initializers. Select stored
parameters explicitly; they retain their stored primal values. In `vmap`,
`None` keeps runtime parameters shared, and initializers are always shared.
`vmap(grad(...))` produces per-example gradients, not gradients of a
batch-reduced loss. On a reverse graph, `jvp` defaults to differentiating the
generated derivatives, so `hvp` above retains the loss and gradient and appends
the HVP.

See the authoritative Python API docstrings for
[`vmap`, `grad`, `jvp`, and `vjp`](https://github.com/justinchuby/onnax/blob/main/src/onnax/_functional.py),
and their compatible
[`VmapPass`](https://github.com/justinchuby/onnax/blob/main/src/onnax/_batching/_transform.py)
and [autodiff passes](https://github.com/justinchuby/onnax/blob/main/src/onnax/_autodiff/_transform.py)
for full operator, shape, selection, and composition contracts.
Runnable examples cover [batching](https://github.com/justinchuby/onnax/blob/main/examples/vmap.py),
[autodiff and nonzero HVPs](https://github.com/justinchuby/onnax/blob/main/examples/autodiff.py),
and [per-example parameter gradients](https://github.com/justinchuby/onnax/blob/main/examples/per_example_gradients.py).

## Experimental support

- Standard ONNX opsets **18-28**, subject to each operator's rule restrictions;
  accepting an opset does not support every operator or dtype. Autodiff supports
  FLOAT/DOUBLE. Shape inference uses `onnx-shape-inference`.
- Supported compositions include batching generated reverse gradients and
  **one forward-over-reverse** `jvp(grad(...))` or `jvp(vjp(...))`, including
  after serialization. Arbitrary higher-order differentiation, nested batching,
  or transform commutation is not promised.
- Unsupported active derivative paths, random operators, control flow, and
  non-inlined local functions fail explicitly. Mapped controls and ragged
  outputs are unsupported; ordinary real-domain and nonsmooth-point limits apply.
- Generated graphs need an external ONNX executor. ONNX Runtime **1.31.1**
  accepts opsets only through **26**; opsets 27-28 are checked and executed with
  ONNX's reference evaluator without downgrading their imports.

## Development

```shell
uv sync --group dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv build
uv run twine check --strict dist/*
```

CI checks Python 3.11 and 3.12 and exercises both built distributions outside
the source tree. No publication workflow or credentials are configured.

### Contributing diagnostics

All user-reachable errors must explain what failed, show the relevant actual
versus expected values, and give a concrete, accurate corrective action.
For operator failures include the transform mode, domain/operator, effective
opset, and node name or output location, with only the offending input's
shape/dtype/axis/control metadata as needed. Never dump tensor contents.
Selection errors should list available graph names; version errors should state
the actual version and supported interval. Keep remedies specific: changing
training modes, reductions, dtypes, or opset labels can change semantics and
must not be suggested as unconditional workarounds for unsupported behavior.

Preserve exception types, chaining, source isolation, and actionable dependency
errors. The inference adapter retains `onnx-shape-inference` exceptions and adds
boundary context in Python exception notes; batching retains its
`PreconditionError` wrapper with the original cause. No fallback inference engine
or public exception hierarchy is needed. Use explicit validation, not assertions,
for user inputs. Regression tests should check diagnostic context and remedies,
not just that something raises; retain numerical and source-preservation checks.

## License

[MIT](https://github.com/justinchuby/onnax/blob/main/LICENSE).
[NOTICE](https://github.com/justinchuby/onnax/blob/main/NOTICE) preserves source
revisions and provenance for the author's upstream transformation work.
ONNX IR is an external dependency, not a vendored fork; dependency licenses
remain their own.
