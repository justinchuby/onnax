# onnax

**JAX-inspired transformations for ONNX graphs.** Batch independent examples
with `vmap`, differentiate with `grad`, `jvp`, and `vjp`, and save the result as
ordinary ONNX. Built on [ONNX IR](https://github.com/onnx/ir-py).

Experimental alpha software.

## Install

Requires Python 3.11+:

```shell
python -m pip install onnax==0.1.0
```

This README describes `main`. Recent additions, including
global rule decorators, `HoistLinearCoefficientsPass`, and GELU/PReLU autodiff, are not included in the
`v0.1.0` release tag. Install from source below to use these development features.

To develop from source:

```shell
git clone https://github.com/justinchuby/onnax.git
cd onnax
python -m pip install .
```

## Transform a model

This complete example builds a scalar loss, then creates new gradient, batched,
and Hessian-vector-product graphs. It does not execute them.

```python
import onnx_ir as ir

from onnax import grad, jvp, vjp, vmap

x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[2])
weight = ir.val("weight", dtype=ir.DataType.FLOAT, shape=[2])
tape = ir.tape.Tape()
prediction = tape.op("Mul", [x, weight])
prediction.name = "prediction"
squared = tape.op("Mul", [prediction, prediction])
squared.name = "squared"
loss = tape.op("ReduceSum", [squared], {"keepdims": 0})
loss.name = "loss"
model = ir.Model(
    ir.Graph(
        [x, weight],
        [loss],
        nodes=tape.nodes,
        opset_imports={"": 24},
        name="linear_loss",
    ),
    ir_version=10,
)
gradient = grad(model, inputs=["weight"], output="loss")
per_example = vmap(gradient, in_axes=(0, None))
pushforward = jvp(model, inputs=["x"], outputs=["loss"])
pullback = vjp(model, inputs=["weight"], outputs=["loss"])
hvp = jvp(gradient, inputs=["weight"])
ir.save(per_example, "per_example.onnx")
ir.save(hvp, "hvp.onnx")
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

Autodiff includes standard GELU and its `approximate="tanh"` variant for ONNX
opsets 20 through 28, including Hessian-vector products and batched gradients.
The derivative uses the same approximation as the source operator.
PReLU supports gradients for both data and its learnable broadcast slope tensor,
including stored-parameter selections, HVPs, and per-example parameter gradients.

Try the [hosted WebGPU jelly playground](https://www.justinchuby.com/onnax/jelly/): drag a spring sheet
with matrix-free Newton-CG, powered by exported
`grad` and `jvp(grad(...))` ONNX graphs. See the [documentation site](https://www.justinchuby.com/onnax/)
and [playground source and run instructions](examples/jelly).

## Optimize generated models

For deployment, we recommend applying the **ONNXScript optimizer** after composing
the desired transforms. Install `onnxscript` separately; it is an optional export
dependency, not required by onnax:

```python
import onnx_ir as ir
from onnxscript import optimizer

per_example = ir.load("per_example.onnx")
optimized = optimizer.optimize(per_example.clone(), onnx_shape_inference=False)
ir.save(optimized, "per_example.optimized.onnx")
```

The example loads the model saved above. `optimizer.optimize` mutates IR models,
so clone first when retaining the original. This disables the optimizer's ONNX
shape-inference option; onnax has already inferred the transformed model's shapes.
Check and numerically validate the optimized graph with representative inputs,
especially before applying another differentiation transform. Keep the original
transform graph for further composition and optimize a deployment copy.

Autodiff already emits fixed-axis broadcast reductions when both operand and
broadcast-result shapes are concrete, and skips reductions when shapes match.
Unknown or symbolic sizes retain runtime shape handling. Deployment optimization
can additionally fold constants and share computation across the composed graph;
it is separate from differentiation. The jelly exporter uses ONNXScript for this
step and disables ONNX Runtime graph optimization during browser execution.

For repeated directional products, use the optional
[`HoistLinearCoefficientsPass`](src/onnax/_linear.py) before ONNXScript optimization:

```python
import onnx_ir as ir
from onnxscript import optimizer

from onnax import HoistLinearCoefficientsPass

hvp = ir.load("hvp.onnx")
# The model above appends one direction input; inspect names for other models.
direction = hvp.graph.inputs[-1].name
factored = HoistLinearCoefficientsPass(inputs=[direction])(hvp).model
optimized = optimizer.optimize(factored, onnx_shape_inference=False)
ir.save(optimized, "hvp.optimized.onnx")
```

The pass factors elementwise linear islands such as `a * v + b * v` into
`v * (a + b)`, exposing direction-independent work for later CSE. Select one
or more runtime inputs; coefficients must be independent of every selection.
It returns a clone and needs no ONNXScript dependency itself. It preserves
nonlinear operations, matrix products, functions, and subgraphs as boundaries;
it does not prove the entire model is linear. `max_terms=64` bounds the number
of dependent bases represented per island.
Overloaded calls remain opaque. Models defining local functions that shadow
emitted standard operators are left unchanged; explicitly inline those
functions first if factoring their bodies is intended.

This is an explicit opt-in to floating-point reassociation: rounding, overflow,
underflow, signed zeros, and non-finite behavior may change, even for finite
inputs. Validate numerical tolerances for the intended workload. Coefficients
are computed from the current runtime inputs, not cached across executions.

## Development

Ruff lint and formatting are both required.
Numerical ONNX Runtime tests disable graph optimization to validate the actual
generated artifacts independently of runtime optimizer cleanup.
Use absolute imports in library modules and native ONNX IR APIs for model and
tensor construction, including README snippets and examples. Direct ONNX schema,
checker, serialization, and reference-evaluator APIs are valid interoperability
boundaries; protobuf construction helpers are reserved for interoperability tests.

### Contributing diagnostics

All user-reachable errors must explain what failed, show the relevant actual
versus expected values, and give a concrete, accurate corrective action.
For operator failures include the transform mode, domain/operator, effective
opset, and node name or output location, with only the offending input's
shape/dtype/axis/control metadata as needed. Never dump tensor contents.
Selection errors should list available graph names; version errors should state
the actual version and supported interval. Keep remedies specific.

## License

[MIT](https://github.com/justinchuby/onnax/blob/main/LICENSE).

## Extend transformation rules

Import a plugin once to register rules globally. Calls stay `onnax.jvp(model)`,
`onnax.grad(model)`, and `onnax.vmap(model, ...)`; pass objects also read the latest
registrations when invoked.

```python
from collections.abc import Mapping, Sequence

import onnx_ir as ir

import onnax


@onnax.register_jvp("example", "Square", since_version=1, until_version=1)
def square_jvp(
    ctx: onnax.AutodiffContext, node: ir.Node, tangents: Mapping[int, ir.Value]
) -> Sequence[ir.Value]:
    x = ctx.input(node)
    return [ctx.op("Mul", ctx.op("Mul", ctx.scalar(x, 2), x), tangents[0])]
```

Use `register_vjp` for reverse rules shared by `vjp` and `grad`, and
`register_batching` for `vmap`. The complete runnable
[custom rules example](examples/custom_rules.py) registers all three and validates
JVP, VJP, gradients, HVPs, and per-example gradients. Extensions use public
`AutodiffContext` and `BatchingContext` helpers; no private imports are needed.

Registrations are keyed by domain, operator name, and an explicit inclusive
opset interval. Later registrations override builtins or earlier extensions within
the new interval; earlier rules remain active outside that interval. Invalid
registrations leave existing rules unchanged.
Each transform takes a stable snapshot, including during concurrent registration.
Import plugins before invoking transforms. Registration is process-local and is
not serialized into exported models.

A JVP callback returns one tangent per output; a VJP callback returns exactly the
requested input-index contributions. Batching callbacks update the cloned node
and register each output's example rank and mapped state. `ctx.elementwise(node)`
handles operators with ordinary elementwise broadcasting; use
`prepare`, `align`, and `register` for other batching contracts. Example ranks
exclude the leading batch axis.

Rules supply transformation semantics, not runtime kernels or inference schemas.
Provide an ONNX function or runtime implementation for custom primal operators,
and annotate tensor dtypes/ranks when inference cannot determine them. Select
floating inputs explicitly when custom operands include nondifferentiable data.
Callbacks should build new values through the context and treat existing tensor
payloads as read-only. Randomness, subgraphs, and overloaded calls remain restricted.
