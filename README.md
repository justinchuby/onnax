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

## Development

Ruff lint and formatting are both required.
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
