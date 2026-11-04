(user-guide)=

# Quickstart

onnax transforms an ONNX graph into another ONNX graph. Start with a prediction
and a scalar loss, select the parameters to differentiate, and save the result.

## Install

```sh
python -m pip install onnax
```

This site describes `main`. To use the current development APIs, install from
source:

```sh
python -m pip install "git+https://github.com/justinchuby/onnax.git"
```

## Generate a gradient graph

```python
import onnx_ir as ir
import onnax

x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[2])
w = ir.val("weight", dtype=ir.DataType.FLOAT, shape=[2])
tape = ir.tape.Tape()
y = tape.op("Mul", [x, w])
squared = tape.op("Mul", [y, y])
loss = tape.op("ReduceSum", [squared], {"keepdims": 0})
loss.name = "loss"
model = ir.Model(
    ir.Graph([x, w], [loss], nodes=tape.nodes, opset_imports={"": 24}, name="squared_loss"),
    ir_version=10,
)
gradient = onnax.grad(model, inputs=["weight"], output="loss")
ir.save(gradient, "gradient.onnx")
```

The exported graph returns the original loss followed by its gradient with
respect to `weight`. An ONNX runtime executes the graph; external code applies
parameter updates. The [LoRA tutorial](lora.md) runs this complete loop.

## Choose a transformation

| Transformation | Result | Typical use |
| --- | --- | --- |
| `grad(model, inputs=..., output=...)` | Scalar-loss gradients | Training |
| `jvp(model, inputs=..., outputs=...)` | Jacobian–vector product | Directional sensitivity |
| `vjp(model, inputs=..., outputs=...)` | Transposed-Jacobian–vector product | External loss gradients |
| `vmap(model, in_axes=...)` | Batched graph | Independent examples or directions |

JVP and VJP append explicit direction inputs. Use the generated input and output
names when connecting graphs. `jvp(grad(model, ...), inputs=...)` produces an
HVP; the [matrix-free fitting tutorial](fitting.md) shows how derivative products
connect to an external solver. For a focused `vmap` example,
[batch sensitivity directions](batched-sensitivity) at one shared input point.

Functions accept `ir.Model`, `ir.Graph`, `ir.GraphView` and `onnx.ModelProto`.
They return the same container type, with `GraphView` returning an independent
`Graph`. Transforms preserve the source and retain its original outputs.
See the [API reference](api.rst) for argument and return contracts.

## Optimize and extend

Apply the [ONNXScript optimizer](optimize-a-deployment-copy)
after composing transforms. Keep a source graph for further differentiation and
optimize a deployment copy.

Register custom rules once with `register_jvp`, `register_vjp` and
`register_batching`. Later registrations override earlier rules in the selected
opset interval. The [custom-rules example](https://github.com/justinchuby/onnax/blob/main/examples/custom_rules.py)
implements and validates all three transformations.

## Start from an array program

These tutorials export a program and differentiate the resulting ONNX graph:

```{toctree}
:maxdepth: 1

ndonnx
onnxscript
```
