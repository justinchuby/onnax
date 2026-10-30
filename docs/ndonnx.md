# Differentiate an ndonnx array program

Write a model with ndonnx's Array API, export it to ONNX, then use onnax to
compute parameter gradients. This example fits a linear regressor with two
features, a weight vector and a scalar bias.

```bash
uv sync --group dev --group authoring
uv run --no-sync python -m examples.ndonnx_autodiff --output-dir ndonnx_models
```

## Build the model

For sample $i$, the prediction and mean squared loss are

$$
p_i=x_i^\mathsf{T}w+b,\qquad
L=\frac1N\sum_i(p_i-y_i)^2.
$$

Gradients tell us how to change $w$ and $b$ to reduce prediction error:

$$
\nabla_w L=\frac2N X^\mathsf{T}(p-y),\qquad
\nabla_b L=\frac2N\sum_i(p_i-y_i).
$$

`ndx.argument` creates symbolic arrays. Multiplication broadcasts the two weights
across samples; summing over the feature axis produces one prediction per row.
The final mean gives the scalar loss used by `onnax.grad`.

```python
import ndonnx as ndx
```

```{literalinclude} ../examples/ndonnx_autodiff.py
:language: python
:pyobject: model
```

`ndx.build` returns an `onnx.ModelProto`. The input and output dictionaries set
the names used for differentiation and runtime feeds.

## Generate the gradients

Convert the protobuf model to ONNX IR, then select the parameters and loss:

```python
import onnx_ir as ir
import onnax
```

```{literalinclude} ../examples/ndonnx_autodiff.py
:language: python
:pyobject: differentiate
```

The gradient model keeps the original predictions and loss and appends two
outputs: a weight gradient of shape `[2]` and a scalar bias gradient. Features,
targets, weights and bias stay explicit runtime inputs. Each training step feeds
the updated parameter arrays into the same graph.

## Execute and update

From `gradient`, the model returned above:

```python
import numpy as np
from onnx.reference import ReferenceEvaluator

_, _, gradient = differentiate()
features = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
target = np.array([2.2, 0.2, 1.7], dtype=np.float32)
weight = np.array([0.2, -0.1], dtype=np.float32)
bias = np.array(0.3, dtype=np.float32)
executor = ReferenceEvaluator(ir.to_proto(gradient))
prediction, loss, dw, db = executor.run(
    None, {"features": features, "target": target, "weight": weight, "bias": bias}
)
weight = weight - 0.1 * dw
bias = bias - 0.1 * db
ir.save(gradient, "gradient.onnx")
```

The complete CLI runs 20 updates on seven samples, reducing loss from about
2.825 to 0.002025. It checks prediction, loss and both gradients against analytic
NumPy formulas and follows an independent parameter-update trajectory.

The output directory contains the original and prepared models, raw and
ONNXScript-optimized gradient graphs, saved parameters, `interface.json` with
actual parameter/gradient names, and `report.json` with the loss history.
Reload checks exercise batch sizes 1, 3 and 7.

See the [full example](https://github.com/justinchuby/onnax/blob/main/examples/ndonnx_autodiff.py),
[ndonnx export documentation](https://ndonnx.readthedocs.io/en/stable/intros/modelconversion.html),
and the [ONNXScript version of this model](onnxscript.md).
