# Differentiate an ONNXScript model

Use ONNXScript to write a typed ONNX function, then generate an executable
gradient graph with onnax. This example fits the same linear regressor as the
[ndonnx tutorial](ndonnx.md), using ONNX operators and a reusable loss function.

```bash
uv sync --group dev --group authoring
uv run --no-sync python -m examples.onnxscript_autodiff --output-dir onnxscript_models
```

## Define the prediction and loss

The model computes $p=Xw+b$ and minimizes the mean squared prediction error,
$L=\operatorname{mean}((p-y)^2)$. `FLOAT["batch", 2]` describes a symbolic sample
count and two features; `FLOAT[()]` describes a scalar.

```python
from onnxscript import FLOAT, script
from onnxscript.onnx_opset import opset18 as op
```

```{literalinclude} ../examples/onnxscript_autodiff.py
:language: python
:pyobject: mean_squared_error
```

```{literalinclude} ../examples/onnxscript_autodiff.py
:language: python
:pyobject: linear_regression
```

`MatMul` applies the weight vector to each row. The separate scripted loss
function becomes a local ONNX function when the model is exported.

## Export, inline and differentiate

`to_model_proto()` produces an ONNX protobuf model. Convert it to ONNX IR and
inline the local loss function, expanding its body into standard operators.
Then select `weight`, `bias` and the scalar `loss`:

```python
import onnx_ir as ir
from onnx_ir.passes.common import InlinePass
import onnax
```

```{literalinclude} ../examples/onnxscript_autodiff.py
:language: python
:pyobject: differentiate
```

The result retains prediction and loss and appends `weight` and `bias` gradients.
The parameter inputs keep their original shapes and FP32 dtype. Use the returned
model's output values or the CLI's `interface.json` for gradient output names.

## Run a parameter update

Execute the generated graph with ONNX ReferenceEvaluator, then update the
parameter arrays in external Python code:

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

The CLI reuses one evaluator for 20 updates. Loss falls from about 2.825 to
0.002025. Analytic NumPy gradients and a separate update trajectory check the
results. The two frontend examples also agree directly on predictions, loss
and parameter gradients.

Exported artifacts include `source.onnx` with the local function,
`prepared.onnx` after inlining, raw and ONNXScript-optimized gradient graphs,
parameter storage and JSON interface/results. Each graph passes full ONNX
checker, and reload checks cover batch sizes 1, 3 and 7.

See the [full example](https://github.com/justinchuby/onnax/blob/main/examples/onnxscript_autodiff.py)
and [ONNXScript's local-function guide](https://microsoft.github.io/onnxscript/auto_examples/06_plot_model_local_funs.html).
