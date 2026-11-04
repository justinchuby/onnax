# Train an ONNX module with PyTorch

`onnax.torch.OnnxModule` connects an ONNX computation to PyTorch autograd.
PyTorch owns the trainable tensors; reusable ONNX sessions compute the forward
outputs and their vector–Jacobian products. Use an ordinary optimizer or wrap the
module in a task model for Hugging Face Trainer.

```sh
pip install 'onnax[torch]'
```

## Select parameters and train

Supply an ONNX model and a dictionary of initial trainable parameters. Names are
ONNX input or initializer names. Selected initializers become explicit parameter
inputs; the other initializers stay frozen.

```python
import torch
from onnax.torch import OnnxModule

with OnnxModule(
    "model.onnx",
    {"lora.A": initial_a, "lora.B": initial_b},
    outputs=["prediction"],
    input_gradients=(),
) as model:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    for features, targets in batches:
        optimizer.zero_grad()
        prediction = model(features=features)
        loss = (prediction - targets).square().mean()
        loss.backward()
        optimizer.step()
    model.export("trained-module")
```

The runnable [Trainer example](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/trainer.py)
trains the LoRA regression model for 200 steps, restores a checkpoint in a fresh
process, and checks the exported ONNX predictions.

Run the [MLP example](https://github.com/justinchuby/onnax/blob/main/examples/torch_mlp.py)
with `python -m examples.torch_mlp --output-dir torch_mlp_models`. It trains
four weight/bias tensors for 50 steps, checks logits, loss, gradients and every
update against PyTorch, and reloads the exported bundle. Tiny Qwen3 also has an
independent three-update regression through this public interface.

For outputs \(y=f(x,\theta)\), PyTorch supplies \(\bar y=\partial L/\partial y\).
The ONNX backward computes \(\bar\theta=J_\theta^T\bar y\) and, when requested,
\(\bar x=J_x^T\bar y\). PyTorch then accumulates these gradients and updates
\(\theta\). This supports losses computed outside ONNX and multiple ONNX outputs.

## Reuse forward activations in backward

Use `save_activations=True` when the loss is computed in PyTorch or a model has
multiple differentiable outputs:

```python
model = OnnxModule(
    "model.onnx",
    {"lora.A": initial_a, "lora.B": initial_b},
    save_activations=True,
)
prediction = model(features=features)
loss = torch.nn.functional.mse_loss(prediction, targets)
loss.backward()
```

The forward session returns the prediction and the intermediate values needed
by the ONNX VJP. Backward consumes those saved values with PyTorch's upstream
gradients. Shape-only intermediates are stored as INT64 shape vectors. Fused
attention pullbacks recompute attention statistics during backward.

PyTorch owns the saved tensors and releases them after backward;
`retain_graph=True` keeps them for another call. Each outstanding forward has its
own activations. In-place changes to inputs requiring gradients, parameters, or
returned outputs are checked before backward. Export/load preserves this mode,
and parameter checkpoints can be exchanged with the default execution mode.
Choose either `save_activations` or `eager_gradients` for a module.

Wide saved boundaries use byte intervals to narrow exact output-overlap checks.
The [real UNet comparison](https://github.com/justinchuby/onnax/tree/main/examples/diffusion/reports/output-ownership-m1max)
records an 8.1% lower complete-step median with identical graph and gradient
hashes; both measured P95 values are about 575 ms.

## Compute a scalar objective and its gradients together

Put the loss in ONNX and select its scalar output with `eager_gradients=True`:

```python
with OnnxModule(
    "loss.onnx",
    {"lora.A": initial_a, "lora.B": initial_b},
    outputs=["loss"],
    input_gradients=(),
    eager_gradients=True,
) as model:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    for features, targets in batches:
        optimizer.zero_grad()
        loss = model(features=features, target=targets)
        loss.backward()
        optimizer.step()
```

One ONNX call returns the loss and its unit-cotangent derivatives. For an outer
scalar objective $F(L)$, PyTorch backward returns
$\frac{dF}{dL}\nabla_\theta L$ from the cached derivatives. Weighted losses and
gradient accumulation follow the ordinary autograd chain rule. The selected
output must be a scalar FLOAT or DOUBLE tensor.

PyTorch saved tensors own the cached gradients: ordinary backward releases them,
while `retain_graph=True` keeps them available for another backward. Each
outstanding forward stores the selected parameter and data gradients. Parameter
and differentiable-input version checks remain active. `torch.no_grad()` uses the
forward session. The execution option is preserved by export/load; the same
scalar source keeps its parameter-checkpoint identity across both schedules.

The [Trainer example](huggingface.md) validates 200 updates and fresh-process
recovery. The [image tutorial](diffusion-images.md) pairs this option with compiled
MLX execution. Loss scaling inside ONNX uses the ordinary VJP schedule, which
supplies the upstream cotangent before derivative execution.

## Compose with other PyTorch layers

Floating data inputs propagate gradients by default. A frozen ONNX block can sit
between trainable PyTorch layers: freezing its weights still allows gradients to
flow through it. `input_gradients=["features"]` selects specific data inputs;
`input_gradients=()` saves the work when only ONNX parameters need gradients.
Integer inputs, such as token IDs, remain data inputs.

One selected output returns a tensor. Multiple outputs return a tuple in the
selected order. Call the module with keyword arguments matching its data input
names. Parameter names and gradient output names are available through
`parameter_names`, `parameters_by_name()`, and `gradient_names`.

## Checkpoint and export

Save `model.state_dict()` together with the external optimizer and scheduler
state. The state includes a source-graph identity, so loading a checkpoint into a
different computation raises an error before replacing parameters.

`model.export(directory)` writes to a new or empty directory:

- `forward.onnx` binds the current parameters for inference.
- `model.onnx`, `execution.onnx`, and `derivative.onnx` preserve the reusable
  training computation and its external tensor files.
- `parameters.npz` stores current parameters.
- `interface.json` records names, shapes, dtypes, gradient mappings, and hashes.

`OnnxModule.load(directory)` restores the bundle and reuses its saved derivative.
The module owns a snapshot of its models and releases sessions and temporary
files on `close()` or context exit. Keep it open through the last backward call.

## Choose an execution backend

The default backend is CPU ONNX Runtime with one thread and graph optimization
disabled. A `session_factory(path)` can provide another backend. It returns an
object with `run(output_names, feeds)` accepting and returning NumPy arrays; an
optional `close()` releases backend resources. Returned buffers remain valid
while referenced. Sessions are created once and reused.

This boundary uses CPU FP32/FP64 PyTorch tensors. Runtime transfers and device
execution belong in the supplied backend. FP32 is the training default; BF16
parameter boundaries and higher-order PyTorch autograd are future work. For
model-specific BF16, NF4, checkpointed decoder execution, and MLX measurements,
see the [Qwen training tutorial](qwen-sft.md).

An `optimize(model)` callback can apply ONNXScript optimization to execution
copies. The source graph and stable parameter interface are preserved.

## Checkpoint large graphs

Use [`CheckpointedModule`](checkpointing.md) with named intermediate boundaries
to retain boundary tensors and recompute block-local activations in backward.
It shares the parameter, session-factory and external-optimizer conventions above.
