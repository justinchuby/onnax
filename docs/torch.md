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
