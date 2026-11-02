# Train through named ONNX checkpoints

`onnax.torch.CheckpointedModule` trains an ONNX graph in blocks. It saves each
block's inputs during the forward pass and recomputes that block's internal
activations when its VJP runs. This bounds saved activations by the chosen
checkpoint interfaces.

Model adapters choose boundaries; the execution layer handles parameters,
shared captures, sessions and autograd. The same interface works with different
attention, normalization and feed-forward architectures.

## Choose boundaries and train

Install `onnax[torch]`. Supply the names of intermediate values already present
in your graph, in forward order:

```python
import torch
from onnax.torch import CheckpointedModule

with CheckpointedModule(
    "decoder.onnx",
    {"adapter.A": initial_a, "adapter.B": initial_b},
    boundaries={
        "first_layers": "layers.3.hidden",
        "last_layers": "layers.7.hidden",
    },
    input_gradients=(),
) as model:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    for batch in batches:
        optimizer.zero_grad()
        logits = model(**batch["inputs"])
        loss = task_loss(logits, batch["labels"])
        loss.backward()
        optimizer.step()
    model.export("trained-decoder")
```

The mapping's keys name blocks; its values identify the last checkpoint value
for each block. Each boundary must depend on the preceding block's computation.
A final `suffix` block contains the remaining output computation. The graph
must be flat, with tensor types and ranks inferable at its boundaries. Floating
checkpoint interfaces use FP32 or FP64; internal projection computation can use
the graph's explicit lower-precision policy.

Each trainable parameter becomes an explicit ONNX input. Other initializers
stay frozen. Freezing a block's weights still propagates activation gradients
through it when earlier parameters need them. Integer and Boolean inputs, such
as token IDs and masks, travel across blocks as data.

## How the backward pass composes

For blocks \(h_{i+1}=f_i(h_i,\theta_i)\), autograd supplies the upstream
\(\bar h_{i+1}\). Each block VJP computes

\[
\bar h_i=J_{h_i}f_i^T\bar h_{i+1},\qquad
\bar\theta_i=J_{\theta_i}f_i^T\bar h_{i+1}.
\]

The partitioner retains every edge crossing a boundary. This includes skip
connections, masks and values reused by several blocks. PyTorch accumulates
contributions for parameters shared across blocks. Multiple graph outputs can
feed an external loss with different upstream gradients.

`parameter_names`, `parameters_by_name()` and `arrays()` use stable ONNX names.
`gradient_names` maps each block name to its input/gradient-output mapping,
so a shared parameter can have contributions from several derivative graphs.

## Export, restore and select a backend

`export()` writes the source model, current-parameter inference model, per-block
forward/VJP modules, parameter arrays and a versioned interface manifest.
`CheckpointedModule.load()` reuses those derivatives directly. Save the module,
optimizer and scheduler state through the external training framework to resume
training. A matching flat `OnnxModule` and checkpointed module can exchange
`state_dict()` because checkpoint placement does not change parameter identity.

Pass `session_factory(path)` and `optimize(model)` as with
[`OnnxModule`](torch.md). Sessions are created once, reused for training and
released on context exit. An optimizer callback can apply ONNXScript to execution
copies. The NumPy boundary accepts CPU PyTorch tensors; the supplied session
factory chooses backend execution and transfers.

`last_residual_bytes` counts saved intermediate checkpoint inputs after a
forward pass, excluding original data and parameters. It is a logical tensor
size, separate from process and runtime memory measurements.

## Run two decoder architectures

```sh
python -m examples.torch_checkpoint --output-dir checkpoint-models
```

The offline example builds two two-layer decoders with causal attention,
residual connections, a frozen head and trainable q/v low-rank adapters:

| Decoder | Attention | Normalization | Feed-forward |
| --- | --- | --- | --- |
| `gqa` | Two query heads, one KV head | RMSNorm | Gated SiLU |
| `mha` | Two query and KV heads | LayerNorm | GELU |

Both compare predictions, losses, every adapter gradient and 12 AdamW updates
against independent PyTorch computations, then export and reload their results.
Tests also cover dynamic shapes, shared parameters, frozen-block gradient flow,
multiple outputs, gradient accumulation and fresh-process optimizer recovery.

The [Qwen workload](qwen-context-training.md) uses the same partitioner and
block autograd executor. Its adapter supplies preserved decoder boundaries and
keeps its existing task-specific export format, chunked output head and PEFT
mapping. Output-head chunking remains in that task adapter for this milestone.
