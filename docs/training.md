# Training tutorials

Choose a small end-to-end example first, then connect your own model and data.
onnax generates forward and derivative graphs; the caller owns parameter updates,
data loading, scheduling and checkpoint storage.

| Workflow | What it teaches |
| --- | --- |
| [Frozen-base LoRA](lora.md) | Select A/B gradients, propagate through frozen weights, update with Array API Adam |
| [PyTorch bridge](torch.md) | Use external losses, autograd, optimizers and reloadable ONNX bundles |
| [Hugging Face Trainer](huggingface.md) | Use datasets, gradient accumulation and training checkpoint recovery |
| [MLP classifier](mlp.md) | Train a complete small model with stable cross-entropy |
| [Implicit hyperparameter tuning](implicit-tuning.md) | Differentiate a fitted model to learn its ridge penalty |
| [Matrix-free fitting](fitting.md) | Connect JVP/VJP products to an external Gauss–Newton solver |

Continue with [language and diffusion models](models.md) for Mobius model
construction, TRL, PEFT adapters and application data. For larger graphs, use
[named checkpoints](checkpointing.md) and the [runtime guide](runtime.md).

```{toctree}
:maxdepth: 1

lora
torch
huggingface
mlp
fitting
implicit-tuning
```
