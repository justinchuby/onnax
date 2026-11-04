# Precision and execution

Choose the arithmetic policy first, then the runtime boundary. FP32 master
parameters, accumulated gradients and optimizer state are the starting point;
BF16/FP16 examples make lower-precision projection operations explicit.

## Precision

- [Microbatches and dtypes](precision.md) defines weighted accumulation and the dtype policy.
- [MLX precision](mlx-precision.md) covers projection arithmetic and deployment choices.
- [BF16 LoRA](bf16.md) and [BF16 classification](mlp-bf16.md) implement mixed-precision training.
- [NF4 LoRA](nf4-lora.md) introduces quantized frozen weights before the larger Qwen workflow.

```{toctree}
:maxdepth: 1

precision
mlx-precision
bf16
mlp-bf16
nf4-lora
```

## Connect a backend

| Boundary | Tutorial |
| --- | --- |
| CPU buffers and ONNX Runtime binding | [I/O binding](iobinding.md) |
| JAX arrays and DLPack | [JAX exchange](dlpack.md) |
| MLX arrays, shared buffers and GPU updates | [MLX execution](mlx-training.md) |
| CoreML execution | [CoreML LoRA](coreml.md) |

The [PyTorch bridge](torch.md) accepts a session factory for backend selection.
[Runtime experiments](runtime-and-memory) contain allocator,
microbatch and device measurements, including their hardware and measurement scope.

```{toctree}
:maxdepth: 1

iobinding
dlpack
mlx-training
coreml
```
