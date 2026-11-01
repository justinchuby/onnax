# Choose precision for MLX fine-tuning

The Qwen training examples offer FP32, BF16 and FP16 projection computation.
Adapters, gradients, AdamW state, attention, normalization and token loss use
FP32. A frozen NF4 decoder keeps its weights packed between executions.

Select the projection policy with `--projection-precision`:

```sh
python -m examples.huggingface.sft --provider mlx \
  --library /absolute/path/to/libonnxruntime_mlx_ep.dylib \
  --mode chunked --checkpoint-layers 1 --nf4 \
  --projection-precision fp16 --steps 100 --output-dir qwen-fp16
```

Use `bf16` for BF16 projection computation or `fp32` for the FP32 baseline.
The [Qwen tutorial](qwen-sft.md) describes training data, external Trainer updates,
checkpoint recovery and adapter export.

## Projection arithmetic

For a frozen decoded weight $W$, let $R_h$ round to the selected half dtype.
The forward projection and its input adjoint are

$$
Y=R_h(R_h(X)R_h(W)),\qquad
\bar X=\bar Y\,\operatorname{float32}(R_h(W))^T.
$$

Trainable LoRA products use the same forward policy and FP32 straight-through
adjoints. The precision choice is part of the exported interface, including
chunked output heads, so loading an artifact restores its computation policy.

MLX EP recognizes the NF4 decode chain through both half casts and FP32 widening
casts. It decodes directly into a half buffer for GEMM, then widens that rounded
buffer when the derivative requires FP32. These buffers expire within the
execution. Other consumers of a decoded weight retain their original outputs.

This avoids the full integer unpack/gather/scale tensor chain and the FP32
weight buffer preceding a half conversion. The portable ONNX function body
continues to express the same arithmetic.

## Correctness and measurement

Native tests compare fused and portable lookup products exactly across FP16,
BF16, forward and adjoint products, transposes, dynamic rows, empty inputs and
shared decoded outputs. Independent NumPy products check the mathematical
result, and runtime traces check which implementation executes.

Tiny Qwen tests compare logits, token loss, every adapter gradient and five
independent AdamW updates with Transformers, for both dense and NF4 storage.
The FP16 whole-model checks use one FP16 epsilon on output/gradient tensor norm
and maximum magnitude. Parameter trajectories use a tensor norm budget; token
loss and fixed-state adapter gradients also retain the FP32 comparison check.

The [MLX EP projection benchmark](https://github.com/onnxruntime/onnxruntime-ep-mlx/blob/main/bench/lookup4_precision.py)
measures NF4 forward/adjoint products and affine INT4/INT8 forward products with
FP32 host boundaries. It records raw synchronous timings, runtime identity and
output arrays. Use the [full training benchmark](attention-vjp.md) to measure
complete steps, runtime transfers and process memory.

NF4 and affine INT4 have different encodings. Keep the format in the model's
interface: an NF4 codebook cannot be passed directly to affine quantized GEMM.
