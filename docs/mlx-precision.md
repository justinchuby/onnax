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

## M1 Max training measurements

Qwen3-0.6B runs a fixed 4096-token batch with NF4 decoder weights, group-4
checkpoints and 128-token output-head chunks. Each precision has ten warmup
updates and 100 recorded updates. Timings cover the complete synchronous
training step, including host exchange and FP32 AdamW; construction and
tokenization are measured separately.

| Projection compute | Step median / P95 (s) | Peak process physical footprint (GiB) |
| --- | ---: | ---: |
| BF16 | 9.00 / 10.29 | 13.84 |
| FP16 | 8.69 / 8.83 | 13.81 |

BF16 improves from the [previous native measurement](attention-vjp.md)
(9.64 s median) to
9.00 s. Its matching PyTorch MPS baseline
is 10.47 s.
Both paths use the same NF4-rounded decoder values; the MPS reference stores
them as dense weights and ONNX stores packed weights.

The process memory peak includes setup, runtime and allocator caches. The
runtime uses a 1 GiB attention score budget and a 20 GiB allocator limit.
The [evidence report](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/qwen3-half-precision-m1max.json)
retains raw timings, full loss trajectories, memory counters, runtime identities,
precision microbenchmarks and the independent-reference checks.

### Reproduce

Prepare the [long-context corpus](qwen-context-training.md), then run:

```sh
ONNAX_TILED_ATTENTION=1 \
ONNXRUNTIME_EP_MLX_NO_COMPILE=1 \
ONNXRUNTIME_EP_MLX_ATTENTION_VJP_DENSE_LIMIT_BYTES=1073741824 \
python -m examples.huggingface.benchmark \
  --mode chunked --real --nf4 --fused-ops --provider mlx \
  --library /absolute/path/to/libonnxruntime_mlx_ep.dylib \
  --projection-precision bf16 --batch-size 1 --sequence-length 4096 \
  --checkpoint-layers 4 --chunk-size 128 --mlx-memory-limit-gib 20 \
  --train-file /tmp/qwen-long-corpus/benchmark.jsonl \
  --warmup 10 --samples 100 --output bf16-benchmark.json
```

Select `--projection-precision fp16` for the FP16 policy.

## Projection microbenchmarks

Forward products use `[M, K, N] = [1024, 1024, 2048]`; NF4 adjoints use
`[1024, 2048, 1024]`. Each case has ten warmups and 100 timed calls, including
FP32 NumPy feeds and outputs. Timing runs leave tracing off; separate calls
collect dispatch evidence.

| NF4 projection policy | Forward, portable → native (ms) | FP32 adjoint, portable → native (ms) |
| --- | ---: | ---: |
| BF16 | 2.46 → 1.18 | 2.33 → 1.11 |
| FP16 | 2.35 → 1.10 | 2.38 → 1.11 |

Fused and portable NF4 outputs are bitwise equal for all measured cases.
The existing affine quantized GEMM path has these forward medians:

| Weight encoding | FP32 compute (ms) | BF16 compute (ms) | FP16 compute (ms) |
| --- | ---: | ---: | ---: |
| INT4 | 1.13 | 1.42 | 1.14 |
| INT8 | 1.15 | 1.35 | 1.13 |

Each encoding uses its own test weights. Its independent NumPy reference decodes
those weights before checking the GPU result. All measured paths execute on
MLX EP with CPU fallback disabled.

## Training and recovery

The FP16 Trainer run completes 100 updates, then repeats steps 50–100 in a fresh
process. Final adapters, optimizer/scheduler state, frozen-weight hashes,
exported predictions, reloaded loss and gradients match exactly. All 24 exported
ONNX modules pass the complete model checker. Training and validation loss
trajectories are retained in the evidence report.
