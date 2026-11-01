# Deploy attention pullbacks on MLX

Attention training spends substantial memory on probabilities over every query/key
pair. The optional deployment pass packages an onnax pullback as a local ONNX
function in the `pkg.onnax` domain. MLX EP then chooses a tiled Metal kernel or a GEMM implementation using
an explicit workspace budget. Other runtimes execute the function's ONNX body.

## The backward computation

For a head with scale $s$, mask $M$ and upstream gradient $G$,

$$
P=\operatorname{softmax}(sQK^T+M),\qquad Y=PV,
$$

$$
D=\operatorname{rowsum}(G\odot Y),\qquad
\bar S=P\odot(GV^T-D),
$$

$$
\bar Q=s\bar S K,\qquad \bar K=s\bar S^TQ,\qquad \bar V=P^TG.
$$

GQA sums the contributions from query heads sharing each KV head. The tiled
implementation recomputes log-sum-exp and probabilities within fixed-size tiles.
Separate dQ and dK/dV kernels avoid atomic updates; causal attention skips tiles
above the diagonal. Fully masked rows contribute zero.

## Build the execution graph

Generate derivatives first, then apply deployment fusion before constant folding
or activation splitting:

```python
import onnax
from onnxscript import optimizer
from examples.huggingface.attention_vjp import fuse_attention_vjp

pullback = onnax.vjp(model, inputs=parameter_names, outputs=output_names)
execution = fuse_attention_vjp(pullback)
optimizer.fold_constants(execution)
```

The pass uses derivative-region annotations and checks every region boundary.
It preserves input, output and gradient names. The source derivative graph stays
available for further transformations. Each generated function records its
geometry and fallback-body digest in its name.

The first contract covers FP32 Q/K/V with head dimensions 16, 32, 64 and 128;
rank-three/rank-four layouts; MHA/GQA/MQA; boolean/additive masks; and top-left
causality. Unsupported configurations retain their portable derivative graph.

## Select the memory policy

The Qwen example pipeline enables this pass with `ONNAX_TILED_ATTENTION=1`.
Use it with `--fused-ops` and an MLX EP library built from
[the attention pullback implementation](https://github.com/onnxruntime/onnxruntime-ep-mlx/pull/100).

`ONNXRUNTIME_EP_MLX_ATTENTION_VJP_DENSE_LIMIT_BYTES` sets the largest individual
score tensor handled by GEMM. Its default is 64 MiB; `0` selects tiled execution.
A 1 GiB budget admits a `[1,16,4096,4096]` FP32 score tensor. GEMM intermediates
have local ownership and expire after their last consumer. The score budget and
the provider's allocator limit control different quantities.

Frozen NF4 products with at least 128 activation rows also use a temporary FP32
decode followed by MLX GEMM. Packed weights remain in the ONNX artifact. Set
`ONNXRUNTIME_EP_MLX_LUT4_GEMM=0` to compare with the small-tile lookup kernel.

Frozen Cast/Transpose/Identity chains retain immutable provenance. The provider
caches large floating precision views and packs a very wide frozen projection's
transposed matrix for backward GEMM. Runtime inputs stay live, including inside
control-flow bodies. `ONNXRUNTIME_EP_MLX_PACK_FROZEN_VIEWS=0` selects the original
view execution for comparison.

## Qualification

Tests compare both attention policies with independent NumPy derivatives and the
portable ONNX function body. Full, chunked/checkpointed and joint tiny Qwen runs
match Transformers gradients and three independent optimizer updates. FP32/BF16
100-step training, fresh-process recovery and complete exported-model checking
cover the training workflow. The [public Torch bridge](torch.md) also has
regression, MLP and Qwen qualification.

(m1-max-measurements)=

## M1 Max measurements

Qwen3-0.6B runs a fixed 4096-token batch with group-4 decoder checkpoints,
128-token output-head chunks, and external FP32 AdamW. Each measurement has
10 warmup updates and 100 timed updates. Model construction and tokenization
are recorded separately; step timing covers forward, loss, backward, exchange
and parameter updates with synchronous device boundaries.

| Projection compute | ONNX/MLX median / P95 (s) | PyTorch MPS median / P95 (s) |
| --- | ---: | ---: |
| FP32 | 8.61 / 8.89 | 9.10 / 9.13 |
| BF16 | 9.64 / 10.45 | 10.47 / 10.51 |

Whole-process peak physical footprint includes setup, runtime and allocator
caches. The machine is an M1 Max with 32 GiB memory.

| Projection compute | ONNX/MLX (GiB) | PyTorch MPS (GiB) |
| --- | ---: | ---: |
| FP32 | 14.32 | 12.89 |
| BF16 | 14.15 | 14.51 |

Both paths use the same NF4-rounded decoder values. ONNX keeps packed weights;
the MPS reference stores their dequantized dense values. Master adapters,
gradients and optimizer state use FP32. BF16 runs round projection computation
and use FP32 adjoints.

The measured runtime uses a 1 GiB score-tensor budget and a 20 GiB allocator
limit. Its frozen precision/layout cache stays with each session. The report
also records tiled execution and idle-cache/head-chunk experiments. A 1024-token
head with an unbounded idle cache exceeded the 24 GiB process budget; the
128-token head is the selected workload.

The [evidence report](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/qwen3-native-pullbacks-m1max.json)
contains all native timing samples, matching 100-sample MPS baselines from November 1, 2026,
fresh MPS repeats, loss trajectories, memory counters, source/binary hashes,
qualification and recovery results. The runtime implementation is merged in
[MLX EP #100](https://github.com/onnxruntime/onnxruntime-ep-mlx/pull/100).

### Reproduce the measurement

Prepare the [long-context corpus](qwen-context-training.md), build MLX EP, then run:

```sh
ONNAX_TILED_ATTENTION=1 \
ONNXRUNTIME_EP_MLX_NO_COMPILE=1 \
ONNXRUNTIME_EP_MLX_ATTENTION_VJP_DENSE_LIMIT_BYTES=1073741824 \
python -m examples.huggingface.benchmark \
  --mode chunked --real --nf4 --fused-ops --provider mlx \
  --library /path/to/libonnxruntime_mlx_ep.dylib \
  --projection-precision fp32 --batch-size 1 --sequence-length 4096 \
  --checkpoint-layers 4 --chunk-size 128 --mlx-memory-limit-gib 20 \
  --train-file /tmp/qwen-long-corpus/benchmark.jsonl \
  --warmup 10 --samples 100 --output fp32-benchmark.json
```

Select `--projection-precision bf16` for the second precision policy.
