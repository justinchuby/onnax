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

## Portable pivot storage

The portable pullback centers each row around its largest-probability entry and
sets that entry's derivative to the negative sum of the remaining entries. This
preserves the zero row sum of a softmax derivative. Flattened row offsets let
`Gather` read each pivot and `ScatterND` replace it, using one index and one zero
per row. The same construction supports broadcast masks and higher-order JVPs.

For a `[1, 8, 4096, 4096]` score tensor, the previous pivot construction used a
512 MiB floating zero matrix and a 128 MiB Boolean mask. Row indexing replaces
both with vectors of 32,768 entries. The probability and score-gradient matrices
remain part of the portable computation.

A matched 512-pixel UNet LoRA run on M1 Max measured 10 warmup steps and
100 training steps in FP32. Row indexing reduced median step time from 1.073 s
to 0.981 s and P95 from 1.098 s to 1.001 s. MLX peak allocation fell from
6.08 GB to 5.41 GB; whole-process lifetime peak physical footprint fell from
6.50 GB to 5.99 GB. Prediction, loss and all 72 adapter gradients matched
bitwise. Both ONNX runs used eager MLX execution, a 512 MiB cache, disabled ORT
optimization and disabled CPU fallback. The native indexed route uses
[MLX EP symbolic ScatterND support](https://github.com/onnxruntime/onnxruntime-ep-mlx/pull/107).
The [measurement report](https://github.com/justinchuby/onnax/blob/main/examples/diffusion/reports/real-images-m1max/mlx-indexed-pivots.json)
records the binary hash, versions and component timings.

Fusing all 18 attention pullbacks in the same UNet workload reduces median/P95
step time to 0.827/0.848 s. MLX peak allocation is 4.32 GB and whole-process
lifetime peak physical footprint is 5.28 GB. All 72 adapter gradients agree with
independent PyTorch MPS at `rtol=1e-4, atol=1e-6`; maximum absolute gradient error
is 6.4e-9. The matched MPS median is 0.320 s. The fused ONNX profile spends
0.168 s in forward and 0.663 s in loss/backward.
The [fused report](https://github.com/justinchuby/onnax/blob/main/examples/diffusion/reports/real-images-m1max/mlx-fused-attention.json)
records the same 10-warmup/100-step protocol, with a 64 MiB per-score GEMM budget.

Raising the per-score GEMM budget to 1 GiB measured 0.808/0.832 s median/P95
with 4.99 GB MLX peak allocation and 5.48 GB whole-process physical peak. This
run uses the same model, data, precision, optimizer and 100-step protocol. Its
[report](https://github.com/justinchuby/onnax/blob/main/examples/diffusion/reports/real-images-m1max/mlx-fused-attention-gemm.json)
records the explicit budget. The 64 MiB default gives lower peak allocation.

## Shape programs in the pullback

For fixed parameter shapes, onnax emits constant broadcast-reduction axes even
when output extents are dynamic. Symbolic target shapes use runtime axes.
[MLX EP shape-program execution](https://github.com/onnxruntime/onnxruntime-ep-mlx/pull/109)
keeps small shape-derived integer and Boolean operations on the MLX CPU stream,
so reduction axes reach their host consumer without a GPU round trip.
`ONNXRUNTIME_EP_MLX_GPU_SHAPE_OPS=1` selects the previous route for comparison.

With the same fused UNet graph and 64 MiB attention budget, 100 measured steps
reach 0.745/0.770 s median/P95, versus 0.827/0.848 s before this runtime change.
MLX peak allocation stays at 4.32 GB; whole-process physical peak is 5.24 GB.
Prediction, loss and all 72 adapter gradients match bitwise. The
[report](https://github.com/justinchuby/onnax/blob/main/examples/diffusion/reports/real-images-m1max/mlx-host-shapes.json)
records the binary hash, memory counters and component timings.

## Build the execution graph

Generate derivatives first, then apply deployment fusion before constant folding
or activation splitting:

```python
import onnax
from onnxscript import optimizer
from examples.huggingface.attention_vjp import fuse_attention_vjp

pullback = onnax.vjp(model, inputs=parameter_names, outputs=output_names)
execution = fuse_attention_vjp(pullback)
execution = optimizer.optimize(execution, inline=False)
```

The pass uses derivative-region annotations and checks every region boundary.
It preserves input, output and gradient names. The source derivative graph stays
available for further transformations. Each generated function records its
geometry and fallback-body digest in its name.

The contract covers FP32 Q/K/V with head dimensions from 8 through 256 in steps
of 8, including UNet dimensions 40, 80 and 160;
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

## Deploy into a saved decoder task

The deployment pass supports 256-wide heads used by Qwen3.5. Its upstream gradient
has the same layout as the attention output. The pass supplies this type and shape
at the function boundary when a dynamic reshape leaves them uninferred. Native
runtimes can then select the function directly.

For an existing export with retained `Attention` primals:

```bash
uv run --group huggingface --group training python -m examples.huggingface.attention_bundle \
  --export qwen_run/export --output qwen_attention_export
```

This creates a private bundle, generates and packages the eligible block VJPs,
checks their parameter/gradient interfaces, and updates artifact hashes. The
frozen weights, forward modules, tokenizer configuration and PEFT mapping remain
part of the same task. `attention_deployment.json` lists the affected blocks.
The derivative functions include ordinary ONNX bodies for portable execution.

Pass the resulting directory to the shared Trainer CLI with
`--load-export qwen_attention_export`. Choose the native score-matrix budget
before creating sessions:

```bash
export ONNXRUNTIME_EP_MLX_ATTENTION_VJP_DENSE_LIMIT_BYTES=1073741824
```

On an M1 Max, a complete 4032-token Qwen3.5-0.8B FP32 block measures:

| Pullback | Paired median / P95 | MLX allocator peak | Process peak physical |
|---|---:|---:|---:|
| Original graph | 329 / 350 ms | 3.97 GiB | 4.30 GiB |
| Tiled, 64 MiB score budget | 376 / 390 ms | 1.74 GiB | 2.43 GiB |
| GEMM, 1 GiB score budget | 260 / 272 ms | 2.54 GiB | 2.95 GiB |

Each strategy uses the same runtime binary, ten warmups and 100 paired
forward/backward calls in an isolated process. Runtime exchange is included.
All five block gradients pass the original `rtol=1e-4, atol=1e-6` check.
[Reports](../examples/huggingface/reports/qwen35-context4k-attention256-summary.json)
include source/library hashes, native dispatch evidence and measurement scope.
The wider tiled K/V kernel uses eight key rows and 20 KiB of shared memory.

The frozen K projection still contributes an activation gradient: adapters in
earlier layers receive gradients through Q, K and V. Selecting adapter parameters
for updates preserves these paths through frozen weights.

The deployed Qwen3.5 task completes four additional 4k updates from a saved
adapter: training loss moves from 2.11727 to 1.90056, validation loss from
2.43249 to 2.43000, and process physical footprint peaks at 8.33 GiB. Trainer
uses the decoder recipe's FP32 AdamW state and `eps=1e-5`. Fresh-process recovery
from update two reproduces update four byte for byte, including 62 state tensors,
Adam, scheduler and PEFT weights. Loaded loss, all 60 adapter gradients and inference
outputs agree on the recorded eight-token replay. The checker validates 114 ONNX
models; 3935 frozen tensor records remain unchanged throughout training.
