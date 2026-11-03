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

## Prepare a saved Torch bundle

Generate a deployment header from a flat `OnnxModule` export:

```sh
python -m examples.torch_deploy --bundle /tmp/unet-pretrained/export \
  --fused-attention --output /tmp/unet-attention-vjp.onnx
```

The header and its JSON manifest reuse the bundle's existing frozen payloads,
matched by dtype, shape and SHA256. Input, seed and gradient names stay unchanged.
The tool runs full ONNX checker validation using private file snapshots. The
[image benchmark](diffusion-images.md) loads it with `--derivative-header`,
checks its source/payload identity and installs it in the runtime's private bundle.
The qualified UNet header is about 2 MB and reuses 149 initializers.

## Package dynamic broadcast adjoints

Broadcasting repeats a tensor along leading or singleton dimensions. Its adjoint
sums along those dimensions: for a target shape $r$, the returned gradient is
$\operatorname{reshape}(\operatorname{sum}_{\text{broadcast axes}} G, r)$.

The deployment tool can package these reductions alongside Attention VJP:

```sh
python -m examples.torch_deploy --bundle /tmp/unet-pretrained/export \
  --fused-attention --broadcast-adjoints --output /tmp/unet-training-vjp.onnx
```

`pkg.onnax::SumToShape_v1(Gradient, Reference)` uses the reference's shape and
returns a tensor of that shape with the gradient's dtype. Reference values are
unused. The portable function discovers axes using ordinary ONNX operators;
MLX specializes the reduction using both live tensor shapes. This covers leading
axes, runtime singleton dimensions, scalars and empty tensors. The deployment
manifest records the number of packaged calls.

Apply `examples.broadcast_adjoint.fuse_broadcast_adjoints` immediately after
differentiation and Attention fusion, before ONNXScript optimization. It matches
the complete generated function body and keeps shared or edited regions intact.
Inline local functions with `onnx_ir.passes.common.InlinePass` before applying
further derivative transforms.

Checkpointed decoder exports accept the same option through
`python -m examples.huggingface.attention_bundle --export SOURCE --output DEST
--broadcast-adjoints`. Offline tests cover Qwen3, Qwen3.5 and Phi3 predictions,
all adapter gradients, three optimizer updates, loading and re-export.

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

## UNet measurements

The matched M1 Max workload uses FP32, one 512-pixel image encoded as 64² latents,
fixed adapters and external AdamW. Each run has 10 warmups and 100 measured
steps; separate ten-sample phase profiles record forward, backward and updates.
MLX uses eager execution except for rows labeled compiled. All
use a 512 MiB idle cache and a 64 MiB attention score budget. ORT graph optimization and CPU fallback are disabled. MPS uses its
default allocator. Process peak includes bundle/model loading, session setup,
warmup and execution.

| Route | Median / P95 | MLX peak allocation | Process peak |
| --- | ---: | ---: | ---: |
| Raw MLX | 1.073 / 1.098 s | 6.08 GB | 6.50 GB |
| Row-indexed pivots | 0.981 / 1.001 s | 5.41 GB | 5.99 GB |
| Fused attention | 0.827 / 0.848 s | 4.32 GB | 5.28 GB |
| Host shape programs | 0.745 / 0.770 s | 4.32 GB | 5.24 GB |
| Direct convolution padding | 0.696 / 0.713 s | 4.32 GB | 5.32 GB |
| Broadcast adjoints, eager | 0.669 / 0.688 s | 4.33 GB | 5.31 GB |
| Broadcast adjoints, compiled | 0.651 / 0.661 s | 4.31 GB | 5.37 GB |
| Joint scalar objective, compiled | **0.474 / 0.482 s** | **4.32 GB** | **5.40 GB** |
| PyTorch MPS eager | 0.320 / 0.323 s | — | 7.39 GB |
| PyTorch MPS checkpointed | 0.436 / 0.441 s | — | 5.82 GB |

All variants pass the prediction/loss/72-adapter-gradient comparison with
independent PyTorch MPS at `rtol=1e-4, atol=1e-6`. The
[reports](https://github.com/justinchuby/onnax/tree/main/examples/diffusion/reports/real-images-m1max)
record model/data/state hashes, native binaries, versions and measurement scope.

For fixed target shapes, broadcast pullbacks use constant reduction axes even
when source extents are dynamic. [MLX host shape programs](https://github.com/onnxruntime/onnxruntime-ep-mlx/pull/109)
execute small shape-derived integer/Boolean operations on the CPU stream.
A same-binary ablation with `ONNXRUNTIME_EP_MLX_GPU_SHAPE_OPS=1` measures
0.839/0.854 s versus 0.745/0.770 s, with bitwise matching captures.
Diagnostic translation/synchronization spans fall from 715 ms to 244 ms
across the same probe, including setup.

UNet's reverse graph has 46 frozen-weight `ConvTranspose` nodes.
[Direct nonnegative padding](https://github.com/onnxruntime/onnxruntime-ep-mlx/pull/110)
computes the requested extent directly: a 16×16 output previously used an
18×18 intermediate followed by cropping. Its captures match the preceding
runtime bitwise. A separate frozen-layout-cache prototype measures
0.705/0.723 s and retains 737 MB more weights, raising active/peak allocation
to 2.23/5.06 GB. Direct padding is the selected implementation.

Before host-shape optimization, raising the attention score budget to 1 GiB
measures 0.808/0.832 s with 4.99 GB peak allocation. The reports retain this
workspace trial alongside the lower-memory 64 MiB default.

Before broadcast-adjoint fusion, default compilation measures 0.963/0.990 s
and 7.16 GB process peak. A same-binary control measures 0.959/0.984 s; packaging
245 broadcast adjoints reaches 0.651/0.661 s, with process peak changing from
7.15 GB to 5.37 GB. All 74 captures match bitwise. Matching diagnostic probes
reduce 1172 subgraph executions to 8 and recorded materialized output bytes
from 4.27 GB to 3.72 MB. Use `--broadcast-adjoints` with
[MLX EP support](https://github.com/onnxruntime/onnxruntime-ep-mlx/pull/111)
and `ONNXRUNTIME_EP_MLX_NO_COMPILE=0`.

For a scalar ONNX loss, [joint objective execution](torch.md) schedules the loss
and its gradients in one call. The UNet example measures 0.474/0.482 s with the
same workload and native binary. Prediction and every adapter gradient match
the two-call route bitwise; the MSE reduction differs by 7.5e-9.

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

(recomputed-pullbacks)=

## Recomputed pullbacks

`fuse_attention_vjp(model, recompute=True)` emits `AttentionVJP_v2_*` for
unmasked, noncausal FP32 heads with widths 32 through 64, in multiples of eight.
The five inputs are Q, K, V, G and an unused mask placeholder. The ordinary ONNX
body recomputes Y from Q/K/V before applying the pullback. This makes the function
self-contained and lets [MLX EP](https://github.com/onnxruntime/onnxruntime-ep-mlx/pull/113)
use its fused SDPA forward/backward implementation.
Rank-three and rank-four layouts, grouped query heads, dynamic batch/sequence
lengths and selected Q/K/V gradient outputs use the same interface.

The v1 function keeps its supplied-Y contract. Other attention forms continue to
use that saved-output implementation. Both versions contain portable ONNX bodies.

Prepare an existing Torch bundle with the recomputed functions:

```sh
python -m examples.torch_deploy --bundle export --output attention.onnx \
  --recompute-attention --broadcast-adjoints
```

For the real-image UNet workflow, pass `--recompute-attention` to
`examples.diffusion.objective`. The resulting bundle uses the usual
`OnnxModule.load` and external optimizer interfaces.
