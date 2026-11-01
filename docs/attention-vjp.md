# Deploy attention pullbacks on MLX

Attention training spends substantial memory on probabilities over every query/key
pair. The optional deployment pass packages an onnax pullback as a local ONNX
function. MLX EP then chooses a tiled Metal kernel or a GEMM implementation using
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

## Qualification

Tests compare both attention policies with independent NumPy derivatives and the
portable ONNX function body. Full, chunked/checkpointed and joint tiny Qwen runs
match Transformers gradients and three independent optimizer updates. FP32/BF16
100-step training, fresh-process recovery and complete exported-model checking
cover the training workflow. The [public Torch bridge](torch.md) also has
regression, MLP and Qwen qualification.

The performance report records matched tokens, precision, checkpoint grouping,
head chunk size, backend versions, timing samples and process physical footprint.
