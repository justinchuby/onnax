# Differentiate fused operators

onnax differentiates `Attention`, `RMSNormalization`, `LayerNormalization`,
`RotaryEmbedding` and `Swish` directly. Their original forward nodes stay in the
graph, so a runtime can select its fused implementations.

```python
import onnx_ir as ir
import onnax

model = ir.load("model.onnx")
backward = onnax.vjp(model, inputs=["lora.A", "lora.B"], outputs=["logits"])
ir.save(backward, "model.vjp.onnx")
```

The same operators work with `jvp` and forward-over-reverse Hessian-vector
products. Builtin rules use the global decorator registry; external registrations
can override them for a selected domain and opset range.

## Attention

For one attention head, write

$$
S = sQK^T + M,\qquad P=\operatorname{softmax}(S),\qquad Y=PV.
$$

Given an upstream gradient $G$, the reverse products are

$$
\bar V=P^TG,\qquad \bar P=GV^T,
$$

$$
\bar S=P\odot\left(\bar P-
\operatorname{sum}_{\mathrm{keys}}(P\odot\bar P)\right),
\qquad \bar Q=s\bar SK,\qquad \bar K=s\bar S^TQ.
$$

GQA and MQA group query heads around their shared KV heads. Their KV gradients
sum across the group; the derivative graph avoids repeating K and V for each
query head. Softmax products center their operands and conserve the row sum,
using the same numerical strategy as onnax's standalone Softmax rule.

Supported training inputs have rank three or four, with known head counts and
dynamic batch and sequence lengths. Rules cover boolean and additive masks,
causal masking, softcap, masks of rank one through four, short masks and opset-25 sliding windows. Floating masks
can be differentiated. Fully masked rows produce zero outputs and derivatives.
The training interface uses the `Y` output and no KV cache.

The original `Attention` remains fused. Its portable derivative graph uses
standard ONNX operations and computes attention probabilities. A tiled attention
backward kernel is the next execution optimization for reducing those temporary
tensors.

## Normalization, RoPE and Swish

For RMSNorm, let $r=(\operatorname{mean}(X^2)+\epsilon)^{-1/2}$,
$N=Xr$, and $W=G\odot\mathrm{scale}$. Then

$$
\bar X=r\left(W-N\operatorname{mean}(WN)\right).
$$

The scale gradient reduces $G\odot N$ over broadcast dimensions. LayerNorm
also subtracts the mean of $W$; its rules cover scale, bias and the optional
mean and inverse-standard-deviation outputs. Statistics follow the operator's
`axis`, `epsilon` and `stash_type` attributes.

RoPE's input adjoint uses another fused `RotaryEmbedding` with the sine cache
negated: this applies the transposed rotation. Cache gradients accumulate across
heads and repeated position IDs. Both layouts, interleaving and partial rotation
are supported.

Swish uses

$$
f'(x)=\sigma(\alpha x)+\alpha x\sigma(\alpha x)(1-\sigma(\alpha x)).
$$

Its derivative evaluates the sigmoid with a stable exponential tail.

## Run Qwen training with MLX

```bash
uv sync --group dev --group training --group huggingface --group native-mlx
export ONNXRUNTIME_EP_MLX_NO_COMPILE=1
uv run python -m examples.huggingface.sft \
  --fused-ops --provider mlx --mode chunked --checkpoint-layers 1 --nf4 \
  --strict-reference --evaluate-train --output-dir qwen_fused
```

Use `--library /path/to/libonnxruntime_mlx_ep.dylib` for a development MLX EP build.
`--projection-precision bf16` selects BF16 projection computation with FP32
attention, normalization, gradients and optimizer state. `--provider reference`
executes the ordinary exported graph with ONNX ReferenceEvaluator.

Attention, RMSNormalization and RoPE require opset 23 or later; Swish requires
24 or later. LayerNormalization follows onnax's opset-18 minimum. With the pinned
CPU ORT version and graph optimization disabled, Swish needs a runtime execution
implementation; the example selects MLX or ReferenceEvaluator for this path.
CPU ORT also lacks the newer sliding-window attributes and short-mask behavior,
which the operator tests exercise through ReferenceEvaluator.

## Native qualification

On an M1 Max with 32 GiB RAM, the two-layer Qwen NF4 example completes 100
updates with strict PyTorch comparisons at the initial and final states:

| Projection compute | Fixed training loss | Validation loss |
| --- | ---: | ---: |
| FP32 | 5.5434 → 5.1406 | 5.5429 → 5.4180 |
| BF16 | 5.5434 → 5.1423 | 5.5429 → 5.4088 |

Recovery from step 50 reproduces the step-100 adapters, AdamW and scheduler
states, loaded predictions and adapter gradients byte for byte. All 24 exported
FP32/BF16 modules pass the full ONNX checker. Native tests also compare three
updates with PyTorch in full-logits, chunked/checkpointed and joint-loss modes.

A Qwen3-0.6B 4096-token smoke comparison uses batch one, four-layer checkpoint
groups, 128-token head chunks, NF4 storage and FP32 projection compute:

| Path | Update median | Update P95 | Peak physical footprint | Session graph nodes |
| --- | ---: | ---: | ---: | ---: |
| Decomposed | 24.64 s | 37.20 s | 23.71 GiB | 36,443 |
| Fused, reviewed build | 19.95 s | 21.41 s | 17.82 GiB | 12,986 |

Timing uses one warmup and three measured updates, with the same token IDs,
weights, optimizer and thread settings. Memory includes construction, host
arrays, runtime state and native caches. MLX compilation and ORT graph
optimization are disabled. All four losses align within `rtol=1e-6, atol=1e-6`.
The {download}`qualification report
<../examples/huggingface/reports/qwen3-fused-operators-m1max.json>` contains the
raw timings, earlier fused smoke run, source hashes, recovery checks and
reproduction commands.

Reproduce the fused measurement with the
[prepared long-context corpus](qwen-context-training.md):

```bash
export ONNXRUNTIME_EP_MLX_NO_COMPILE=1
uv run python -m examples.huggingface.guard --output-dir fused_4k.guard -- \
  .venv/bin/python -m examples.huggingface.benchmark --mode chunked \
  --real --nf4 --fused-ops --provider mlx --checkpoint-layers 4 \
  --sequence-length 4096 --batch-size 1 --chunk-size 128 --mlx-memory-limit-gib 20 \
  --train-file /path/to/benchmark.jsonl --warmup 1 --samples 3
```
