# FP32 loss and derivative accuracy

Training uses FP32 parameters, gradients, reductions and optimizer state. Higher
precision calculations serve as diagnostic oracles. Lower precision compute uses
explicit boundaries with FP32 loss and accumulation, as in the [BF16 example](bf16.md).

## Check the reference as well as the runtime

A reference evaluator is an implementation with its own rounding and kernels.
The same-input kernel audit runs both ONNX ReferenceEvaluator and ORT CPU, with
ORT graph optimization disabled:

```bash
uv run python -m examples.huggingface.numerics --output fp32-kernels.json
```

The recorded ONNX 1.23.2 / ORT 1.31.1 results show two different failures:

| FP32 calculation | ONNX ReferenceEvaluator | ORT CPU | Scalar oracle |
| --- | --- | --- | --- |
| Sigmoid(-18) | 1.5229979e-8 | 0 | 1.52299795e-8 |
| LogSoftmax([0, -200]), second element | -inf | -200 | -200 |

ReferenceEvaluator computes this LogSoftmax through Softmax followed by Log;
the probability underflows before Log. ORT's Sigmoid loses a representable small
value. These checks compare kernels at identical inputs, independently of the
model's accumulated rounding differences.

On Apple M1 Max, MLX EP 0.31.0 preserves both negative-tail Sigmoid values
and returns finite LogSoftmax results. The audit disables CPU fallback and records
MLX-only kernel events from ORT profiling:

```bash
uv sync --group dev --group training --group native-mlx
uv run python -m examples.huggingface.numerics --mlx --output fp32-mlx.json
ONNAX_TEST_MLX_EP=1 uv run pytest -q tests/test_mlx_precision.py
```

Native MLX regressions cover the kernel audit, confident Softmax/LogSoftmax
JVP/VJP and Softplus gradients/HVP. The development MLX provider also executes
the complete hard-label loss and gradient graph with CPU fallback disabled,
including a 151,936-class case and ignored NaN rows.
The CPU Sigmoid reproduction is tracked in [onnax issue #95](https://github.com/justinchuby/onnax/issues/95).

The Qwen validation reference is the corresponding **PyTorch Transformers model**.
Its report is distinct from these ONNX ReferenceEvaluator diagnostics.

## Preserve small gradients

[A Fix for Softmax and Cross-Entropy](https://arxiv.org/html/2610.31642v1)
by Barron, Gnanapragasam and Tschannen describes cancellation when a probability
rounds to one, and when nearby student and teacher probabilities are subtracted.

onnax's Softmax and LogSoftmax derivatives select one maximum logit, center the
incoming direction there, and recover its gradient from the negative sum of the
other entries. A constant logit shift has zero derivative; a confident target
retains its small gradient. Ties select one pivot consistently.

Softplus differentiation uses a bounded exponential expression for its sigmoid
slope. It preserves negative-tail gradients on ORT CPU and composes with JVP for
higher derivatives.

The Qwen chunked-head example computes hard-label loss from the target's margin
against all other classes:

$$
L=\operatorname{softplus}\left(\log\sum_{j\ne t} e^{z_j}-z_t\right).
$$

It retains a positive FP32 loss beyond the usual probability-rounding threshold,
ignores labels `-100`, and accumulates chunk losses in FP32.

The loss uses a portable ONNX local function and analytic JVP/VJP rules. Let
$u=\log\sum_{j\ne t}e^{z_j}-z_t$ and $q=\sigma(u)$. Its target derivative is
$-q$; each other derivative is $q e^{z_j}/\sum_{k\ne t}e^{z_k}$. Computing
these values directly avoids differentiating a numerically cancelling
stabilization shift.

For 151,936 classes with target logit 10 and all other logits -4, the previous
expanded derivative gave the chosen non-target pivot a gradient of -1.17e-5.
Its analytic gradient is positive, 7.38e-7. The registered rules preserve this
sign and match every component against a scalar oracle on CPU and MLX. Ignored
rows are replaced by zeros before arithmetic, producing zero loss and gradients
even when their input logits contain NaNs. JVP, HVP and vmap checks cover the
composed rules.
The [CPU/MLX before-and-after report](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/hard-ce-fp32-pivot.json)
records the scalar oracle, all checks, runtime versions and library hash.

Call `stable_losses.install(model)` to attach the standard-ONNX function bodies
after building a graph with these helpers. The chunked head installs them
automatically. Keep the local functions intact during optimization before AD.

`examples.huggingface.stable_losses` also provides a fixed-teacher logit loss,
using compensated Log1p/Expm1 ONNX local functions and global JVP/VJP rules.
Equal teacher and student logits produce zero gradients. Analytic function rules
preserve curvature at zero. Keep local functions intact while optimizing before
further differentiation. Exported function bodies use standard ONNX operations.

## Acceptance work remaining

| Workload | Completed | Next acceptance |
| --- | --- | --- |
| External regression LoRA + Trainer | 200 steps, checkpoint replay, export | Maintain regression coverage |
| Tiny Qwen3 | Training, masking, adapter exchange, chunked head; rounded BF16/FP16 logits with FP32 loss tests | Full mixed-precision model execution |
| Qwen3-0.6B FP32 | 30 updates, loss decrease, checkpoint replay, export | Locate layerwise FP32 differences at identical intermediate inputs; repeat paired updates at the original tolerance |
| BF16 Qwen3 | Dtype policy and smaller mixed-precision examples | Full model gradients and optimizer trajectory |
| NF4 | Tiny Qwen SFT/recovery/PEFT; real Qwen 30-step training; isolated tiny benchmarks | Real-model gradient qualification, decoding lifetime and fused backend |

The real-model FP32 elementwise comparison remains open. FP64 diagnostics help
locate arithmetic differences; the acceptance target is FP32 and lower precision
compute with FP32 accumulation.

## Locate Qwen3 differences at identical inputs

```bash
uv sync --group dev --group training --group huggingface
HF_HUB_OFFLINE=1 uv run python -m examples.huggingface.layer_audit \
  --real --output qwen3-fp32-layers.json
```

The audit exposes inputs and outputs of the source Mobius MatMul, RMSNorm and
Swish operators. Each operator is replayed in PyTorch FP32 using the **observed
ONNX input**, so upstream rounding is held fixed. The frozen Qwen3-0.6B run uses
64 token IDs and the pinned model revision. Selected MatMul elements also have
an independent scalar dot-product oracle using `math.fsum` over exact FP32
products.

The recorded 338-operator audit found:

| Operator | Nodes | Elements exceeding the original tolerance before the Swish fix | After |
| --- | --- | --- | --- |
| RMSNorm | 113 | 0 | 0 |
| Swish | 28 | 10 | 0 |
| MatMul | 197 | 31,756 | 32,589 |

Swish now uses a bounded exponential sigmoid, preserving small negative-tail
activations and derivatives. Changing its output also changes downstream MatMul
inputs, so the MatMul columns describe two different observed trajectories.
Full-logit relative L2 error against PyTorch changed from 4.63e-6 to 4.29e-6;
strict elementwise violations changed from 220,670 to 210,219 out of 9,723,904.

The lm_head contains most local MatMul violations. At one of its largest scaled
mismatches, the independent dot product was `-0.0009705360527334506`.
ORT's absolute error was `4.12e-6`; PyTorch's was `7.07e-6`. This provides a
concrete case where the FP32 reference contributes more error than ORT.

The before/after JSON reports in `examples/huggingface/reports/` retain source
node names, local metrics and selected dot-product oracles. The next comparison
is adapter-gradient propagation at fixed activations, followed by the independent
optimizer trajectories.

(qwen-stable-recovery)=
### Real-model training and recovery

The stable-Swish Qwen3-0.6B run completed 30 updates in FP32. Validation loss
went from **7.435994 to 6.264874**. A new process resumed checkpoint 15 and
reproduced all 112 adapter tensors, all 113 checkpoint tensors, AdamW moments
and scheduler state byte for byte. Both frozen ONNX exports were loaded and
executed; logits and hidden states were byte-identical. Their 336 frozen
initializer hashes also matched.

Run and verify recovery with:

```bash
uv run python -m examples.huggingface.sft --real --steps 30 \
  --diagnose-rounding --save-activations --peft-roundtrip --output-dir run
uv run python -m examples.huggingface.sft --real --steps 30 \
  --diagnose-rounding --save-activations --peft-roundtrip \
  --resume run/checkpoint-15 --output-dir resumed
uv run python -m examples.huggingface.replay \
  --uninterrupted run --resumed resumed --step 30 --output replay.json
```

`qwen3-stable-swish-replay.json` records the training, recovery and original
numerical comparisons. At the final PEFT comparison, two gradient elements
in layer 20's q-projection B exceed `rtol=1e-4, atol=1e-6`; independent AdamW
updates also exceed the threshold. Diagnostic mode records each failing
parameter. Strict mode raises on the first failure. Completed training and
export results are saved before optional audits run.
