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
| NF4 | Example bridge and Trainer regression | Multi-layer Qwen execution, memory and throughput measurements |

The real-model FP32 elementwise comparison remains open. FP64 diagnostics help
locate arithmetic differences; the acceptance target is FP32 and lower precision
compute with FP32 accumulation.
