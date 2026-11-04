# Experiments and reports

These articles explain measured behavior, numerical investigations and design
tradeoffs. Start with the [training tutorials](training.md) or [model workflows](models.md)
for a runnable task; use the reports below to investigate a specific question.

## Training quality and numerical accuracy

| Question | Report |
| --- | --- |
| How does BF16/NF4 training behave across decoder families? | [Family audits and 100-step recovery](decoder-bf16-projections) |
| Do independent training runs reach similar quality? | [Training quality](training-quality.md) |
| How closely do reference logits and optimizer trajectories agree? | [Reference accuracy](qwen-reference-accuracy.md) |
| Where does forward or backward rounding enter? | [Qwen runtime audit](qwen-runtime-audit.md) |
| How are small loss gradients preserved? | [FP32 loss accuracy](softmax-precision.md) |
| How large can BF16 matrix error be? | [Product bounds](bf16-bounds.md) and [composed bounds](composed-bounds.md) |

```{toctree}
:maxdepth: 1

training-quality
qwen-sft-qualification
qwen-reference-accuracy
qwen-runtime-audit
softmax-precision
bf16-bounds
composed-bounds
```

(runtime-and-memory)=

## Runtime and memory

The measurements below cover particular workloads and machines. Each article
records the execution boundary, precision and validation used for its comparison.

| Question | Measurements |
| --- | --- |
| How does Qwen3 4K ONNX training compare with PyTorch MPS? | [Current Qwen performance](current-4k-performance) |
| How much does compiling the output head save? | [Head layout and session policy](selective-head-compilation) |
| How does Qwen3.5 compare with PyTorch MPS at 4K? | [Matched FP32 measurement](qwen35-mps-comparison) |
| How do fused recurrent scans change Qwen3.5 execution? | [Full-model 4K measurement](qwen35-state-scan-performance) |
| Where do fused attention pullbacks help? | [Native pullback measurements](m1-max-measurements) |

```{toctree}
:maxdepth: 1

block-runtime
classifier-device
metal-microbatch
memory-lifecycle
activation-liveness
mlx-bf16
```

## Development notes

The [roadmap](roadmap.md) lists current priorities. Historical measurements and
handoff notes are collected here:

```{toctree}
:maxdepth: 1

development-history
ort-training-handoff
```
