# Experiments and reports

These articles explain measured behavior, numerical investigations and design
tradeoffs. Start with the [training tutorials](training.md) or [model workflows](models.md)
for a runnable task; use the reports below to investigate a specific question.

## Training quality and numerical accuracy

| Question | Report |
| --- | --- |
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
