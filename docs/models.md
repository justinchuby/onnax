# Language and diffusion models

Mobius builds ONNX model graphs and preserves source weight names. onnax adds
adapters and derivative graphs. External libraries provide tokenization,
training loops, sampling and adapter storage.

## Language models

Start with the [decoder-family workflow](decoder-families.md) for supported
architectures and pretrained sources, or [Qwen3 SFT](qwen-sft.md) for a complete
TRL example. Choose subsequent topics by workload:

- [Qwen3.5](qwen35-training.md): recurrent linear-attention state and chunk replay.
- [Adapters and memory](qwen-adapters.md): PEFT exchange, chunked loss and saved activations.
- [Long contexts](qwen-context-training.md): data preparation, capacity and recovery at 4k tokens.
- [NF4](qwen-nf4.md): packed frozen weights and trainable adapters.

```{toctree}
:maxdepth: 1

decoder-families
qwen-sft
qwen35-training
qwen-adapters
qwen-context-training
qwen-nf4
```

## Diffusion models

The first tutorial builds and fine-tunes a conditional denoiser. The second
connects encoded image-caption batches, the external scheduler and checkpoint
recovery.

```{toctree}
:maxdepth: 1

diffusion
diffusion-images
```

For numerical audits and matched performance comparisons, use the
[experiment index](experiments.md).
