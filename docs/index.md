# onnax

Generate, compose and optimize ONNX differentiation graphs. Use `grad`, `jvp`,
`vjp` and `vmap`, then export ordinary ONNX models. External tools run training,
update parameters and manage data and checkpoints.

## Choose a starting point

| Goal | Start here |
| --- | --- |
| Differentiate an ONNX graph | [Quickstart](guide.md) |
| Train adapters or a small model | [Training tutorials](training.md) |
| Fine-tune a language or diffusion model | [Model workflows](models.md) |
| Control graph memory and fusion | [Graph optimization](fused-autodiff.md) |
| Choose precision and execution backends | [Runtime guide](runtime.md) |
| Compare measured results | [Experiments and reports](experiments.md) |

Explore gradients and matrix-free Newton-CG in the
[Jelly playground](https://www.justinchuby.com/onnax/jelly/).

```{toctree}
:maxdepth: 1
:caption: Learn and build

guide
training
models
```

```{toctree}
:maxdepth: 1
:caption: Optimize and execute

fused-autodiff
runtime
```

```{toctree}
:maxdepth: 1
:caption: Reference and results

api
experiments
roadmap
Jelly playground <https://www.justinchuby.com/onnax/jelly/>
```
