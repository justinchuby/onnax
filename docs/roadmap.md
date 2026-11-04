(workload-driven-development-roadmap)=

# Development roadmap

onnax generates, composes and optimizes ONNX differentiation graphs for
fine-tuning, small-model training and scientific computing. External tools own
optimizers, data loading, schedulers and checkpoint storage.

## Current priorities

1. **General training interfaces.** Consolidate model-specific wrappers around
   the public `OnnxModule` and `CheckpointedModule`, which already provide saved
   activations and named recomputation boundaries. Preserve parameter mappings,
   upstream gradients, accumulation, export and checkpoint recovery while
   carrying qualified execution policies across decoder and denoiser tasks.
2. **MLX execution and memory.** Profile the complete decoder and denoiser paths,
   then improve the measured bottlenecks. Compare saving, recomputing and fused
   pullbacks at the same precision, context and effective batch.
3. **Precision and quantized bases.** Keep FP32 gradients and external optimizer
   state while extending qualified BF16/FP16 projections and packed frozen weights.
   [BF16 family training](decoder-bf16-projections) now covers NF4, PEFT exchange
   and fresh-process recovery, including a pinned Qwen3.5-0.8B run. Next trace
   whole-model BF16 reference differences at shared activation points, qualify
   FP16 adapter exchange and measure long-context memory and execution costs.
4. **Adapter curvature.** Explore matrix-free HVP diagnostics for LoRA, connecting
   the existing higher-order transforms to practical fine-tuning questions.

## Working baselines

| Area | Entry point | Established workflow |
| --- | --- | --- |
| LoRA | [Tutorial](lora.md) | Frozen base, selected adapter gradients, external Array API Adam |
| Small models | [MLP](mlp.md) | Stable loss, independent updates and reloaded execution |
| Scientific computing | [Fitting](fitting.md) | Matrix-free JVP/VJP products and an external solver |
| Decoder training | [Model families](decoder-families.md) | Mobius graphs, TRL, PEFT and checkpoint recovery |
| Diffusion | [Image training](diffusion-images.md) | Real image-caption data, external scheduling and adapter updates |
| Graph memory | [Saved values](vjp-memory.md) | Inspectable replay boundaries and compact runtime shape vectors |
| Fused execution | [Attention pullbacks](attention-vjp.md) | Portable ONNX functions with native MLX implementations |

## Delivery requirements

Each workload change should include a runnable command, an independent reference,
parameter and gradient mappings, frozen-weight checks, exported-model validation
and recovery where training state is involved. Performance reports record the
hardware, precision, effective batch, execution boundary, warmup, median/P95 and
process memory.

Keep the functional API and global decorator registration. Later registrations
override earlier rules. Export ordinary ONNX and use ONNXScript for graph
optimization. New operators and rewrites should address a workload requirement
or a measured bottleneck.

For completed investigations and their measurements, see
[experiments and reports](experiments.md). The [development history](development-history.md)
preserves the earlier milestone log and the [ORT Training handoff](ort-training-handoff.md)
records its design input.
