# LoRA adapters, chunked loss and saved activations

The [Mobius Qwen3 workload](qwen-sft.md) has two complementary memory controls:
bound the output projection by token chunks, or save the primal values needed
by the exported backward graph.

```bash
uv run python -m examples.huggingface.sft --mode chunked --save-activations --peft-roundtrip --fp64-audit --output-dir qwen_chunked
uv run python -m examples.huggingface.trainer --nf4 --output-dir nf4_trainer
uv run python -m examples.huggingface.benchmark --output qwen-benchmark.json
```

## Chunked output projection and loss

The transformer exports hidden states H. A separate ONNX module computes
`H_chunk @ W_head`, stable cross-entropy and the hidden-state gradient.
The wrapper processes 128 token rows at a time and feeds the assembled hidden
cotangent into the transformer's VJP. The output head remains frozen.

For N valid shifted target tokens, each chunk contributes its loss sum divided
by N. Padding labels are -100; an all-ignored batch produces zero loss and
zero gradients. Tests cover chunks of 1, 7 and 128 rows, a partial final chunk,
all adapter gradients and prediction through the full output head.

Training returns the scalar loss. Calling the wrapper without labels requests
full logits. A small SFTTrainer subclass uses Trainer's loss handling directly
for the chunked path, avoiding a full-logits pass for TRL token metrics.

## Save or recompute

`prepare(save_activations=True)` cuts the flat VJP at the exact primal/derivative
boundary. Forward exports the required intermediate values; backward accepts
them as explicit ONNX inputs. Each autograd invocation owns its saved values.
Values sharing the returned prediction's storage are snapshotted, so caller
mutations cannot corrupt backward. Parameters retain PyTorch version checks.

The default recomputes those values inside the VJP. Both paths preserve the
low-rank matrix chain and share common products through CSE. In particular,
`U = G @ B.T` feeds both A's gradient and the low-rank contribution to X's
gradient. No dense A@B matrix is constructed. The runtime's optimized backward
exports only gradients, removing primal outputs unused by that execution.
Three independent updates from saved and recomputed graphs must be identical.

For each projection with T token rows, input width d, output width h and rank r,
saving `X @ A` needs `4*T*r` logical bytes in FP32 and avoids its `2*T*d*r`
recomputation FLOPs. The full cut also saves other backward dependencies;
the benchmark reports their logical extents and the process's measured RSS.

## PEFT files

The SFT command writes `adapter/adapter_model.safetensors`, `adapter_config.json`
and a versioned `adapter_mapping.json`. Each entry maps a source Qwen module to
its explicit ONNX A/B inputs. ONNX A `[in,rank]` and B `[rank,out]` are transposed
to PEFT Linear weight layouts. Rank, alpha and scaling remain unchanged.

`export_adapter(model, directory)` and `import_adapter(model, directory)` in
`examples.huggingface.peft_adapter` exchange standard q/v LoRA adapters. Import
validates the complete target/name/shape/dtype mapping before changing any
parameter. The example uses one adapter, uniform rank/alpha and zero dropout.

## Measurements

The benchmark starts a separate process for PyTorch, raw ONNX, optimized ONNX,
saved activations and chunked loss. Defaults are batch four, length 64, FP32,
one CPU thread, 10 warmups and 100 measured AdamW updates. It reports valid
tokens/s, forward/loss, backward, AdamW, complete-step median/P95 and peak RSS.
RSS includes construction, runtime and optimizer allocations. CPU input-view
conversion is recorded separately; snapshots and parameter/output views are
included in the execution boundaries. ORT graph optimization is disabled.

Use `--real --batch-size 1 --samples 5 --warmup 3` for a short 0.6B measurement.
The JSON records the actual sample count and hardware/software versions.

### Recorded tiny Qwen3 CPU comparison

Apple M1 Max, batch 4, length 64, vocabulary 256, 252 valid shifted targets per
update; one CPU thread, 10 warmups and 100 measured updates:

| Path | Step median (ms) | P95 (ms) | Peak process RSS (MiB) |
| --- | ---: | ---: | ---: |
| PyTorch | 2.698 | 3.180 | 536.9 |
| Raw ONNX | 7.995 | 8.270 | 581.4 |
| Optimized ONNX | 7.941 | 8.528 | 591.7 |
| Saved activations | 6.724 | 6.809 | 596.7 |
| Chunked loss | 7.511 | 7.836 | 592.1 |

PyTorch is fastest in this CPU configuration. Saving activations reduces the
ONNX step time and retains 8.26 MiB of logical residual arrays. The small
256-entry vocabulary makes output-head storage a small part of this workload.
The JSON in `examples/huggingface/reports/tiny-qwen-benchmark.json` includes
node counts, parameter bytes, boundary timing, sample counts and versions.
