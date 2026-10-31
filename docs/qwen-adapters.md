# LoRA adapters, chunked loss and saved activations

The [Mobius Qwen3 workload](qwen-sft.md) offers three execution paths: bound the output
projection by token chunks, save primal values for a separate backward graph,
or execute loss and adapter gradients together.

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

## Joint loss and adapter gradients

```bash
uv run python -m examples.huggingface.sft --mode joint --nf4 --steps 100 --peft-roundtrip --output-dir qwen_joint
uv run python -m examples.huggingface.benchmark --mode joint --nf4 --warmup 10 --samples 100
```

The joint path adds causal cross-entropy to the forward graph, then asks
`onnax.grad` for every adapter parameter's gradient. ONNXScript optimizes the
combined graph. One persistent ONNX session returns the summed token loss
and all A/B gradients; intermediate activations stay inside that execution.

For valid shifted targets $\mathcal V$, the exported scalar is

$$
L_{\mathrm{sum}}=\sum_{t\in\mathcal V}-\log p(y_t\mid x_{\le t}).
$$

The PyTorch wrapper divides this sum by the effective token count. Backward
multiplies the returned gradients by the upstream scalar, so Trainer can
normalize accumulated microbatches with different lengths. Ignored labels
(`-100`) contribute zero. A sequence with no shifted targets also returns
zero loss and gradients.

Trainer owns the adapter tensors, AdamW state, scheduler and checkpoints.
Calling the wrapper without labels creates a temporary forward session for
full-logit prediction. Training continues to reuse the joint session.

The export directory contains `loss_gradient.onnx` and its external data,
`adapter.npz`, and `interface.json`. The versioned interface maps each parameter
to its gradient output, shape and dtype and records the frozen-weight hashes.
A bound `forward.onnx` restores the trained adapter values for inference.
Checkpoint loading checks the frozen base identity before restoring parameters.

Use the numerical audit to compare every update with the corresponding
Transformers model. It also evaluates the reference at the ONNX parameters,
separating local gradient differences from accumulated optimizer drift:

```bash
uv run python -m examples.huggingface.mlx_validation --mode joint --provider cpu --stable-reference --steps 100 --output joint-cpu-audit.json
uv run python -m examples.huggingface.mlx_validation --mode joint --library /path/to/libonnxruntime_mlx_ep.dylib --stable-reference --steps 100 --output joint-mlx-audit.json
```

Add `--real` for the pinned Qwen3-0.6B model. `--segmented --segment-nodes 1024`
bounds graph size per runtime session. Both paths use the same parameter
mapping, FP32 arithmetic and `rtol=1e-4, atol=1e-6` checks. The CLI saves all
per-step failures and exits nonzero when a check fails.

The 100-step tiny NF4 audit at the original tolerance gives:

| Check | ORT CPU | Native MLX |
| --- | --- | --- |
| Same-parameter logits, loss and all gradients | All 100 pass | All 100 pass |
| Independently updated parameters | All 100 pass | Steps 1–94 pass; 95–100 fail |
| Independent trajectory loss | All 100 pass | All 100 pass |
| First independent gradient failure | Step 87 | Step 83 |
| First independent logit failure | Step 93 | Step 92 |

Frozen buffers remain unchanged. These results separate agreement at identical
parameters from differences accumulated along two training trajectories. The
[CPU report](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/tiny-nf4-joint-cpu-100.json)
and [MLX report](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/tiny-nf4-joint-mlx-100.json)
retain every step and failed tensor. The MLX parameter divergence remains part
of the FP32 qualification work.

The benchmark reports joint loss/gradient execution, PyTorch gradient scaling,
AdamW and complete-step median/P95, plus process peak RSS and logical session
input/output bytes. It constructs the selected provider's sessions directly.
Run each path in its own process with the same provider, weights, batch and
thread settings to compare time and memory.

A native MLX run on Apple M1 Max used tiny NF4 Qwen3, batch four, sequence
length 64 and 252 valid targets per update. After 10 warmups, 100 measured
updates took 87.14 ms median and 95.62 ms P95; process peak RSS was 633.5 MiB.
The loss decreased from 5.55 to 5.09 over all 110 updates. The session accepted
15,360 logical input bytes and returned 7,172 bytes, including 7,168 bytes of
adapter gradients. The [measurement JSON](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/tiny-qwen-joint-native-m1max.json)
records the source commit, native library digest, versions and timing scope.

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
saved activations, chunked loss and joint loss/gradients. Defaults are batch four, length 64, FP32,
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
