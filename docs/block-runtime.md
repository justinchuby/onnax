# Measuring exported decoder blocks

Capture a block's inputs during a complete task backward, then replay its exported
recomputation/VJP module in a fresh process. This connects the
[saved-value analysis](vjp-memory.md) with gradients and native allocator peaks
from the graph used by training.

## Capture and replay

Start with a trained decoder export from the [family tutorial](decoder-families.md).
Select named blocks from its execution interface:

```bash
uv run --group huggingface --group training python -m examples.huggingface.block_runtime capture \
  --export qwen_run/export --block decoder_0_1 --block decoder_3_4 \
  --sequence 512 --output captures --provider mlx --library "$MLX_EP_LIBRARY"

uv run --group huggingface --group training python -m examples.huggingface.block_runtime measure \
  --bundle qwen_run/export/execution --captures captures --block decoder_0_1 \
  --provider mlx --library "$MLX_EP_LIBRARY" --warmup 10 --samples 100 \
  --output linear-runtime.json

uv run --group huggingface --group training python -m examples.huggingface.block_liveness \
  --bundle qwen_run/export/execution --captures captures --block decoder_0_1 \
  --output linear-logical.json
```

Capture runs one complete task forward/loss/backward with fixed parameters. Each
selected block records the actual inputs and gradient outputs. Replay verifies
bundle files and captured array hashes before constructing a session, then checks
first and final outputs at `rtol=1e-4, atol=1e-6`. Input hashes must remain unchanged.
Use a new capture directory and a separate process for each measured case.

MLX runs disable ORT graph optimization, CPU fallback and MLX compile, with a
512 MiB cache limit. Add `--unfused` to disable only the gated-delta reverse kernel.
CPU replay uses `--provider cpu` without a library argument.

## What the reports measure

Execution time includes the session's host/device exchange. Reports separate
session creation, first execution and warmed median/P95. Each session reuses fixed
captured parameters; other blocks, task loss and optimizer updates run during
capture, outside isolated replay timing.

After warmup, the worker resets the MLX allocator peak and records active and
cached bytes separately. Reset can report zero peak while weights remain live,
so the measurement-window peak includes the starting active allocation. The
incremental peak subtracts that baseline. Process RSS and physical footprint
include worker imports, verification, CPU arrays and runtime storage.

Logical analysis binds captured shapes to a copy of the executed module, expands
local fallbacks and optimizes it with ONNXScript. It includes nested Scan bodies
and preallocated Scan outputs. Its per-value storage model helps identify tensor
contributors; native counters also include backend workspaces and allocations.
The saved-activation pullbacks in the earlier study have their own graphs and
memory totals. Here, logical analysis follows the exported recomputation module.

## Qwen3.5-0.8B on M1 Max

The pinned revision is `2fc06364715b967f1860aea9cf38778875588b17`.
These FP32 measurements use batch one, trained rank-four adapters and identical
captured inputs across each pair. Each case has ten warmups and 100 timed calls.

| Tokens / block | Generic reverse median / P95 | Fused reverse median / P95 | Active peak, generic / fused |
|---|---:|---:|---:|
| 64 / linear attention | 37.60 / 38.90 ms | 27.63 / 28.37 ms | 387.94 / 348.95 MiB |
| 512 / linear attention | 194.92 / 200.36 ms | 110.77 / 113.22 ms | 1183.10 / 958.36 MiB |
| 64 / full attention | 16.62 / 17.09 ms | 16.74 / 17.33 ms | 174.63 / 174.63 MiB |
| 512 / full attention | 24.33 / 24.70 ms | 25.81 / 26.17 ms | 522.18 / 522.18 MiB |

At 512 tokens, gated-delta reverse fusion reduces linear-block median time by
43.2% and active peak by 19.0%. Full-attention rows serve as controls: the switch
changes the recurrent kernel, leaving their allocator peaks unchanged. All first
and final gradients meet the original tolerance. Fused outputs match capture
byte for byte.

The optimized logical recomputation totals at 512 tokens are 200.50 MiB for the
linear block and 214.26 MiB for full attention. This gap to native peaks directs
further profiling toward runtime storage and workspaces.

The {download}`measurement index
<../examples/huggingface/reports/qwen35-block-runtime-m1max/summary.json>` records
hardware, guard completion, paired hashes and source provenance. The
[report directory](https://github.com/justinchuby/onnax/tree/main/examples/huggingface/reports/qwen35-block-runtime-m1max)
contains all eight runtime reports and four logical reports. Each runtime report
identifies the measured worker source, ONNX module, inputs, outputs and EP library.

## Persistent weight storage across sessions

A separate full-task probe records allocator counters after all 51 session calls.
At completion, active memory is 6.68 GiB, closely matching the 6.68 GiB of declared
session initializer payload. Distinct immutable initializer regions occupy
3.81 GiB. The repeated payload is about 2.87 GiB. Closing all sessions releases
active memory to zero; the allocator retains about 512 MiB of cache.

For this workload, persistent active storage plus the isolated linear block's
incremental peak exactly reproduces the observed full-task allocator peak of
7.54 GiB. The {download}`initializer probe
<../examples/huggingface/reports/qwen35-block-runtime-m1max/full-model-initializers.json>`
includes each call, the probe source and successful process guard record.

The next runtime change is shared ownership of immutable GPU initializers across
sessions. Its tests will cover overridable inputs, different contents/shapes/dtypes,
concurrent sessions and release after the final owner closes. Measured training
and recovery will establish the resulting memory savings. Selective block
recomputation then targets the remaining transient peak.
