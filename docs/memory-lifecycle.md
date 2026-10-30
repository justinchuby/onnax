# Release idle runtime caches

Inspect the memory held between classifier updates, release cached buffers at a
pause, and resume from the same parameters and Adam state. The example uses the
[activation-scale classifier](metal-microbatch.md), with 16,384 samples and 2,048
hidden units.

```bash
uv sync --group dev --group training --group native-mlx
python -m examples.mlx_memory_lifecycle --output-dir memory_lifecycle
```

Use `--microbatch 4096` for four microbatches, or `--precision bf16` for BF16 matrix
computation with FP32 gradients and master state. Each CLI invocation runs in its
own process.

## Observe both allocators

The recorded native provider bundles its own MLX library. Python MLX's memory
counters observe the library used by external arrays and Adam updates. The
provider counters observe the library used for ONNX graph execution.

`ProviderMemory` reads the provider's bundled `libmlxc.dylib` through the public
[MLX C memory interface](https://github.com/ml-explore/mlx-c/blob/main/mlx/c/memory.h).
It reports active bytes, cached bytes and the allocator's peak counter. Whole-process
[physical footprint](metal-microbatch.md) supplies a third view of memory.

The diagnostic performs 10 warmup updates and 100 timed updates, then captures a
reference for update 111. At that fixed input checkpoint it:

1. Samples both allocator counters and the process footprint.
2. Clears Python MLX's cache and samples immediately and after 250 ms.
3. Clears the provider's cache and repeats those samples.
4. Executes update 111 again and compares every output and state array byte for byte.

The fixed wait samples delayed kernel accounting outside the timed steps.
`--settle-ms` sets its duration. The report retains both immediate and delayed
samples, the steady-step median/P95 and the single first-resumed-step time.

## Pause and resume

At an idle boundary, finish runtime submissions and synchronize external arrays
before releasing buffers:

```python
import mlx.core as mx
from examples.mlx_binding import synchronize
from examples.mlx_provider_memory import ProviderMemory

provider_memory = ProviderMemory()  # After creating the native runtime session.
synchronize(params, state)
mx.synchronize()
provider_memory.clear_cache()
```

The cache belongs to the provider allocator shared by sessions in the process;
coordinate their submissions at the same idle boundary. Parameters and optimizer
state stay live. The next execution allocates buffers again and repopulates the
cache. The normal training loop retains its cache for reuse.

The reference comparison checks 17 finite arrays: loss, four gradients, four
updated parameters and both sets of four Adam moments. It also checks the step
counter and verifies that the input checkpoint remains unchanged. Equality
includes dtype, shape and bytes, so signed-zero differences are detected.

## Recorded M1 Max results

In all four runs, clearing Python MLX's cache leaves the provider cache counter
unchanged. Clearing the provider cache reduces that counter to zero. The table
uses physical-footprint samples after each 250 ms settling interval.

| Compute | Microbatch | Provider cache before release (MiB) | Current footprint before → after (MiB) | Steady median / first resume (ms) |
|---|---:|---:|---:|---:|
| FP32 | 16,384 | 398.70 | 572.27 → 173.58 | 7.65 / 37.86 |
| FP32 | 4,096 | 101.70 | 274.34 → 172.66 | 11.84 / 20.26 |
| BF16 | 16,384 | 539.89 | 715.64 → 175.77 | 8.51 / 39.20 |
| BF16 | 4,096 | 137.51 | 310.84 → 173.34 | 12.31 / 29.69 |

The steady median covers 100 updates; the resume time is one first step after
release. All four runs preserve the input checkpoint and reproduce update 111's
loss, gradients, parameters, moments and step counter byte for byte.

Idle release reduces current memory residency. The lifetime peak still records
the largest earlier allocation, and resumed execution repopulates the cache.
Microbatch size controls the training working set; cache release controls retained
buffers during a pause. Recomputing activations addresses the working set within
an update, which is the next graph-memory analysis target.

The [complete lifecycle record](https://github.com/justinchuby/onnax/blob/main/examples/measurements/allocator-lifecycle-apple-arm64.json)
includes immediate and delayed samples, provider/Python counters, lifetime peaks,
model sizes, thread settings and package versions. These diagnostic processes
also retain CPU copies used for byte comparisons. The
[batch-size benchmark](metal-microbatch.md) measures uninterrupted training in
separate processes.
