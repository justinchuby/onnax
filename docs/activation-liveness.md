# Select a recomputation boundary from activation lifetimes

Checkpoint selection needs the peak live tensor set as well as lifetime length.
This experiment analyzes the exported [activation-scale classifier](metal-microbatch.md),
recomputes its hidden preactivation near backward use, and measures both graphs.

```bash
python -m examples.mlp_rematerialize --output-dir recompute_models
python -m examples.activation_liveness \
    recompute_models/baseline/gradient-16384.onnx --alias-views > liveness.json
```

The first command runs the complete native MLX experiment, including 150 updates,
independent gradients, checkpoint recovery and separate benchmark processes.
Use the [native MLX dependencies](metal-microbatch.md) for it. The second command
reads the exported ordinary ONNX graph with static shapes and fixed-width dtypes.

## Account for tensor lifetimes

The analyzer follows the serialized node order. It allocates each node's outputs
before releasing inputs whose final use is that node. Inputs and initializers
persist throughout the invocation; graph outputs persist until the graph ends.
External storage and the combined peak are reported separately.
The JSON contains birth/final-use indices, tensor sizes, storage groups, a
per-node timeline and the groups present at peak.

Two accounting scenarios are available:

- Default: each produced tensor has separate storage.
- `--alias-views`: equal-byte, equal-dtype Identity, Reshape, Flatten, Squeeze,
  Unsqueeze and Transpose outputs share their input's storage group. Any alias
  extends that group's lifetime through its final consumer.

These scenarios describe tensor storage in node order. The native allocator
measurement captures the runtime's execution plan. Kernel workspaces, fusion and
allocation reuse are accounted for by the runtime measurement.

The analyzer requires a flat, topologically ordered graph with known sizes.
Inline local functions first. Unknown dimensions/dtypes and nested control flow
produce errors, so every reported tensor has an explicit byte count. Packed
four-bit types use rounded-up packed storage sizes; zero-element tensors use zero
bytes. Analysis leaves the input graph unchanged.

## Recompute the hidden preactivation

For this classifier,

$$
Z=XW_1+b_1,\qquad H=\tanh(Z).
$$

The original 128 MiB $Z$ is produced at node 1 and last read at node 32. The
candidate duplicates `MatMul` and `Add` immediately before the backward consumers.
The forward copy of $Z$ then ends its lifetime at node 2; a fresh copy supplies
the later tanh derivative.

Both graphs reach the same tensor-accounting peak while computing $|Z|$ for the
stable tanh derivative. The live set contains:

| Live tensor | Size (MiB) |
|---|---:|
| Preactivation $Z$ | 128 |
| Upstream hidden cotangent | 128 |
| Sign mask | 32 |
| Negated $Z$ | 128 |
| Selected absolute value | 128 |
| Retained loss and output-layer gradients | 0.125 |

The peak activation storage is **544.125 MiB** in both view scenarios and both graphs. Recreating
$Z$ closer to its use shortens its lifetime while leaving this peak set intact.

## Validate and measure the candidate

All 150 updates produce byte-identical loss, four gradients, four parameters,
both Adam moment sets and the step counter. Independent NumPy analytic gradients
are checked at every state. The exported candidate and final checkpoint reproduce
update 151 in fresh sessions. Native validation loss reaches about 0.682.

Each benchmark runs in a fresh process with 10 warmup and 100 measured updates.
ORT optimization and CPU fallback are disabled; ONNXScript optimizes the baseline
before the candidate is created.

| Graph | Exported nodes | Activation storage peak (MiB) |
|---|---:|---:|
| Baseline | 44 | 544.125 |
| Recompute $Z$ | 46 | 544.125 |

Three pairs of independent M1 Max benchmark processes produced:

| Repetition | Baseline allocator peak (MiB) | Candidate allocator peak (MiB) | Baseline median / P95 (ms) | Candidate median / P95 (ms) |
|---|---:|---:|---:|---:|
| 1 | 408.893 | 408.893 | 7.730 / 7.838 | 7.646 / 7.808 |
| 2 | 408.893 | 817.724 | 8.340 / 9.073 | 11.579 / 235.933 |
| 3 | 408.893 | 408.893 | 8.200 / 8.639 | 8.338 / 8.821 |

The candidate's peak matches the baseline in repetitions 1 and 3. Repetition 2
records a larger allocator peak and latency spikes. All pairs remain in the
record; load-average context accompanies the third pair. This boundary leaves
the logical peak intact and provides no observed native peak reduction.

Repeat benchmarks of the exported graphs with:

```bash
for variant in baseline candidate; do
    python -m examples.mlp_rematerialize --benchmark-only \
        --output-dir "recompute_models/$variant" > "$variant-benchmark.json"
done
```

Running ONNXScript optimization on the candidate returns 44 nodes again. The
experiment keeps the 46-node export for its native comparison.

This candidate stays in the example for reproduction. The next checkpoint
workload needs a deeper activation chain whose stored tensors overlap the peak;
the lifetime report supplies the data to choose that boundary. The
[complete record](https://github.com/justinchuby/onnax/blob/main/examples/measurements/preactivation-recompute-apple-arm64.json)
includes both timelines, storage groups, all measurement phases and versions.
