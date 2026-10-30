# Accumulate classifier gradients on Metal

Train a classifier with a large effective batch while executing smaller ONNX
loss/gradient graphs. FP32 gradient sums, master parameters and Adam moments stay
in MLX Metal arrays. The external Python loop owns accumulation and updates.

```bash
uv sync --group dev --group training --group native-mlx
python -m examples.mlp_microbatch --profile medium --microbatch 300 \
    --output-dir microbatch_models
```

The medium classifier has 74,256 parameters and 1,024 training samples. Its batches
are `300 + 300 + 300 + 124`. The large profile has 1,116,192 parameters; use
`--profile large --microbatch 600` for batches `600 + 600 + 600 + 248`.
`--precision bf16` selects BF16 matrix computation with FP32 gradients and state.

## Weight the sample means

Each exported graph returns a sample-mean loss $L_j$ and gradients $g_j$ for a
microbatch containing $n_j$ samples. Keep parameters fixed throughout the batch,
then perform one Adam update using

$$
L=\frac{\sum_j n_jL_j}{\sum_j n_j},\qquad
 g=\frac{\sum_j n_jg_j}{\sum_j n_j}.
$$

Weighting by sample count gives the shorter final batch its correct contribution.
This classifier uses per-sample tanh activations and cross-entropy, so each
sample's forward calculation is independent of other samples.

The shared [Array API accumulator](https://github.com/justinchuby/onnax/blob/main/examples/lora/accumulation.py)
accepts a generator of `(sample_count, mean_loss, gradients)` tuples. Supply
`evaluate=mx.eval` to materialize running sums after each microbatch. This releases
previous accumulation expressions before requesting the next gradient graph.
The final normalization also executes on Metal.

```{literalinclude} ../examples/mlp_microbatch.py
:language: python
:pyobject: Workload.gradients
```

The example exports one ordinary ONNX graph for each distinct batch size and
reuses its runtime session. A versioned `interface.json` maps each of the four
parameter inputs to its gradient output. `MlxBinding` checks shared input/output
storage at the Python/runtime boundary. The native provider handles graph
execution and output exchange; ORT optimization and CPU fallback are disabled.

## Check training and recovery

The 150-step acceptance loop compares accumulated loss and all four gradients
with a full-batch native graph at the same parameters, and independently with
NumPy analytic derivatives. It also follows an independent NumPy Adam trajectory
and checks the native update and both moments using the actual accumulated
gradients. Every step verifies FP32 Metal storage and unchanged input parameters.

The checkpoint contains parameters, both Adam moments and the step counter.
Recovery loads the exported graphs and checkpoint into fresh runtime sessions,
then requires byte-identical loss, gradients, parameters and moments at update
151. A separately exported validation graph executes the final parameters.

The report records each comparison's first failure with the original
`rtol=1e-4, atol=1e-6` tolerance. The [forward-value diagnostic](classifier-device.md)
helps investigate BF16 output-layer differences.

## Measure the execution boundary

Each configuration measures full and microbatch execution in separate processes:
10 warmup steps followed by 100 timed updates. Timings split binding/execution/
output import, synchronized accumulation, synchronized Adam and the complete
training step. They exclude analytic references and validation. Data creation and
session setup occur before timing; session creation and the first step are
reported separately.

Peak RSS includes Python, graph construction, runtime sessions and process memory.
Input batches are prepared on Metal before training. Both measurements use one
intra-op and one inter-op runtime thread. The report includes model sizes, node
counts, parameter bytes and package versions.

### Recorded M1 Max results

All four runs complete 150 updates and restore update 151 exactly. Training loss
falls from about 2.774 to 0.679 for the medium profile and 3.472 to 0.785 for the
large profile. Final native validation loss is about 0.683 and 0.786 respectively.

Both FP32 profiles pass all same-parameter loss/gradient comparisons against the
full graph and independent NumPy reference. BF16 output-layer comparisons and
independent parameter trajectories retain the differences listed in `report.json`.
The actual-gradient Adam checks pass for parameters and both moments in all runs.

| Profile | Compute | Batch path | Step median / P95 (ms) | Peak RSS (MiB) |
|---|---|---|---:|---:|
| Medium | FP32 | Full | 0.977 / 1.048 | 162.88 |
| Medium | FP32 | 300 + 300 + 300 + 124 | 4.035 / 4.549 | 165.81 |
| Medium | BF16 | Full | 1.011 / 1.164 | 163.98 |
| Medium | BF16 | 300 + 300 + 300 + 124 | 4.088 / 4.311 | 167.03 |
| Large | FP32 | Full | 2.933 / 3.139 | 199.22 |
| Large | FP32 | 600 + 600 + 600 + 248 | 8.459 / 8.848 | 204.88 |
| Large | BF16 | Full | 2.940 / 3.200 | 200.45 |
| Large | BF16 | 600 + 600 + 600 + 248 | 8.593 / 8.748 | 205.39 |

At these sizes, full-batch execution uses less time and lower peak RSS. The
microbatch path adds runtime calls, gradient accumulation and a second session
for the shorter final batch. The next memory experiment increases the activation
workload to locate the useful batch-size tradeoff.

`accumulate_sync` measures evaluation of running sums and final normalization.
Python expression construction is included in the complete step. The
[measurement record](https://github.com/justinchuby/onnax/blob/main/examples/measurements/metal-microbatch-apple-arm64.json)
contains phase timings, cold-step costs, numerical failures and software versions.
