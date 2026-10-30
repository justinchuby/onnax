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

## Scale the activation workload

The `activation` profile keeps 296,976 trainable parameters and increases the
batch to 16,384 samples with 2,048 hidden units. A single FP32 hidden tensor then
occupies 128 MiB. Train it with four 4,096-sample microbatches:

```bash
python -m examples.mlp_microbatch --profile activation --microbatch 4096 \
    --output-dir activation_models
```

The complete 150-update run checks all four gradients against the full native
batch and an independent NumPy backward calculation at each parameter state.
It also checks Adam moments and master parameters, then reloads the exported
model and checkpoint to reproduce update 151. Training loss decreases from
2.770 to 0.681; native validation loss decreases from 2.769 to 0.682.
Same-parameter checks and actual-gradient Adam checks pass. Independent parameter
trajectory differences remain in the report with their first failing coordinates.

### Run a batch-size sweep

Run each batch size in a fresh process to compare memory and synchronized step
time. `--benchmark-only` performs 10 warmup and 100 measured updates:

```bash
for batch in 16384 4096 2048 1000; do
    python -m examples.mlp_microbatch --profile activation --microbatch "$batch" \
        --benchmark-only --output-dir "activation_bench_$batch" > "batch_$batch.json"
done
```

Alongside peak RSS, macOS runs record current and lifetime-peak **physical
footprint**, read with `proc_pid_rusage(RUSAGE_INFO_V4)`. This is the kernel's
whole-process memory accounting. Both counters include setup, graphs, data and
runtime allocations. They are sampled after synchronized training, outside the
timed steps. The JSON report also records the analytical size of one FP32 hidden
tensor, so tensor size, process footprint and RSS remain distinct quantities.

| Microbatch | Step median / P95 (ms) | Peak physical footprint (MiB) | Peak RSS (MiB) |
|---:|---:|---:|---:|
| 16,384 (full) | 7.706 / 7.939 | 570.64 | 213.36 |
| 4,096 | 11.439 / 11.798 | 274.22 | 213.36 |
| 2,048 | 14.282 / 14.803 | 227.74 | 214.78 |
| 1,000, plus 384-sample tail | 22.881 / 23.233 | 209.95 | 214.61 |

On this M1 Max, four 4,096-sample microbatches reduce peak physical footprint by
about 52% while increasing step time by about 48%. Smaller microbatches reduce
footprint further and add runtime calls. RSS alone remains nearly constant
across this sweep, while physical footprint reveals the batch-size tradeoff.

The [activation-scale record](https://github.com/justinchuby/onnax/blob/main/examples/measurements/activation-memory-apple-arm64.json)
contains complete training validation, both memory counters, all phase timings,
model sizes and software versions. For this model, microbatch size provides a
measured memory control before adding graph recomputation.
