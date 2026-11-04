# Symbolic broadcast adjoints

The Qwen3.5-0.8B training graph uses NF4 frozen weights, BF16 projection computation
and FP32 adapter gradients. Matching named dimensions let onnax determine
broadcast-gradient reduction axes during graph generation.

| Backward block | Baseline nodes | Simplified nodes |
| --- | ---: | ---: |
| Full attention, layer 3 | 567 | 367 |
| Linear attention, layer 8 | 633 | 389 |

The representative blocks retain their input/output names and frozen-weight
bytes. The linear-attention block keeps the MLX EP state, replay and reverse
Scan kernels at 64 and 4032 tokens.

The whole-model comparison replaces all 25 backward blocks, reuses the saved
forward graph and frozen output head, and runs one warmup and one checked step
at each sequence length. All 60 adapter gradients are byte equal; token losses
match at `rtol=1e-4, atol=1e-6`. ORT graph optimization is disabled.

`correctness.json` records input and adapter digests, per-gradient digests and
errors, graph operation counts, frozen-weight digests and the source revision
of the shape-inference fix used for these runs.

## 64-token native execution

The paired full-model run uses the same saved input, adapter, frozen forward
and head graphs, and MLX EP library. Each independent process runs ten warmups
and 100 measured forward/loss/gradient steps with fixed parameters. The candidate
loads the 25 simplified backward blocks through the recorded session overrides.

| ONNX backward graph | Step median | Step P95 | Prediction targets/s |
| --- | ---: | ---: | ---: |
| Original | 0.3821 s | 0.5021 s | 164.9 |
| Symbolic adjoints | 0.3263 s | 0.3684 s | 193.1 |

This pair improves throughput by 17.1%. All 100 token losses agree at the
recorded tolerance, and all 60 final adapter gradients are byte equal. Peak physical footprint for the process, including setup and runtime, is
4,883,353,440 bytes for the original and 4,871,262,072 bytes for the candidate.
The workload has 63 shifted prediction targets per step.

`sequence64/` contains raw samples, environment/version metadata, per-gradient
checks, session overrides, process guard observations and artifact digests.
Both process guards completed successfully with zero observation errors and
maximum observation gaps below 1.02 seconds. A `caffeinate -i` assertion covers
the measurement queue.

## 4032-token native execution

The longer sequence uses the same protocol: independent original/candidate
processes, ten warmups and 100 measured steps, fixed inputs and adapters.

| ONNX backward graph | Step median | Step P95 | Prediction targets/s |
| --- | ---: | ---: | ---: |
| Original | 10.6717 s | 10.8508 s | 377.7 |
| Symbolic adjoints | 10.5258 s | 10.8208 s | 383.0 |

Throughput changes by 1.4% in this pair. The whole-process lifetime peak physical
footprint rises from 8,336,707,736 to 8,569,835,696 bytes, including setup and
runtime. All 100 losses and all 60 final adapter gradients are byte equal.
The 4032-token input has 4031 shifted prediction targets.

Body backward takes a median 7.1560 seconds of the candidate's 10.5258-second
step. This is the next path to profile for larger long-sequence gains.

`sequence4032/` retains the complete raw samples and process guards. Maximum
observation gaps are below 1.03 seconds, with zero observation errors. Both
source checkouts remain clean at the recorded commits throughout the pair.
`measurement-sources.json` and the copied driver sources identify the exact
implementation used for the run.

## Graph and weight audit

Across all 25 backward blocks, nodes fall from 14,489 to 8,951. Every input and
output name remains identical. `all-blocks.json` records each graph's digest and
node counts; `frozen-weights.json` compares the shape, dtype and payload digest
of every `model.*` frozen initializer in the saved backward graphs.

The rebased implementation also passes 39 FP32 layout and mixed-product tests
on native MLX EP with CPU fallback disabled. The copied test plugin, JUnit
results and library digest record that backend qualification.
