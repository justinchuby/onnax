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
