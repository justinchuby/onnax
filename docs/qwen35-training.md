# Fine-tune Qwen3.5 recurrent graphs

Qwen3.5 uses three Gated DeltaNet layers followed by one full-attention layer.
The first real target is the text backbone of
[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B): 24 layers, hidden width
1024 and a tied 248,320-token embedding/head. An offline four-layer model
exercises the same 3:1 layout with hidden width 64 and a local tokenizer.

Mobius already expresses the recurrent blocks as `com.microsoft::LinearAttention`
and `CausalConvWithState`, with portable function bodies. The
[linear-attention function](https://github.com/onnxruntime/mobius/blob/304622181f6418520b332de4f5de0694f2ab576f/src/mobius/functions/linear_attention.py)
uses `Scan` for the sequence recurrence. onnax provides direct JVP and VJP rules
for both operators, preserving these function calls in the forward graph.

## State update and pullback

For one batch element and value head, let $S_t$ have shape $[d_k,d_v]$.
Queries and keys arrive from the model's per-head normalization. With per-key
log-decay $g_t\in\mathbb R^{d_k}$, update rate $\beta_t$ and query scale $s$,
write $D_t=\operatorname{diag}(e^{g_t})$. The update is

\[
A_t=D_tS_{t-1},\quad
r_t=v_t-A_t^T k_t,\quad
S_t=A_t+\beta_t k_t r_t^T,\quad
y_t=s S_t^T q_t.
\]

`linear_attention_reference.py` implements this recurrence in FP32 NumPy.
Its pullback takes cotangents for both the sequence output and final state,
and returns gradients for all six inputs: query, key, value, initial state,
log-decay and update rate. Mobius accepts decay as `[batch,time,value_heads*d_k]`
or `[batch,time,value_heads]`, broadcasting the latter across key channels.
Qwen3.5 uses the per-head scalar form. Beta has shape `[batch,time,value_heads]`.
The generated derivatives and reference support both decay layouts.

Write $G_t$ for the sequence-output cotangent and $H_t$ for the state cotangent
arriving from later tokens. The reverse update is

\[
\begin{aligned}
M_t &= H_t+s q_tG_t^T, & \bar q_t &= s S_tG_t,\\
\bar r_t &= \beta_tM_t^Tk_t, & \bar v_t &= \bar r_t,\\
\bar\beta_t &= k_t^TM_tr_t, & \bar k_t &= \beta_tM_tr_t-A_t\bar r_t,\\
\bar A_t &= M_t-k_t\bar r_t^T, & \bar g_{t,i} &= \sum_j\bar A_{t,ij}A_{t,ij},\\
H_{t-1} &= D_t\bar A_t.
\end{aligned}
\]

When value heads outnumber query/key heads, forward execution repeats each
query/key head into its value-head group. The pullback sums those groups into
the original query/key layout. A chunk's initial-state gradient becomes the
preceding chunk's final-state cotangent. Frozen projection weights still pass
this gradient to earlier trainable adapters.

(adaptive-recurrent-checkpoints)=

## Recompute a chunk from its boundary

Saving every recurrent state costs $O(Td_kd_v)$ elements per head. The oracle
keeps a state at each chunk boundary, then reconstructs one chunk's states during
reverse execution. For chunk length $C$, retained state storage is

\[
\left(\lceil T/C\rceil+\min(C,T)\right)\operatorname{bytes}(S).
\]

For a single Qwen3.5-0.8B linear-attention layer, batch one and FP32 state,
$S$ occupies 1 MiB. At 8192 tokens, a complete state history contains 8193 MiB;
128-token chunks retain 192 MiB of boundary and reconstructed states. These
are state-array budgets; process measurements will include activations,
gradients, workspaces and the runtime.

The portable VJP chooses a chunk length from the sequence shape. It compares
32-token multiples on either side of $\sqrt{T}$ using the retained-state count
$\lceil T/C\rceil+C$, choosing the smaller chunk on ties. Short sequences retain
32-token chunks. A 4,032-token sequence uses 64-token chunks (127 states versus
158 with fixed 32); 8,192 tokens use 96-token chunks (182 versus 288).

The shape calculation uses ordinary ONNX operations and adapts when a session
receives a different sequence length. The last chunk is padded with identity
state updates, and padded gradients are removed before returning them. The JVP
carries just the primal state and its tangent through a single `Scan`.

The reference tests compare both outputs and all six gradients with independent
PyTorch autograd at `rtol=1e-4, atol=1e-6`. Cases include grouped heads, nonzero
initial states, independent output/state cotangents, partial final chunks,
external chunk composition and the state-storage ledger. Graph tests execute
the actual Mobius function with ONNX Runtime, compare its derivatives with
PyTorch, and reuse a session across batch and sequence sizes.

### Measured checkpoint choices on M1 Max

A real Qwen3.5-0.8B recurrent block uses 64-token chunks at 4,032 tokens and
96-token chunks at 8,192. Both policies execute the fused state/replay/reverse
kernels. Each case ran in a fresh process with 10 warmups and 100 samples:

| Sequence | Chunk | Block median / P95 | MLX allocator window peak | Process peak physical footprint |
| --- | ---: | ---: | ---: | ---: |
| 4,032 | 32 | 284.54 / 293.73 ms | 3.44 GiB | 4.00 GiB |
| 4,032 | 64 | 264.02 / 269.06 ms | 3.17 GiB | 3.74 GiB |
| 8,192 | 32 | 584.20 / 597.22 ms | 5.62 GiB | 6.26 GiB |
| 8,192 | 96 | 538.27 / 549.86 ms | 5.54 GiB | 6.15 GiB |

The 4K case replays a captured training block; the 8K case repeats its arguments
and upstream direction. Both adapter gradients agree at `rtol=1e-4, atol=1e-6`.
Runtime array exchange is included, with ORT optimization and CPU fallback
disabled, eager MLX execution, one CPU thread and a 512 MiB allocator cache.

The complete 24-layer 4K model also ran with 10 warmups and 100 samples:

| Recurrent policy | Forward/loss/all-gradient median / P95 | Process peak physical footprint |
| --- | ---: | ---: |
| Fixed 32 | 12.922 / 13.114 s | 8.07 GiB |
| Shape-selected 64 | 10.553 / 10.660 s | 8.46 GiB |

The shape-selected run is **1.22×** the fixed-32 throughput, and **1.99×** the
original generic-state-scan baseline. The input, adapter snapshot, eager 128-row
head and native library are the same; parameters stay fixed. All 60 adapter
gradients are byte-identical. This run loads 18 regenerated derivative modules
with their own immutable frozen-weight payload; process footprint includes
that setup and the runtime. The block allocator and whole-process peaks above
measure these two different scopes.

A separate four-layer 4K training check compares three AdamW updates against
fixed-32 execution. Predictions, losses, all adapter gradients, parameters and
optimizer moments are identical. Shared-state predictions/loss/gradients also
match Transformers at the original tolerance. Frozen weights remain unchanged,
exported ONNX modules pass the full checker, and loaded logits/gradients match
byte for byte. A fresh process restores checkpoint one through update three,
reproducing all 80 parameter/state arrays byte for byte. The offline CI regression
uses 2,113 tokens, including a partial 64-token tail and recovery across policies.

Generate the derivative with current onnax to include the shape-driven plan.
The exported ONNX graph retains this calculation and adapts to later sequence
lengths. A reused MLX session also matches fixed-32 gradients byte for byte
across 4,032 → 8,192 → 4,032 tokens with changing adapters and inputs.
The [raw samples, traces, training/recovery checks and drivers](https://github.com/justinchuby/onnax/tree/main/examples/huggingface/reports/adaptive-recurrent-checkpoints-m1max)
record the qualified implementation and runtime hashes.

## Implement the training graph

The offline four-layer text task and operator rules are implemented.
Call-level rules own nested functions and `Scan` bodies; registration is
mode-specific and randomness checks traverse the complete function tree. Each
training call starts with zero convolution and recurrent states and returns
full-sequence logits and hidden states. The task sets the Q/K L2-normalization
epsilon to `1e-6` to match Transformers.

Each linear layer receives one rank-four adapter on fused `in_proj_qkv`.
It updates Q/K/V together. The full-attention layer receives adapters on
`q_proj` and `v_proj`; `q_proj` retains the source's query/output-gate layout.
This gives ten trainable A/B tensors across all four layers.

Run the FP32 CPU qualification with external AdamW:

```bash
uv sync --group huggingface --group training --group dev
uv run python -m examples.huggingface.qwen35_train --output-dir /tmp/qwen35 --steps 100
```

The CLI compares logits, causal token loss and every adapter gradient against
Transformers at the same parameters (`rtol=1e-4, atol=1e-6`). A second reference
trajectory performs independent AdamW updates. Its parameters use the tensor
norm criterion `L2(error) <= 1e-6 + 1e-4 * L2(reference)`; the report also records
coordinate errors and pointwise tolerance violations. Frozen weights are
hashed, and exported modules pass the full ONNX checker and execute after
loading. The plain `forward.onnx` includes the trained adapter weights.

Resume the external optimizer in a new process:

```bash
uv run python -m examples.huggingface.qwen35_train \
  --resume /tmp/qwen35/checkpoint.pt --output-dir /tmp/qwen35-continued --steps 120
```

Tests compare continuous training with fresh-process recovery byte for byte.
Dynamic right-padding tests compare valid-token logits and adapter gradients;
padded tokens are excluded from causal loss with `-100` labels. The CLI's
fixed token patterns exercise training and validation independently of
tokenizer downloads.

On an M1 Max, the [100-step FP32 report](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/tiny-qwen35-fp32-cpu-m1max.json)
records training loss `5.61375 → 4.42861` and validation loss
`5.61941 → 4.95735`. The largest same-parameter gradient error was `1.42e-7`.
Independent AdamW trajectories ended with maximum parameter error `2.03e-5`;
their final training/validation losses agreed within `3e-6`. The full CLI took
8.98 seconds and peaked at 665 MiB RSS, including graph construction, three
tiny model copies, reference autograd, checker and export/load checks.

The same [100-step MLX run](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/tiny-qwen35-fp32-mlx-m1max.json)
produced training/validation losses `4.42860 / 4.95735`, with maximum
same-parameter gradient error `1.34e-7`. Its forward and VJP sessions require
complete MLX assignment. PyTorch holds FP32 adapter parameters and AdamW state;
the bridge exchanges NumPy arrays at the runtime boundary. Exported execution
reloaded exactly and the bound forward also executed on CPU.

```bash
ONNXRUNTIME_EP_MLX_NO_COMPILE=1 uv run --group native-mlx \
  python -m examples.huggingface.qwen35_train --provider mlx \
  --library "$MLX_EP_LIBRARY" --output-dir /tmp/qwen35-mlx --steps 100
```

`--library` selects a development build; the report records its SHA256. This
qualification took 19.90 seconds including reference checks and export/loading,
with 706 MiB peak process RSS. Steady-state GPU timing will separate those costs
from forward, backward, host exchange and external updates.

The [shared decoder adapter](decoder-families.md) now connects this task to
TRL SFTTrainer, datasets and PEFT, with decoder-block checkpointing and fresh-process
trainer recovery. Select `--family qwen35` in `examples.huggingface.decoder_sft`.
The next model milestone pins the real model/tokenizer revision and qualifies 0.8B.
Native MLX work will use measured state lifetimes to select recurrent kernel and recomputation boundaries.
Portable derivative functions will use the `pkg.onnax` domain.

Trainer and TRL continue to own training loops and checkpoints. ONNXScript
optimizes the generated graph; the model adapter owns architecture conventions.
