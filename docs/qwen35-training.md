# Qwen3.5 recurrent training design

Qwen3.5 uses three Gated DeltaNet layers followed by one full-attention layer.
The first real target is the text backbone of
[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B): 24 layers, hidden width
1024 and a tied 248,320-token embedding/head. An offline four-layer model will
exercise the same 3:1 layout with hidden width 64 and a local tokenizer.

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

The portable VJP uses 32-token chunks. It pads the last chunk with identity
state updates and removes padded gradients before returning them. The JVP
carries just the primal state and its tangent through a single `Scan`.

The reference tests compare both outputs and all six gradients with independent
PyTorch autograd at `rtol=1e-4, atol=1e-6`. Cases include grouped heads, nonzero
initial states, independent output/state cotangents, partial final chunks,
external chunk composition and the state-storage ledger. Graph tests execute
the actual Mobius function with ONNX Runtime, compare its derivatives with
PyTorch, and reuse a session across batch and sequence sizes.

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

Next, connect this task to the shared Trainer/TRL and PEFT examples, pin the
real model/tokenizer revision, and qualify 0.8B. Native MLX work will use measured
state lifetimes to select recurrent kernel and recomputation boundaries.
Portable derivative functions will use the `pkg.onnax` domain.

Trainer and TRL continue to own training loops and checkpoints. ONNXScript
optimizes the generated graph; the model adapter owns architecture conventions.
