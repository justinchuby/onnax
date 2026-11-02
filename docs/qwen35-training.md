# Qwen3.5 recurrent training design

Qwen3.5 uses three Gated DeltaNet layers followed by one full-attention layer.
The first real target is the text backbone of
[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B): 24 layers, hidden width
1024 and a tied 248,320-token embedding/head. An offline four-layer model will
exercise the same 3:1 layout with hidden width 64 and a local tokenizer.

Mobius already expresses the recurrent blocks as `com.microsoft::LinearAttention`
and `CausalConvWithState`, with portable function bodies. The
[linear-attention function](https://github.com/onnxruntime/mobius/blob/304622181f6418520b332de4f5de0694f2ab576f/src/mobius/functions/linear_attention.py)
uses `Scan` for the sequence recurrence. These two operators are the next graph
boundaries to differentiate for the full training task.

## State update and pullback

For one batch element and value head, let $S_t$ have shape $[d_k,d_v]$.
Queries and keys arrive from the model's per-head normalization. With scalar
log-decay $g_t$, update rate $\beta_t$ and query scale $s$, the update is

\[
A_t=e^{g_t}S_{t-1},\quad
r_t=v_t-A_t^T k_t,\quad
S_t=A_t+\beta_t k_t r_t^T,\quad
y_t=s S_t^T q_t.
\]

`linear_attention_reference.py` implements this recurrence in FP32 NumPy.
Its pullback takes cotangents for both the sequence output and final state,
and returns gradients for all six inputs: query, key, value, initial state,
log-decay and update rate.

Write $G_t$ for the sequence-output cotangent and $H_t$ for the state cotangent
arriving from later tokens. The reverse update is

\[
\begin{aligned}
M_t &= H_t+s q_tG_t^T, & \bar q_t &= s S_tG_t,\\
\bar r_t &= \beta_tM_t^Tk_t, & \bar v_t &= \bar r_t,\\
\bar\beta_t &= k_t^TM_tr_t, & \bar k_t &= \beta_tM_tr_t-A_t\bar r_t,\\
\bar A_t &= M_t-k_t\bar r_t^T, & \bar g_t &= \langle\bar A_t,A_t\rangle_F,\\
H_{t-1} &= e^{g_t}\bar A_t.
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

The reference tests compare both outputs and all six gradients with independent
PyTorch autograd at `rtol=1e-4, atol=1e-6`. Cases include grouped heads, nonzero
initial states, independent output/state cotangents, partial final chunks,
external chunk composition and the state-storage ledger.

## Implement the training graph

1. Add direct pullbacks for `LinearAttention` and `CausalConvWithState`. The
   latter combines a depthwise causal convolution, SiLU and a convolution-state
   output; gradients from both outputs contribute to the concatenated history.
2. Preserve deterministic recurrent function bodies at registered operator
   boundaries. The current autodiff preflight rejects `Scan` bodies globally;
   refine that boundary check while retaining explicit validation of randomness
   and generic control flow.
3. Build the full-sequence Mobius text task with explicit masks and positions.
   Use one source adapter for each linear layer's fused `in_proj_qkv`, and q/v
   adapters for full-attention layers. The fused adapter updates Q/K/V together;
   its source names and layout must agree with the PyTorch PEFT target.
   Full-attention `q_proj` also contains the output-gate projection, so retain
   that fused source layout when mapping the adapter.
4. Qualify the four-layer task against Transformers through loss, all adapter
   gradients, external updates, held-out loss, checkpoint recovery and exported
   execution. Then pin the real model/tokenizer revision and repeat on 0.8B.
5. Retain portable derivative functions under `pkg.onnax`; use measured state
   lifetimes to select native MLX recomputation and kernel boundaries.

Trainer and TRL continue to own training loops and checkpoints. ONNXScript
optimizes the generated graph; the model adapter owns architecture conventions.
