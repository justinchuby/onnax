# Workload-driven development roadmap

onnax generates, composes and optimizes ONNX differentiation graphs for fine-tuning,
small-model training and scientific computing. External programs own optimization,
scheduling, data loading and checkpoint storage.

## P0 — Frozen-base LoRA fine-tuning

The [LoRA tutorial](lora.md) is the first acceptance workload: load a frozen ONNX
base, inject adapters, export selected parameter gradients, and train using an
external Array API Adam update. Independent reference trajectories, reloaded
artifacts, frozen-weight checks and bounded performance measurements are required.

## P1 — Small-model training

Build a complete MLP classification example using the same explicit parameter
and named-gradient boundary. Prioritize stable classification losses, batch
reductions and operators demonstrated necessary by that model. Export loss and
gradient modules; keep the training loop external. Operator count is not a goal.

## P2 — Matrix-free scientific computing

Extend the existing [Jelly Newton-CG example](https://www.justinchuby.com/onnax/jelly/)
with a parameter-fitting or inverse-problem workload. Compose JVP/VJP for
Gauss–Newton products and HVPs for Newton-CG without materializing dense
Jacobians or Hessians. Check solver progress against an independent reference.

## P3 — Memory, precision and device execution

Define BF16 compute, FP32 gradient accumulation and external optimizer-state
policies before enabling mixed precision. Measure memory and runtime behavior
before choosing device-resident bindings, DLPack exchange, recomputation,
batched directions or structured operators. Array API update portability alone
does not establish zero-copy ONNX execution.

## Engineering boundaries

- Preserve model-first transforms and global decorator registration; later
  registrations override overlapping versions. No mandatory per-call rules argument.
- Export ordinary ONNX, document backend requirements, and optimize copies with
  ONNXScript. ORT graph optimization is disabled in numerical acceptance tests.
- Preserve parameter names and record actual collision-resolved gradient names.
- Require workload evidence for new operators or local graph rewrites.
- Separate measured execution, exchange and update costs; never infer acceleration
  from executability or a smaller graph.
- FP8, QLoRA and distributed execution are outside the first milestone.
