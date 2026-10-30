# LoRA training over an NF4 base

Train FP32 LoRA adapters while storing the frozen ONNX base as packed NF4 codes.
This example extends the [LoRA workload](lora.md) with blockwise quantization and
input-gradient rules for quantized matrix multiplication.

```bash
uv sync --group training
uv run python -m examples.lora.nf4_train --output-dir nf4_lora_models
```

## Model and derivatives

The model computes

\[
Z=XQ(W)+\tfrac14XAB,\qquad \hat Y=\tanh(Z)Q(C),\qquad
L=\operatorname{mean}((\hat Y-Y)^2).
\]

Here, \(Q\) reconstructs each block of 64 weights from a 16-value NF4 codebook
and an FP32 absolute-maximum scale. Two codes occupy one byte. Both base matrices
stay frozen; Adam updates only `lora.A` and `lora.B`.

For a frozen matrix product \(U=XQ(W)\), reverse differentiation computes
\(\bar X=\bar UQ(W)^T\). This derivative lets gradients pass through the frozen
output matrix `base.C` to the adapters. Freezing weights removes their updates;
stopping gradients would also remove this path.

The example registers these JVP/VJP rules with global decorators. The resulting
ONNX models contain local functions built from standard ONNX operations. Their
reference implementation unpacks NF4, reconstructs an FP32 matrix, and executes
MatMul. Vector, matrix and batched-token activations are covered by tests.

## Execution and artifacts

CPU ONNX Runtime executes one reused gradient session with graph optimization
disabled. ONNXScript optimizes a separate copy with `inline=False`, preserving
packed initializers and function boundaries. Inlining enables constant folding
of the reconstructed matrices into persistent FP32 weights.

The external Array API Adam uses FP32 parameters, gradients and moments. Runtime
feeds and outputs are NumPy arrays. The exported directory contains:

- `base.fp32.onnx`: the original base, saved and loaded before adaptation.
- `forward.onnx`: the quantized base and explicit adapter inputs.
- `gradient.onnx` and `gradient.optimized.onnx`: loss and adapter gradients.
- `interface.json`: codebook, packing layout, parameter/gradient names, shapes,
  precision policy and frozen-buffer hashes.
- `checkpoint.npz`: adapter parameters, step and Adam moments.
- `report.json`: correctness results and isolated timing/memory measurements.

The code uses its own row-major ONNX packing layout. To differentiate a reloaded
forward model, call `examples.lora.nf4.register_rules(model)` before `onnax.grad`.
Executing an exported gradient model requires only the ONNX runtime.

## Validation and measurements

The 200-step regression compares predictions, loss, both adapter gradients and
parameter trajectories against independent NumPy arithmetic. Central finite
differences check a state with nonzero B. It verifies initially zero A gradients,
nonzero B gradients, gradient flow through C, dynamic batches, unchanged packed
weights, and exact checkpoint update replay. Training and validation loss must
each fall by at least 50%.

`report.json` records session creation, gradient execution, Adam and complete-step
median/P95 after 10 warmups and 100 measurements. Peak RSS includes the isolated
process and runtime. Frozen code/scale byte counts describe stored weights;
runtime also allocates reconstructed matrices and unpacking temporaries.

## Next integration

This reference establishes NF4 adapter training on CPU. A fused NF4 kernel is the
next step for GPU training, followed by a PyTorch autograd bridge for Trainer and
TRL. Importing bitsandbytes checkpoints also needs an explicit weight-layout and
quantization-state conversion. Double quantization and paged optimizers can be
added at those boundaries.
