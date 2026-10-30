# Fine-tune a frozen ONNX base with LoRA

This runnable, offline example trains adapters through an ONNX differentiation
graph. It uses a controlled teacher/student regression task rather than a
pretrained language model, tokenizer or external dataset. The graph boundary and
frozen-weight traversal are the capabilities being validated.

## Run the complete acceptance workload

From a source checkout:

```shell
uv sync --group dev --group training
uv run --no-sync python -m examples.lora.train --output-dir lora_models
uv run --no-sync pytest tests/test_lora_training.py
```

The command runs 200 full-batch Adam updates, checks an independently evolving
NumPy reference, saves the adapted models and parameters, reloads them, and writes
`report.json`. The benchmark runs in a separate child process. The optional
training dependencies do not become onnax runtime dependencies.

## Model and mathematics

For 16 features, a 24-unit hidden layer, 8 outputs and rank 4:

$$
Z=XW+\tfrac14XAB,\qquad \hat Y=\tanh(Z)C,
\qquad L=\frac{\|\hat Y-Y\|_F^2}{N\cdot8}.
$$

Only $A\in\mathbb R^{16\times4}$ and $B\in\mathbb R^{4\times24}$ are trained.
The frozen base contains $W$ and the post-adapter head $C$. Training has 256
examples; validation has 128 independent examples from the same fixed teacher.
All data and initialization use a fixed seed. A is random and B starts at zero,
so the initial adapted prediction equals the base prediction.

With $H=\tanh(Z)$, the independent reference computes

$$
G_Z=\left[\frac{2(\hat Y-Y)}{8N}C^T\right]\odot(1-H^2),\quad
\nabla_A L=\tfrac14 X^TG_ZB^T,\quad
\nabla_B L=\tfrac14(XA)^TG_Z.
$$

**Freezing a weight does not stop gradients through its operation.** C is never
updated and has no requested gradient output, but its transpose is essential to
propagating the loss back to A and B. Detaching the hidden activation before C
would break that path. There is no stop-gradient API in this example.
Initially A's gradient is zero because B is zero; after B updates, A receives a
nonzero gradient. The acceptance checks test both facts and compare selected
gradient coordinates with independent central finite differences away from the
zero-B initialization.

## Load, adapt and export

The example first saves `base.onnx`, then loads it and injects the adapter with
ONNX IR. The injector intentionally recognizes this example's exact
MatMul → Tanh → MatMul architecture; it is not a general arbitrary-model LoRA
rewriter. Unsupported structures and reserved-name collisions are rejected.

The loss module is differentiated through the existing public API:

```python
import onnax

gradient = onnax.grad(model, inputs=["lora.A", "lora.B"], output="loss")
```

A and B are explicit FLOAT inputs with no initializer defaults. W and C remain
initializers. The dynamic batch dimension is verified at batch sizes 1, 7 and
128 after serialization. The external loop reuses one ONNX Runtime CPU session
for all training steps; it does not rebuild the graph or session after updates.

Artifacts include:

| File | External use |
| --- | --- |
| `base.onnx` | Frozen starting model |
| `forward.onnx` | Prediction from features and current adapters |
| `loss.onnx` | Prediction and mean-squared loss |
| `gradient.onnx` | Prediction, loss and selected adapter gradients |
| `gradient.optimized.onnx` | Independently checked ONNXScript-optimized copy |
| `interface.json` | Version 1 parameter/gradient bindings, shapes, dtypes and frozen hashes |
| `adapter.npz` | Example-owned final adapter storage, not a core checkpoint format |
| `benchmark.npz`, `report.json` | Reproduction inputs and numerical/measurement results |

The interface records actual output names after collision resolution. Validate
it against the loaded gradient model before creating runtime bindings. Do not
guess a gradient suffix or use an output name from another graph. Invalid
parameter mappings and changed frozen tensors fail validation.

## External Array API Adam

`examples/lora/adam.py` is example code, not an onnax training API. Its functional
`adam_step(params, grads, state, *, xp, ...)` returns new parameter/state dictionaries.
It uses Array API arithmetic, `sqrt` and `zeros_like`, with no NumPy conversions,
array mutation or array scalar reads inside the update. Names, shapes, dtypes and
devices must agree. Parameters, gradients and moments are FP32; the learning rate
is 0.03 and Adam uses beta1=0.9, beta2=0.999 and epsilon=1e-8.

Namespace selection happens once using `array-api-compat` or the backend's native
Array API namespace. Conversion and synchronization belong at the execution
boundary, not inside Adam. The tests compare 20 updates with a separately written,
flattened NumPy Adam calculation.

| Backend | Acceptance scope |
| --- | --- |
| NumPy | Complete CPU ONNX training and portable update tests |
| array-api-strict | Required portable update tests |
| JAX CPU | Required portable update tests; no JAX autodiff used |
| MLX | Optional Apple Silicon GPU update test; install MLX separately |
| CuPy | Optional CUDA update test; requires a matching CuPy/CUDA installation |

Missing optional hardware/dependencies are reported as skipped, not validated.
MLX GPU updates were checked on the development Apple Silicon machine; full
ONNX+MLX GPU training is not implemented. CuPy was not validated on that machine.

## Numerical and execution acceptance

The run compares prediction, loss and A/B gradients with independent NumPy
formulas, and compares both evolving training trajectories at every step using
`rtol=1e-4, atol=1e-6`. Each final training/validation loss must fall by at least
50%. It verifies byte-for-byte frozen weights, complete ONNX checker validity,
ONNXScript optimization equivalence, and restored adapter execution.

ORT graph optimization is disabled. ONNXScript optimizes a copy offline;
it has no relationship to the external Adam parameter optimizer.

## Measurement scope and limitations

The report records model-file reading and session creation separately, then 10 warmup and 100 measured
training updates. Gradient execution, boundary preparation/conversion, Adam and
total step times each have median/P95 values. These timings exclude graph export
and independent-reference acceptance checks. Gradient timing includes the Python
feed/interface checks, ORT call and returned CPU arrays; it is not kernel-only time. It also records model bytes/node
counts, trainable parameter bytes, runtime versions, platform and ORT thread settings.

The measured full loop is NumPy + CPUExecutionProvider, with one intra-op and one
inter-op thread. Its exchange measurement covers CPU feed preparation and array
wrapping; it is not a measurement of GPU transfer or zero-copy execution.
Peak RSS comes from an isolated benchmark process and includes Python imports,
runtime, weights and allocator overhead. It is neither a tensor-allocation peak
nor device memory. RSS measurement currently targets Unix/macOS via `resource`.
There is no performance pass/fail threshold and no claim of acceleration over
another training framework. Fewer graph nodes alone are not evidence of speedup.

BF16/mixed precision, device-resident ONNX bindings, arbitrary base architectures,
classification losses, FP8, QLoRA and distributed training remain later work.

### Recorded development-machine run

A recorded Apple M1 Max CPU run (batch 256, FP32, ORT 1.31.1, one intra/inter-op
thread) measured the following after warmup:

| Component | Median (ms) | P95 (ms) |
| --- | ---: | ---: |
| Gradient execution | 0.1676 | 0.1773 |
| CPU boundary wrapping/preparation | 0.0026 | 0.0030 |
| Array API Adam | 0.0158 | 0.0164 |
| Complete update | 0.1859 | 0.1961 |

Model-file reading took 0.048 ms and session creation 3.58 ms. Peak process RSS
was 143.5 MB (decimal), including imports and runtime. The adapters occupy only
640 bytes; this illustrates why process RSS must not be presented as tensor memory.
The gradient model had 229 nodes before ONNXScript and 89 afterward; execution
of the unoptimized graph was validated, but no unoptimized timing comparison was
made. These measurements establish a baseline, not a speedup claim.

The [recorded JSON](https://github.com/justinchuby/onnax/blob/main/examples/lora/measurements/apple-m1-max.json)
contains loss endpoints, artifact sizes and runtime versions. Rerun on your
machine rather than assuming these small-model timings generalize.
