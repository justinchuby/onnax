# Fine-tune a frozen ONNX base with LoRA

This runnable, offline example trains adapters through an ONNX differentiation graph. A
fixed teacher/student regression task demonstrates parameter bindings, gradient
propagation through frozen layers and external parameter updates.

## Run the example

From a source checkout:

```shell
uv sync --group dev --group training
uv run --no-sync python -m examples.lora.train --output-dir lora_models
uv run --no-sync pytest tests/test_lora_training.py
```

The command runs 200 full-batch Adam updates, checks an independently evolving NumPy
reference, saves the adapted models and parameters, reloads them, and writes
`report.json`. The benchmark runs in a separate child process. The optional training
dependencies are installed separately from the core package.

## Model and mathematics

For 16 features, a 24-unit hidden layer, 8 outputs and rank 4:

$$
Z=XW+\tfrac14XAB,\qquad \hat Y=\tanh(Z)C,
\qquad L=\frac{\|\hat Y-Y\|_F^2}{N\cdot8}.
$$

Only $A\in\mathbb R^{16\times4}$ and $B\in\mathbb R^{4\times24}$ are trained. The frozen
base contains $W$ and the post-adapter head $C$. Training has 256 examples; validation
has 128 independent examples from the same fixed teacher. All data and initialization
use a fixed seed. A is random and B starts at zero, so the initial adapted prediction
equals the base prediction.

With $H=\tanh(Z)$, the independent reference computes

$$
G_Z=\left[\frac{2(\hat Y-Y)}{8N}C^T\right]\odot(1-H^2),\quad
\nabla_A L=\tfrac14 X^TG_ZB^T,\quad
\nabla_B L=\tfrac14(XA)^TG_Z.
$$

**Frozen layers still propagate gradients.** C stays fixed while its transpose
propagates the loss gradient to A and B. Stopping the gradient at the hidden activation
would cut this path. Initially A's gradient is zero because B is zero; after B updates,
A receives a nonzero gradient. The acceptance checks test both facts and compare
selected gradient coordinates with independent central finite differences away from the
zero-B initialization.

## Load, adapt and export

The example first saves `base.onnx`, then loads it and injects the adapter with ONNX IR.
The injector accepts the MatMul → Tanh → MatMul architecture shown here and checks its
structure and reserved names.

The loss module is differentiated through the existing public API:

```python
import onnax

gradient = onnax.grad(model, inputs=["lora.A", "lora.B"], output="loss")
```

A and B are explicit FLOAT inputs with no initializer defaults. W and C remain
initializers. The dynamic batch dimension is verified at batch sizes 1, 7 and 128 after
serialization. The external loop reuses one ONNX Runtime CPU session for all training
steps.

Artifacts include:

| File | External use |
| --- | --- |
| `base.onnx` | Frozen starting model |
| `forward.onnx` | Prediction from features and current adapters |
| `loss.onnx` | Prediction and mean-squared loss |
| `gradient.onnx` | Prediction, loss and selected adapter gradients |
| `gradient.optimized.onnx` | Independently checked ONNXScript-optimized copy |
| `interface.json` | Version 1 parameter/gradient bindings, shapes, dtypes and frozen hashes |
| `adapter.npz` | Saved final adapters |
| `benchmark.npz`, `report.json` | Reproduction inputs and numerical/measurement results |

The interface records actual output names after collision resolution. Validate it
against the loaded gradient model before creating runtime bindings. Validation checks
parameter mappings and frozen tensor hashes.

## External Array API Adam

`examples/lora/adam.py` provides the external update function `adam_step(params, grads,
state, *, xp, ...)`. It returns new parameter/state dictionaries using Array API
arithmetic, `sqrt` and `zeros_like`. Arrays stay on their device, and inputs are
preserved. Names, shapes, dtypes and devices must agree. Parameters, gradients and
moments are FP32; the learning rate is 0.03 and Adam uses beta1=0.9, beta2=0.999 and
epsilon=1e-8.

Namespace selection happens once using `array-api-compat` or the backend's native Array
API namespace. The execution adapter handles conversion and synchronization. The tests
compare 20 updates with a separately written, flattened NumPy Adam calculation.

| Backend | Acceptance scope |
| --- | --- |
| NumPy | Complete CPU ONNX training and portable update tests |
| array-api-strict | Required portable update tests |
| JAX CPU | Required portable update tests |
| MLX | Optional Apple Silicon GPU update test; install MLX separately |
| CuPy | Optional CUDA update test; requires a matching CuPy/CUDA installation |

Optional tests are skipped when their hardware or dependencies are unavailable. The
development Apple Silicon run verified MLX GPU updates. Complete ONNX training uses
NumPy and CPU ORT; CUDA validation is pending.

## Correctness checks

The run compares prediction, loss and A/B gradients with independent NumPy formulas, and
compares both evolving training trajectories at every step using `rtol=1e-4, atol=1e-6`.
Each final training/validation loss must fall by at least 50%. It verifies byte-for-byte
frozen weights, complete ONNX checker validity, ONNXScript optimization equivalence, and
restored adapter execution.

ONNXScript optimizes a graph copy offline, and ORT graph optimization is disabled during
execution. Adam updates the trainable parameter arrays.

## Measurements

The report records model-file reading and session creation separately, followed by 10
warmup and 100 measured training updates. Gradient execution, boundary
preparation/conversion, Adam and total step times each have median/P95 values. These
timings exclude graph export and independent-reference acceptance checks. Gradient
timing includes the Python feed/interface checks, ORT call and returned CPU arrays. The
report also records model bytes/node counts, trainable parameter bytes, runtime
versions, platform and ORT thread settings.

The measured full loop is NumPy + CPUExecutionProvider, with one intra-op and one
inter-op thread. Its exchange measurement covers CPU feed preparation and array
wrapping. Peak process RSS includes Python imports, runtime, weights and allocator
overhead, measured on Unix/macOS via `resource`.

See the [MLP classifier](mlp.md) for classification, [microbatch
accumulation](precision.md) for memory measurements, and [BF16 LoRA](bf16.md) for mixed
precision. The [MLX GPU training example](mlx-training.md) keeps external Adam
state in MLX and measures shared runtime boundaries. The [roadmap](roadmap.md) tracks upcoming runtime and model support.

### Recorded development-machine run

A recorded Apple M1 Max CPU run (batch 256, FP32, ORT 1.31.1, one intra/inter-op thread)
measured the following after warmup:

| Component | Median (ms) | P95 (ms) |
| --- | ---: | ---: |
| Gradient execution | 0.1676 | 0.1773 |
| CPU boundary wrapping/preparation | 0.0026 | 0.0030 |
| Array API Adam | 0.0158 | 0.0164 |
| Complete update | 0.1859 | 0.1961 |

Model-file reading took 0.048 ms and session creation 3.58 ms. Peak process RSS was
143.5 MB (decimal), including imports and runtime; the adapters occupy 640 bytes.
ONNXScript reduced the gradient graph from 229 nodes to 89. The table reports execution
times for the optimized graph.

The [recorded
JSON](https://github.com/justinchuby/onnax/blob/main/examples/lora/measurements/apple-m1-max.json)
contains loss endpoints, artifact sizes and runtime versions. Run the CLI to measure
your own configuration.
