# Train with MLX GPU updates and shared runtime buffers

Keep LoRA adapter parameters and Adam moments as MLX Metal arrays while
executing the generated ONNX gradient graph through the MLX ONNX Runtime
provider. This example uses the [FP32 LoRA model](lora.md), shares storage at
the Python/runtime boundary, and checks a complete training trajectory.

```bash
uv sync --group dev --group training --group native-mlx
uv run --no-sync python -m examples.lora.mlx_training --output-dir mlx_training_models
```

The CLI trains for 200 steps, compares predictions, loss, gradients and updated
parameters with independent NumPy implementations, and checks validation data.
It exports separate forward and loss/gradient ONNX graphs, a versioned parameter/gradient interface and
a checkpoint containing adapter masters, both Adam moments and the step count.
Reloading reproduces execution and the next parameter update. The forward
graph also executes batches 1, 7, 128 and 256; the training graph uses batch 256.

On the recorded Apple M1 Max run, training MSE fell from **0.01896 to 3.10e-9**
and validation MSE from **0.01626 to 4.60e-9**. Comparisons use
`rtol=1e-4, atol=1e-6`. Frozen W/C weights remain byte-identical, and the
initial zero/nonzero adapter-gradient checks verify propagation through C.

## Execute an external GPU update

The optimizer is the same [Array API Adam](lora.md) used by the
CPU example. Its array arithmetic receives `xp=mlx.core`. A GPU stream context
places the update on MLX GPU; parameters and moments keep their Metal storage.
Synchronization stays at the execution and timing boundaries.

```python
import mlx.core as mx
from examples.lora.adam import adam_step
from examples.lora.mlx_exchange import MlxDLPack, synchronize
from examples.lora.mlx_runtime import session
from examples.lora.mlx_training import gradient_model
from examples.lora.train import problem

w, c, x, y, _, _, initial = problem()
graph, interface = gradient_model(w, c, batch=256)
bound = MlxDLPack(session(graph), interface)
with mx.stream(mx.gpu):
    features, target = mx.array(x), mx.array(y)
    params = {name: mx.array(value) for name, value in initial.items()}
    state = None
    bound.bind(features, target, params)
    _, loss, grads = bound.run()
    params, state = adam_step(params, grads, state, xp=mx)
    synchronize(params, state)
```

## Compile the GPU update

Add `--compiled-adam` to run the same external Array API arithmetic through
`mx.compile`:

```bash
uv run --no-sync python -m examples.lora.mlx_training \
  --compiled-adam --output-dir compiled_mlx_models
```

The wrapper validates names, shapes and state before calling a pure array
kernel. Parameters and both moments are explicit arguments. Each step passes
two FP32 bias-correction values as data, so the Python step counter can advance
while the compiled graph is reused. Hyperparameters are fixed when creating
the wrapper; checkpoint state stays in the existing `AdamState` format.

```python
from examples.lora.mlx_adam import CompiledAdam

update = CompiledAdam()
with mx.stream(mx.gpu):
    params, state = update(params, grads, state)
    synchronize(params, state)
```

[MLX compilation](https://ml-explore.github.io/mlx/build/html/usage/compile.html)
traces pure functions and specializes them for input structure and shapes.
The tests verify one trace across 80 updates, a new specialization after a
shape change, independent moment/parameter values, preserved inputs and
checkpoint replay with a newly created wrapper. The ONNX model hash is the
same for eager and compiled updates.

## Share storage across the boundary

MLX arrays report a Metal DLPack device. The adapter explicitly requests a
CPU-address view of that unified-memory storage, with `copy=False`, for ORT's
CPU tensor interface. Pointer checks verify that ORT sees the same allocation.

```{literalinclude} ../examples/lora/mlx_exchange.py
:language: python
:pyobject: CpuView
```

Each execution allocates fresh ORT outputs. MLX imports them with
`from_dlpack(..., copy=False)` and checks pointer identity again. Imported
arrays retain their storage after later runs and session destruction.
Inputs are rebound every step because indexed assignment can replace an MLX
array's storage while preserving its Python identity.

MLX's [conversion guide](https://ml-explore.github.io/mlx/build/html/usage/numpy.html)
describes shared imports and synchronization. The adapter evaluates input
arrays before ORT reads them and waits for parameters and moments when timing
an update. It rejects wrong names, dtypes, shapes and noncontiguous inputs.

## Generate simpler matrix adjoints

For a vector or rank-two matrix product, reverse mode uses the direct
adjoints, with vector-axis promotion/restoration as needed:

$$
\bar A=\bar YB^\mathsf{T},\qquad \bar B=A^\mathsf{T}\bar Y.
$$

These operands have no broadcast batch dimensions. onnax now skips their
broadcast-reduction scaffolding, removing twelve nodes from this optimized
LoRA graph and allowing all 30 nodes to execute as one MLX partition.
Higher-rank batched products retain broadcast reduction. Tests cover vectors,
matrices, symbolic row counts, empty dimensions and batched products.

## Measure complete steps and internal copies

Each row below is a separate process, run sequentially with 10 warmups and
100 measured updates. All paths execute the same native gradient graph and
share dataset views. Batch 16,384 repeats the 256 training samples. `dlpack` uses the adapter above; `numpy_view` uses ordinary
runtime feeds from `np.asarray(..., copy=False)` and shared MLX output imports.
Both run GPU Adam. `numpy_adam` keeps parameter updates and state in NumPy on CPU.
Benchmark steps fetch only loss and gradients. The correctness CLI checks
predictions with a separate forward session.

| Batch | Path | First Adam (ms) | Adam median (ms) | Step median (ms) | Step P95 (ms) | Peak process RSS (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 256 | dlpack eager | 1.896 | 0.365 | 0.952 | 1.042 | 156.3 |
| 256 | dlpack compiled | 3.731 | 0.290 | 0.884 | 0.992 | 157.9 |
| 256 | numpy_view eager | 1.914 | 0.367 | 0.936 | 0.993 | 156.5 |
| 256 | numpy_view compiled | 3.146 | 0.275 | 0.808 | 0.923 | 157.5 |
| 256 | numpy_adam | 0.083 | 0.017 | 0.551 | 0.575 | 155.7 |
| 16384 | dlpack eager | 2.473 | 0.391 | 1.453 | 1.740 | 161.6 |
| 16384 | dlpack compiled | 3.176 | 0.273 | 1.032 | 1.132 | 162.2 |
| 16384 | numpy_view eager | 2.001 | 0.392 | 1.405 | 1.547 | 163.0 |
| 16384 | numpy_view compiled | 3.252 | 0.277 | 1.044 | 1.331 | 161.9 |
| 16384 | numpy_adam | 0.079 | 0.019 | 1.024 | 1.374 | 160.9 |

At batch 256, compilation reduces the DLPack path's Adam median from
0.365 to 0.290 ms and its full-step median from 0.952 to 0.884 ms. The first
update increases from 1.90 to 3.73 ms. First-update time includes state
initialization and, for compiled paths, tracing and compilation.

CPU Adam gives the shortest steps for these 160 parameters at batch 256.
At batch 16,384, recorded CPU and compiled DLPack medians are both about
1.03 ms. The report separates preparation, execution/exchange, output import
and synchronized Adam. Benchmark workers omit independent-reference checks.
Session creation, versions and model hashes are recorded separately.

RSS includes imports, graph generation, runtime and allocators. Python MLX
allocator counters include the active initialization baseline and subsequent
allocation maximum; plugin allocations have their own runtime.

Separate tracing runs show that the plugin copies **70,840 output bytes over
110 executions**, including loss and both gradients. This is
internal provider work after the Python arrays share storage. The trace also
records input-wrapper alignment and sampled process Metal allocations.

```bash
uv run --no-sync python -m examples.lora.mlx_benchmark \
  --mode dlpack --compiled-adam --batch 256 --output dlpack.json
uv run --no-sync python -m examples.lora.mlx_benchmark \
  --mode numpy_adam --batch 256 --output numpy_adam.json
```

The [compiled-update report](https://github.com/justinchuby/onnax/blob/main/examples/measurements/compiled-mlx-updates-apple-arm64.json)
contains eager/compiled training endpoints, all benchmark variants, first-update
costs and parameter comparisons. The [shared-buffer report](https://github.com/justinchuby/onnax/blob/main/examples/measurements/mlx-gpu-updates-apple-arm64.json)
records input alignment and sampled Metal memory. Larger adapters are the next
workload for evaluating GPU update placement.
