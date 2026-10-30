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

| Batch | Path | Session creation (ms) | Step median (ms) | Step P95 (ms) | Peak process RSS (MiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| 256 | dlpack | 27.36 | 0.973 | 1.002 | 157.2 |
| 256 | numpy_view | 29.24 | 0.940 | 0.987 | 157.0 |
| 256 | numpy_adam | 27.52 | 0.552 | 0.580 | 155.4 |
| 16384 | dlpack | 27.56 | 1.102 | 1.438 | 162.3 |
| 16384 | numpy_view | 27.32 | 1.185 | 1.415 | 161.7 |
| 16384 | numpy_adam | 27.35 | 0.830 | 1.014 | 160.2 |

For these 160 adapter parameters, CPU Adam gives the shortest complete steps.
At batch 256, the Adam phase takes about 0.37 ms on GPU and 0.017 ms on CPU.
The report separates preparation, execution/exchange, output import and
synchronized Adam. Benchmark workers omit independent-reference checks.
RSS includes imports, graph generation, runtime and allocators. Python MLX
allocator counters include the active initialization baseline and subsequent
allocation maximum; plugin allocations have their own runtime.

Separate tracing runs show that the plugin copies **70,840 output bytes over
110 executions**, including loss and both gradients. This is
internal provider work after the Python arrays share storage. The trace also
records input-wrapper alignment and sampled process Metal allocations.

```bash
uv run --no-sync python -m examples.lora.mlx_benchmark \
  --mode dlpack --batch 256 --output dlpack.json
uv run --no-sync python -m examples.lora.mlx_benchmark \
  --mode numpy_adam --batch 256 --output numpy_adam.json
```

The [recorded report](https://github.com/justinchuby/onnax/blob/main/examples/measurements/mlx-gpu-updates-apple-arm64.json)
contains model hashes, versions, memory scopes, trace summaries and parameter
comparisons across all benchmark paths. These measurements motivate evaluating
compiled array updates and larger adapters next.
