# Exchange ONNX gradients with JAX through DLPack

Train the [LoRA model](lora.md) using ONNX Runtime gradients and the external
Array API Adam on JAX CPU. DLPack shares parameter and gradient storage between
the runtimes; the adapter checks their buffer addresses during execution.

From a source checkout:

```bash
uv sync --group dev --group training
uv run --no-sync python -m examples.lora.jax_training --output-dir dlpack_models
```

The example uses JAX 0.10.2 or newer and ORT 1.31.1 or newer, FP32 arrays, one
local JAX CPU device and CPUExecutionProvider. The ONNX graph and parameter names
are the same as in the NumPy LoRA example.

## Share inputs and own each output

Export a JAX array directly into an ORT tensor:

```python
import onnxruntime as ort

value = ort.OrtValue.from_dlpack(array)
assert value.data_ptr() == array.unsafe_buffer_pointer()
binding.bind_ortvalue_input(name, value)
```

The adapter retains the JAX array and its ORT handle while they are bound.
Unchanged array objects reuse their input handles; new parameter arrays are
imported and rebound after each functional Adam update.

ORT allocates fresh output storage on every execution. JAX imports that storage
with `copy=False`:

```python
import jax

gradient = jax.dlpack.from_dlpack(ort_gradient, copy=False)
assert gradient.unsafe_buffer_pointer() == ort_gradient.data_ptr()
```

The imported array retains the DLPack storage owner. Its value stays stable when
later executions allocate new gradients, and it can outlive the adapter and
runtime session. Tests exercise both lifetimes.

This differs from the [NumPy binding adapter](iobinding.md), whose fixed output
buffers are overwritten by the next call. JAX arrays are immutable, so the JAX
adapter clears output bindings and requests fresh CPU buffers before every run.
This also lets pending optimizer work safely consume earlier gradients.

## External updates and synchronization

```python
import jax
import jax.numpy as jnp
from examples.lora.adam import adam_step
from examples.lora.dlpack import JaxCpuDLPack, synchronize

features, target, params = jax.device_put((features, target, params), jax.devices("cpu")[0])
bound = JaxCpuDLPack(executor, interface)
state = None
for _ in range(200):
    bound.bind(features, target, params)
    loss, grads = bound.run()
    params, state = adam_step(params, grads, state, xp=jnp)
    synchronize(params, state)
```

`bind` waits for input computations before ORT reads their storage. CPU ORT
execution is synchronous. After Adam, `synchronize` waits for both parameters
and moments before the next runtime boundary. The optimizer itself uses the
shared Array API implementation.

The same gradients can feed [streaming weighted accumulation](precision.md)
with `xp=jnp`. Each gradient has its own storage, so queued JAX sums keep the
values they consume while ORT runs the next microbatch.

NumPy is used for data setup, independent correctness checks and checkpoint I/O.
The timed DLPack path exports and imports arrays directly between JAX and ORT.

## Validate training and restore state

At each of 200 updates, tests compare loss and gradients with analytic NumPy at
the same parameters. Two separate JAX parameter trajectories use DLPack and
NumPy API boundaries, while a third reference uses independent NumPy Adam.
All comparisons use `rtol=1e-4, atol=1e-6`. Validation MSE falls from about
0.01626 to `4.60e-9`.

The CLI exports ordinary ONNX models, the parameter/gradient interface, final
adapters and a checkpoint containing the FP32 Adam moments and step count.
Restoring the checkpoint reproduces update 201 exactly. Reloaded bindings check
batch sizes 1, 7 and 128; frozen weight hashes stay unchanged. ONNXScript
optimizes a copy, with ORT graph optimization disabled during execution.

Ownership tests retain old gradients across reexecution and after producer
collection, check shared input/output pointers, and exercise Adam consumption
before its explicit boundary wait. A microbatch test combines sizes 63, 63, 63,
63 and 4 through JAX FP32 accumulation.

## Measure the exchange boundary

The benchmark runs each mode in a separate process with the same model, data,
JAX CPU parameters and Array API Adam. DLPack mode exports inputs and imports
fresh ORT outputs directly. NumPy mode uses `np.asarray` feeds, `session.run`
and `jnp.asarray` outputs.

After 10 warmups, 100 updates record median/P95 input exchange, execution,
output exchange, Adam and complete-step times. Imported arrays and optimizer
state are synchronized within their measured stages. ORT uses one intra-op and
one inter-op thread; the report records JAX device and thread environment
settings. Process RSS includes imports, JAX compilation/runtime and allocators.

Recorded Apple M1 Max results (milliseconds):

| Boundary | Input median | Output median | Adam median | Step median | Step P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DLPack | 0.0359 | 0.1450 | 0.1638 | 0.4386 | 0.4927 |
| NumPy API | 0.0041 | 0.0395 | 0.1578 | 0.2899 | 0.3024 |

The NumPy API boundary costs less on this small CPU workload. Input checks and
DLPack output import account for most of the difference. The fetched loss and
gradients total 644 bytes; parameters occupy 640 bytes and Adam moments 1280 bytes.

The [recorded report](https://github.com/justinchuby/onnax/blob/main/examples/measurements/jax-dlpack-apple-arm64.json)
includes execution timings, process RSS, model size, node count and software and
hardware versions. Native GPU bindings and BF16 execution are the next runtime
milestones in the [roadmap](roadmap.md).

See the [ORT DLPack API](https://onnxruntime.ai/docs/api/python/api_summary.html#onnxruntime.OrtValue.from_dlpack)
and [JAX DLPack API](https://docs.jax.dev/en/latest/_autosummary/jax.dlpack.from_dlpack.html)
for the exchange protocol.
