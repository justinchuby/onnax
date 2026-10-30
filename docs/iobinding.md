# Reuse CPU gradient buffers

Run the [LoRA gradient graph](lora.md) with ONNX Runtime I/O binding. This example
reuses loss and parameter-gradient storage across 200 external Adam updates and
compares it with ordinary `session.run` execution.

From a source checkout:

```bash
uv sync --group dev --group training
uv run --no-sync python -m examples.lora.iobinding --output-dir iobinding_models
```

## Bind owned NumPy storage

A CPU `OrtValue` can share a contiguous NumPy array. Allocate an output array,
wrap it, and bind it to the gradient output name from `interface.json`:

```python
import numpy as np
import onnxruntime as ort

binding = executor.io_binding()
row = interface["parameters"][0]
output = np.empty(row["shape"], dtype=np.float32)
value = ort.OrtValue.ortvalue_from_numpy(output)
assert value.data_ptr() == output.ctypes.data
binding.bind_ortvalue_output(row["gradient"], value)
```

`examples.lora.binding.CpuGradientBinding` manages all three outputs:

| Output | Shape | Bytes |
| --- | --- | ---: |
| Loss | scalar | 4 |
| A gradient | `[16, 4]` | 256 |
| B gradient | `[4, 24]` | 384 |

These 644 bytes of output storage keep the same addresses throughout training.
The output shapes depend on the parameters, so the same buffers also work for
different sample counts.

The adapter retains both NumPy owners and `OrtValue` handles. It checks parameter
names, shapes and dtypes against the runtime interface and verifies shared CPU
pointers. Pass C-contiguous FP32 arrays; convert strided views explicitly with
`np.ascontiguousarray` at the data boundary. Keep bound arrays' shape and storage
fixed, and supply new arrays when changing their layout.

## Update parameters and rebind

```python
from examples.lora.adam import adam_step
from examples.lora.binding import CpuGradientBinding

bound = CpuGradientBinding(executor, interface)
state = None
for _ in range(200):
    bound.bind(features, target, params)
    loss, grads = bound.run()
    params, state = adam_step(params, grads, state, xp=xp)
```

`bind` reuses handles for unchanged array objects. Adam returns new parameter
arrays, so the next iteration creates and binds new parameter handles. Changes
to values within an existing input array are visible through shared storage.
Each training loop owns its binding and buffers.

`run` returns borrowed arrays: the next run overwrites their contents. Consume
the gradients before another call; use `.copy()` when retaining a snapshot.
Adam creates new parameter and moment arrays, so its state owns the values it
needs for subsequent updates.

Streaming microbatch accumulation follows the same order:

```python
from examples.lora.accumulation import accumulate_gradients


def chunks():
    for start in range(0, len(features), 63):
        xb = features[start : start + 63]
        yb = target[start : start + 63]
        bound.bind(xb, yb, params)
        loss, grads = bound.run()
        yield len(xb), loss, grads


loss, grads = accumulate_gradients(chunks(), xp=xp)
params, state = adam_step(params, grads, state, xp=xp)
```

The accumulator consumes each borrowed result before requesting the next chunk.
Its running sums own separate FP32 arrays. The sample-count weights handle the
short final microbatch; see [weighted accumulation](precision.md).

## Correctness and exports

Each of 200 updates compares loss and both gradients with analytic NumPy at the
same parameter point. Ordinary ONNX execution and independent NumPy updates
maintain separate parameter trajectories and agree with the bound path using
`rtol=1e-4, atol=1e-6`. Validation MSE falls from about 0.01626 to `4.60e-9`.

Tests verify input ownership after caller references are dropped, parameter
rebinding, shared-storage value changes, output reuse and microbatch consumption.
Invalid feeds require a successful rebind before execution; overlapping input
and output buffers are rejected.

The CLI saves ordinary ONNX models, the parameter/gradient interface, final
adapters and a JSON report. Full ONNX checker validates the optimized graph;
reloaded bindings exercise sample counts 1, 7 and 128. Frozen weight hashes stay
unchanged. ONNXScript optimizes a graph copy, and ORT graph optimization is
disabled during execution.

## Measured runtime costs

Two separate processes run the same CPU model, sample count 256, initial
parameters, loss/gradient outputs and FP32 Adam update. Each uses one intra-op
and one inter-op thread, with 10 warmups and 100 measured steps. The stages are
feed/binding preparation, execution/result handling and Adam. Binding preparation
includes interface, contiguity and overlap checks plus changed-input handle
creation; ordinary feed preparation builds a dictionary.

Recorded Apple M1 Max results (milliseconds):

| Mode | Prepare median | Execute median | Step median | Step P95 |
| --- | ---: | ---: | ---: | ---: |
| session.run | 0.0002 | 0.0865 | 0.1014 | 0.1047 |
| I/O binding | 0.0118 | 0.0825 | 0.1094 | 0.1121 |

Input checks and parameter rebinding added about 0.012 ms per step in this run.
The standard CPU training example continues to use `session.run`.

Peak process RSS was about 143.0 MB for `session.run` and 143.0 MB for binding,
including Python imports, model/runtime state and allocator overhead. The
[recorded report](https://github.com/justinchuby/onnax/blob/main/examples/measurements/cpu-binding-apple-arm64.json)
also includes model/data/parameter sizes, session creation, output-binding setup, versions and thread
settings. These measurements cover the CPU adapter above.

Next steps are device-native buffers and transfer measurements, using the same
parameter names and ownership contract. See the [development roadmap](roadmap.md)
and [ONNX Runtime I/O binding API](https://onnxruntime.ai/docs/api/python/api_summary.html#iobinding).
