# Train LoRA adapters with CoreML

Run the LoRA loss/gradient graph through ONNX Runtime's CoreML provider, then
update adapter parameters with the external Array API Adam. The base weights
remain initializers. The generated graph propagates gradients through the frozen
output matrix to both adapters.

```bash
uv sync --group dev --group training
uv run --no-sync python -m examples.lora.coreml --output-dir coreml_models
```

This command uses the [LoRA model and dataset](lora.md), checks 200 updates
against independent NumPy forward/backward and Adam implementations, and saves
`gradient.onnx`, `adapter.npz`, a runtime profile and `report.json`. It reloads
the graph and adapter into a fresh CoreML session and checks execution again.
The command requires a macOS ONNX Runtime installation with CoreML enabled.

## Prepare the gradient graph

The example fixes the batch size before differentiation, then optimizes with
ONNXScript. For the stable Tanh slope,

$$
t=\exp(-2|z|),\qquad \tanh'(z)=\frac{4t}{(1+t)^2},
$$

it expresses the magnitude as `Relu(z) + Relu(-z)`, using multiplication by
`-1` for negation. This lets CoreML execute the complete graph. The adapter lives
in the example's graph factory; the general onnax differentiation rules retain
their existing representation. Tests compare gradients and HVPs at zero,
ordinary and saturated activations.

```{literalinclude} ../examples/lora/coreml.py
:language: python
:pyobject: gradient_model
```

The session uses `MLProgram`, `CPUAndGPU`, static input shapes and FP32
accumulation. ORT graph optimization and CPU EP fallback are disabled.
The runtime profile verifies that executed partitions belong to CoreML EP.
CoreML chooses CPU/GPU placement within those partitions. See the
[provider options](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
for the compute-unit policy and compute-plan profiling.

## Recorded run

On Apple M1 Max, training loss fell from **0.01896 to 3.10e-9**; validation loss
fell from **0.01626 to 4.60e-9**. Prediction, loss, both gradients and parameters
matched the independent reference throughout 200 updates at
`rtol=1e-4, atol=1e-6`.

Each benchmark below runs in a separate process, sequentially. It performs
10 warmup updates and measures 100 updates with the same static graph and
NumPy Adam. Larger batches repeat the training data.

| Batch | Provider | Session creation | Step median | Step P95 | Peak process RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| 256 | CPU | 2.44 ms | 0.068 ms | 0.072 ms | 128.3 MiB |
| 256 | CoreML | 70.56 ms | 0.136 ms | 0.148 ms | 142.7 MiB |
| 16384 | CPU | 2.53 ms | 2.353 ms | 2.532 ms | 144.7 MiB |
| 16384 | CoreML | 189.64 ms | 1.466 ms | 1.535 ms | 163.0 MiB |

CoreML reduced step time for the larger batch; CPU was faster for batch 256.
Session creation includes native model compilation. Execution timing includes
host/device exchange, prediction/loss/gradient retrieval and Python outputs.
RSS includes imports, graph generation, compilation, runtime and allocators.
The report records feed preparation, execution/exchange and Adam separately.

```bash
uv run --no-sync python -m examples.lora.coreml_benchmark \
  --provider cpu --batch 16384 --output cpu.json
uv run --no-sync python -m examples.lora.coreml_benchmark \
  --provider coreml --batch 16384 --output coreml.json
```

The [recorded report](https://github.com/justinchuby/onnax/blob/main/examples/measurements/coreml-apple-arm64.json)
contains timings, versions, thread settings and training history. This path
uses FP32 NumPy parameters and host-facing runtime feeds. Device-resident
updates and native BF16 execution remain the next precision/device workloads.
