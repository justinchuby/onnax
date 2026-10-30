# Evaluate native BF16 training with MLX

Run the exported [BF16 LoRA gradient graph](bf16.md) on Apple Silicon using the
[MLX ONNX Runtime plugin](https://github.com/onnxruntime/onnxruntime-ep-mlx).
The example measures native execution and compares every step with independent
NumPy forward/backward and Adam implementations.

```bash
uv sync --group dev --group training --group native-mlx
uv run --no-sync python -m examples.lora.mlx_bf16 --output-dir mlx_bf16_models
```

The recorded run completes 200 updates and checkpoint replay. Its strict
numerical comparison reports `False`, so the CLI returns exit status 1 after
saving the report. The difference is reproducible at a BF16 rounding boundary;
the original `rtol=1e-4, atol=1e-6` remains in effect.

## Export and execute

The graph uses BF16 matrix operands/results and FP32 nonlinearities, loss,
master parameters, derivatives and Adam state. The example fixes batch size
before differentiation, inlines the local BF16 functions and optimizes with
ONNXScript. This resolves runtime shape computations for the plugin.

```{literalinclude} ../examples/lora/mlx_bf16.py
:language: python
:pyobject: gradient_model
```

The native session disables ORT optimization and CPU EP fallback. Runtime
profiles check that executed partitions belong to `MLXExecutionProvider`.
Training and validation use batches 256 and 128, respectively. Parameters enter
and gradients leave through NumPy runtime feeds; Adam runs on the host.

## What the run shows

On Apple M1 Max with plugin 0.31.0 and ORT 1.31.1:

| Quantity | Value |
| --- | ---: |
| Initial training MSE | 0.0188992 |
| Final training MSE | 0.000002909 |
| Initial validation MSE | 0.0162153 |
| Final validation MSE | 0.000002887 |
| Independent NumPy trajectory final validation MSE | 0.000002706 |
| Steps with a same-parameter comparison mismatch | 2 / 200 |
| First mismatch, zero-based step | 68 |

The evaluator checks two comparisons separately. **Same-parameter comparisons**
feed native parameters into the independent forward/backward reference.
**Trajectory comparisons** evolve a second parameter set with independent
NumPy Adam. A small execution difference can move those trajectories across
later BF16 rounding boundaries. The full report retains every comparison.

Frozen FP32 weights remain byte-identical. The checkpoint stores FP32 adapter
masters, both Adam moments and the step count. Reloading the exported graph and
checkpoint reproduces predictions, gradients and the next parameter update.

## Reproduce the rounding difference

At step 68, prediction `[50, 7]` differs by one BF16 interval. The diagnostic
exposes both operands of the final BF16 MatMul as FP32 outputs and compares them
between runtimes: **the operands are identical**.

| Calculation | Value |
| --- | ---: |
| Operand dot product computed in FP64 | -0.1870117038488388 |
| NumPy FP32 dot product | -0.18701171875 |
| Native BF16 result | -0.1865234375 |
| Reference BF16 result | -0.1875 |

The FP32 dot product lands exactly at the midpoint between adjacent BF16
values. Nearest-even rounding selects `-0.1875`. The higher-precision sum lies
on the other side of that midpoint and rounds to `-0.1865234375`, matching the
native result. This case isolates matrix accumulation and rounding behavior.

The CLI saves `first-mismatch.npz` with the inputs and parameters. Run the
standalone diagnostic after evaluation:

```python
from examples.lora.mlx_bf16 import diagnose

print(diagnose("mlx_bf16_models"))
```

## Inspect matrix-accumulation intervals

Add `--inspect-products` to check all forward BF16 products with exact-integer
accumulation bounds. The [bound derivation and recorded results](bf16-bounds.md)
explain the arithmetic model and midpoint handling. The report keeps these
local checks alongside the original strict numerical comparisons.

## Artifacts and measurements

The output directory contains checked `gradient.onnx` and `validation.onnx`,
`checkpoint.npz`, `first-mismatch.npz`, a runtime profile and `report.json`.
The report records the parameter/gradient mapping, frozen-weight digests,
versions, per-step errors, session creation, execution/exchange time and peak
process RSS. Timing excludes references and Adam; the process also holds
validation/replay sessions and executes the references. RSS includes all of
that process's imports, compilation and allocators.

The [recorded summary](https://github.com/justinchuby/onnax/blob/main/examples/measurements/mlx-bf16-apple-arm64.json)
contains aggregate errors and the first mismatch diagnostic. The [matrix-accumulation checks](bf16-bounds.md) now validate a local error
contract at observed inputs. Next compose error propagation through the
nonlinearities and backward graph, alongside device-resident updates.
