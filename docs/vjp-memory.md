# Inspect VJP saved values and local memory peaks

A checkpoint decides which block inputs survive from forward to backward.
Within a block, the VJP also reads forward intermediates. Comparing those saved
values with the block's live tensor set helps choose where to save or recompute.

The example analysis reads an exported `CheckpointedModule` bundle by block name:

```bash
uv run --group huggingface --group training python -m examples.huggingface.saved_values \
  --bundle qwen_run/export/execution --block decoder_3_4 \
  --dimension batch=1 --dimension sequence=64 --optimize \
  --output block-memory.json
```

Use a name from `interface.json`. The bundle supplies active VJP inputs, output
cotangents and parameter names. To analyze an ordinary ONNX model directly:

```bash
uv run --group training python -m examples.huggingface.saved_values model.onnx \
  --inputs adapter.A adapter.B --outputs hidden \
  --parameters adapter.A adapter.B --dimension batch=1 \
  --output model-memory.json
```

## Read the report

The tool clones the source, specializes named dimensions and generates its VJP.
It splits the graph at the forward/backward boundary and lists values actually
read by the pullback. `--optimize` applies the existing ONNXScript training
pipeline before counting those reads.

| Category | Meaning |
|---|---|
| `trainable_parameter` | Selected explicit parameter inputs read by backward |
| `frozen_initializer` | Source initializers read by backward |
| `folded_constant` | Constants created by differentiation or constant folding, including frozen layouts |
| `data_input` | Original inputs such as hidden states and masks |
| `saved_activation` | Forward-produced values read by backward |
| `cotangent` | Upstream gradients supplied to the block |

Each entry records its name, dtype, shape, bytes, forward producer and backward
consumers. Shared reads count the value once. Declared inputs with no backward
use are removed from the accounting copy. A parameter can require a gradient
without backward needing to read its current value: for $u=x\,a$, computing
$\bar a=\bar u\,x$ reads $x$, while the forward pass reads $a$.

`extra_forward_outputs` counts saved values beyond the original forward outputs.
An output already returned across a checkpoint boundary can also serve backward.
The source model and its interfaces remain unchanged.

## Attribute a local peak

For a flat pullback with concrete shapes, `backward_liveness` follows node order:
allocate outputs, then release inputs at their last use. Its inputs and
initializers persist for the invocation. The report includes the peak node,
contributing tensor groups, a timeline and totals by category. Each value has
separate logical storage in this model.

Use runtime allocator and process counters to measure complete training-step
memory, including kernels, sessions and overlapping blocks. The existing
[activation-liveness experiment](activation-liveness.md) shows how to pair these
measurements. Unknown sizes appear in `unknown_values`; nested control flow or
opaque functions produce an `unresolved` liveness result with the required next
analysis step.

## Qwen3.5 block study

The pinned `Qwen/Qwen3.5-0.8B` export supplies a linear-attention block
(`decoder_0_1`) and a full-attention block (`decoder_3_4`). Both use FP32 and rank-4
adapters. Batch is one, with sequence lengths 64, 256 and 512.

For the optimized full-attention pullback:

| Sequence | Saved activations | Backward temporary peak | Total local tensor peak |
|---|---:|---:|---:|
| 64 | 6.03 MiB | 5.72 MiB | 82.38 MiB |
| 256 | 24.32 MiB | 22.88 MiB | 119.37 MiB |
| 512 | 49.14 MiB | 45.75 MiB | 169.25 MiB |

The total includes backward inputs and frozen/constant storage. The linear block
has 22 saved forward values; its Scan body requires a control-flow memory model
before reporting a local peak. This identifies the next extension for recurrent
blocks: separate retained sequence outputs, per-iteration temporaries and carry
storage.

The {download}`complete block reports
<../examples/huggingface/reports/qwen35-block-saved-values.json>` contain source
revision and bundle identity, shapes, producers/consumers, timelines and peak sets.
They provide inputs for selecting a recomputation boundary, followed by gradient,
checkpoint-recovery and runtime-memory comparisons.
