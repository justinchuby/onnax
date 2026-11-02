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
  --parameters adapter.A adapter.B --dimension batch=1 --optimize \
  --output model-memory.json
```

## Read the report

The tool clones the source and generates its VJP, then specializes named dimensions.
This order preserves derivative choices made for the original graph. An
inference-only copy expands local functions to recover missing boundary shapes.
The generated VJP uses those shapes for accounting.
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

Saved boundaries include forward values captured by nested pullback bodies,
even when those values are implicit inputs of a Scan. Forward-produced captures
become explicit saved-value inputs.

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

Add `--scan-bodies` to apply the {ref}`nested Scan storage schedule <scan-memory>`
to generated recurrences.

## Qwen3.5 block study

The pinned `Qwen/Qwen3.5-0.8B` export supplies a linear-attention block
(`decoder_0_1`) and a full-attention block (`decoder_3_4`). Both use FP32 and rank-4
adapters. Batch is one, with sequence lengths 64, 256 and 512.

For the optimized full-attention pullback:

| Sequence | Saved activations | Backward temporary peak | Total local tensor peak |
|---|---:|---:|---:|
| 64 | 6.03 MiB | 5.97 MiB | 82.63 MiB |
| 256 | 24.32 MiB | 23.88 MiB | 120.37 MiB |
| 512 | 49.14 MiB | 47.75 MiB | 171.25 MiB |

The total includes backward inputs and frozen/constant storage. The linear block
has 22 saved forward values. Analyze its nested recurrences with:

```bash
uv run --group huggingface --group training python -m examples.huggingface.saved_values \
  --bundle qwen_run/export/execution --block decoder_0_1 \
  --dimension batch=1 --dimension sequence=512 --optimize --scan-bodies \
  --output recurrent-memory.json
```

Its existing VJP uses 32-token chunks. An FP32 state with shape `[1,16,128,128]`
occupies 1 MiB. It retains one state per chunk, then reconstructs a 32-state
history window inside the backward loop:

| Sequence | Chunks | Retained boundary states | Local replay history | Local tensor peak |
|---|---:|---:|---:|---:|
| 64 | 2 | 2 MiB | 32 MiB | 105.97 MiB |
| 65 | 3 | 3 MiB | 32 MiB | 108.88 MiB |
| 256 | 8 | 8 MiB | 32 MiB | 150.69 MiB |
| 512 | 16 | 16 MiB | 32 MiB | 210.32 MiB |

The 65-token case includes a padded tail chunk. The replay window stays bounded
as sequence length grows; parameter, activation and gradient storage also
contribute to the block total. The {download}`Scan memory reports
<../examples/huggingface/reports/qwen35-scan-memory.json>` contain both outer
loops, their inner Scans, capture lifetimes and the tensors present at peak.

The {download}`complete block reports
<../examples/huggingface/reports/qwen35-block-saved-values.json>` contain source
revision and bundle identity, shapes, producers/consumers, timelines and peak sets.
They provide inputs for selecting a recomputation boundary, followed by gradient,
checkpoint-recovery and runtime-memory comparisons.

The [block runtime study](block-runtime.md) measures these exported recomputation
modules with captured task inputs and native allocator counters.

## Execute a selective replay region

The example layer can now choose saved values by their ONNX names. Given a split
VJP boundary, `split_vjp(..., recompute=names)` moves the required producer slice
into backward. Traversal stops at other saved values, original inputs and frozen
weights. Shared producers execute once. The source and joint derivative remain
unchanged.

The replay contract currently covers deterministic arithmetic, layouts and
standard normalization. Random operators, control flow and custom functions need
an explicit state contract before entering a replay region. Saved tensors retain
PyTorch's parameter version checks across the autograd boundary.

Inspect a candidate with the existing report command:

```bash
uv run --group huggingface --group training python -m examples.huggingface.saved_values \
  --bundle qwen_run/export/execution --block decoder_3_4 \
  --dimension batch=1 --dimension sequence=4032 --optimize \
  --recompute v_model.layers.3.mlp.gate_proj.MatMul_395 \
              v_model.layers.3.mlp.Swish_396 \
              v_model.layers.3.mlp.up_proj.MatMul_398 \
  --output replay-plan.json
```

Names come from the saved-value report for that exact export. The JSON includes
retained values, replay inputs and topologically ordered producer nodes. The
liveness report accounts for the resulting backward graph. Dynamic broadcast
paths can leave value-dependent temporary sizes unresolved; concrete-shape
reports list those unknowns.

`examples.huggingface.block_recompute` executes paired forward/backward sessions
against recorded checkpoint arguments. Use the same bundle, captures and block
as `block_runtime`; omit `--recompute` for the save-all control, then supply the
selected names for the candidate. Both routes check every captured gradient,
verify immutable feeds, run the full ONNX checker and measure ten warmups plus
one hundred calls. Reports separate forward/backward/paired times, retained CPU
arrays, MLX allocator peak and whole-process physical footprint.

For small training experiments, the example `bridge.OnnxModule` accepts
`save_activations=True, recompute=names`. Tests cover dynamic batches, shared
producer slices, non-unit upstream gradients, unequal microbatch accumulation,
multi-step updates and Adam checkpoint restoration. The decoder checkpoint
executor continues to use its qualified whole-block replay policy while these
block-local alternatives are measured.
