# NF4 Qwen fine-tuning with SFTTrainer

Store Qwen's frozen decoder matrices in NF4 and train FP32 LoRA adapters through
ordinary ONNX forward/VJP modules. The example combines Mobius model construction,
onnax differentiation, TRL SFTTrainer and PEFT adapter files.

```bash
uv sync --group dev --group training --group huggingface
uv run python -m examples.huggingface.sft --nf4 --steps 100 \
  --save-activations --peft-roundtrip --output-dir tiny_nf4
```

Use `--real --steps 30 --diagnose-rounding` for the pinned Qwen3-0.6B model.
The [Qwen tutorial](qwen-sft.md) describes the corpus, tokenizer, right padding,
causal shift and effective-token normalization.

## Frozen decoder, trainable adapters

Every decoder layer quantizes seven Linear matrices: q/k/v/o projections and
MLP gate/up/down projections. For a q- or v-projection,

$$
Y=XQ(W)+\tfrac14XAB,
\qquad \bar X=\bar YQ(W)^T+\tfrac14(\bar YB^T)A^T.
$$

Only A and B are updated. Gradient propagation through the frozen quantized
matrices connects adapters across layers. Embeddings, normalization and lm_head
stay FP32. Loss, adapter gradients, AdamW parameters and moments are FP32.

Each block of 64 weights has an FP32 absmax and 64 NF4 codes; two codes occupy a
byte. ONNX stores the transposed Linear matrix in row-major blocks, high nibble
first. `export/quantization.json` records the codebook, shapes, buffer names and
storage sizes. Reconstructing that layout reproduces the ONNX base weights.

The CPU reference functions unpack each matrix and run FP32 MatMul. Exported
models contain standard ONNX function bodies. ONNXScript preserves these function
boundaries during optimization. The runtime uses ORT graph optimization disabled.
Saved-activation splitting retains function calls and saves their output boundary.

## Independent validation and adapter exchange

The Transformers reference uses exactly the stored NF4 values, independently
decoded in NumPy and copied into its Linear weights. Tests compare predictions,
token loss, every adapter gradient and multiple independent AdamW updates using
`rtol=1e-4, atol=1e-6`. Both full-logit and chunked-loss paths are covered.

PEFT adapter tensors keep the standard Linear layout: ONNX A/B are transposed
when saved. `adapter_mapping.json` also records the base quantization. Adapter
exchange compares the same reconstructed base in ONNX and PyTorch.

Tiny Qwen runs 100 updates, exports a fully checked ONNX graph and resumes
checkpoint 50 in a fresh process. The replay verifier compares adapter bytes,
checkpoint tensors, AdamW and scheduler state, frozen-buffer hashes and outputs
from both loaded ONNX exports:

```bash
uv run python -m examples.huggingface.sft --nf4 --steps 100 \
  --save-activations --resume tiny_nf4/checkpoint-50 --output-dir resumed_nf4
uv run python -m examples.huggingface.replay --uninterrupted tiny_nf4 \
  --resumed resumed_nf4 --step 100 --output replay.json
```

The two-layer model packs 294,912 decoder-weight bytes into 41,472 bytes including
scales. This storage count measures the packed buffers. Runtime RSS includes
function execution, model construction and optimizer state.

## Measure the execution paths

```bash
uv run python -m examples.huggingface.benchmark --nf4 --samples 100 \
  --warmup 10 --output nf4-benchmark.json
```

Each execution path runs in its own process: PyTorch, raw ONNX, optimized ONNX,
saved activations and chunked loss. PyTorch uses the same reconstructed NF4
weight values stored as FP32 tensors. Reports identify weight values and storage
separately, and include step median/P95, effective tokens/s, CPU array-view time,
peak process RSS, thread counts and package versions.

The next performance work is decoding lifetime and backend fusion, guided by
these measurements.

## Recorded CPU runs

On Apple M1 Max with 32 GiB RAM, Qwen3-0.6B completed 30 NF4 adapter updates.
Validation loss decreased from **7.750500 to 5.731173**. Its 196 decoder matrices
use 247,726,080 packed/scales bytes, compared with 1,761,607,680 FP32 bytes.
The exported weight data is 893,583,360 bytes, including the FP32 embedding and
other frozen buffers. The full ONNX checker validates the export. A fresh process resumed checkpoint
15 and reproduced all 112 adapters, all 113 checkpoint tensors, optimizer and
scheduler state byte for byte. Both loaded exports returned byte-identical
logits and hidden states; all 532 frozen initializer hashes matched.

`qwen3-nf4.json` retains the original elementwise comparisons. At the final
state, three gradient elements in layer 16's q-projection B exceed the original
threshold; independent AdamW update differences are also recorded. Tiny Qwen
passes the strict gradient, update and PEFT comparisons.

The isolated tiny-model benchmark uses batch four, sequence length 64 and 252
valid shifted tokens per update. Each path warms up for 10 updates and measures
100 updates on the same NF4-reconstructed base values:

| Execution | Step median (ms) | P95 (ms) | Effective tokens/s | Peak process RSS (MiB) |
| --- | ---: | ---: | ---: | ---: |
| PyTorch, reconstructed FP32 storage | 3.29 | 16.66 | 76,683 | 541.2 |
| Raw ONNX NF4 | 13.61 | 45.73 | 18,516 | 597.5 |
| Optimized ONNX NF4 | 10.56 | 12.18 | 23,856 | 604.4 |
| Saved activations | 9.10 | 9.19 | 27,700 | 611.5 |
| Chunked loss | 11.51 | 12.14 | 21,898 | 605.8 |

The JSON report records per-phase times, CPU array-view costs and versions.
Process RSS includes imports, model construction and runtime. A sample of the
real CPU run identifies codebook Gather as a decoding hotspot; backend fusion
and decoding lifetime are the next performance targets.

## Run forward and VJP on Apple GPUs

The native MLX validation uses the same two-layer NF4 Qwen and independently
updates its ONNX and Transformers adapters for four AdamW steps. It compares
logits, token loss, all adapter gradients and parameters at every step. ONNX
forward and VJP sessions require complete MLX assignment and disable ORT graph
optimization. PyTorch computes loss and AdamW on the CPU; NumPy arrays cross the
session boundary.

Use an MLX EP build containing [dynamic Range and ScatterElements support](
https://github.com/onnxruntime/onnxruntime-ep-mlx/pull/90):

```bash
uv run python -m examples.huggingface.mlx_validation \
  --library /path/to/libonnxruntime_mlx_ep.dylib --output mlx-qwen.json
```

Omit `--library` to use the installed plugin. The report records the library,
package versions, execution boundary and numerical tolerances. The checked-in
M1 Max report has loss decreasing from 5.50393 to 5.49281 across four steps.
Run the native regression with `ONNAX_TEST_MLX_EP=1` and set
`ONNAX_MLX_EP_LIBRARY` for a development dylib:

```bash
ONNAX_TEST_MLX_EP=1 ONNAX_MLX_EP_LIBRARY=/path/to/libonnxruntime_mlx_ep.dylib \
  uv run pytest tests/test_qwen_mlx.py -q
```
