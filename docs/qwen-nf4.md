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
