# Qwen fine-tuning at 4096 tokens

Decoder checkpoints make the activation lifetime explicit. The forward graph
maps hidden states through consecutive decoder groups,

\[
h_{i+1}=F_i(h_i;\theta_i),\qquad
\bar h_i=\operatorname{VJP}(F_i,h_i,\bar h_{i+1}).
\]

The bridge saves each group's inputs and recomputes its internal operations
when its VJP runs. Frozen projections still propagate hidden-state gradients
to adapters in earlier groups. `--checkpoint-layers 1`, `2` or `4` selects the
group size. The output head computes loss and hidden-state gradients in
128-token chunks.

The training precision policy keeps LoRA parameters, accumulated gradients and
AdamW moments in FP32. `--precision bf16` uses BF16 projection arithmetic with
FP32 boundaries and adjoints. The frozen decoder uses packed NF4 weights.
Trainer owns updates, gradient accumulation and checkpoint storage.

## Prepare complete long examples

The pinned LongAlpaca revision supplies complete QA pairs. Selection preserves
newlines and appends the Qwen EOS token before counting tokens. Training has
96 short and 32 long examples; validation has 24 short and 8 long examples.
Short examples have at most 1024 tokens and long examples have 3072–4096.
Shared contexts and source files form document groups held out together.
The manifest records source indices, document identities, token counts and
file digests. A separate `benchmark.jsonl` supplies a naturally long example
for fixed 4096/8192-token performance prefixes. LongAlpaca uses CC BY-NC 4.0; see its
[dataset card](https://huggingface.co/datasets/Yukang/LongAlpaca-12k).

```bash
uv run --extra huggingface python -m examples.huggingface.long_corpus \
  --output-dir /tmp/qwen-long-corpus
```

The JSONL files contain one `{"text": ...}` record per complete example. The
SFT CLI also accepts ordinary text files with one example per line.
Checkpoints record the data, tokenizer and maximum-length identities.

## Run training and resume

Run each precision in a separate process. The CLI checks every update for
finite FP32 parameters, gradients and AdamW state, and records memory counters
in `qualification.json`. `report.json` contains fixed training-set and
validation loss before and after training.

For evaluation, each example's loss averages its valid shifted tokens; Trainer
then averages these example losses. Training accumulation normalizes the
summed token loss over the valid tokens in its four microbatches.

```bash
ONNXRUNTIME_EP_MLX_NO_COMPILE=1 \
ONNXRUNTIME_EP_MLX_CACHE_LIMIT_BYTES=536870912 \
uv run --extra huggingface --extra native-mlx python -m examples.huggingface.long_context \
  --corpus /tmp/qwen-long-corpus --library /path/to/libonnxruntime_mlx_ep.dylib \
  --precision fp32 --output-dir /tmp/qwen-4k-fp32
```

The defaults are 30 updates, maximum length 4096, microbatch 1, accumulation 4,
learning rate 0.003, rank 4 and alpha 1 in all q/v projections. The base model
and tokenizer use Qwen3-0.6B revision
`c1899de289a04d12100db370d81485cdf75e47ca`.

Repeat with `--precision bf16` and a new output directory. For a fresh-process
recovery run, add `--resume /tmp/qwen-4k-fp32/checkpoint-15` and select another
output directory. Compare the completed runs using
`python -m examples.huggingface.replay --help`.

The allocator cap is 20 GiB. The qualification runner records process physical
footprint and allocator active/cache/peak bytes after updates. Use the external 24 GiB process-footprint guard with a 5 GiB free-disk
reserve:

```bash
uv run --extra huggingface --extra native-mlx python -m examples.huggingface.guard \
  --output-dir /tmp/qwen-4k-fp32-guard -- \
  python -m examples.huggingface.long_context \
  --corpus /tmp/qwen-long-corpus --library /path/to/libonnxruntime_mlx_ep.dylib \
  --precision fp32 --output-dir /tmp/qwen-4k-fp32
```

Set the environment variables above before invoking the guard. The
512 MiB setting is the allocator's free-cache target.

## Capacity results

On M1 Max with 32 GiB unified memory, both FP32 and BF16 compute completed a
4096-token forward, backward and AdamW update with all 112 adapter gradients
finite. Whole-process peak physical footprint was 13.5 GiB and 14.4 GiB,
respectively. These include cold model construction and native runtime memory.
The [capacity report](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/qwen3-4k-checkpoint-capacity-m1max.json)
records input identities, versions, allocator counters and phase boundaries.

## Training results

Both computation policies completed 30 updates and reduced loss on the fixed
training and validation sets:

| Compute | Train before | Train after | Validation before | Validation after |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 3.1839 | 2.2026 | 2.9834 | 2.2320 |
| BF16 | 3.1845 | 2.2050 | 2.9840 | 2.2325 |

In each run, all 112 adapter tensors changed during the second half of
training, while all 532 frozen initializer hashes stayed unchanged. All 64
exported ONNX modules passed the full checker. Trainable parameters remain
explicit FP32 inputs with matching FP32 gradient outputs. Whole-process peak
physical footprint was 14.6 GiB for FP32 and 13.4 GiB for BF16, including
construction, evaluation, training and export.

The [FP32 report](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/qwen3-4k-nf4-fp32-training-m1max.json)
and [BF16 report](https://github.com/justinchuby/onnax/blob/main/examples/huggingface/reports/qwen3-4k-nf4-bf16-training-m1max.json)
record exact losses, corpus identities, source hashes, checkpoint audits and
per-update memory observations.

Fresh processes resumed checkpoint 15 under both computation policies and
reproduced all final adapters, AdamW state, scheduler and RNG state byte for
byte at update 30. Loaded inference and training modules for both policies reproduced
complete forward outputs, loss
and all 112 gradients on a 4096-token input with seven right-padding tokens.
PEFT exchange preserved all 112 adapter arrays under both policies; the stable FP32 reference
control aligned all adapter gradients at rtol=1e-4 and atol=1e-6.

## Strategy measurements

The M1 Max measurements below use FP32 projection computation, packed NF4
decoder weights, microbatch 1 and 128-token output-head chunks. Each strategy
runs in a fresh process with 10 warmups and 100 measured updates. A training
step includes forward, backward and one AdamW update. Peak process footprint
includes model construction and runtime initialization.

### FP32, 512 tokens

| Strategy | Step median | Step P95 | Valid tokens/s | Peak process footprint |
| --- | ---: | ---: | ---: | ---: |
| Whole-body recomputation | 2.917 s | 3.005 s | 175.2 | 10.44 GiB |
| Checkpoint every layer | 2.780 s | 2.854 s | 183.8 | 7.05 GiB |
| Checkpoint every 2 layers | 2.837 s | 2.870 s | 180.1 | 6.86 GiB |
| Checkpoint every 4 layers | 2.846 s | 2.916 s | 179.5 | 7.40 GiB |

### FP32, 1024 tokens

| Strategy | Step median | Step P95 | Valid tokens/s | Peak process footprint |
| --- | ---: | ---: | ---: | ---: |
| Whole-body recomputation | 4.586 s | 4.689 s | 223.1 | 15.98 GiB |
| Checkpoint every layer | 4.531 s | 4.631 s | 225.8 | 7.26 GiB |
| Checkpoint every 2 layers | 4.495 s | 4.608 s | 227.6 | 7.04 GiB |
| Checkpoint every 4 layers | 4.561 s | 4.602 s | 224.3 | 8.03 GiB |

At 512 and 1024 tokens, all four strategies produced identical loss values
across all 110 updates. Saving all activations exceeded the 24 GiB process
budget at both lengths and was stopped by the guard.

### FP32, 4096 tokens

| Strategy | Step median | Step P95 | Valid tokens/s | Peak process footprint |
| --- | ---: | ---: | ---: | ---: |
| Checkpoint every layer | 23.605 s | 23.749 s | 173.5 | 12.62 GiB |
| Checkpoint every 2 layers | 21.899 s | 22.040 s | 187.0 | 14.19 GiB |
| Checkpoint every 4 layers | 21.123 s | 21.385 s | 193.9 | 19.36 GiB |

All three checkpoint group sizes produced identical loss values across all
110 updates.

Whole-body recomputation and saving all activations exceeded the 24 GiB
process budget during their capacity probes.
The 8192-token capacity probe with checkpoints every layer also exceeded this
process budget.

## Export and measure

The export directory contains a bound ordinary ONNX forward model, shared
frozen weight data, adapter arrays and grouped forward/VJP modules. See
[checkpoint loading](qwen-adapters.md) for direct execution with an external
optimizer. The training directory also contains a standard PEFT adapter.
Validate its exchange through PyTorch PEFT:

```bash
python -m examples.huggingface.checkpoint_adapter \
  --training-dir /tmp/qwen-4k-fp32 --output-dir /tmp/qwen-peft-check \
  --provider mlx --library /path/to/libonnxruntime_mlx_ep.dylib
```

The audit reloads the saved training modules and restores the trained adapter
after its one-step update comparison. Add `--stable-reference` to include the
FP32 pairwise output-head reference control.

Benchmark the same graph strategy independently from quality training:

```bash
ONNXRUNTIME_EP_MLX_NO_COMPILE=1 \
ONNXRUNTIME_EP_MLX_CACHE_LIMIT_BYTES=536870912 \
uv run --extra huggingface --extra native-mlx python -m examples.huggingface.benchmark \
  --mode chunked --checkpoint-layers 1 --real --nf4 --provider mlx \
  --library /path/to/libonnxruntime_mlx_ep.dylib \
  --train-file /tmp/qwen-long-corpus/benchmark.jsonl \
  --sequence-length 4096 --batch-size 1 --chunk-size 128 \
  --mlx-memory-limit-gib 20 \
  --warmup 10 --samples 100
```

Each benchmark step executes one microbatch, its backward pass and one AdamW
update. The benchmark takes a naturally long example's token prefix and uses every
shifted token. It reports step median/P95, tokens/s, output-head and body
execution, AdamW time, host-input preparation, process memory and graph sizes.
ONNX Runtime execution is synchronous, so session times include runtime
transfers. ORT graph optimization is disabled; ONNXScript optimizes the graphs.
Use fixed input tokens, precision and batch size when comparing strategies.
