# Compare training quality across runtimes

The Qwen quality experiment trains independent LoRA adapters on each runtime and
compares held-out token loss across seeds. Each matched seed starts from the same
adapter arrays and follows the same shuffled data schedule. Parameters and Adam
states evolve independently after initialization.

```bash
ONNXRUNTIME_EP_MLX_NO_COMPILE=1 uv run python -m examples.huggingface.training_impact \
  --seeds 17 29 43 71 101 --steps 100 \
  --library /path/to/libonnxruntime_mlx_ep.dylib --output quality.json
```

The default engine list includes PyTorch CPU, PyTorch CUDA, ONNX ReferenceEvaluator
CPU and MLX. The report lists the engines executed and those unavailable on the
host. `--engines torch_cpu torch_cuda` selects a same-framework CPU/CUDA baseline.

## Matched training conditions

Mobius constructs the canonical Qwen graph and source parameter mapping. The
experiment reconstructs the same NF4 base values for PyTorch. The base stays fixed;
only rank-four q/v adapters are updated. `--projection-precision bf16` uses BF16
projection operands/results and FP32 adapters, gradients and Adam state. The
PyTorch policy wrapper implements the same FP32 surrogate adjoints. The reference
head uses pairwise FP32 accumulation over 128-class blocks.

Each optimizer update accumulates four microbatches of one sequence by default.
Right padding, explicit position IDs and `-100` labels keep the objective aligned.
Each engine normalizes summed stable cross-entropy by the total valid shifted
tokens across the update. `--microbatch` and `--accumulation` select other sizes.

The first 18 corpus lines form the training split and the final six form the
held-out split. The tiny tokenizer learns the corpus vocabulary. Seeds vary the
initial A matrices and data order; B starts at zero. The report checks matching
source weights, adapter initialization and schedules, and verifies frozen weights
after training.

## Recorded tiny BF16/NF4 comparison

On M1 Max, five seeds each ran 100 updates with identical AdamW settings
(`lr=0.003`, `eps=1e-8`, zero weight decay):

| Engine | Mean final held-out loss | Seed standard deviation |
| --- | ---: | ---: |
| PyTorch CPU | 5.455727 | 0.003744 |
| ONNX ReferenceEvaluator CPU | 5.455727 | 0.003702 |
| ONNX / MLX | 5.455682 | 0.003715 |

The largest matched-seed PyTorch CPU/MLX loss gap is **0.000209**, about 5.6% of
the observed seed standard deviation. Sampled trajectory gradients have a minimum cosine
similarity of approximately **0.999966**. All training losses decrease; frozen weights
remain unchanged. Final adapter entries differ by as much as 0.00336 between
PyTorch CPU and MLX while their held-out losses remain close.

This experiment finds similar training outcomes across the three engines on the
tiny workload. CUDA was unavailable on the recorded Mac. Real-model and CUDA
comparisons are the next quality measurements.

The {download}`recorded report <../examples/huggingface/reports/tiny-nf4-bf16-training-impact-m1max.json>`
contains every seed, training loss, validation curve, gradient norm and paired
snapshot comparison, with source hashes, versions and the MLX library digest.

## Compare trajectories and checkpoint replay

`--interval` controls held-out evaluations and gradient snapshot spacing. Snapshot
comparisons use independently evolving parameters at the same update and data
batch. The [fixed-primal audit](qwen-runtime-audit.md) checks derivative accuracy
at a common state. The [checkpoint verifier](qwen-sft.md) checks exact recovery on
the same backend.

Use `--capture snapshots.npz` to save FP32 gradient vectors and final adapters in
sorted parameter-name order. The report records their checksum. Host snapshots
are part of this quality experiment; the [benchmark](qwen-runtime-audit.md)
measures execution and exchange costs separately.

Add `--real --steps 30` for the pinned Qwen3-0.6B model. A matched comparison uses
the same precision, data schedule, effective batch and optimizer on each engine.
Validation curves and seed variation provide the training-quality baseline;
pointwise numerical checks provide diagnostic detail.


## Run large-model engines in separate processes

Each engine can write its own report and snapshot file, keeping runtime caches
within one process at a time:

```bash
uv run python -m examples.huggingface.training_impact --real --steps 30 \
  --seeds 17 29 43 --engines torch_cpu --capture cpu.npz --output cpu.json
ONNXRUNTIME_EP_MLX_NO_COMPILE=1 uv run python -m examples.huggingface.training_impact \
  --real --steps 30 --seeds 17 29 43 --engines mlx \
  --library /path/to/libonnxruntime_mlx_ep.dylib --capture mlx.npz --output mlx.json
uv run python -m examples.huggingface.training_quality_compare \
  --reports cpu.json mlx.json --captures cpu.npz mlx.npz --output comparison.json
```

The comparison verifies matching model/data/precision/optimizer settings, source
weights, initial adapters, schedules and snapshot checksums. It reports every
seed's validation-curve gap, gradient direction and parameter difference. Replacing
`mlx` with `torch_cuda` selects a CUDA component on a CUDA-equipped host.
