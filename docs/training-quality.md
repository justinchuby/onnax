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

## No Robots instruction-data comparison

The pinned No Robots split uses 128 training sequences and 32 official-test
sequences. Three seeds each train Qwen3-0.6B NF4 adapters for 30 updates, with
FP32 projections, microbatch one and four accumulation steps.

| Engine | Initial held-out loss | Mean final held-out loss | Seed standard deviation |
| --- | ---: | ---: | ---: |
| PyTorch CPU | 3.411631 | 2.666917 | 0.013719 |
| ONNX / MLX | 3.411631 | 2.666826 | 0.013719 |

Held-out loss decreases at every sampled checkpoint for all three seeds.
The largest paired loss gap across checkpoints is **0.000255**; the minimum
sampled trajectory-gradient cosine similarity is **0.999945**. This FP32 run
shows closely matched training outcomes on instruction data. CUDA was
unavailable on the M1 Max host.

The {download}`FP32 report
<../examples/huggingface/reports/qwen3-no-robots-nf4-fp32-training-quality-m1max.json>`
records the matched settings, validation curves, adapter comparisons, source
hashes and process journals. The [corpus preparation](pinned-instruction-corpus)
records the dataset revision and selected examples. Memory observations cover
construction, training, evaluation and host snapshots.

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
tiny workload. CUDA was unavailable on the recorded Mac. The real-model comparison follows below; CUDA remains a separate measurement.

The {download}`recorded report <../examples/huggingface/reports/tiny-nf4-bf16-training-impact-m1max.json>`
contains every seed, training loss, validation curve, gradient norm and paired
snapshot comparison, with source hashes, versions and the MLX library digest.

## Recorded Qwen3-0.6B comparison

Three seeds ran 30 updates with the same NF4/BF16 policy and optimizer settings.
The two processes used identical base weights, initial adapters and data schedules.

| Seed | CPU validation loss, step 10 | MLX validation loss, step 10 | CPU final validation loss | MLX final validation loss |
| --- | ---: | ---: | ---: | ---: |
| 17 | 5.217939 | 5.222630 | 6.893252 | 6.797369 |
| 29 | 5.325381 | 5.312928 | 7.161592 | 7.586224 |
| 43 | 5.252580 | 5.280543 | 7.578716 | 7.768860 |

Final held-out loss averages **7.211187** on PyTorch CPU and **7.384151** on MLX,
with seed standard deviations of 0.345413 and 0.516308. The largest paired final
loss gap is **0.424632**. At step 10, the largest gap is 0.027963. Both runtimes
start near 6.749 and reach their lowest sampled held-out loss at step 10. Training
loss continues down to 1.35–1.38 while held-out loss rises over the remaining
updates: this 18-line training split overfits the six-line held-out split.

The independently evolving trajectories diverge more than in the tiny experiment.
Gradient cosine similarities start above 0.9997; by step 30 they are 0.892, 0.877
and -0.076 for seeds 17, 29 and 43. These snapshots evaluate different parameter
states. The [fixed-primal audit](qwen-runtime-audit.md) compares derivatives at a
shared state. The shared-parameter replay below separates evaluation differences from
trajectory divergence.

All frozen-base checks pass. Both runs exit successfully. CUDA was unavailable
on this host. The {download}`full comparison report
<../examples/huggingface/reports/qwen3-nf4-bf16-training-quality-m1max.json>` records
every validation checkpoint, paired gradient and parameter metric, source and
capture hashes, and process memory observations. Its memory observations include
model construction, evaluation and host snapshots; use the dedicated benchmark
for step timing and throughput.

## Recorded FP32 control

The FP32 control keeps the same three seeds, data, adapter initialization,
schedules and Adam settings, and uses FP32 projection operands and results.
Both engines complete 30 updates and preserve their frozen weights.

| Projection precision | CPU final held-out mean ± seed std | MLX final held-out mean ± seed std | Largest paired loss gap |
| --- | ---: | ---: | ---: |
| FP32 | 7.310223 ± 0.494285 | 7.310187 ± 0.494544 | 0.000327 |
| BF16 | 7.211187 ± 0.345413 | 7.384151 ± 0.516308 | 0.424632 |

```{figure} _static/qwen-precision-quality.svg
:alt: FP32 and BF16 held-out token loss over 30 updates, comparing CPU and MLX across three seeds.

Mean validation curves on the 18/6 local corpus. Shaded bands show the sample
standard deviation across three seeds.
```

Every sampled FP32 validation gap stays below 0.000327. Sampled trajectory
gradients have a minimum cosine similarity of **0.999991**. Final adapter vectors
have relative L2 differences of 0.000183–0.000756; individual entries differ by
up to 0.002638. These parameter differences coexist with closely aligned
validation curves. Both FP32 curves also reach their lowest sampled held-out
loss at step 10 and then overfit the small corpus.

The BF16 experiment shows substantially greater cross-runtime trajectory
variation on this workload. Use FP32 as the comparison baseline when developing
or testing a training path, then evaluate BF16 with the same data and optimizer.
The {download}`FP32 report
<../examples/huggingface/reports/qwen3-nf4-fp32-training-quality-m1max.json>`
contains all paired losses, gradients, parameter metrics, execution journals and
source/capture hashes. The CPU/CUDA baseline remains a measurement for a
CUDA-equipped host.

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
weights, initial adapters, schedules and snapshot checksums. It also requires
ORT optimization to be disabled, the highest FP32 MatMul precision, FP32 CUDA
BF16 partial reductions and every scheduled validation checkpoint. It reports every
seed's validation-curve gap, gradient direction and parameter difference. Replacing
`mlx` with `torch_cuda` selects a CUDA component on a CUDA-equipped host.

## Replay both adapters at a shared state

Cross-evaluate the final adapters from both training paths on each engine:

```bash
uv run python -m examples.huggingface.training_quality_replay \
  --reports cpu.json mlx.json --captures cpu.npz mlx.npz \
  --engine torch_cpu --capture replay-cpu.npz --output replay-cpu.json
ONNXRUNTIME_EP_MLX_NO_COMPILE=1 uv run python -m examples.huggingface.training_quality_replay \
  --reports cpu.json mlx.json --captures cpu.npz mlx.npz \
  --engine mlx --library /path/to/libonnxruntime_mlx_ep.dylib \
  --capture replay-mlx.npz --output replay-mlx.json
uv run python -m examples.huggingface.training_quality_replay_compare \
  --reports replay-cpu.json replay-mlx.json \
  --captures replay-cpu.npz replay-mlx.npz --output shared-states.json
```

Each process reuses one model and evaluates every saved final adapter. The report
records its parameter checksum, held-out loss and gradient on the final scheduled
training batch. Gradients use the saved post-update parameters. The comparison
requires matching parameter checksums, data, source weights and gradient scope.

Same-engine replay checks whether the saved adapters reproduce their recorded
validation losses. Cross-engine replay holds the adapters fixed while comparing
loss and gradient direction. Evaluating both trained adapters on one engine then
shows the effect of their different parameter trajectories.

### Recorded shared-state replay

All six final adapter states were evaluated on both CPU and MLX. Same-engine
replay reproduces every saved validation loss exactly. At shared parameters,
the largest CPU/MLX held-out loss gap is **0.014166**. For seed 29, evaluating
both trained adapters on CPU retains a loss gap of **0.410466**, compared with
0.424632 when each adapter is evaluated on its training runtime. Most of this
seed's final quality gap therefore persists under a common evaluator.

Shared-state gradient cosine similarities range from **0.988867 to 0.999676**;
relative L2 differences range from **0.025454 to 0.151191**. These common-parameter
gradients differ less in direction than the independently evolving step-30
trajectories. The FP32 control above keeps the data and optimizer fixed and produces
closely aligned validation curves across the two runtimes.

The {download}`shared-state report
<../examples/huggingface/reports/qwen3-nf4-bf16-shared-state-replay-m1max.json>`
contains every paired metric, component result, parameter/capture checksum and
execution journal. Both processes completed successfully, preserved frozen
weights and used matching computation-source hashes.


## Trace a strict five-step parameter difference

The tiny FP32-storage BF16 test first fails at step five for entry `[40, 1]`
of `model.layers.0.self_attn.q_proj.lora.A`, with a parameter gap of 2.51e-6.
Using the pairwise reference head preserves this result. All preceding loss and
gradient checks pass at `rtol=1e-4, atol=1e-6`.

Replaying each side's five captured gradients through a scalar AdamW reproduces
both full-tensor parameter updates exactly, at the original learning rate and
epsilon. The trace separates the optimizer's response to its gradient sequence
from differences between the two training trajectories. The
{download}`coordinate trace
<../examples/huggingface/reports/tiny-bf16-pairwise-adam-trace-m1max.json>` records
the gradients, moments, scalar replays, source hashes and the original failed
parameter check.

## Compare your own corpus

Use a UTF-8 file with one sequence per nonempty line. `--train-lines` selects the
first N sequences for training; the remainder forms the held-out split:

```bash
uv run python -m examples.huggingface.training_impact --real \
  --projection-precision fp32 --corpus my-corpus.txt --train-lines 128 \
  --engines torch_cpu --seeds 17 29 43 --steps 30 \
  --capture cpu.npz --output cpu.json
```

Run the other engine with the same corpus, split and training settings, then
compare the report/capture pairs as above. Sequences are truncated to 64 tokens.
The real model uses its pinned Qwen tokenizer; a custom tiny run builds its
local tokenizer from the training portion.

Pass `--corpus my-corpus.txt` to shared-state replay as well. It restores the
saved split and checks the normalized corpus and tokenizer identities before
building the execution model. The tokenizer identity includes its vocabulary,
merges, normalization, preprocessing and special-token configuration.


(pinned-instruction-corpus)=
### Prepare a pinned instruction corpus

[No Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) provides human-written
instructions and answers under CC BY-NC 4.0. This preparation command selects
128 training and 32 test examples from its official splits at revision
`e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b`:

```bash
uv run python -m examples.huggingface.quality_corpus   --output no-robots.txt --manifest no-robots.json
```

It selects unique single-turn user/assistant pairs in a fixed shuffled order.
Prompts occupy at most 32 tokens, leaving room for the answer in the 64-token
training sequence. The text uses `User: … Assistant: …`; loss includes both
parts. The manifest records source split, sample ID, retained token count,
corpus/tokenizer hashes and package versions. The downloaded text stays in the
chosen output file.

Use `--corpus no-robots.txt --train-lines 128` for training and pass the same
file to replay. The {download}`recorded selection
<../examples/huggingface/reports/qwen3-no-robots-quality-corpus.json>` fixes the
160 sample IDs used by the extended experiment.


## Plot recorded comparisons

The plotting script reads the saved JSON reports in an isolated environment:

```bash
uv run --isolated --no-project --with matplotlib==3.11.2 python tools/plot_training_quality.py \
  examples/huggingface/reports/qwen3-nf4-fp32-training-quality-m1max.json \
  examples/huggingface/reports/qwen3-nf4-bf16-training-quality-m1max.json \
  --output quality.svg
```

It checks matching data, optimizer settings and seeds before plotting the means
and seed standard deviations. Matplotlib is a plotting dependency; the training
runtime uses its existing environment.
