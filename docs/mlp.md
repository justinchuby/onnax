# Train a small ONNX classifier

This P1 workload trains a 2 → 16 → 3 MLP with Tanh activation and four explicit
FP32 parameter inputs. It is an offline, fixed-seed three-cluster classification
task, not an accuracy benchmark on a public dataset.

```shell
uv sync --group dev --group training
uv run --no-sync python -m examples.mlp --output-dir mlp_models
uv run --no-sync pytest tests/test_mlp_training.py
```

## Stable loss and derivatives

For one-hot targets, the model computes

$$
H=\tanh(XW_1+b_1),\quad Z=HW_2+b_2,\quad
L=-\frac1N\sum_{n,c}Y_{nc}\operatorname{LogSoftmax}(Z_n)_c.
$$

The graph uses LogSoftmax directly rather than taking the logarithm of an
exponentially computed probability. A class reduction is a sum; the batch
reduction is a mean. A regression verifies finite loss and gradients for logits
[-1000, 0, 1000], and invariance when adding 10000 to every class logit.

The independent NumPy reference subtracts each row's maximum before computing
log-sum-exp and implements analytic backpropagation. Central finite differences
check a coordinate of every parameter. During 150 full-batch updates, the ONNX
prediction, loss, all four gradients and evolving parameters must agree with an
independently evolving reference at rtol=1e-4, atol=1e-6. The external Array API
Adam is reused from the LoRA milestone; the reference update is a separately
written flattened NumPy calculation.

## Export and external updates

The CLI writes loss.onnx, gradient.onnx and an ONNXScript-optimized gradient copy.
interface.json records actual parameter/gradient names, shapes, FP32 dtype and
loss reduction. There are no parameter initializer defaults. The CPU loop keeps
one session and supplies current weights and biases on every call.

Final parameters are saved by example-owned NumPy storage and restored into a
reloaded loss module. Both original and optimized gradient artifacts are checked
at batch sizes 1, 7 and 192 against the reference. All exported modules pass the
full ONNX checker and run with ORT graph optimization disabled.

Both training and validation cross-entropy must decrease by at least 50%, and
validation accuracy must reach 95%. A development run decreased training loss
from 1.083 to 0.00125 and validation loss from 1.076 to 0.00210, with 100% accuracy
on the 192 held-out synthetic examples. These easy-cluster results do not imply
performance on image, text or real-world classification datasets.

## Measurement boundaries

report.json contains all loss values and an isolated benchmark-process report.
The benchmark warms up for 10 updates and measures 100 updates using a single
CPU intra/inter-op thread. It reports median/P95 gradient-call, external Adam
and complete-step times, session creation and peak process RSS. Gradient-call
time includes feed preparation, runtime invocation and returned CPU gradient
arrays; only gradient outputs are requested. Peak RSS includes imports and the
runtime, not just tensors. Measurements exclude the acceptance reference loop.
The Unix/macOS resource API is used for RSS, as in the LoRA example.

No GPU training, zero-copy exchange, mixed precision, or speedup over another
framework is claimed. Data loading, optimizer state and parameter storage remain
outside onnax. The next workload priority is matrix-free parameter fitting in
the [roadmap](roadmap.md).
