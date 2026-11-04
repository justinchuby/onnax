# Output ownership checks in the saved-activation bridge

The real pretrained UNet exposes 505 NumPy outputs at its forward boundary.
The previous ownership guard compared every output with every feed and sibling
output. An interval sweep now excludes disjoint byte ranges before querying
exact NumPy overlap. Strided views, read-only outputs and shared buffers retain
the same copy decisions; small result lists keep the direct checks.

These independent Apple M1 Max 32 GB workers use the same FP32 model, image
batch, adapters, native MLX binary and benchmark driver. Each warms up ten times
and measures 100 fixed-parameter steps: forward, external MSE, backward and
AdamW with learning rate zero. Session construction and whole-VJP validation
happen before timing. Task-owned GPU replay sessions were closed before both runs.

| Measurement | Pairwise guard | Interval sweep |
| --- | ---: | ---: |
| Forward median | 288.45 ms | 242.98 ms |
| Backward median | 272.29 ms | 272.98 ms |
| Complete step median | 562.44 ms | 516.64 ms |
| Complete step P95 | 574.68 ms | 576.40 ms |
| Window MLX peak | 4.146 GB | 4.146 GB |
| Process physical-footprint peak | 5.748 GB | 5.679 GB |

The complete-step median decreases by 8.1%. Graph, input and all 72 adapter-gradient
hashes match exactly. Both workers check the gradients against the whole-VJP
control before and after timing, validate all three graphs with the full ONNX
checker, and verify that parameters remain byte-identical. The physical peak
includes setup and the control execution; MLX peak counters reset after warmup.

## Reproduce

Use the baseline core at `f079de71be66ffe615770924bd29a163c7c2dbdd`
and the candidate checkout. Run the **same candidate benchmark driver** from
each checkout so both reports record the same driver hash. Supply the existing
pretrained UNet export, cached image batch and whole-VJP artifacts:

```sh
HF_HUB_OFFLINE=1 PYTHONPATH=src:. \
ONNXRUNTIME_EP_MLX_NO_COMPILE=0 \
ONNXRUNTIME_EP_MLX_CACHE_LIMIT_BYTES=536870912 \
python -m examples.huggingface.guard --output-dir ownership-worker -- \
  python /path/to/candidate/tools/benchmark_saved_bridge.py \
  --source-dir /tmp/onnax-unet-pretrained-qualification-v2 \
  --batch-file /tmp/onnax-unet-real-images-512/example-0.pt \
  --golden-path /tmp/onnax-unet-saved-vector-shapes-measured/joint.onnx \
  --library /path/to/libonnxruntime_mlx_ep.dylib \
  --warmup 10 --samples 100 --output ownership.json
```

Use fresh output paths for each worker. ORT graph optimization and CPU fallback
are disabled. `baseline.json` and `candidate.json` include the raw timings,
software versions, hashes, byte counts and allocator readings;
`comparison.json` records the matched fields and paired results.
