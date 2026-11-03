# Compact saved-shape boundaries on a real UNet

This experiment measures paired forward/vector-VJP execution of the pinned
pretrained UNet on one cached 512-pixel image latent, with FP32 parameters and
fixed inputs. Both schedules use fused attention pullbacks, broadcast adjoints
and shape-only saved-value compression. They differ in the broadcast contract:
`SumToShape_v1` accepts a full reference activation; `SumToShape_v2` accepts its
INT64 shape vector. The portable function bodies preserve the same reduction.

Each independent worker warms up 10 times and measures 100 paired calls, using
the same binary, input identities, thread settings and 512 MiB MLX cache limit.
All 72 parameter gradients match the whole-VJP control at
`rtol=1e-4, atol=1e-6`, before and after measurement. Inputs remain unchanged;
all three ONNX graphs in each bundle pass the full checker.

| Quantity | Reference tensor | Shape vector |
|---|---:|---:|
| Retained boundary arrays | 1.465 GB | 0.744 GB |
| Forward median | 338.01 ms | 239.00 ms |
| Backward median | 268.91 ms | 267.57 ms |
| Paired median / P95 | 606.99 / 620.28 ms | 507.65 / 519.69 ms |
| Window MLX peak allocation | 4.867 GB | 4.146 GB |
| Whole-process physical peak | 6.439 GB | 5.860 GB |

The compact contract reduces retained host arrays by 49.2% and paired median
execution time by 16.4%. Shape compression handles 250 saved tensors with the
compact contract, compared with 57 when reference tensors are hidden inside
local-function calls. Forward timing includes host output materialization and
boundary exchange. The workload uses a fixed cotangent derived from MSE;
optimizer updates and task-loss computation are outside these timings.

Allocator peaks are reset after warmup. Whole-process physical peaks include
source graphs, whole-VJP control execution, session setup and runtime caches.
ORT graph optimization and CPU fallback are disabled. Each report records
source identity, input content hashes, graph and library checksums, versions,
retained byte counts and the complete shape-compression plan.

The recorded measurements use driver commit `fca005d` and native source
`c7fa4e4`. The current CLI reads only the pinned scheduler configuration during
setup, eliminating the earlier unused PyTorch UNet weight load.
