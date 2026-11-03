# ONNX Runtime Training：给 onnax 的设计交接

日期：2026-11-02。本文来自 side conversation，供主 session 评估和安排后续工作。
这次交付仅新增本文，没有修改实现、现有路线或主 session 的工作。

## 结论

最值得借鉴的是训练图的保存值分析、选择性重算、梯度生命周期，以及
前向与反向配套的融合设计。这些能力属于 onnax 的微分图生成与组合范围；
训练循环、优化器、scheduler 和 checkpoint 存储继续由外部工具管理。

建议先完成一个小而完整的改进：在现有分块 LM head 前收集有效标签位置，
减少冻结投影的计算。随后把现有 checkpoint 和 activation-liveness 工具连接起来，
建立可测量的保存值与重算规划。

## 已检查的项目基础

- [通用 checkpoint](checkpointing.md)：已有按命名边界分块的 VJP、共享参数梯度累加、
  session 复用和导出接口。`src/onnax/_checkpoint.py` 保留跨块的所有边，
  目前对 flat graph 分区。
- [激活生命周期](activation-liveness.md)：已有逻辑 live-set 分析和独立进程测量。
  现有分类器实验也说明：缩短一个张量的生命周期，未必降低峰值 live set。
- [融合算子求导](fused-autodiff.md)：已有 Attention、normalization、RoPE 和 Swish
  的直接求导规则。后续重点是保存值接口与执行融合。
- `examples/huggingface/chunked_loss.py`：已有分块冻结 head、loss sum 和 hidden-state
  梯度。检查时，`_ChunkLoss.forward` 在 causal shift 后遍历全部行，
  包括标签为 `-100` 的位置；可以在投影前收集有效行。

以上是本次定向阅读的观察，主 session 实施前应按当前代码再次确认。

## 主 session 对照结果与当前排期

已按当前实现核对：`chunked_loss.py` 在 causal shift 后收集有效标签位置，
反向散回完整 hidden 梯度，全忽略标签跳过 head。该路径已接入
Qwen3、Qwen3.5 和 Phi-3。保存值分析、嵌套 Scan live-set 报告以及
MLX block 运行时测量也已具备；共享冻结权重和通用 bounded head 已交付。

完整 4k Qwen3.5 已完成保存值、MLP 重算与整块重算对照；shape-only 边界将
425 MiB 激活替换为 384 字节的 shape 元数据。整块重算保持当前最低内存策略。
256 维 attention VJP 已通过真实训练及新进程恢复验收，PR #152 等待 CI 合并。

本轮先缩短 checkpoint 激活的生命周期：删除 VJP 未使用的输入，只保存实际需要的
张量，并验证下一块执行前的释放时点、skip connection 和导出恢复。随后推进
预训练 Diffusers denoiser，将保存值和融合接口用于第二类真实工作负载。
具体排期见{ref}`训练图与内存路线 <training-graph-memory-priorities>`。

下文保留最初的设计建议，实施状态以上述对照和路线图为准。

## 1. 有效 token 的分块 head

ORT Training 提供 label sparsity 优化开关，体现了利用监督位置稀疏性减少计算的方向。
这里建议的 gather/scatter 是针对 onnax 现有实现的方案，并非对 ORT 具体算法的复述。
参见 [ORT label sparsity 配置][ort-guidelines]。

设 causal shift 后标签为 t，hidden rows 为 H，有效位置集合为

\[
I=\{i\mid t_i\ne -100\}.
\]

收集 H[I] 与 t[I]，仅对这些行分块计算冻结 lm_head、稳定交叉熵与 hidden 梯度，
再把梯度 scatter 回完整序列，其余位置为零。主体仍处理完整上下文，
所以这些位置可以通过 attention 等路径收到来自后续有效 token 的梯度。

实施位置先放在现有示例的 head 调度层，复用已有 ONNX session 和稳定 loss。
保留 loss sum；有效 token 数由现有训练接口用于归一化和梯度累加。

验收：

- 对齐完整 head 的 loss、hidden 梯度、全部 adapter 梯度与多步更新。
- 覆盖 causal shift、右 padding、prompt masking、全忽略标签、单个有效 token、
  分块尾块和不等长梯度累加。
- 全忽略标签返回零 loss sum 和零 hidden 梯度，跳过 head 执行。
- 固定输入序列与有效 token 数，记录不同监督密度下的 head 执行、收集/散回、
  完整训练步 median/P95 与独立进程峰值内存。
- 分别覆盖 CPU FP32 和当前 MLX 路径；记录运行时边界的数据交换成本。

## 2. 保存值分析与选择性重算

ORT Training 会识别可重算子图，以减少需要保存的激活；其配置还区分不同重算级别。
参见 [ORT 重算配置][ort-guidelines]。

onnax 已有分块重算，下一步可以细化到块内部：

1. 从 VJP 实际使用的前向值计算保存值集合，分开记录参数、原始输入和激活。
2. 利用现有 live-set 分析，找出训练执行中真正形成峰值的集合。
3. 对候选值比较保存字节数、重算子图、重算次数及实测耗时。
4. 输出明确的执行计划：保存哪些值、在哪里重算、最后一次使用后何时释放。

先选择一个 decoder block 验证，再扩展到多层 Qwen 和 diffusion。
重算规划需要处理跨块共享值、skip connection、动态 shape 和控制流。
对于随机或有状态操作，执行计划必须保留对应状态或保存原值，保证重算语义一致。

验收使用现有 checkpoint 基线，比较梯度、更新与恢复结果，同时记录逻辑 live set、
运行时分配峰值和进程 footprint。候选选择以峰值集合和实测成本为依据。

## 3. 生成梯度时选择计算形式

ORT 的 MatMul 梯度会根据 shape 和所需梯度分支选择实现。
例如二维权重梯度可以把批量维展平后使用 Gemm；无需的输入梯度分支可以省略。
参见 [gradient_builder.cc][ort-gradient] 的 `GetMatMulGradient`。

对 onnax 的应用：

- 检查活动梯度分析是否在生成阶段排除了无用分支，同时保持穿过冻结权重的激活梯度。
- 对真实模型中的 batched MatMul，比较展平/Gemm 与 MatMul/reduction 的执行成本。
- 对 LoRA 共享低秩反向乘积，保持低秩计算形式，复用公共结果。

先检查现有规则和 ONNXScript passes，针对实测图补充改写。
具体计算形式由数值对齐和 MLX 测量决定。

## 4. 前向与反向配套的融合接口

ORT Training 同时提供 attention fusion，以及逐元素/reduction 子图的融合执行路径。
参见 [ORT attention 与 Triton 配置][ort-guidelines]。

onnax 可以明确表达融合操作的前向结果、反向所需保存值和 VJP。
例如 normalization 的统计量、attention 的行统计量，以及递推操作的状态边界。
具有组合语义的实现可用 `pkg.onnax` local function 表达，由 ONNXScript 改写，
MLX EP 选择融合执行。

这一设计应衔接主 session 当前的融合优化工作。新增接口前，先核对现有规则与
运行时识别模式，统一保存值和 dtype 约定，并验证 VJP、JVP 及高阶组合语义。

## 5. 梯度与临时张量的及时释放

ORT 的 memory-efficient gradient management 会在参数梯度生成后触发对应的
PyTorch backward，以缩短 ORT 梯度缓冲的生命周期。
参见 [ORT 梯度管理配置][ort-guidelines]。

我们的桥接层可以检查每块 VJP 完成后是否及时交还参数梯度，释放该块的
cotangent、重算激活和临时输出。共享参数的贡献仍由外部 autograd 正确累加。
LoRA 梯度较小，实施时优先关注大激活和隐藏状态梯度的持有者。

测试应覆盖共享参数、多输出、异常退出、连续训练与 session 复用。
通过对象生命周期和 allocator 数据定位额外持有，再修改释放策略。

## 建议交付顺序

| 顺序 | 可审查交付 | 主要验证 |
|---|---|---|
| 1 | 有效 token 收集与分块 head | 完整梯度、多步训练、不同监督密度测量 |
| 2 | VJP 保存值报告与峰值归因 | 单块/多块、共享值、逻辑与运行时内存 |
| 3 | 一个 block 的选择性重算 | 梯度与恢复一致、内存/耗时对照 |
| 4 | 融合算子的保存值接口 | 衔接现有 MLX kernel 与高阶求导 |
| 5 | 分块梯度交还与释放改进 | 共享参数与长时间训练的内存轨迹 |

具体排期由主 session 结合正在进行的 MLX 优化结果决定。
图优化继续使用 ONNXScript；数值验收关闭 ORT graph optimizer。
性能报告分开记录执行、交换与外部更新，并保持相同数据、精度和有效 token 数。

## 参考来源与阅读范围

- [ORTModule Training Guidelines][ort-guidelines]：本次查看了重算、梯度管理、
  label sparsity、attention fusion 与 Triton 配置说明。
- [ORT gradient_builder.cc][ort-gradient]：本次查看了 MatMul 的 shape 特化、
  所需梯度分支判断，以及若干直接梯度规则。
- [ONNX Runtime Training technical deep dive][ort-deep-dive]：训练图优化的架构背景。

本次没有逐行审计 ORT 的 memory optimizer 和 label sparsity pass 源码。
后续若采用具体算法，应补充源码阅读并固定引用 commit。

[ort-guidelines]: https://github.com/microsoft/onnxruntime/blob/main/docs/ORTModule_Training_Guidelines.md
[ort-gradient]: https://github.com/microsoft/onnxruntime/blob/main/orttraining/orttraining/core/graph/gradient_builder.cc
[ort-deep-dive]: https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/onnx-runtime-training-technical-deep-dive/1398310
