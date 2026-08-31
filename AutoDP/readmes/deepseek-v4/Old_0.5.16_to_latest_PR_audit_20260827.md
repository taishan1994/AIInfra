# 旧版 0.5.16 高性能实现向新版 SGLang 的 PR 移植核对

日期：2026-08-27

对比对象：

- 旧版：`/data/ssd2/sglang_v0.5.16_pr32281_fix7_repro_20260823`
- 新版：`/data/ssd2/gongoubo/sglang_pr/sglang_pd`
- 旧版高性能配置（以实际服务日志中的 `server_args` 为准）：Prefill `MegaMoE + moe_runner_backend=auto`，Decode `MegaMoE + deep_gemm + DSpark`

> 重要口径修正：本报告后面的“旧版高结果”指
> `dspark_stepwise_ablation_20260824/variants/dspark_deepgemm/` 这一组结果。
> 实际服务日志 `logs/services/prefill/prefill_20260824_131654_pid4117672.log` 和
> `logs/services/decode/decode_20260824_131656_pid4117768.log` 中记录的启动参数为：
> Prefill `moe_a2a_backend='megamoe'`、`moe_runner_backend='auto'`；Decode
> `moe_a2a_backend='megamoe'`、`moe_runner_backend='deep_gemm'`。因此不能将这组
> 旧版结果标注为“Prefill flashinfer_mxfp4”。

## 结论摘要

旧版 26 个修改文件不能整体移植。新版已经吸收了 DSpark embedding/KV projection、DeepSeek-V4 W4A4 MegaMoE、共享专家融合基础实现以及大量 PD/TBO 修复。直接搬迁会与新版 MoE、配置解析和 CUDA Graph 架构冲突。

## 已经上游化、无需重复提 PR 的部分

| 旧版内容 | 新版证据 | 结论 |
|---|---|---|
| DSpark embedding 折叠到 draft graph | `8ae0eb83f`，新版 `dspark.py` 有 `forward_embed` | 已有 |
| DSpark 逐层 ctx KV projection 合并 | `024639a37`，新版有 stacked ctx KV 路径 | 已有 |
| DeepSeek-V4 W4A4 MegaMoE | `b7f856df7`、`3b5909de0` | 已有 |
| FlashInfer MXFP4 共享专家融合 | `4b4bf3d2a`，新版有量化重打包和单测 | 已有 |
| DSpark + MegaMoE + DP attention | `6eb941a34` | 已有 |
| DSpark draft H2D、verify 等修复 | `151a31482`、`f2c84de02` | 已有 |

## 共享专家融合验证

曾在新版 W4A4 MegaMoE 上临时让 `--enable-w4a4-mxfp4-megamoe` 自动请求 fusion。验证结果：

- Prefill/Decode 均打印 `Shared experts fusion optimization enabled.`；
- FP8 shared expert 成功量化并装入 FP4 fused MoE weights；
- `你是谁`：HTTP 200，`WHOAMI_VALID=True`；
- Smoke：10/10 成功；
- 性能没有显示出足够收益，因此自动强制 fusion 的改动已回退，不作为 PR。

新版原有 fusion 策略是 DeepSeek-V4 显式 opt-in，保守关闭是为了避免量化精度、EP expert slot 和 CUDA Graph shape 不兼容。

## 仍值得核对的候选

### 1. Mooncake 异构 KV stride / descriptor 对齐

旧版 `conn.py` 增加了源/目标 KV item length 分离、不同 buffer stride 以及 64 KiB descriptor 对齐。新版已经支持异构 TP 和多种 state layout，但主 KV envelope 路径仍保留严格 layout 校验，未完全等价覆盖旧版的 per-buffer stride 处理。

这是一个偏正确性和兼容性的候选 PR，适合补充单测和异构 TP/DSA KV transfer 用例；不能宣称为吞吐优化。

### 2. FlashInfer MXFP4 的 DeepEP low-latency / grouped 路径

旧版 `mxfp4_flashinfer_trtllm_moe.py` 新增了约 1000 行，包括 `flashinfer_grouped`、`flashinfer_cutlass`、DeepGEMM low-latency、DeepEP LL layout 转换和 FP8 scale 适配。

新版已把 MXFP4 主路径重构到 `mxfp4.py`，并通过统一 MoE runner 处理 DeepGEMM；但旧版的 `SGLANG_MXFP4_LL_BACKEND` 和 grouped/cutlass 可切换实现没有一一保留。它可以作为独立 PR 候选，但需要按新版 `Mxfp4MoEMethod` 拆分，不能直接复制旧文件。

注意：旧版高吞吐结果的 Prefill 是标准 `flashinfer_mxfp4`，Decode 是 `deep_gemm`，因此旧版 grouped/FlashInfer LL 分支并不是该结果的直接证据。

### 3. TBO + DSpark target-verify

旧版增加了 DFlash/DSpark verify metadata 对 TBO splitter 的兼容处理。但旧版高吞吐部署没有开启 TBO，所以该修改不能解释旧版高吞吐结果。只有在目标是“DSpark 与 TBO 同时可用”时，才值得单独整理为兼容性 PR。

## 当前验证结果

新版共享专家融合相关单测：`11 passed`，包括 fusion policy、DSpark fused slot loading 和 FP8→MXFP4 requant layout。

因此当前优先级为：

1. 先评估 Mooncake 异构 KV stride 是否需要补齐测试和边界修复；
2. 再把旧版 MXFP4 LL/grouped 代码按新版接口拆成小范围候选；
3. 不再重复移植已上游化的 DSpark stacking、W4A4 MegaMoE 和共享专家融合基础实现。

## 旧版高并发 / 新版低 TPOT 差异分析

旧版高结果与新版 W4A4 结果的关键对比：

| ISL/C | 旧版 Total tok/s | 新版 Total tok/s | 旧版 TTFT ms | 新版 TTFT ms | 旧版 TPOT ms | 新版 TPOT ms |
|---|---:|---:|---:|---:|---:|---:|
| 1024/C1 | 697.31 | 748.12 | 190.44 | 216.16 | 2.68 | 2.46 |
| 1024/C16 | 8032.88 | 8125.97 | 339.35 | 412.40 | 3.50 | 3.36 |
| 1024/C256 | 70621.19 | 72020.68 | 661.99 | 757.52 | 6.25 | 6.04 |
| 1024/C512 | 102278.66 | 99010.14 | 1116.16 | 1603.82 | 8.35 | 8.19 |
| 8192/C1 | 3251.79 | 3375.89 | 208.96 | 219.58 | 2.56 | 2.45 |
| 8192/C16 | 35928.66 | 37391.93 | 485.64 | 519.55 | 3.39 | 3.21 |
| 8192/C256 | 74394.51 | 73897.47 | 26214.34 | 26843.63 | 3.90 | 3.48 |
| 8192/C512 | 75017.53 | 74067.37 | 55827.19 | 56923.22 | 3.95 | 3.61 |

### 解释

TTFT 包含 Prefill 计算、PD/Mooncake KV transfer、Decode 请求接纳和排队等待；TPOT 主要反映首 token 之后的稳定 Decode 步耗时。新版 Decode kernel/Graph 的 TPOT 更低，说明稳态单 token 路径更快，但新版 Prefill 到 Decode 的首 token 链路更慢。

低并发时，首 token 固定开销不能被重叠或摊薄，所以新版 TTFT 增大直接降低 Req/s。高并发时 TTFT 可以部分摊薄，TPOT 优势开始体现；但 1024/C512 中新版 TTFT 仍比旧版高 487.66 ms，TPOT 只低 0.16 ms，最终新版 Req/s 和 Total tok/s 反而下降。

### 优先排查项

1. **先统一实验口径。** 旧版高结果实际是 Prefill MegaMoE/auto、Decode MegaMoE/deep_gemm、DSpark、Graph128、chunked-prefill=16384/256，且旧版 `enable_w4a4_mxfp4_megamoe=False`；新版 W4A4 结果是在相同主干配置上额外开启 `--enable-w4a4-mxfp4-megamoe`。因此当前表格不是“旧版 flashinfer_mxfp4 对新版 W4A4”，而是“旧版 MegaMoE/auto + deep_gemm 对新版 W4A4 MegaMoE + deep_gemm”。后续若要把收益归因于新版源码或 W4A4，必须补做新版关闭 W4A4 的 A/B。
2. **拆 TTFT 时间线。** 分别记录 Prefill forward、KV transfer、Decode bootstrap、首个 decode forward 和 scheduler queue wait；只看平均 TTFT 无法定位瓶颈。
3. **核对旧版 PD hidden transfer 改动。** 旧版有 hidden pool/queue、streaming transfer、Mooncake descriptor 和异步释放相关改动；这些更可能解释 TTFT，而不是 TPOT。新版应优先做 tracing/计时移植，不应直接搬整个 `conn.py`。
4. **核对 routed scaling 是否被推迟到输出端。** 旧版 MXFP4 TRT-LLM quant method 声明 `fuse_routed_scaling_factor_in_topk = True`，新版主 `Mxfp4MoEMethod` 没有同名声明；在 MegaMoE 路径上新版可能额外执行一次输出缩放。该差异是一个值得做 micro-benchmark 和数值单测的候选，但尚未证明一定是 TTFT/TPOT 差异来源。
5. **检查新版 W4A4/FP8 shared-expert 处理。** 新版可能在模型加载时把 FP8 shared expert 重新量化为 FP4；这会影响加载而非稳态 TPOT，但若改变了 Prefill 首批 warmup 或 workspace 初始化，也可能抬高早期 TTFT。

当前最有希望形成 PR 的不是“强制开启 shared fusion”，而是：补齐 PD/Decode 首 token 分段 tracing，并针对已确认的 Mooncake per-buffer stride 或 routed-scale 融合差异提交小范围修复。

## 2026-08-27：候选 PR 验证——CUDA Graph warmup 捕获边界同步

旧版 `full_cuda_graph_backend.py` 在两次 warmup 完成后、创建
`torch.cuda.CUDAGraph()` 前额外执行一次 device synchronize 和 TP barrier；当前新版原代码
只有每轮 warmup 开始前的同步。这个差异的作用是建立最后一次 warmup 到 graph capture
之间的明确边界，避免异步首用/JIT kernel 或 launch metadata 初始化仍在进行时进入捕获，
并避免多 TP rank 进入 capture 的时序不一致。

本次按新版接口移植为最小 7 行改动，未改变模型、MoE、DSpark 或调度行为：

```text
python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py
```

验证：

```text
PYTHONPATH=.../sglang_pd/python python3 -m pytest -q \
  test/registered/unit/model_executor/runner_backend/test_full_cuda_graph_backend.py
4 passed, 15 warnings（单独运行）；随后与 graph-pool 相邻单测合计 `10 passed, 15 warnings`
git diff --check                 # passed
```

CPU mock 单测确认调用顺序的数量为：两次 warmup 前各一次同步/屏障，加上 capture 前的
最后一次同步/屏障，共 3 次；实际 CUDA 多卡收益尚未宣称，仍需在提 PR 前做一次单卡/多卡
graph capture 回归验证。修改前文件已备份到：
`backups/latest_pr_candidate_graph_sync_20260827/full_cuda_graph_backend.py`。

该候选属于 CUDA Graph 捕获正确性/稳定性 PR，而不是已证明的吞吐优化 PR。与旧版的
DeepSeek-V4 专用 decode-TBO、Waterfill、LPLB 和 MXFP4 grouped 大改动相比，它耦合面最小，
也不依赖本轮不稳定的性能结果，建议优先整理成独立 PR。

## 旧版其它改动的当前结论

| 旧版改动 | 当前新版核对结果 | 是否建议直接移植 |
|---|---|---|
| DSpark embedding 折叠、逐层 ctx-KV projection 合并 | 新版已有对应实现和 parity 单测 | 否，已上游化 |
| DSpark + DP attention / MegaMoE 入口放开 | 新版已允许 `moe_a2a_backend='megamoe'`，但要求 static ragged verify | 否，旧版“完全取消限制”不能直接搬 |
| DeepSeek-V4 decode TBO | 新版仍明确只实现 V4 prefill TBO；旧版注释也说明 decode TBO 曾回退/需要 graph capture 工作 | 暂不移植；缺少独立正确性与收益证据 |
| Waterfill/LPLB 低并发 fallback | 属于特定 workload 策略，不是通用正确性修复 | 不作为当前 PR |
| MXFP4 grouped/DeepEP LL 大文件 | 新版 MXFP4 已重构到统一 `mxfp4.py`，旧接口不能直接套用 | 需拆成独立 kernel/layout PR |
| Mooncake per-buffer KV stride | 新版仍是 scalar `dst_kv_item_len` 主协议；但旧版新增字段位置与新版 staging/DCP 字段冲突 | 有价值，但必须重设计协议后再提 |
| 最后 warmup 后 CUDA Graph 边界同步 | 新版缺少，已完成最小实现和 CPU mock 验证 | 当前首选候选 |

特别注意：旧版 Mooncake 补丁的 `msg[14]` 在旧协议中可作为新增字段，但新版已经使用
`msg[14]`/`msg[15]` 表示 staging、`msg[16]`/`msg[17]` 表示 DCP。若直接 cherry-pick，
decode 会把 staging 指针误当作 KV stride 数组，属于协议级错误。正确做法应是把新 vector
追加到当前协议末尾（例如 `msg[18]`），同时保留缺失字段时的 scalar fallback，并为旧/新
peer rolling restart 增加 wire compatibility 单测；在完成这一步前不应提交旧版 `conn.py`
整文件。

## 2026-08-27：Mooncake stride 候选的协议级验证

按上述新版协议重新实现了一个候选 patch（当前工作树未提交）：

- `KVArgsRegisterInfo` 从新增的 `msg[18]` 解析每个目标 KV buffer 的 stride；没有该字段时
  自动将旧的 `dst_kv_item_len` 扩展到每个目标 pointer，兼容旧 peer。
- Decode 注册端在现有 staging/DCP 字段之后追加 packed `uint64` stride vector，不改变
  `msg[14:18]` 的含义。
- 通用 KV transfer 将源 stride 用于源地址偏移和传输长度，将目标 stride 仅用于目标地址
  偏移；MHA 的 K/V 拆分、MLA 的 PP 映射和 layer-id 映射均沿用新版 helper。
- DCP relayout 暂时显式拒绝异构目标 stride，避免 DCP 仍按旧 scalar stride 写入而产生
  静默数据损坏；等独立 DCP token-layout 设计完成后再扩展。

新增的 CPU 单测覆盖：

```text
test_mooncake_registration_staging_fields                 # legacy scalar fallback
test_mooncake_registration_accepts_appended_per_buffer_strides
test_mooncake_registration_rejects_stride_pointer_mismatch
test_mooncake_transfer_uses_independent_source_and_destination_strides
```

与 CUDA Graph 候选联合运行结果：`32 passed, 15 warnings, 7 subtests passed`，且
`git diff --check` 通过。Mooncake 原文件备份在：
`backups/latest_pr_candidate_mooncake_stride_20260827/conn.py`。

这证明了 wire 兼容和 CPU 侧地址规划，但还不能证明真实 Mooncake/RDMA 在异构 page stride
下可用；提交前仍需用 staging 开关、不同 attention TP/PP 和 DSV4 indexer KV 做最小多卡
transfer 回归。若该回归通过，它可以作为独立的“heterogeneous KV buffer stride” PR；
如果只需要统一 KV envelope，则应关闭该候选，避免扩大协议面。

## 2026-08-27：当前候选 patch 相同条件 8 组回归

为确认候选 patch 不影响完整 PD 推理链路，使用新版源码工作树
`/data/ssd2/gongoubo/sglang_pr/sglang_pd`，并保持上一轮相同配置：Prefill/Decode
均为 MegaMoE，Prefill `moe-runner-backend=auto`，Decode
`moe-runner-backend=deep_gemm`，Decode 开启 DSpark、CUDA Graph batch
`1 2 4 8 16 32 64 128` 和 `--enable-w4a4-mxfp4-megamoe`，PD 使用 Mooncake。
本轮未开启 FP4 indexer。启动脚本、三端服务日志、校验结果和 benchmark 原始结果均保存在：

`logs/runs/pr_candidates_same_condition_full8_20260827/`

“你是谁”校验为 HTTP 200，语义校验通过；8/8 组 benchmark 均成功，所有请求完成：

| ISL | OSL | Concurrency | Req/s | Out tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 0.37 | 375.14 | 750.27 | 199.99 | 2.47 |
| 1024 | 1024 | 16 | 3.98 | 4080.32 | 8160.65 | 348.12 | 3.42 |
| 1024 | 1024 | 256 | 34.44 | 35262.58 | 70525.15 | 680.54 | 6.19 |
| 1024 | 1024 | 512 | 47.98 | 49135.88 | 98271.77 | 1412.76 | 8.47 |
| 8192 | 1024 | 1 | 0.37 | 376.71 | 3390.36 | 211.34 | 2.45 |
| 8192 | 1024 | 16 | 4.01 | 4106.97 | 36962.75 | 523.30 | 3.24 |
| 8192 | 1024 | 256 | 8.09 | 8281.94 | 74537.43 | 26342.80 | 3.70 |
| 8192 | 1024 | 512 | 8.13 | 8326.94 | 74942.45 | 56040.83 | 3.78 |

Decode 日志在服务刚启动时记录了 `post-warmup freeze_gc` 的连接竞态 traceback，原因是
后台 freeze 请求早于 Uvicorn 监听完成；随后 Decode 正常启动，所有正式请求均返回 200，未
发现 OOM、CUDA error、运行时异常或请求失败。因此该现象属于启动日志噪声，仍建议后续单独
修复启动 warmup 的时序，避免在 PR 回归日志中造成误判。
