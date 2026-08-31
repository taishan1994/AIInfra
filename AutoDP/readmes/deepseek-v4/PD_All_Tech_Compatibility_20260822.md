# PD 分离全技术组合兼容性记录

## 当前目标

Prefill 使用 baseline + MegaMoE；Decode 组合 DeepEP 或 MegaMoE、DeepGEMM、DSpark、Waterfill、LPLB、TBO 和 FP4 indexer，并保留 DeepEP low_latency 与 Decode CUDA Graph。

## 当前源码审计结论

| 组合 | 当前结论 | 证据/限制 |
|---|---|---|
| DeepEP + DeepGEMM + Waterfill | 可组合 | 已有正式 PD 结果 |
| DeepEP + DeepGEMM + LPLB | 可组合 | 已完成 DeepSeek-V4 适配和 baseline 对比 |
| DeepEP + DSpark | 可组合 | 已完成 8 组 PD 验证 |
| DeepEP + FP4 indexer | 参数路径可组合 | 需要 SM100，需真实服务验证 |
| DeepEP + TBO + Decode | 当前不支持 | `operations_strategy.py` 明确对 DeepSeek-V4 Decode 抛出 `NotImplementedError` |
| DSpark + TBO | 当前不支持完整热路径 | DeepSeek-V4 会因 DSpark per-layer hidden capture 主动跳过 TBO |
| MegaMoE + LPLB | 代码层面可走标准 TopK/A2A 路径 | 尚未完成本机端到端验证 |
| MegaMoE + TBO + Decode | 未验证且不应直接假设可用 | 当前 DSV4 TBO Decode 实现缺失 |

## 新增入口

- Prefill：`flash_prefill_all_megamoe.sh`
- Decode 最大组合探针：`flash_decode_all_tech_deepep.sh`

两个入口都单独记录服务日志，并固定原生 `/data/ssd2/sglang_v0.5.16/python`，避免 Prefill/Decode 加载不同源码树。

## 当前真正需要修复的瓶颈

不是简单补参数，而是 DeepSeek-V4 Decode TBO 的执行策略，以及 DSpark hidden capture 与 TBO 分层执行的冲突。已完成 DSV4 Decode/TARGET_VERIFY 的 TBO 策略、DSpark 辅助状态携带和 DFlash metadata 适配；DSpark TARGET_VERIFY 保留独立 CUDA Graph，普通 Decode 走 TBO。

## 2026-08-22 启动验证

### Prefill MegaMoE

新入口第一次启动发现历史 `flash_prefill_megamoe.sh` 使用了不存在的 `/data/ssd2/checkpoints/...` 路径；当前机器实际 checkpoint 为 `/data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash`。新入口已修正并固定原生 SGLang `PYTHONPATH`，随后完成模型配置解析、MegaMoE EP 初始化和 Mooncake bootstrap server 启动。

### Decode 最大组合

入口使用：

```text
DeepEP + low_latency + DeepGEMM + DSpark + Waterfill + LPLB + TBO + FP4 indexer + Decode CUDA Graph
```

参数解析日志确认：

```text
moe_a2a_backend='deepep'
moe_runner_backend='deep_gemm'
deepep_mode='low_latency'
ep_dispatch_algorithm='lp'
enable_waterfill=True
enable_two_batch_overlap=True
enable_deepseek_v4_fp4_indexer=True
speculative_algorithm='DSPARK'
cuda_graph_config.decode.backend='full'
```

第一次探针还发现 Decode 入口不能同时设置 `CUDA_VISIBLE_DEVICES=4,5,6,7` 和 `--base-gpu-id=4`；前者会把进程内编号重映射为 0--3，触发 NUMA/NVML 的 `device 4 is not visible`。入口已删除 Decode 的 `CUDA_VISIBLE_DEVICES`，沿用已验证的物理 GPU 编号方式。

### TBO/DSpark 兼容修复

1. `operations_strategy.py` 增加 DSV4 Decode 和 TARGET_VERIFY 的 TBO operation schedule。
2. `deepseek_v4.py` 允许 DSV4 Decode/TARGET_VERIFY 进入 TBO；每个捕获层将 post-mHC hidden mean 放入 TBO child output，并在两个 child 合并时按 token range 重建，保持 DSpark 返回接口不变。
3. `two_batch_overlap.py` 为 `DFlashVerifyInput` 增加线性 token/position 切分；其不含 Eagle 的 `retrieve_index` 等树结构字段。
4. DSpark/DFlash verify 不准备 TBO child batch，避免 draft/verify CUDA Graph 捕获时触发 `TboAttnBackend` 断言；其 verify CUDA Graph 仍保留。

### 已验证结果与失败记录

- 无 DSpark 的最大组合已成功启动，完成 Decode CUDA Graph 捕获，并通过 PD `你是谁`：HTTP 200，回复正常。
- DSpark 最大组合已成功完成 target verify、draft verify CUDA Graph 捕获，PD warmup 的 4 个 `/generate` 请求全部 HTTP 200。
- 曾出现 `DFlashVerifyInput has no attribute retrieve_index`，已修复。
- 曾出现 DSpark TBO 辅助 tensor 合并时误当作 output dict，已修复。
- 曾出现 draft verify graph 对非 TBO attention backend 的断言，已通过 DFlash verify 旁路 TBO 修复。
- router 曾因 decode `/server_info` 内部状态控制通道超时而将 decode 识别为不可用；`http_server.py` 现在对内部状态查询设置 2 秒超时，并按 `dp_size` 返回占位状态，避免空 `internal_states` 被识别为 DP=0。

当前最后一轮正在重新启动 Decode 以加载 server-info 修复；完成后还需重新启动 router，再验证完整 PD `你是谁` 请求。服务日志位于 `logs/services/decode_all_tech_deepep/`，每次启动单独保存。

## 2026-08-22 运行时隔离结果（重要）

本轮没有把“服务 ready”当作组合验证成功，而是实际通过 router 发送了 PD 请求，并用 py-spy 检查卡住的 scheduler。

### 1. TBO 与 Decode DP attention 的 collective 死锁

完整组合（日志 `decode_20260822_100122_pid2149523.log`）在服务 ready 后收到真实请求时出现：

- Decode DP0 停在 `broadcast_reqs`；
- DP1/DP3 停在 `dp_attn.all_gather`；
- Prefill 停在 `pop_bootstrapped` 的 TP `all_reduce`；
- 最终 watchdog 触发，scheduler 以 `-3` 退出。

这不是启动参数或 router 端口问题，而是当前 DSV4 Decode TBO operation schedule 与 PD/DP attention 的 collective 顺序不一致。关闭 TBO 后请求不再死锁，因此当前 DSV4 Decode TBO 适配仍不能宣称完成。

### 2. DSpark 与路由/FP4/DeepGEMM 组合的输出正确性问题

隔离结果如下：

| Decode 配置 | PD 请求结果 |
|---|---|
| DeepEP low_latency + DeepGEMM + Waterfill + LPLB + FP4 indexer + CUDA Graph，关闭 DSpark/TBO | HTTP 200，内容可返回，服务稳定 |
| 在上项基础上开启 DSpark、关闭 TBO，draft backend=`deep_gemm` | HTTP 200，但输出从无关数学题开始，语义错误 |
| 在上项基础上把 DSpark draft backend 改为 `auto` | 仍然语义错误 |
| 临时把主 Decode backend 改为 `auto`，仍保留 Waterfill/LPLB/FP4 + DSpark | 仍然语义错误 |
| 历史 DSpark 基线（无 Waterfill/LPLB/FP4，主 backend=`auto`） | 已有日志显示 `HTTP=200` 且返回 DeepSeek 身份回答 |

因此当前问题不是单纯的 draft backend 继承，而是 DSpark target-verify 与 Waterfill/LPLB/FP4 路由组合的 token/专家路径不一致。`HTTP 200` 不能作为 DSpark 组合正确性的依据。

### 3. 当前脚本和备份

Decode 脚本现在支持通过 `SGLANG_DECODE_MOE_RUNNER_BACKEND` 做 A/B，默认主 Decode 仍为 `deep_gemm`；DSpark draft 显式使用 `--speculative-moe-runner-backend auto`。本轮脚本备份位于：

- `backups/all_tech_before_dspark_draft_backend_fix_20260822/`
- `backups/router_explicit_port_before_revert_20260822/`

当前结论：非 DSpark、非 TBO 的 DeepEP 最大组合已经可用；DSpark target-verify 正确性和 DSV4 Decode TBO collective 同步仍需源码级修复，不能将当前组合用于正式性能矩阵。

## 2026-08-22 后续隔离实验补充

为区分 DSpark draft graph、TBO 和 DeepEP dispatch 上限，又进行了以下实验：

| 实验 | 结果 | 诊断 |
|---|---|---|
| 全组合 + TBO，DeepEP 最大 dispatch=512 | Graph 捕获阶段断言 `x.size(0) <= num_max_dispatch_tokens_per_rank` | TBO 会放大 Graph capture 的 dispatch token 数，512 不足 |
| 全组合 + TBO，最大 dispatch=1024 | Graph capture 后首个 PD warmup 在 draft buffer copy 报 CUDA illegal memory | TBO/target Graph 状态已污染，不能只靠提高上限修复 |
| 全组合关闭 TBO，保留 DSpark、Waterfill、LPLB、FP4、DeepGEMM、DeepEP | 首个 draft forward 前仍报 CUDA illegal memory | DSpark 与当前 target Graph/路由状态仍不兼容 |
| 恢复历史 draft metadata、关闭 DSpark draft Graph | 仍未形成可用完整组合 | draft Graph 不是唯一根因 |
| 最小 DSpark、A2A=`none`、`flashinfer_mxfp4`、关闭 Waterfill/LPLB/FP4/TBO | target Graph/JIT 长时间未进入 ready，已终止诊断 | 当前源码下 DSpark 独立基线也未能在本轮复现 |

本轮新增诊断日志目录：

```text
logs/services/decode_all_tech_draft_eager_diag/
logs/services/decode_all_tech_draft_eager_diag2/
logs/services/decode_all_tech_draft_eager_diag3/
logs/services/decode_all_tech_dspark_no_tbo_diag/
logs/services/decode_dspark_minimal_diag/
logs/services/decode_dspark_minimal_diag2/
```

因此目前可以确认的边界是：Prefill MegaMoE 与“关闭 DSpark/TBO 的 Decode 最大技术组合”可以组合，且已有正常 PD 请求证据；但 Prefill MegaMoE + Decode 同时启用 DSpark、TBO 及全部路由优化尚未兼容，当前不能称为成功，也不应据此跑性能对比。

## 2026-08-22 DSpark draft backend 根因定位

在最小 DSpark 功能性请求中，当前脚本曾显式传入：

```text
--speculative-moe-runner-backend auto
```

这会使 draft 路径选择 Triton。实际报错为：

```text
Hidden size mismatch: hidden=[5,4096], w1=[64,4096,2048]
```

历史 DSpark 成功脚本没有强制 `auto`，而是沿用 `flashinfer_mxfp4`。将 Decode 脚本改为：

```text
--speculative-moe-runner-backend ${SGLANG_SPECULATIVE_MOE_RUNNER_BACKEND:-flashinfer_mxfp4}
```

后，DSpark draft 权重加载和服务启动均成功，不再出现 hidden-size mismatch；真实请求进入 FlashInfer MxFP4 首次 `nvcc/ptxas` JIT。该修复已备份在 `backups/all_tech_before_draft_hidden_mismatch_diag_20260822/`（诊断文件）以及脚本历史备份目录中。后续需要等待 JIT 完成并重新做 target CUDA Graph、DeepEP 和全部技术的逐项合入验证。

## 2026-08-22 DeepGEMM draft A/B 补充

进一步隔离得到：

| Target Decode | DSpark draft backend | PD `你是谁` | 结论 |
|---|---|---|---|
| `deep_gemm` | `deep_gemm` | HTTP 200，但语义错误 | DeepGEMM 不能直接作为当前 DSpark draft backend |
| `deep_gemm` | `flashinfer_mxfp4` | draft 权重加载成功，真实请求进入 FlashInfer SM100 JIT | 这是当前应继续验证的 target/draft 组合 |

第二组已将 Decode watchdog 临时提高到 1800 秒；首次 `fused_moe_trtllm_sm100` 的 `ptxas` 编译超过 5 分钟仍在进行，尚未产生最终语义结果。脚本现在支持：

```text
SGLANG_SPECULATIVE_MOE_RUNNER_BACKEND
SGLANG_WATCHDOG_TIMEOUT
SGLANG_CUDA_GRAPH_BACKEND_DECODE
SGLANG_SKIP_SERVER_WARMUP
```

这些开关只用于 A/B 和首次编译诊断，正式全技术实验仍必须恢复 Decode CUDA Graph，并通过语义 `whoami` 后才能计入结果。

## 2026-08-22 DSpark 与目标 DeepGEMM 的干净 A/B

为排除首次 JIT 编译和旧 PD 状态影响，先重启 Decode/Router，并复用已生成的 FlashInfer TRT-LLM SM100 JIT 缓存。在相同的 TP4/DP4、DP attention、PD Mooncake、目标 `deep_gemm`、Decode CUDA Graph disabled、A2A=`none` 配置下，仅切换 DSpark：

| Target Decode | DSpark | PD `你是谁` | 结果 |
|---|---:|---|---|
| `deep_gemm` | 关闭 | `HTTP=200`, `WHOAMI_VALID=True` | 目标 DeepGEMM + PD 链路正常 |
| `deep_gemm` | 开启，draft=`flashinfer_mxfp4` | `HTTP=200`, `WHOAMI_VALID=False`，返回网页/cookie 内容 | DSpark draft 组合当前语义错误 |

证据文件：

- 正常 A/B：`logs/services/router_dspark_draft_flashinfer/whoami_target_only_20260822.log`
- DSpark 失败：`logs/services/router_dspark_draft_flashinfer/whoami_clean_20260822.log`
- Decode 日志：`logs/services/decode_ab_target_deepgemm/decode_20260822_122101_pid2303455.log`、`logs/services/decode_dspark_draft_flashinfer/decode_20260822_121814_pid2298610.log`

该 A/B 说明：问题不是 FlashInfer 首次编译超时，也不是 Prefill 或目标 DeepGEMM 本身。DSpark 开启后，draft/verify 或其 DP attention 同步路径产生了错误 token；此前观察到的各 rank 分别停在 `request_receiver.broadcast` 与 `dp_attn.all_gather`，与客户端超时后的状态删除同时出现。当前不能把 DSpark 与目标 DeepGEMM 组合用于 MegaMoE 或正式性能测试，后续应先在关闭 Waterfill/LPLB/FP4/TBO 的最小组合中修复并通过语义验证。

## 2026-08-22 DSpark + DeepEP 最小组合修复

继续按目标组合做控制实验，固定目标 Decode 为 `deep_gemm`、A2A=`deepep`、DeepEP `low_latency`、DP attention、PD Mooncake，关闭 Waterfill/LPLB/FP4 indexer/TBO 和 Decode Graph，仅切换 DSpark：

| 配置 | 结果 |
|---|---|
| 目标 DeepGEMM + DeepEP，DSpark 关闭 | `HTTP=200`, `WHOAMI_VALID=True` |
| 目标 DeepGEMM + DeepEP，DSpark 开启（修复前） | 首请求 `bonus_tokens` 为空，随后错误；修复后又暴露 idle rank 空 FP8 scale 的 stride 错误 |
| 目标 DeepGEMM + DeepEP + DSpark（修复后） | `HTTP=200`, `WHOAMI_VALID=True` |

本轮确认并修复两个实际问题：

1. 调度器先创建空 `DFlashDraftInputV2` 时，worker 不能只判断 `spec_info is None`；active batch 还必须检查 `bonus_tokens.numel()==0`，并用 `batch.input_ids` 或 PD request 最后输出 token 初始化首个 draft anchor。
2. DeepEP low-latency 的 idle DP rank 会进入 collective，但没有专家 token；MxFP4 量化层不能把空 FP8 scale 直接重解释为 `uint8`，需要返回与 DeepEP combine 约定一致的空 expert-major BF16 buffer。

修复备份：

- `backups/all_tech_before_restore_first_anchor_20260822/dspark_worker_v2.py`
- `backups/all_tech_before_deepep_zero_token_guard_20260822/mxfp4_flashinfer_trtllm_moe.py`
- `backups/all_tech_before_skip_dp_mlp_sync_ab_20260822/`（已验证不适用；该参数仅支持 EAGLE）

成功证据：`logs/services/router_dspark_draft_flashinfer/whoami_target_deepgemm_deepep_dspark_20260822.log`；Decode 日志：`logs/services/decode_dspark_deepgemm_deepep_eager/decode_20260822_124712_pid2335523.log`。这说明 Prefill 侧切换 MegaMoE 具备继续验证的基础，但完整技术组合仍需逐项恢复并验收。

## 2026-08-22 完整技术组合验收

在上述修复基础上，按单变量顺序恢复 Decode 特性：CUDA Graph → Waterfill → LPLB → FP4 indexer → TBO。每一步都使用独立 Decode 日志目录，并在真实 PD 请求后执行语义 `whoami`。

最终 Decode 参数包含：

```text
--moe-a2a-backend deepep
--deepep-mode low_latency
--moe-runner-backend deep_gemm
--speculative-algorithm DSPARK
--speculative-moe-runner-backend flashinfer_mxfp4
--enable-waterfill
--ep-num-redundant-experts 16
--ep-dispatch-algorithm lp
--enable-deepseek-v4-fp4-indexer
--enable-two-batch-overlap
--cuda-graph-backend-decode full
--cuda-graph-bs-decode 1 2 4 8 16 32 64 128
```

最终组合结果：`HTTP=200`、`WHOAMI_VALID=True`。证据文件：

- Router：`logs/services/router_dspark_draft_flashinfer/whoami_all_tech_tbo_20260822.log`
- Decode：`logs/services/decode_all_tech_full_tbo/decode_20260822_130106_pid2355315.log`
- Prefill baseline 当前仍使用独立服务日志 `logs/services/prefill_all_baseline/`。

这证明完整 Decode 技术栈在当前源码和 PD 拓扑下已经能与 Prefill baseline 联通；下一步仍需用同一 Decode 配置切换 Prefill MegaMoE，并分别执行 whoami 与完整请求成功率检查，之后才能开始正式性能矩阵。

### Prefill MegaMoE 验收

切换 Prefill 为 `flash_prefill_all_megamoe.sh` 后，Decode 保持完整技术组合不变，真实 PD `你是谁` 请求同样通过：

- Router：`logs/services/router_dspark_draft_flashinfer/whoami_all_tech_tbo_prefill_megamoe_20260822.log`
- Prefill：`logs/services/prefill_all_megamoe_full_tbo/prefill_20260822_130409_pid2359465.log`
- Decode：`logs/services/decode_all_tech_full_tbo/decode_20260822_130106_pid2355315.log`

结果：`HTTP=200`、`WHOAMI_VALID=True`。因此两种 Prefill 方案目前都已通过完整 Decode 技术栈的功能性联通验收：Prefill baseline 和 Prefill MegaMoE。

### TBO 高并发事件循环修复

完整组合单请求通过后，第一次 16 并发短测出现 collective 顺序错位：DP0/2/3 已进入 `dp_attn.all_gather`，DP1 仍停在 `request_receiver.broadcast`。问题来自 TBO 让不同 DP rank 在事件循环中推进速度不同，CPU 请求广播与下一轮 CUDA MLP sync 的顺序发生交叉。

在 `disaggregation/decode.py` 的 normal/overlap decode event loop 开头增加 TBO+DP attention 条件下的 `tp_cpu_group` barrier，使所有 rank 完成上一轮后再进入请求接收。修改前备份：`backups/all_tech_before_tbo_event_loop_barrier_20260822/decode.py`。

修复后短验收（Prefill MegaMoE、完整 Decode 技术组合、ISL=1024、OSL=16、C=16、16 prompts）：

```text
Successful requests: 16/16
Total input tokens: 16384
Total generated tokens: 256
Total token throughput: 14203.59 tok/s
Mean TTFT: 504.46 ms
Mean TPOT: 33.62 ms
```

原始结果文件：`logs/results/all_tech_full_tbo_megamoe/isl1024_osl16_c16_p16_barrier.jsonl`。该结果是功能/稳定性验收，不作为 OSL=1024 正式性能对比数据。

## 2026-08-22 高并发 Verify 路径修复与正式测试进展

在 MegaMoE Prefill + 全技术 Decode 组合下，`1024/1024` 的 `C=1/16/256` 已完成且全部请求成功：

| ISL | OSL | Concurrency | Num prompts | Total tok/s | Mean TTFT | Mean TPOT | 成功 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 10 | 87.23 | 205.00 ms | 22.75 ms | 10/10 |
| 1024 | 1024 | 16 | 160 | 1018.66 | 295.38 ms | 29.75 ms | 160/160 |
| 1024 | 1024 | 256 | 2560 | 9454.83 | 691.07 ms | 51.21 ms | 2560/2560 |

结果目录：`logs/results/all_tech_full_tbo_megamoe_20260822/`。

`C=512` 首次正式测试触发了新的路径错误：DSpark target-verify 在高并发下进入 eager runner，但 DSV4 attention backend 收到 `DSV4RawVerifyMetadata` 后直接访问不存在的 `core_attn_metadata`。这不是请求数据错误，也不是 DeepEP collective 错误。

修复方式是在 attention `forward()` 入口补充 Raw verify/decode metadata 的兜底 materialize，使 eager verify 路径与 CUDA Graph metadata hook 使用同一套 raw→full 转换。修改前备份：`backups/all_tech_before_raw_verify_materialize_20260822/deepseek_v4_backend.py`。

修复后验证：`C=512、512 请求、OSL=32` 全部成功，`Peak concurrency=512`，`rc=0`，Mean TTFT `4487.03 ms`、Mean TPOT `36.33 ms`，无新的 Traceback/NCCL/Gloo 错误。结果文件：`logs/results/all_tech_full_tbo_megamoe_20260822/raw_verify_fix/isl1024_osl32_c512_n512.jsonl`。

随后已重新启动 Decode 服务并重新执行正式 `1024/1024、C=512、5120 请求`，使用独立服务日志：`logs/services/decode_all_tech_full_tbo_barrier_rawfix/`；旧的失败测试不计入正式结果。

## 2026-08-22 DSpark + CUDA Graph + 全技术组合最终修复

前一轮出现“C4/C128 全是 skip”的结果不计入实验。skip 是临时保护：TBO 子 batch 的压缩输入行数与父 batch 的 `out_cache_loc` 不一致时直接跳过写回，避免了崩溃，但也使结果无效。该保护已移除。

本轮定位到完整组合下 DSpark 的两类元数据问题：

1. PD decode/overlap 后，scheduler 的 GPU/CPU `seq_lens` 可能不是当前请求的真实长度；DSpark `assign_extend_cache_locs` 因此访问 draft KV pool 越界。planner、draft forward、KV 注入和 verify batch 现在统一使用 `Req.seqlen` 重建 prefix length。
2. target verify 构造 `ForwardBatch` 时仍会携带失真的 GPU `seq_lens`，导致 CUDA Graph replay 使用错误 metadata。`DFlashVerifyInput.prepare_for_verify()` 现在在 verify 入口按请求重建 GPU `seq_lens`。
3. TBO/DSpark 的压缩路径在尾部 padding 行数不一致时，改为截断 padding 的 input/buffer/APE/plan，并保留有效 token 写回，不再 skip 整个压缩阶段。

相关源码备份：

- `backups/all_tech_before_dspark_seq_lens_fix_20260822/`
- `backups/all_tech_before_final_dspark_seq_lens_verify_20260822/`
- `backups/all_tech_before_row_alignment_20260822/`

当前最终服务配置仍为：Prefill MegaMoE；Decode DeepEP `low_latency` + DeepGEMM + DSpark（draft MxFP4）+ Waterfill + LPLB + TBO + FP4 indexer + Decode CUDA Graph。

功能验证：

| ISL | OSL | Concurrency | 请求数 | 成功 | Total tok/s | Mean TTFT | Mean TPOT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 32 | 16 | 160 | 160/160 | 13674.91 | 432.83 ms | 24.05 ms |
| 1024 | 32 | 256 | 256 | 256/256 | 51169.06 | 2392.42 ms | 33.27 ms |

结果文件：`logs/results/all_tech_full_tbo_megamoe_20260822/fix3_validation/`。两组短测均返回 `exit=0`，完整 Decode 服务未出现 Traceback、illegal access 或 NCCL 错误；正常服务下 `你是谁`请求也通过 `HTTP=200`、`WHOAMI_VALID=True`。

这两组是修复后的功能/稳定性验证，不替代 OSL=1024 的正式 baseline 性能矩阵；确认稳定后再继续 C=512 和 8192 输入的正式测试。
## 2026-08-22：全技术组合 C128/DeepEP Graph 兼容性修复（进行中）

### 本轮 skip 的真实原因

上一轮正式 `1024/1024/C512` 并不是测试完成后得到的 skip：Decode 在 C128 压缩路径触发
`expected 128 but got 96` 后退出，benchmark 收到不完整 HTTP payload，后续样例因 Decode
端口不可用才被脚本跳过。该轮结果不计入性能表。

### 已完成的修复

1. C4/C128 compressor 的 `ape` 是固定 kernel lookup table，分别要求 `[8, head_dim]` 和
   `[128, head_dim]`，不能随 TBO/Graph query 行数裁剪。移除错误的 `ape[:actual_q]` 裁剪；
   修改前源码保存在 `backups/all_tech_before_row_alignment_20260822/`，修复后备份
   在 `backups/all_tech_after_ape_fixed_before_c512_20260822/`。
2. DeepEP low_latency 在本版本硬限制
   `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK <= 1024`。DSpark verify 每请求 6 个
   token，因此 Decode CUDA Graph 最大 bucket 调整为 `128`，使最大 verify dispatch 为
   `128*6=768`，仍保留 CUDA Graph；没有用关闭 Graph 规避问题。

### 当前已验证结果

| ISL | OSL | Concurrency | 请求 | 状态 | Output tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 1024 | 32 | 512 | 512 | 512/512 成功 | 1716.29 | 56637.61 | 4551.56 ms | 33.78 ms |

原始结果：`logs/results/all_tech_full_tbo_megamoe_20260822/ape_fix_validation/isl1024_osl32_c512_n512.jsonl`。
服务日志：`logs/services/decode_all_tech_ape_fix2_20260822/decode_20260822_160944_pid2529574.log`。

正式 `1024/1024/C512、5120 请求` 已启动，待完成后再写入正式结果；运行期间未出现服务端
Traceback 或 scheduler exception。

## 2026-08-22：全技术组合正式 C512 长测完成

在上述修复后的配置下，Prefill MegaMoE + Decode DeepEP `low_latency`、DeepGEMM、DSpark、
Waterfill、LPLB、TBO、FP4 indexer 和 Decode CUDA Graph（最大 bucket 128）完成正式
`ISL=1024、OSL=1024、Concurrency=512、Num prompts=5120` 测试。

| ISL | OSL | Concurrency | Num prompts | 成功 | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT | Duration |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 512 | 5120 | 5120/5120 | 1450.17 | 2900.34 | 1209.11 ms | 349.57 ms | 3615.36 s |

正式结果：`logs/results/all_tech_full_tbo_megamoe_20260822/plan_cache_key_final/isl1024_osl1024_c512_n5120.jsonl`。
benchmark 日志：`logs/results/all_tech_full_tbo_megamoe_20260822/plan_cache_key_final/isl1024_osl1024_c512_n5120.log`。
本轮 Decode 服务日志：`logs/services/decode_all_tech_plan_cache_key_20260822/decode_20260822_163845_pid2551126.log`。

本轮正式长测累计请求全部成功，服务日志没有新的 `num_q < num_w`、Traceback、watchdog、Xid
或 Killed。长测耗时较短测明显增加，主要是 OSL=1024 下 DSpark verify、TBO 和 PD 传输共同作用；
该结果可作为修复后全技术组合的正式功能/性能记录。

## 2026-08-22：最终修复版本低并发与 8192 输入复核

为排除旧结果和旧服务日志影响，在 `plan_cache_key` 修复后的同一服务上重新执行了低并发
和长输入验证。两组均由 benchmark JSON 确认全部完成，不能把旧日志中的 `Skip C4/C128 store`
当作本轮请求跳过；当前服务日志中该类真实 skip 计数为 0。

| ISL | OSL | Concurrency | Num prompts | 成功 | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 10 | 10/10 | 48.67 | 97.35 | 194.21 ms | 20.37 ms |
| 1024 | 1024 | 16 | 160 | 160/160 | 588.94 | 1177.89 | 277.98 ms | 25.42 ms |
| 1024 | 1024 | 256 | 2560 | 2560/2560 | 5173.11 | 10346.21 | 614.40 ms | 46.85 ms |
| 8192 | 1024 | 1 | 10 | 10/10 | 47.91 | 431.17 | 214.53 ms | 20.68 ms |
| 8192 | 1024 | 16 | 160 | 160/160 | 582.05 | 5238.49 | 375.19 ms | 25.87 ms |
| 8192 | 1024 | 256 | 2560 | 2560/2560 | 8392.63 | 75533.64 | 27.17 ms | 29.01 ms |

结果目录：`logs/results/all_tech_full_tbo_megamoe_20260822/final_matrix/`。
本轮 Decode 日志未出现 `Scheduler hit an exception`、Traceback、`invalid prefill plan`、
`num_q < num_w`、Xid 或 Killed。

## 2026-08-22：高并发 Mooncake 瞬时 descriptor 失败修复（复测中）

上一轮 8192/1024/C512 的“完成”结果不能作为有效结果：约 19:43 起 Mooncake 出现
`Failed to get segment descriptor`，随后 Decode/Prefill 传输级联失败。Prefill 和 Decode
进程本身没有退出，原有 failed-session probe 可以恢复 session，但恢复前已经失败的请求仍会
污染整组结果。

本轮在保留全部技术开关的前提下增加了两层恢复：

1. Prefill 的 failed-session probe 周期从 10 s 调为 1 s；
2. Mooncake 单次发送遇到非零返回时，主动 `send_probe` 并最多重试 3 次，每次间隔 0.2 s，
   重试成功不再把 session 标记为 failed。

源码备份：`backups/mooncake_retry_20260822/`。当前服务日志：
`logs/services/prefill_retry_20260822/`、`logs/services/decode_retry_20260822/`。
当前 1024/1024/C512 长测正在执行，结果落盘前不计入正式性能表。

## 2026-08-22：Mooncake 重试修复后 C512 完整复测通过

在上述重试修复版本上，使用同一套完整技术组合重新执行
`ISL=1024、OSL=1024、Concurrency=512、Num prompts=5120`。本次 benchmark JSON 确认
`completed=5120`，Prefill/Decode 日志均未出现 KVTransferError、segment descriptor 失败、
failed session、Traceback、Xid 或 Killed。

| ISL | OSL | Concurrency | Num prompts | 成功 | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT | Duration |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 512 | 5120 | 5120/5120 | 1429.92 | 2859.84 | 1220.23 ms | 354.59 ms | 3666.55 s |

结果文件：`logs/results/all_tech_full_tbo_megamoe_20260822/retry_validation/isl1024_osl1024_c512_n5120.jsonl`。
服务日志：`logs/services/prefill_retry_20260822/`、`logs/services/decode_retry_20260822/`。
长测结束后再次发送“你是谁”请求，HTTP payload、text、output_ids 和 finish_reason 均正常。

## 2026-08-22：NIXL/UCX 传输层验证（初始化成功，但 PD KV 不兼容）

为排查 Mooncake 在 8192 输入高并发下的 descriptor 失败，临时将 PD 传输后端切换为
NIXL/UCX，并保持 Prefill MegaMoE、Decode DeepEP `low_latency`、DeepGEMM、DSpark、
Waterfill、LPLB、TBO、FP4 indexer 和 Decode CUDA Graph 全部开启。两端 NIXL agent 和
UCX backend 均成功初始化，服务也完成启动 warmup。

但第一条真实 PD 请求在 Prefill 建立 KV dlist 时失败：

```text
ValueError: NIXL prepared dlist transfer length exceeds item stride:
xfer_len=8448, item_len=4352, mem_kind=VRAM
```

这表示 Prefill 与 Decode 的 DSV4 KV layout 在对应层上的 slot stride 不一致，当前 NIXL
equal-TP 直传路径仍按“目标 stride 至少容纳源传输长度”建立 dlist；不能把长度截断来规避，
否则会造成 KV 地址错位，结果不可信。因此本轮请求没有形成性能结果，外层 benchmark/请求
看起来像是全部 skip，其实是 PD bootstrap 被传输异常阻塞后超时/取消。

NIXL 日志：`logs/services/prefill_nixl_20260822/`、`logs/services/decode_nixl_20260822/`。
本轮 NIXL 配置和脚本改动备份：`backups/nixl_transport_20260822/`。
结论：NIXL/UCX 当前只能证明 backend 初始化成功，不能证明该 DSV4 PD KV layout 已适配；
正式性能表仍以 Mooncake 修复后通过的结果为准。

恢复 Mooncake 后的端到端冒烟验证：`ISL=1024、OSL=32、Concurrency=16、128 请求`
全部 `128/128` 成功，benchmark duration 7.30 s，Out tok/s 285.63，Total tok/s 9186.61，
Mean TTFT 526.09 ms，Mean TPOT 19.53 ms。结果文件：
`logs/results/all_tech_full_tbo_megamoe_20260822/restore_validation/isl1024_osl32_c16_n128.jsonl`；
服务日志：`logs/services/prefill_restore_mooncake_20260822/`、
`logs/services/decode_restore_mooncake_20260822/`。

## 2026-08-22：8192 长输入 illegal memory access 的隔离结论

恢复服务后，8192/1024/C16 的短测在默认 PD overlap schedule 下仅完成 9/160，Decode
scheduler 在 `process_batch_result()` 等待 `result.copy_done` 时触发
`CUDA error: an illegal memory access`，随后 Prefill 请求统一表现为 `AbortReq`。关闭 DSpark
后仍能复现同一问题，因此不能把它归因于 DSpark 单项。

进一步使用 `SGLANG_DISABLE_OVERLAP_SCHEDULE=1`，保留 Decode DeepEP `low_latency`、
DeepGEMM、Waterfill、LPLB、TBO、FP4 indexer 和 decode CUDA Graph，8192/128/C16/32
全部成功；重新打开 DSpark 后同样全部成功：

| 配置 | ISL | OSL | Concurrency | 请求 | 成功 | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DSpark 关闭，PD overlap 关闭 | 8192 | 128 | 16 | 32 | 32/32 | 202.33 | 13189.89 | 774.77 ms | 42.97 ms |
| 全技术组合，PD overlap 关闭 | 8192 | 128 | 16 | 32 | 32/32 | 289.54 | 18875.75 | 722.03 ms | 26.94 ms |

结果文件分别为：
`logs/results/all_tech_full_tbo_megamoe_20260822/isolation_nooverlap/isl8192_osl128_c16_n32.jsonl`、
`logs/results/all_tech_full_tbo_megamoe_20260822/full_nooverlap/isl8192_osl128_c16_n32.jsonl`。
日志分别在 `logs/services/decode_isolation_nooverlap_20260822/` 和
`logs/services/decode_full_nooverlap_20260822/`。

因此当前可用兼容配置是：保留 TBO，但对 DSV4 长输入关闭通用 PD overlap schedule；该开关
通过 `SGLANG_DISABLE_OVERLAP_SCHEDULE=1` 已在启动脚本中支持。默认 overlap 路径仍不能作为
8192 长输入的正式性能配置，后续应继续修复 `result.copy_done` 对应的异步 buffer 生命周期，
而不是把失败请求计入吞吐或误判为 benchmark skip。

## 2026-08-22：全量测试脚本的 skip 判定修复

此前 `run_lplb_deepgemm_baseline_ab_20260821.sh` 只要发现旧的 JSONL 和旧日志中存在
`Successful requests:`，就直接将当前样例记为 `skipped`。这会把不同源码、服务配置或实验批次
的旧结果误认为当前实验已完成，从而出现“全部 skip”。这不是请求失败，也不是服务没有运行，
而是结果复用条件过于宽松。

已将复用改为显式行为：默认新运行不复用旧结果；只有设置 `FORCE_RERUN=0` 且旧日志的成功数
精确匹配当前 `num_prompts` 时才允许 skip，设置 `FORCE_RERUN=1` 可强制重测。修改前源码已备份至
`backups/skip_guard_fix_20260822/`。

## 2026-08-22：FP4 indexer + full CUDA Graph 长上下文修复验证

隔离测试确认：DeepEP `low_latency` + DeepGEMM + FP4 indexer + full CUDA Graph 在
`ISL=1024、OSL=1024、C256` 下此前会在 Graph replay 阶段触发 illegal memory access；关闭
CUDA Graph 时同一 FP4 配置可完成 256/256，因此问题集中在 Graph padding/idle row 的压缩长度
元数据，而不是 Mooncake、Prefill、DeepEP 或 FP4 KV cache 基础布局。

修复内容：在 `PagedIndexerMetadata` 和 indexer 的消费路径对 `c4_seq_lens` 做 `clamp_min(0)`。
DSV4 的 Graph padding 行可能携带负的原始 C4 长度，而 DeepGEMM paged-MQA/topk-v2 会把负值
按无符号长度解释为超大长度，导致越界访问；将 idle 行按 0 长度处理是安全的 Graph 语义。
修改前源码已备份至 `../sglang_v0.5.16/backups/all_tech_fp4_graph_clamp_before_20260822/`。

修复后重新启动完整 Graph 服务，并通过 Router `http://127.0.0.1:13784` 测试，配置为
Decode DeepEP `low_latency`、DeepGEMM、FP4 indexer、full CUDA Graph，TBO/Waterfill/LPLB/DSpark
仅在本隔离中关闭。结果如下：

| ISL | OSL | Concurrency | 请求 | 成功 | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 256 | 256 | 256/256 | 9795.92 | 19591.83 | 2469.90 ms | 22.12 ms |

结果文件：`logs/results/all_tech_full_tbo_megamoe_20260822/c256_isolation/fp4_graph_clamp_isl1024_osl1024_c256_n256.jsonl`；
服务日志：`logs/services/decode_fp4_graph_clamp_20260822/`。该结果证明 FP4 Graph 修复已生效，
后续全技术矩阵可以继续使用 CUDA Graph，不应再以“关闭 Graph”作为最终方案。

## 2026-08-22：全技术配置正式矩阵复测进度

在同一套 PD 服务上执行全技术配置：Prefill MegaMoE；Decode DeepEP `low_latency`、DeepGEMM、
DSpark、Waterfill、LPLB、TBO、FP4 indexer 和 full CUDA Graph（graph bucket 最大 128，
并关闭 DSV4 通用 PD overlap schedule）。本轮使用全新结果目录并设置 `FORCE_RERUN=1`，
没有复用旧结果。

已完成的 1024 输入样例：

| ISL | OSL | Concurrency | 请求 | 成功 | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 10 | 10/10 | 42.43 | 84.85 | 211.65 ms | 23.38 ms |
| 1024 | 1024 | 16 | 160 | 160/160 | 497.22 | 994.43 | 288.88 ms | 30.55 ms |
| 1024 | 1024 | 256 | 2560 | 2560/2560 | 4648.97 | 9297.93 | 620.17 ms | 52.17 ms |

结果目录：`logs/results/all_tech_full_tbo_megamoe_20260822/final_matrix_graph128/`。
C512 样例已启动，完成后再补充到本表；8192 输入四组仍待本轮服务完成后执行。
