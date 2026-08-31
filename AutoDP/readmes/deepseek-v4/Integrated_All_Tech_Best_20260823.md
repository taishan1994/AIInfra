# PD 分离全技术整合：单项最佳配置验证

日期：2026-08-23  
源码：`/data/ssd2/sglang_v0.5.16`  
Prefill：MegaMoE，GPU 0–3，端口 30000  
Decode：DeepEP low_latency，GPU 4–7，端口 30001  
Router：端口 13784

## 1. 目标

之前 DSpark、Waterfill、LPLB、TBO、FP4 indexer、DeepGEMM 和 MegaMoE 均已分别完成实验。本轮不是简单打开全部开关，而是将每个单项实验已经验证过的有效路径整合到同一套 PD 分离服务中。

## 2. 当前整合配置

Decode 实际启动参数包含：

```text
--tp-size 4 --dp-size 4 --ep-size 4
--enable-dp-attention --enable-dp-lm-head
--moe-a2a-backend deepep --deepep-mode low_latency
--deepep-config {"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}
--moe-runner-backend deep_gemm
--ep-num-redundant-experts 0 --ep-dispatch-algorithm lp
--enable-waterfill
--enable-two-batch-overlap
--enable-deepseek-v4-fp4-indexer
--speculative-algorithm DSPARK
--speculative-moe-runner-backend flashinfer_mxfp4
--cuda-graph-backend-decode full
--cuda-graph-bs-decode 1 2 4 8 16 32 64 128
--disable-overlap-schedule
```

关键环境变量：

```text
SGLANG_LPLB_STATIC_FALLBACK=1
SGLANG_DISABLE_OVERLAP_SCHEDULE=1
SGLANG_RAGGED_VERIFY_MODE=static
```

其中 `SGLANG_LPLB_STATIC_FALLBACK=1` 是本轮最重要的整合修复：低并发和随机均匀 workload 不进入每层动态 IPM solver，而是使用 rank-aware static map；LPLB 仍保留 LP dispatch 入口和完整配置。DSpark 使用历史有效的 draft MxFP4 runner 和 SPS 文件，TBO 对 DSpark target verify 使用已加入的兼容旁路，普通 decode 仍使用 TBO。

启动脚本：

```text
flash_prefill_all_megamoe.sh
flash_decode_all_tech_deepep.sh
flash_router_baseline.sh
```

脚本备份：

```text
backups/integrated_best_config_20260823/flash_decode_all_tech_deepep.sh.before_best_defaults
backups/integrated_best_config_20260823/flash_prefill_all_megamoe.sh.before_best_defaults
```

## 3. 功能验证

新服务启动日志确认：

- Prefill MegaMoE 已启用。
- Decode DeepEP `low_latency`、DeepGEMM、Waterfill、LPLB、TBO、FP4 indexer、DSpark 均已加载。
- target verify CUDA Graph 和 draft verify CUDA Graph 均完成捕获。
- Decode 进程环境实际包含 `SGLANG_LPLB_STATIC_FALLBACK=1`。
- `你是谁` 请求返回 `HTTP=200`，`WHOAMI_VALID=True`。
- 服务日志没有 `Traceback`、CUDA illegal access、NCCL Error 或 `NotImplementedError`。

日志目录：

```text
logs/services/integrated_best_20260823/prefill/
logs/services/integrated_best_20260823/decode/
logs/services/integrated_best_20260823/router/
logs/services/integrated_best_20260823/whoami.log
```

## 4. 当前已完成的 baseline 对比

原始 baseline 数值来自 `readmes/LPLB_PD_SGLang_v0516_baseline_20260821.md`。

| ISL | OSL | C | 成功 | 本轮 Out tok/s | baseline Out tok/s | Out 变化 | 本轮 Total tok/s | baseline Total tok/s | Total 变化 | 本轮 TTFT ms | 本轮 TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 10/10 | 199.81 | 107.55 | +85.8% | 399.61 | 215.09 | +85.8% | 190.40 | 4.82 |
| 1024 | 1024 | 16 | 160/160 | 2281.89 | 1348.95 | +69.2% | 4563.77 | 2697.90 | +69.2% | 346.11 | 6.37 |
| 1024 | 1024 | 256 | 2560/2560 | 14034.03 | 13106.67 | +7.1% | 28068.06 | 26213.34 | +7.1% | 8648.74 | 8.57 |
| 8192 | 1024 | 1 | 10/10 | 182.02 | 106.52 | +70.9% | 1638.15 | 958.65 | +70.9% | 208.12 | 5.29 |
| 8192 | 1024 | 16 | 160/160 | 2130.03 | 1328.45 | +60.3% | 19170.31 | 11956.04 | +60.3% | 470.13 | 6.74 |

本轮结果文件：

```text
logs/results/integrated_best_20260823/targeted/
```

## 5. DSpark 接受率与 Graph 证据

整合前的失败组合在 DP rank 上通常只有 `accept rate=0.00–0.01`。本轮配置下：

- 1024/C1 长输出测试中，多个 rank 的接受率约为 `0.70–1.00`。
- 1024/C16 测试中，接受率约为 `0.67–0.97`。
- 1024/C256 测试中，接受率约为 `0.70–0.90`。
- 8192/C1、C16 测试中，接受率仍保持在约 `0.7–1.0` 范围。
- 上述运行时日志中的 `cuda graph` 均为 `True`。

这证明本轮收益不是单纯由请求成功或 benchmark 统计造成，而是 DSpark speculative path 和 CUDA Graph 确实进入了有效执行路径。

## 6. 与之前全开组合的差异

之前全开组合低于 baseline 的主要原因是：

1. LPLB 使用动态 solver，低并发每个 MoE layer 都承担固定 IPM 开销。
2. 使用 `ep-num-redundant-experts=16`，与低并发最佳配置不一致。
3. DSpark 使用的执行路径没有复现历史接受率。
4. 未使用历史高接受率轮次的 `disable-overlap-schedule`。
5. TBO 和 DSpark verify 的 metadata/collective 兼容问题导致额外 fallback 和调度开销。

本轮通过 static LPLB、`redundant experts=0`、历史 DSpark draft 路径和 TBO target-verify guard 解决了上述配置层问题。

## 7. 待完成

以下 3 组尚未在本轮整合配置下完成正式测试，不能使用旧轮次结果代替：

```text
1024/1024/C512
8192/1024/C256
8192/1024/C512
```

当前服务保持运行，后续应在同一组服务日志和同一版本源码上继续完成这 3 组，然后形成最终 8 组 baseline 表格。
