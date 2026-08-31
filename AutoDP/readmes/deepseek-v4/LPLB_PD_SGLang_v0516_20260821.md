# 原生 SGLang 0.5.16 PD 分离 LPLB 实验报告

日期：2026-08-21

## 1. 实验目标与源码基线

目标是在 `/data/ssd2/sglang_v0.5.16` 上将 LPLB 适配到 DeepSeek-V4-Flash 的 PD 分离 Decode，使用 DeepEP `low_latency`、DeepGEMM 和 Decode CUDA Graph，并与原始 baseline 对比。

本报告的正式结果使用默认 `SGLANG_LPLB_IPM_ITERS=5`；第 6 节另列出 IPM=1、IPM=2 和 mapping refresh interval 的受控 A/B，均未混入第 3 节正式结果。

原始 baseline 使用 Decode TP4/DP4/EP4、DeepEP low_latency、原生 DeepGEMM masked 路径和 PD Router。baseline 历史数据为：

| ISL | OSL | Concurrency | Out tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 107.55 | 215.09 | 332.51 | 8.98 |
| 1024 | 1024 | 16 | 1348.95 | 2697.90 | 631.91 | 11.25 |
| 1024 | 1024 | 256 | 13106.67 | 26213.34 | 2284.50 | 16.82 |
| 1024 | 1024 | 512 | 13243.63 | 26487.25 | 16554.06 | 21.32 |
| 8192 | 1024 | 1 | 106.52 | 958.65 | 348.65 | 9.05 |
| 8192 | 1024 | 16 | 1328.45 | 11956.04 | 607.99 | 11.23 |
| 8192 | 1024 | 256 | 6861.28 | 61751.53 | 21913.55 | 14.34 |
| 8192 | 1024 | 512 | 7110.57 | 63995.17 | 55535.20 | 14.42 |

## 2. LPLB 适配内容

### 2.1 DeepSeek-V4 模型支持

原始 LPLB 模型白名单不包含 `DeepseekV4ForCausalLM`，因此旧服务会在第一次请求时抛出：

```text
NotImplementedError: DeepseekV4ForCausalLM does not support --ep-dispatch-algorithm lp
```

DeepSeek V4 复用了 DeepSeek V2 的 MoE/TopK 路径，且空 token rank 会进入 LPLB solver 的 all-reduce 路径。适配后将 `DeepseekV4ForCausalLM` 加入 `_LPLB_SUPPORTED_MODEL_ARCHS`，满足 DP Attention 下所有 EP rank 必须参与 collective 的约束。

修改文件：

```text
/data/ssd2/sglang_v0.5.16/python/sglang/srt/eplb/lplb_solver.py
```

修改前备份：

```text
backups/lplb_20260821/lplb_solver.py.before_deepseek_v4_support
```

### 2.2 Decode 配置

```text
--tp-size 4 --dp-size 4 --ep-size 4
--enable-dp-attention --enable-dp-lm-head
--moe-a2a-backend deepep
--deepep-mode low_latency
--moe-runner-backend deep_gemm
--ep-num-redundant-experts 16
--ep-dispatch-algorithm lp
--cuda-graph-backend-decode full
--cuda-graph-bs-decode 1 2 4 8 16 32 64 128 256 512
--max-running-requests 1024
```

启动脚本显式固定源码路径，避免误用旧容器源码：

```bash
export PYTHONPATH=/data/ssd2/sglang_v0.5.16/python:${PYTHONPATH:-}
```

### 2.3 Prefill、Decode、Router

```text
Prefill: GPU 0–3, port 30000, TP4/DP1/EP1
Decode:  GPU 4–7, port 30001, TP4/DP4/EP4
Router:  port 13784
Transfer: Mooncake
```

服务启动完成后日志确认：

```text
Initialized LPLB solvers for 43 layers
Capture target decode CUDA graph end
PD warmup completed
```

每次重启后执行 `validate_pd_whoami.sh`，Router 返回 HTTP 200，且“你是谁”语义验证通过。完整矩阵 8/8 组请求均成功。

## 3. LPLB 与 baseline 对比

每组请求数为 `10 × concurrency`。本轮 benchmark 通过 Router `http://127.0.0.1:13784` 发送，统计 Out tok/s、Total tok/s、Mean TTFT 和 Mean TPOT。

| ISL | OSL | Concurrency | 本轮 Out tok/s | baseline Out tok/s | Out 变化 | 本轮 Total tok/s | baseline Total tok/s | Total 变化 | 本轮 TTFT ms | baseline TTFT ms | 本轮 TPOT ms | baseline TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 59.71 | 107.55 | -44.48% | 119.43 | 215.09 | -44.47% | 290.60 | 332.51 | 16.48 | 8.98 |
| 1024 | 1024 | 16 | 824.28 | 1348.95 | -38.89% | 1648.56 | 2697.90 | -38.89% | 531.10 | 631.91 | 18.81 | 11.25 |
| 1024 | 1024 | 256 | 9247.49 | 13106.67 | -29.44% | 18494.98 | 26213.34 | -29.44% | 1181.50 | 2284.50 | 26.26 | 16.82 |
| 1024 | 1024 | 512 | 14914.37 | 13243.63 | **+12.62%** | 29828.74 | 26487.25 | **+12.62%** | 4331.48 | 16554.06 | 28.79 | 21.32 |
| 8192 | 1024 | 1 | 59.65 | 106.52 | -44.00% | 536.88 | 958.65 | -44.00% | 241.35 | 348.65 | 16.54 | 9.05 |
| 8192 | 1024 | 16 | 821.61 | 1328.45 | -38.15% | 7394.51 | 11956.04 | -38.15% | 584.28 | 607.99 | 18.79 | 11.23 |
| 8192 | 1024 | 256 | 7099.22 | 6861.28 | **+3.47%** | 63892.97 | 61751.53 | **+3.47%** | 9462.45 | 21913.55 | 25.38 | 14.34 |
| 8192 | 1024 | 512 | 7370.88 | 7110.57 | **+3.66%** | 66337.89 | 63995.17 | **+3.66%** | 41942.99 | 55535.20 | 25.45 | 14.42 |

结论：默认 IPM=5 下，8 组中 3 组超过 baseline，分别是 1024/C512、8192/C256 和 8192/C512。8192 输入的高并发两组超过 baseline；低并发 5 组未超过 baseline。

## 4. 结果文件与服务日志

结果目录：

```text
logs/flash_decode_lplb/results_original_baseline_ab_20260821/
```

各组 benchmark 日志和 JSONL 文件：

```text
isl1024_osl1024_c1_n10.log/jsonl
isl1024_osl1024_c16_n160.log/jsonl
isl1024_osl1024_c256_n2560.log/jsonl
isl1024_osl1024_c512_n5120.log/jsonl
isl8192_osl1024_c1_n10.log/jsonl
isl8192_osl1024_c16_n160.log/jsonl
isl8192_osl1024_c256_n2560.log/jsonl
isl8192_osl1024_c512_n5120.log/jsonl
```

状态文件：

```text
logs/flash_decode_lplb/results_original_baseline_ab_20260821/status.tsv
```

默认 IPM=5 服务日志：

```text
logs/services/lplb_decode/decode_lplb_20260821_065952_pid1384512.log
logs/services/prefill/prefill_20260821_045958_pid1344778.log
```

当前 `IPM=1 + refresh_interval=2` 服务日志：

```text
logs/services/lplb_decode/decode_lplb_20260821_083204_pid1427613.log
```

## 8. 低并发最终优化结果

后续针对 HashTopK LP 分支、DeepEP 96-SMS 配置以及 baseline 参数一致性完成了低并发优化。最终低并发对照表和逐项日志见独立报告：

[LPLB_PD_SGLang_v0516_baseline_20260821.md](LPLB_PD_SGLang_v0516_baseline_20260821.md)

最终配置下，1024 输入 C1/C16/C256 和 8192 输入 C1/C16 均超过原始 baseline，所有请求成功。该低并发模式通过 `SGLANG_LPLB_STATIC_FALLBACK=1` 使用缓存的 rank-aware static map；高并发动态 LPLB IPM 结果仍以本报告第 3 节为准。

默认测试脚本：

```text
run_lplb_deepgemm_baseline_ab_20260821.sh
```

## 5. 性能瓶颈分析

当前 LPLB 的在线路径在每个 MoE layer、每个 forward batch 中执行：

```text
1. local expert count
2. EP all-reduce
3. LP input preparation kernel
4. fused IPM solver
5. log2phy probability extraction kernel
6. probability-based logical-to-physical dispatch
```

DeepSeek V4 有 43 个 MoE layer，因此低并发时每个 token 都会承担大量固定 solver 和 collective 开销；随机均匀 workload 又没有足够的专家热点供 LPLB 平衡，导致低中并发明显低于 baseline。高并发时专家负载和 DP rank 利用率改善，才出现 1024/C512、8192/C256、8192/C512 的吞吐收益。

从当前结果看，LPLB 已经实际生效，但不能把高并发收益外推到低并发。TTFT 在 8192/C512 达到 41942.99 ms，说明高并发下 Prefill/PD 排队仍是主要延迟问题；LPLB 当前主要改善的是 Decode 输出吞吐，不是首 token 延迟。

## 6. IPM solver 优化 A/B

为验证 IPM barrier iteration 是否是主要固定开销，新增了可配置环境变量：

```text
SGLANG_LPLB_IPM_ITERS=1..N
```

修改文件：

```text
/data/ssd2/sglang_v0.5.16/python/sglang/jit_kernel/lplb/cuda_solver.py
```

修改前备份：

```text
backups/lplb_20260821/cuda_solver.py.before_ipm_iters
```

默认值仍为 5，不影响第 3 节正式结果。IPM=1 服务已按同一 LPLB 配置重启，并通过 whoami 验证，独立服务日志为：

```text
logs/services/lplb_decode/decode_lplb_20260821_075051_pid1406539.log
```

IPM=1 的受控 A/B 结果如下：

| ISL | OSL | C | IPM iters | Out tok/s | Total tok/s | TTFT ms | TPOT ms | 相对原始 baseline |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1024 | 1024 | 16 | 1 | 984.13 | 1968.27 | 418.77 | 15.80 | Out/Total -27.04% |
| 1024 | 1024 | 256 | 1 | 10438.46 | 20876.91 | 1032.40 | 23.25 | Out/Total -20.36% |
| 1024 | 1024 | 512 | 1 | 16216.72 | 32433.44 | 3862.60 | 25.91 | Out/Total **+22.45%** |

与同一服务的 IPM=5 对比：

| C | IPM=5 Out tok/s | IPM=1 Out tok/s | Out 变化 | IPM=5 Total tok/s | IPM=1 Total tok/s | Total 变化 |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 824.28 | 984.13 | +19.39% | 1648.56 | 1968.27 | +19.39% |
| 256 | 9247.49 | 10438.46 | +12.88% | 18494.98 | 20876.91 | +12.88% |
| 512 | 14914.37 | 16216.72 | +8.73% | 29828.74 | 32433.44 | +8.73% |

IPM=1 仍未使 C16/C256 超过 baseline，但使 C512 的领先幅度从 IPM=5 的 +12.62% 提升到 +22.45%。因此当前更合理的默认候选是 IPM=1：它保留了高并发 LPLB 的负载均衡收益，并显著减少 LP solver 固定开销；是否采用更高 IPM 迭代次数，应在偏斜专家 workload 上验证映射质量后决定。

IPM=1 benchmark 文件：

```text
logs/flash_decode_lplb/results_ipm1_ab_retry_20260821/isl1024_osl1024_c16_n160.log/jsonl
logs/flash_decode_lplb/results_ipm1_ab_retry_20260821/isl1024_osl1024_c256_n2560.log/jsonl
logs/flash_decode_lplb/results_ipm1_ab_retry_20260821/isl1024_osl1024_c512_n5120.log/jsonl
```

### 6.2 IPM=2 与同步 mapping 复用 A/B

IPM=2 的三组 1024 输入测试全部成功，但没有优于 IPM=1：

| ISL | OSL | C | IPM iters | Refresh interval | Out tok/s | Total tok/s | TTFT ms | TPOT ms | 相对原始 baseline |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1024 | 1024 | 16 | 2 | 1 | 929.60 | 1859.19 | 660.20 | 16.55 | Out/Total -31.14% |
| 1024 | 1024 | 256 | 2 | 1 | 10160.59 | 20321.18 | 940.97 | 24.04 | Out/Total -22.48% |
| 1024 | 1024 | 512 | 2 | 1 | 15465.53 | 30931.06 | 4315.66 | 27.12 | Out/Total **+16.78%** |

结果目录：

```text
logs/flash_decode_lplb/results_ipm2_ab_20260821/
```

为减少每层每 batch 的 solver 和 EP all-reduce 固定开销，在所有 rank 保持相同调用序列的前提下增加：

```text
SGLANG_LPLB_REFRESH_INTERVAL=1..N
```

非刷新步复用上一轮 `log2phy_prob`，刷新步仍执行完整 count、all-reduce、IPM 和 dispatch。当前源码备份为：

```text
backups/lplb_20260821/lplb_solver.py.before_refresh_interval
```

`IPM=1 + SGLANG_LPLB_REFRESH_INTERVAL=2` 的 1024 输入结果如下：

| ISL | OSL | C | IPM iters | Refresh interval | Out tok/s | Total tok/s | TTFT ms | TPOT ms | 相对原始 baseline |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1024 | 1024 | 16 | 1 | 2 | 984.79 | 1969.57 | 396.80 | 15.81 | Out/Total -27.07% |
| 1024 | 1024 | 256 | 1 | 2 | 10488.79 | 20977.58 | 968.09 | 23.23 | Out/Total -19.97% |
| 1024 | 1024 | 512 | 1 | 2 | 16802.44 | 33604.87 | 3871.33 | 24.41 | Out/Total **+26.88%** |

该配置比原 IPM=1 在 C16、C256、C512 分别变化约 +0.07%、+0.48%、+3.61%。因此 refresh interval 能改善高并发，但不足以消除低中并发与 baseline 的差距；低中并发还需要继续分析 DeepEP/PD 调度和 MoE dispatch 的固定开销。

测试目录：

```text
logs/flash_decode_lplb/results_refresh2_ipm1_ab_20260821/
```

在相同 `IPM=1 + refresh_interval=2` 配置下，8192 输入已完成 C16/C256：

| ISL | OSL | C | Out tok/s | baseline Out tok/s | Out 变化 | Total tok/s | baseline Total tok/s | Total 变化 | TTFT ms | TPOT ms | 成功请求 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 1024 | 16 | 973.49 | 1328.45 | -26.72% | 8761.38 | 11956.04 | -26.72% | 545.59 | 15.80 | 160/160 |
| 8192 | 1024 | 256 | 7159.77 | 6861.28 | **+4.35%** | 64437.90 | 61751.53 | **+4.35%** | 12712.48 | 22.01 | 2560/2560 |
| 8192 | 1024 | 512 | 7417.21 | 7110.57 | **+4.31%** | 66754.87 | 63995.17 | **+4.31%** | 45015.14 | 22.13 | 5120/5120 |

8192 输入测试目录和状态文件为：

```text
logs/flash_decode_lplb/results_refresh2_ipm1_8192_20260821/
```

8192/C512 完成耗时较长，期间 Prefill 日志曾显示 `inflight-req=4` 且队列约 334--366，Decode 仍保持 CUDA Graph=True；最终所有请求成功。这表明 8192/C512 的主要首 token 延迟限制转移到 Prefill→Decode transfer/排队，而不是 LPLB solver 本身。

## 7. 失败经验与修复

1. 旧源码未将 `DeepseekV4ForCausalLM` 加入 LPLB 白名单，触发 `NotImplementedError`；已补充模型适配并备份原文件。
2. 早期启动使用 `CUDA_VISIBLE_DEVICES=4,5,6,7` 同时传入物理 `--base-gpu-id 4`，导致 device 4 不可见；最终移除该 CVD 限制，保持物理 GPU 映射与 IB 设备一致。
3. 使用 `deepep_mode=normal` 时 Decode CUDA Graph 会自动关闭，不符合正式目标；最终固定 `--deepep-mode low_latency` 和完整 Decode Graph。
4. Decode 必须使用 `--moe-runner-backend deep_gemm`；未显式固定 runner 的历史 auto baseline 与本轮 LPLB 配置不能混淆。
5. 服务启动成功不等于请求链路正确；每次重启均执行 whoami 验证，并检查 Prefill/Decode 日志中的 HTTP 200 和请求成功数。

## 8. 当前结论与后续方向

当前已完成 DeepSeek-V4 + LPLB + DeepEP low_latency + DeepGEMM + Decode CUDA Graph + PD 分离的可运行适配，并完成默认 IPM=5 的 8 组 baseline 对比。默认 IPM=5 配置在 3/8 组超过 baseline，且 8/8 组请求成功；优化后的 `IPM=1 + refresh_interval=2` 在 1024/C512 上 Total tok/s 为 33604.87、提升 26.88%，在 8192/C256 和 C512 上分别提升 4.35% 和 4.31%，三组均完成全部请求。

要进一步扩大超过 baseline 的范围，优先级为：

1. 对 `refresh_interval=2` 补齐 8192 输入矩阵，确认长输入下的 Decode 吞吐和 TTFT 变化。
2. 保证 LP mapping 的质量不明显下降后，再评估 refresh interval=4；必须保持所有 DP/EP rank 的 collective 顺序一致，不能简单跳过空 rank 的 all-reduce。
3. 对随机均匀 workload 之外的专家热点/偏斜 workload 做测试，验证 LPLB 的核心负载均衡收益。
4. 单独优化 8192 高并发 TTFT 和 Prefill 排队，避免把 Decode 吞吐提升误认为端到端首 token 延迟提升。
