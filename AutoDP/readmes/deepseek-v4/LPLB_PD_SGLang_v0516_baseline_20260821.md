# 原生 SGLang 0.5.16 PD 分离 LPLB：baseline 对比记录

日期：2026-08-21

## 1. 实验范围

本报告记录基于 `/data/ssd2/sglang_v0.5.16` 的 DeepSeek-V4-Flash PD 分离 Decode 实验，并统一整理为 baseline 对比表格。

Decode 侧配置为：DeepEP `low_latency`、DeepGEMM、Decode full CUDA Graph、TP4/DP4/EP4、Mooncake PD 分离。Prefill 使用 GPU 0–3，Decode 使用 GPU 4–7，Router 使用端口 13784。

源码通过启动脚本显式固定：

```bash
export PYTHONPATH=/data/ssd2/sglang_v0.5.16/python:${PYTHONPATH:-}
```

每组请求数为 `10 × concurrency`，指标包括 Out tok/s、Total tok/s、Mean TTFT 和 Mean TPOT。

## 2. 原始 baseline

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

## 3. LPLB 正式结果与 baseline 对比

正式矩阵使用 LPLB IPM=5，结果全部 8/8 组请求成功。该表是当前 LPLB 的正式 baseline 对照：

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

正式 LPLB IPM=5 结果中，8 组有 3 组超过 baseline：1024/C512、8192/C256、8192/C512。低并发仍是主要短板。

结果目录：

```text
logs/flash_decode_lplb/results_original_baseline_ab_20260821/
```

## 4. 低并发专项隔离结果

为定位低并发损失，先验证 DeepGEMM、PD 和 CUDA Graph 本身的影响，再隔离 LP 映射和 HashTopK solver。

| 配置 | ISL/OSL/C | 成功请求 | Out tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms | 相对 baseline Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepGEMM + static，冗余 0 | 1024/1024/16 | 160/160 | 1360.98 | 2721.96 | 459.65 | 11.27 | **+0.89%** |
| DeepGEMM + static，冗余 16 | 1024/1024/16 | 160/160 | 1356.06 | 2712.12 | 447.15 | 11.33 | **+0.53%** |
| LP + first-copy fallback，冗余 16 | 1024/1024/16 | 160/160 | 1283.12 | 2566.25 | 435.98 | 12.02 | -4.88% |
| LP + rank-aware fallback，冗余 16 | 1024/1024/16 | 160/160 | 1248.37 | 2496.75 | 765.57 | 12.02 | -7.45% |
| LP + HashTopK fallback，rank-aware，冗余 0 | 1024/1024/16 | 160/160 | 1326.92 | 2653.84 | 430.33 | 11.62 | -1.63% |

最后一项是当前低并发最佳配置。它已经证明 HashTopK 中的 LP solver 是重要瓶颈，但仍略低于原始 baseline；因此不能把当前 fallback 结果标为“超过 baseline”。

## 5. 已完成的适配与修复

1. 将 `DeepseekV4ForCausalLM` 加入 LPLB 支持白名单，修复原始启动时的 `NotImplementedError`。
2. 保留 DeepEP `low_latency`、Decode CUDA Graph 和 DeepGEMM 配置，确保测试不是切换到 DSpark、TBO 或其它历史修改路径。
3. 在 LP solver 中加入 IPM iteration 和 refresh interval 参数，用于受控 A/B：`SGLANG_LPLB_IPM_ITERS`、`SGLANG_LPLB_REFRESH_INTERVAL`。
4. 为低并发 fallback 增加 rank-aware static expert 映射，复用原生 `compute_logical_to_rank_dispatch_physical_map`，并按 metadata/layer/rank 缓存。
5. 修复 DeepSeek-V4 默认 fused `HashTopK` 路径仍然调用 LP solver 的遗漏；fallback 模式下 HashTopK 现在跳过 solver，统一使用 static 映射。
6. 所有服务重启后执行 `validate_pd_whoami.sh`；最近一次返回 `HTTP=200`、`WHOAMI_VALID=True`，160/160 benchmark 请求成功。

## 6. 日志、结果和备份

正式 LPLB benchmark：

```text
logs/flash_decode_lplb/results_original_baseline_ab_20260821/
```

当前 HashTopK fallback 低并发结果：

```text
logs/flash_decode_lplb/results_hash_fallback_r0_20260821/isl1024_osl1024_c16_n160.log
logs/flash_decode_lplb/results_hash_fallback_r0_20260821/isl1024_osl1024_c16_n160.jsonl
```

对应 decode 服务日志：

```text
logs/services/lplb_decode/decode_lplb_20260821_100309_pid1479349.log
```

实际文件名以目录中最新日志为准；日志中应同时包含 CUDA Graph capture、DeepEP low_latency、DeepGEMM warmup 和 PD warmup。

源码备份：

```text
backups/lplb_20260821/lplb_solver.py.before_deepseek_v4_support
backups/lplb_20260821/lplb_solver.py.before_refresh_interval
backups/lplb_20260821/lplb_solver.py.before_static_fallback
backups/lplb_20260821/expert_location_dispatch.py.before_rank_aware_static_fallback
backups/lplb_20260821/hash_topk.py.before_static_fallback
backups/lplb_20260821/flash_decode_lplb_v0516.sh.before_control_params
```

## 7. 当前结论与后续方向

当前 LPLB 的高并发收益已经确认，但低并发仍未超过原始 baseline。隔离结果表明：

- DeepGEMM + static dispatch 在 C16 已经超过 baseline，说明 PD、DeepGEMM 和 CUDA Graph 不是低并发损失的主因。
- LP 动态映射及其 solver/HashTopK 路径是主要损失来源。
- 跳过 HashTopK solver 后性能明显恢复，但仍有约 1.63% 差距。

后续若要证明动态 LP solver 在低并发下本身带来独立收益，仍需单独比较 solver 开启与 fallback 的 A/B；但作为当前 LPLB PD 部署的低并发工作配置，最终结果见下一节。

## 8. 低并发优化后的最终结果（2026-08-21）

针对上一节的低并发瓶颈，最终采用以下配置：

```text
SGLANG_LPLB_STATIC_FALLBACK=1
SGLANG_LPLB_REDUNDANT_EXPERTS=0
--ep-dispatch-algorithm lp
--deepep-config {"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}
--enable-dp-attention
--enable-dp-lm-head 未启用（与原始 baseline 一致）
--moe-a2a-backend deepep --deepep-mode low_latency
--moe-runner-backend deep_gemm
--cuda-graph-backend-decode full
```

低并发 fallback 通过 LP 部署入口使用缓存的 rank-aware static physical expert map，并把 fused HashTopK 直接切到 static hot path，避免每个 forward 进入 LP solver 分支。高并发正式矩阵仍使用动态 LPLB IPM 路径；因此这里的结果应准确称为“LPLB 部署下的低并发 static fallback”，不能误称为低并发动态 LP solver 的独立收益。

| ISL | OSL | Concurrency | 成功请求 | 本轮 Out tok/s | baseline Out tok/s | Out 变化 | 本轮 Total tok/s | baseline Total tok/s | Total 变化 | 本轮 TTFT ms | baseline TTFT ms | 本轮 TPOT ms | baseline TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 10/10 | 108.14 | 107.55 | **+0.55%** | 216.28 | 215.09 | **+0.55%** | 294.72 | 332.51 | 8.97 | 8.98 |
| 1024 | 1024 | 16 | 160/160 | 1372.53 | 1348.95 | **+1.75%** | 2745.07 | 2697.90 | **+1.75%** | 422.65 | 631.91 | 11.18 | 11.25 |
| 1024 | 1024 | 256 | 2560/2560 | 14061.82 | 13106.67 | **+7.29%** | 28123.64 | 26213.34 | **+7.29%** | 1426.20 | 2284.50 | 16.48 | 16.82 |
| 8192 | 1024 | 1 | 10/10 | 107.89 | 106.52 | **+1.29%** | 971.00 | 958.65 | **+1.29%** | 235.81 | 348.65 | 9.05 | 9.05 |
| 8192 | 1024 | 16 | 160/160 | 1356.19 | 1328.45 | **+2.09%** | 12205.67 | 11956.04 | **+2.09%** | 547.10 | 607.99 | 11.13 | 11.23 |
| 1024 | 1024 | 512 | 5120/5120 | 17586.51 | 13243.63 | **+32.80%** | 35173.03 | 26487.25 | **+32.80%** | 4956.08 | 16554.06 | 21.92 | 21.32 |
| 8192 | 1024 | 256 | 2560/2560 | 7460.47 | 6861.28 | **+8.73%** | 67144.21 | 61751.53 | **+8.73%** | 18757.27 | 21913.55 | 14.51 | 14.34 |
| 8192 | 1024 | 512 | 5120/5120 | 7384.66 | 7110.57 | **+3.85%** | 66461.92 | 63995.17 | **+3.85%** | 52941.89 | 55535.20 | 14.50 | 14.42 |

最终 8 组配置均超过原始 baseline，且所有请求成功。C256/C512 的最终配置结果仍保留较大的吞吐优势；低并发优化的主要收益来自：

1. HashTopK 不再执行无效的 LP solver 查找/分支。
2. rank-aware static map 与原生 static dispatch 保持一致。
3. DeepEP low_latency 使用 baseline 的 96-SMS dispatch/combine 配置。
4. 去掉 baseline 中不存在的 `enable-dp-lm-head`，减少 C1 固定开销。

最终配置 benchmark 文件：

```text
logs/flash_decode_lplb/results_baseline_flags_effective_static_20260821/
```

该目录现在包含 8 个最终配置样例的 `.log` 和 `.jsonl` 文件，覆盖：

```text
1024/C1, 1024/C16, 1024/C256, 1024/C512
8192/C1, 8192/C16, 8192/C256, 8192/C512
```

最终服务日志：

```text
logs/services/lplb_decode/decode_lplb_20260821_102953_pid1499315.log
```

该日志确认 Decode CUDA Graph capture、DeepEP low_latency、DeepGEMM warmup 和 PD disaggregation warmup 均完成；重启后的 `validate_pd_whoami.sh` 返回 `HTTP=200`、`WHOAMI_VALID=True`。
