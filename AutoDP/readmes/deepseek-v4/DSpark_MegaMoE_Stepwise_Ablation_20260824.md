# DSpark + MegaMoE 逐项消融与全量整合实验（2026-08-24）

## 结论

本轮严格以 `DSpark + Prefill MegaMoE + Decode MegaMoE` 为第 0 阶段，每个后续变体只增加一项技术：DeepGEMM、FP4 indexer、Waterfill、LPLB、TBO。五个单项变体均通过服务启动、Decode CUDA Graph、`你是谁` 和 10 请求 Smoke 验证，并产生 8 组 benchmark 结果；FP4 indexer 的原始 `8192/C256`、`8192/C512` 因 TTFT=0/KV transfer 异常不计入有效结果，已由第 8 节双端 FP4 修复复测替代。

随后尝试整合全部已验证技术（DeepGEMM + FP4 indexer + Waterfill + LPLB + TBO），以及排除 FP4 后的候选最终配置（DeepGEMM + Waterfill + LPLB + TBO）。两次配置的服务和 CUDA Graph 捕获都成功，但第一条 `你是谁` 请求均在 120 秒内超时，因此没有运行整合配置的性能矩阵。当前不能宣称存在一个稳定的全量最终配置，说明“每个单项可运行”不等价于“所有技术可以直接叠加”。

## 1. 统一基线和部署

- 源码：`/data/ssd2/sglang_v0.5.16_pr32281_fix7_repro_20260823`
- 模型：`/data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash`
- Prefill：GPU 0–3，TP4/DP1/EP1，Prefill MegaMoE，Mooncake，IB `mlx5_0..3`，`mem-fraction-static=0.9`
- Decode：GPU 4–7，TP4/DP4/EP4，Decode MegaMoE，Mooncake，IB `mlx5_4,mlx5_9,mlx5_10,mlx5_11`
- Decode 必选：`--deepep-mode low_latency`、DSpark、`--cuda-graph-bs-decode 1 2 4 8 16 32 64 128`
- Router：Prefill `30000`，Decode `30001`，PD Router `13784`
- Workload：随机输入/输出，`ISL/OSL = 1024/1024` 和 `8192/1024`，并发 `1/16/256/512`，请求数为 `10 * concurrency`
- 所有变体均关闭其它未声明技术；每个服务的 stdout/stderr 单独保存。

第 0 阶段的完整基线及 8 组结果见 [DSpark_MegaMoE_PD_Experiment_20260824.md](DSpark_MegaMoE_PD_Experiment_20260824.md)。

## 2. 逐项配置

| 阶段 | Decode runner | 单独打开的技术 | 结果目录 |
|---|---|---|---|
| 0 基线 | `auto` | 无（DSpark + MegaMoE） | `variants/base` |
| 1 | `deep_gemm` | DeepGEMM | `variants/dspark_deepgemm` |
| 2 | `auto` | FP4 indexer | `variants/dspark_fp4_indexer` |
| 3 | `auto` | Waterfill | `variants/dspark_waterfill` |
| 4 | `auto` | LPLB | `variants/dspark_lplb` |
| 5 | `auto` | TBO | `variants/dspark_tbo` |
| 整合 | `deep_gemm` | FP4 indexer + Waterfill + LPLB + TBO | `variants/dspark_all_valid` |

各项技术通过环境开关传入 Decode：`SGLANG_MOE_RUNNER_BACKEND`、`SGLANG_ENABLE_FP4_INDEXER`、`SGLANG_ENABLE_WATERFILL`、`SGLANG_ENABLE_LPLB`、`SGLANG_ENABLE_TBO`。部署脚本副本位于各变体目录中。

## 3. 结果摘要

下面列出最能反映高并发 Decode 行为的 `8192/1024/C512`；每个变体的 8 组原始 benchmark 日志均保留在对应 `logs/results` 目录，包含 total token throughput、TTFT、TPOT 和成功请求数。

| 配置 | 成功请求 | Req/s | Out tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| 第 0 阶段 DSpark + MegaMoE | 5120 | 8.108 | 8302.53 | 74722.79 | 56097.16 | 3.91 | 有效基线 |
| + DeepGEMM | 5120 | 8.140 | 8335.28 | 75017.53 | 55827.19 | 3.95 | 有效 |
| + FP4 indexer | 5120 | 8.178 | 8374.01 | 75366.07 | 0.00 | 58.21 | 请求统计完成，但 TTFT=0、日志含 KV transfer 异常，不纳入严格 TTFT 结论 |
| + Waterfill | 5120 | 8.053 | 8245.96 | 74213.67 | 56071.15 | 4.33 | 有效 |
| + LPLB | 5120 | 8.099 | 8293.17 | 74638.53 | 54688.27 | 5.34 | 有效 |
| + TBO | 5120 | 8.086 | 8287.87 | 74590.85 | 54089.90 | 5.93 | 有效 |
| 全量整合（含 FP4） | — | — | — | — | — | `你是谁` 超时，未跑矩阵 |
| 候选最终（排除 FP4） | — | — | — | — | — | `你是谁` 超时，未跑矩阵 |

单项高并发结果与基线相比没有形成稳定的叠加收益：DeepGEMM 略高，Waterfill/LPLB/TBO 的 total throughput 接近或略低；FP4 indexer 的高并发日志存在请求级 KV transfer 异常，不能只看表面吞吐。

## 4. 功能验证和日志

每个有效变体均满足：

1. Prefill/Decode `/health` 返回 200；
2. Decode 日志出现 `Capture draft verify CUDA graph end`；
3. `你是谁` 请求返回正常；
4. Smoke 10/10 成功。

整合变体的服务和 Graph 捕获也成功，但 `你是谁` 请求超时，Router 日志记录请求开始而没有正常完成。因此两种整合变体均被判定为功能失败，不继续运行正式矩阵。对应目录为 `variants/dspark_all_valid` 和 `variants/dspark_final_valid`。

日志位置：

```text
dspark_stepwise_ablation_20260824/variants/<variant>/logs/services/{prefill,decode,router}
dspark_stepwise_ablation_20260824/variants/<variant>/logs/validation
dspark_stepwise_ablation_20260824/variants/<variant>/logs/results
```

## 5. 备份

本轮脚本、每个变体的部署参数、服务日志、验证结果和 benchmark 原始结果已备份到：

`backups/dspark_stepwise_ablation_20260824`

## 6. 下一步建议

全量整合失败的直接边界是：单项技术分别能通过初始化和低负载请求，但联合启用后请求路径发生阻塞；排除 FP4 后仍然复现，说明问题不只由 FP4 indexer 单独造成。下一步应先用 profiling/服务日志定位 DeepGEMM、Waterfill、LPLB、TBO 的 dispatch/Graph replay 与 Mooncake KV transfer 之间的组合冲突，再做二项组合消融；在整合配置通过 `你是谁` 和 10 请求 Smoke 前，不应运行高并发矩阵。

## 7. 2026-08-25：8192/C512 FP4 indexer + DSpark KV transfer 修复复测

### 7.1 根因

此前 FP4 indexer 的 8192/C512 结果出现 `Mean TTFT=0`，并伴随 Mooncake
`KVTransferError`。该结果是 HTTP 200 空流被 benchmark 错误统计造成的，不是真实性能。

本次定位确认：

1. DSV4 的 62 个 KV buffer 具有不同 stride，需要传递完整的
   `dst_kv_item_lens`，并在源/目标地址计算中分别使用对应 stride。
2. 原先只修了 staging/slice fallback，正常同 TP 的 `send_kvcache` 路径仍使用
   Prefill stride 计算 Decode 地址；已补齐普通路径。
3. 更关键的是，原部署 Prefill 的
   `enable_deepseek_v4_fp4_indexer=False`，Decode 为 `True`，两端 C4/indexer KV
   layout 不一致。Prefill 脚本现支持 `SGLANG_ENABLE_FP4_INDEXER=1`，本轮两端均开启。

此外，hybrid MLA 的 pointer 与 item-lens 现在共用相同的 PP/compression-bucket 映射，
避免按未映射的 `item_lens[layer_id]` 计算 stride。

### 7.2 验证配置

Prefill 为 MegaMoE TP4/DP1/EP1，Decode 为 MegaMoE TP4/DP4/EP4 + DSpark + FP4
indexer；Decode CUDA Graph full，buckets 为 `1 2 4 8 16 32 64 128`。

稳定性验证额外使用：

```text
SGLANG_PD_HIDDEN_RECV_POOL_TOKENS=131072
SGLANG_DSPARK_PD_HIDDEN_POOL_TOKENS=131072
SGLANG_DSPARK_SYNC_PD_HIDDEN_INJECT=1
SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0
SGLANG_MOONCAKE_ALIGN_KV_REGISTRATION=0
```

### 7.3 修复后结果

`你是谁`：HTTP 200，`WHOAMI_VALID=True`；8192/C16 smoke：32/32 成功，输出非空，
TTFT 全部大于 0。

| ISL | OSL | Concurrency | Num prompts | Req/s | Out tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms | 完整性 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8192 | 1024 | 512 | 5120/5120 | 15.357 | 7856.59 | 70728.55 | 29578.64 | 4.01 | 全部成功、无空流 |

结果文件：
`logs/runs/fp4_indexer_kv_transfer_fix_20260825/results/fixed_both_fp4_isl8192_osl1024_c512_n5120.jsonl`

校验结果：`output_lens=5120`、空文本为 0、TTFT 为 0 的请求为 0、
`total_output_tokens_retokenized=2619299`；Prefill/Decode 日志无
`KVTransferError`、`remote mooncake session`、`CUDA error`、`illegal memory` 或
`Traceback`。

### 7.4 备份

- `backups/fp4_indexer_kv_transfer_fix_20260825/conn.py.with_mla_lens_mapping.py`
- `backups/fp4_indexer_kv_transfer_fix_20260825/flash_prefill_megamoe.sh.with_fp4_both_sides`
- 服务日志：`logs/runs/fp4_indexer_kv_transfer_fix_20260825/services/`
- whoami 日志：`logs/runs/fp4_indexer_kv_transfer_fix_20260825/validation/whoami_both_fp4.log`

## 8. 2026-08-25：双端 FP4 indexer 全量 8 组有效复测

第 7 节之后重新启动了干净的 Prefill/Decode 服务，并修正了两处复现问题：

1. Decode 明确使用原实验的 `--moe-a2a-backend megamoe`，避免脚本默认回到
   DeepEP；
2. Prefill 和 Decode 均显式开启 `--enable-deepseek-v4-fp4-indexer`，保证两端
   C4/indexer KV layout 一致。

本轮使用 `conn.py` 的 per-buffer `dst_kv_item_lens`、普通 `send_kvcache` 路径的
source/destination stride 映射，以及 hybrid MLA pointer/lens 映射修复。配置为：

- Prefill：MegaMoE，TP4/DP1/EP1，Mooncake，`mem-fraction-static=0.9`；
- Decode：MegaMoE，TP4/DP4/EP4，DSpark，CUDA Graph full，batch
  `1/2/4/8/16/32/64/128`；
- `SGLANG_ENABLE_FP4_INDEXER=1`（两端）；
- `SGLANG_DSPARK_SYNC_PD_HIDDEN_INJECT=0`、
  `SGLANG_MOONCAKE_ALIGN_KV_REGISTRATION=1`；
- 每组请求数为 `10 × concurrency`，随机输入长度为 1024 或 8192，输出长度为
  1024。

### 8.1 全量结果

| ISL | OSL | Concurrency | Requests | Req/s | Out tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms | 状态 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1024 | 1024 | 1 | 10/10 | 0.334 | 342.42 | 684.84 | 189.07 | 2.74 | PASS |
| 1024 | 1024 | 16 | 160/160 | 3.925 | 4019.71 | 8039.42 | 344.29 | 3.50 | PASS |
| 1024 | 1024 | 256 | 2560/2560 | 34.606 | 35436.57 | 70873.15 | 736.20 | 6.19 | PASS |
| 1024 | 1024 | 512 | 5120/5120 | 48.807 | 49978.86 | 99957.72 | 1178.24 | 8.54 | PASS |
| 8192 | 1024 | 1 | 10/10 | 0.340 | 348.64 | 3137.74 | 204.87 | 2.67 | PASS |
| 8192 | 1024 | 16 | 160/160 | 3.888 | 3980.82 | 35827.37 | 506.80 | 3.40 | PASS |
| 8192 | 1024 | 256 | 2560/2560 | 8.051 | 8244.25 | 74198.22 | 26281.21 | 3.92 | PASS |
| 8192 | 1024 | 512 | 5120/5120 | 8.084 | 8278.07 | 74502.64 | 56218.03 | 3.98 | PASS |

### 8.2 有效性审计

8/8 组均满足：完成请求数等于目标请求数、输出吞吐大于 0、Mean TTFT/TPOT
均大于 0、结果中的非空错误数为 0。两端 server args 均确认
`enable_deepseek_v4_fp4_indexer=True`；Decode 日志持续显示 `cuda graph: True`。

Prefill/Decode 服务日志中未发现 `KVTransferError`、`Failed to get kvcache`、
`Decode instance could be dead`、CUDA illegal memory、`Traceback` 等错误；本轮
`你是谁`验证为 `HTTP=200`、`WHOAMI_VALID=True`。

结果目录：`logs/runs/fp4_indexer_full_retest_20260825/results/` ；逐组有效性记录：
`logs/runs/fp4_indexer_full_retest_20260825/results/status.tsv`；服务日志：
`logs/runs/fp4_indexer_full_retest_20260825/services/`。

本轮部署脚本、源码快照和复测脚本备份于：
`backups/fp4_indexer_full_retest_20260825/`。

## 9. 2026-08-24：其余技术变体完整矩阵结果

以下结果均来自 `dspark_stepwise_ablation_20260824/variants/` 下的原始 JSONL，统一使用
随机 workload、OSL=1024、每组 `10 × concurrency` 请求。除 FP4 indexer 的两组长输入高并发
结果外，各表中的请求均完成；FP4 的有效修复结果以第 8 节为准。

### 9.1 第 0 阶段基线：DSpark + MegaMoE

部署：Prefill/Decode 使用 `--moe-a2a-backend megamoe`，Decode
`--moe-runner-backend auto`、DSpark、DeepEP low-latency 和 CUDA Graph。

| ISL | OSL | C | Requests | Req/s | Out tok/s | Total tok/s | TTFT ms | TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 10/10 | 0.336 | 344.09 | 688.18 | 183.97 | 2.73 |
| 1024 | 1024 | 16 | 160/160 | 3.841 | 3933.39 | 7866.77 | 324.25 | 3.59 |
| 1024 | 1024 | 256 | 2560/2560 | 34.676 | 35508.11 | 71016.22 | 658.82 | 6.20 |
| 1024 | 1024 | 512 | 5120/5120 | 45.133 | 46216.09 | 92432.17 | 1378.47 | 8.70 |
| 8192 | 1024 | 1 | 10/10 | 0.350 | 358.91 | 3230.18 | 204.16 | 2.59 |
| 8192 | 1024 | 16 | 160/160 | 3.914 | 4008.20 | 36073.81 | 495.21 | 3.36 |
| 8192 | 1024 | 256 | 2560/2560 | 8.076 | 8269.72 | 74427.49 | 26291.04 | 3.82 |
| 8192 | 1024 | 512 | 5120/5120 | 8.108 | 8302.53 | 74722.79 | 56097.16 | 3.91 |

原始结果：`variants/base/logs/results/`。

### 9.2 + DeepGEMM

在基线 Decode 配置上增加 `--moe-runner-backend deep_gemm`。

| ISL | OSL | C | Requests | Req/s | Out tok/s | Total tok/s | TTFT ms | TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 10/10 | 0.340 | 348.66 | 697.31 | 190.44 | 2.68 |
| 1024 | 1024 | 16 | 160/160 | 3.922 | 4016.44 | 8032.88 | 339.35 | 3.50 |
| 1024 | 1024 | 256 | 2560/2560 | 34.483 | 35310.59 | 70621.19 | 661.99 | 6.25 |
| 1024 | 1024 | 512 | 5120/5120 | 49.941 | 51139.33 | 102278.66 | 1116.16 | 8.35 |
| 8192 | 1024 | 1 | 10/10 | 0.353 | 361.31 | 3251.79 | 208.96 | 2.56 |
| 8192 | 1024 | 16 | 160/160 | 3.899 | 3992.07 | 35928.66 | 485.64 | 3.39 |
| 8192 | 1024 | 256 | 2560/2560 | 8.072 | 8266.06 | 74394.51 | 26214.34 | 3.90 |
| 8192 | 1024 | 512 | 5120/5120 | 8.140 | 8335.28 | 75017.53 | 55827.19 | 3.95 |

原始结果：`variants/dspark_deepgemm/logs/results/`。

### 9.3 + Waterfill

在基线 Decode 配置上增加 `--enable-waterfill`。

| ISL | OSL | C | Requests | Req/s | Out tok/s | Total tok/s | TTFT ms | TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 10/10 | 0.294 | 301.38 | 602.76 | 185.73 | 3.14 |
| 1024 | 1024 | 16 | 160/160 | 3.296 | 3375.36 | 6750.73 | 360.86 | 4.24 |
| 1024 | 1024 | 256 | 2560/2560 | 31.680 | 32440.37 | 64880.73 | 753.30 | 6.72 |
| 1024 | 1024 | 512 | 5120/5120 | 39.535 | 40483.73 | 80967.47 | 1174.30 | 10.88 |
| 8192 | 1024 | 1 | 10/10 | 0.305 | 312.58 | 2813.18 | 207.73 | 3.00 |
| 8192 | 1024 | 16 | 160/160 | 3.454 | 3536.64 | 31829.74 | 475.99 | 3.89 |
| 8192 | 1024 | 256 | 2560/2560 | 8.039 | 8232.20 | 74089.84 | 25909.40 | 4.30 |
| 8192 | 1024 | 512 | 5120/5120 | 8.053 | 8245.96 | 74213.67 | 56071.15 | 4.33 |

原始结果：`variants/dspark_waterfill/logs/results/`。

### 9.4 + LPLB

在基线 Decode 配置上增加 `--ep-dispatch-algorithm lp`。

| ISL | OSL | C | Requests | Req/s | Out tok/s | Total tok/s | TTFT ms | TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 10/10 | 0.257 | 263.22 | 526.43 | 178.67 | 3.63 |
| 1024 | 1024 | 16 | 160/160 | 2.977 | 3048.45 | 6096.91 | 352.08 | 4.68 |
| 1024 | 1024 | 256 | 2560/2560 | 28.564 | 29249.59 | 58499.18 | 611.53 | 7.73 |
| 1024 | 1024 | 512 | 5120/5120 | 31.693 | 32453.50 | 64907.01 | 1021.75 | 14.20 |
| 8192 | 1024 | 1 | 10/10 | 0.274 | 280.91 | 2528.22 | 205.51 | 3.36 |
| 8192 | 1024 | 16 | 160/160 | 3.090 | 3164.67 | 28482.02 | 493.63 | 4.37 |
| 8192 | 1024 | 256 | 2560/2560 | 8.041 | 8233.90 | 74105.11 | 24972.68 | 5.25 |
| 8192 | 1024 | 512 | 5120/5120 | 8.099 | 8293.17 | 74638.53 | 54688.27 | 5.34 |

原始结果：`variants/dspark_lplb/logs/results/`。本轮 LPLB 已能运行，但在短输入和低/中并发下
TPOT 与 Total throughput 均明显劣于基线。

### 9.5 + TBO

在基线 Decode 配置上增加 `--enable-two-batch-overlap`。

| ISL | OSL | C | Requests | Req/s | Out tok/s | Total tok/s | TTFT ms | TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 10/10 | 0.216 | 221.47 | 442.93 | 184.62 | 4.34 |
| 1024 | 1024 | 16 | 160/160 | 2.722 | 2787.43 | 5574.86 | 320.30 | 5.18 |
| 1024 | 1024 | 256 | 2560/2560 | 26.842 | 27486.69 | 54973.39 | 639.57 | 8.18 |
| 1024 | 1024 | 512 | 5120/5120 | 31.296 | 32047.08 | 64094.17 | 933.58 | 14.28 |
| 8192 | 1024 | 1 | 10/10 | 0.223 | 228.58 | 2057.20 | 205.34 | 4.18 |
| 8192 | 1024 | 16 | 160/160 | 2.780 | 2847.06 | 25623.57 | 490.88 | 4.93 |
| 8192 | 1024 | 256 | 2560/2560 | 8.017 | 8209.19 | 73882.75 | 24362.10 | 5.88 |
| 8192 | 1024 | 512 | 5120/5120 | 8.094 | 8287.87 | 74590.85 | 54089.90 | 5.93 |

原始结果：`variants/dspark_tbo/logs/results/`。TBO 在本轮 workload 下没有覆盖其额外调度开销，
未形成独立收益。

### 9.6 变体有效性与整合失败记录

上述五个单项变体的 `status.tsv` 均为 8/8 `completed`，并且服务日志和 smoke 日志已留存于
各自的 `logs/services/`、`logs/validation/` 和 `logs/results/`。其中原始 FP4 indexer
长输入高并发两组虽 benchmark 进程退出 0，但结果为 TTFT=0 并伴随 KV transfer 问题，故不作为
有效性能结果；第 8 节的双端 FP4 复测已覆盖并修复该问题。

`dspark_all_valid` 和 `dspark_final_valid` 只完成服务启动、CUDA Graph 捕获和部分 smoke，
`你是谁` 未在规定时间内正常完成，因此没有可加入的正式吞吐矩阵，继续保留为整合失败记录。

## 10. 2026-08-25：逐项优化闸门复测

本轮按严格顺序执行：固定 Prefill MegaMoE、Decode DSpark + MegaMoE + CUDA Graph，
每轮只启用一个 Decode 技术；只有当前技术超过同一 baseline，才允许进入下一项。
Waterfill 未通过后，按用户要求切换到 TBO 单项继续验证；TBO 仍不与 Waterfill/LPLB 叠加，
因此本节的 TBO 结果不构成联合配置，也不能推进到“已超过 baseline”的结论。

### 10.1 Waterfill 优化尝试

本轮只在 Decode 加 `--enable-waterfill`，没有启用 TBO 或 LPLB。新增参数：
`SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE=4096`，使小 batch 跳过 routed-count 和
fused materialize，改走本地 shared-expert expansion；默认阈值和默认 Waterfill 行为不变。

| ISL | OSL | C | Requests | Req/s | Out tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms | 相对 baseline |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 16 | 160/160 | 3.28 | 3357.53 | 6715.06 | 350.57 | 4.22 | -14.62% |
| 1024 | 1024 | 16 | 160/160 | 3.46 | 3542.38 | 7084.76 | 355.31 | 4.00 | -10.51%（torch.compile） |
| 1024 | 1024 | 1 | 10/10 | 0.31 | 317.16 | 634.32 | 191.40 | 2.97 | -7.85%（torch.compile） |

baseline 对应 `variants/base` 的 `1024/1024/C16`：Total tok/s=7866.77、TTFT=324.25 ms、
TPOT=3.59 ms。Waterfill 仍然低于 baseline，未通过闸门。

失败原因不是单一的 routed-count kernel：Waterfill 在模型初始化阶段把 fused shared expert
改成额外 routed slot，即使跳过 balance/materialize 的部分开销，Decode 仍承担额外的 routed
expert 调度和计算路径。当前随机 workload 没有共享专家热点，Waterfill 没有可抵消该成本的
负载均衡收益。

结果与日志：
`logs/runs/stepwise_opt_waterfill_20260825/`；源码备份：
`backups/stepwise_opt_20260825/`。

### 10.2 TBO 单项复测

TBO 配置：Decode 使用 `--enable-two-batch-overlap`、DSpark、MegaMoE、Decode CUDA Graph；
Prefill 保持 MegaMoE，未加 TBO。正式复测服务日志位于
`logs/runs/stepwise_opt_tbo_20260825_v3/services/`，结果位于
`logs/runs/stepwise_opt_tbo_20260825_v3/results/`。

| ISL | OSL | C | Requests | Req/s | Out tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms | 相对 baseline |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 10/10 | 0.21 | 214.47 | 428.94 | 187.83 | 4.48 | -37.67% |
| 1024 | 1024 | 16 | 160/160 | 2.71 | 2770.70 | 5541.39 | 327.13 | 5.23 | -29.00% |

对照 baseline 为同一阶段的 `1024/1024/C1=688.18`、`C16=7866.77` Total tok/s。
两组请求均全部成功，Decode 日志确认 `cuda graph: True`。结果表明当前 DSV4 的模型级
TBO 实现只允许 Prefill：`DeepseekV4ForCausalLM._can_run_tbo()` 要求
`is_extend_without_speculative()`，所以 Decode 端不会执行 TBO 的层级重叠计算，但仍会
承担 TBO 调度/dispatcher 相关路径开销；这正是低并发和中并发均低于 baseline 的主要原因。

另做了 Decode 专用门控尝试（源码备份为
`backups/stepwise_opt_20260825/two_batch_overlap.py.tbo_decode_gate_20260825`），结果导致
Decode CUDA Graph 未命中，故未计入性能表并已恢复完整 TBO 配置。

结论：在“只允许 Decode 使用 TBO、同时必须保持 CUDA Graph”的约束下，当前 TBO 没有通过
baseline 闸门；要获得正收益，需要实现真正的 DSV4 Decode TBO，或允许 TBO 用于 Prefill，
而不是继续调大 Decode 端 TBO 参数。

### 10.3 LPLB 状态

Waterfill 和 TBO 均未超过 baseline；LPLB 的动态 LP 路径随后单独进行了稳定性验证，结果和
static fallback 复测见 10.5。

### 10.5 在 DSpark + MegaMoE + TBO 上继续适配 LPLB

本轮固定上一节已经完成 8 组测试的配置，只增加 LPLB：
`SGLANG_ENABLE_LPLB=1`、`--ep-dispatch-algorithm lp`，保留 DSpark、MegaMoE、Decode
TBO、FP4 indexer 和 Decode CUDA Graph。Prefill 没有增加 LPLB。

#### 动态 LP 稳定性结果

动态配置使用 `SGLANG_LPLB_STATIC_FALLBACK=0`、`SGLANG_LPLB_REFRESH_INTERVAL=1`。
服务启动成功，43 层 LPLB solver 和 CUDA IPM solver 均完成初始化；但第一条通过 PD 路由的
`你是谁` 请求一直等待，Decode 服务没有产生完成响应日志。该问题不是启动时的
`NotImplementedError`，而是动态 LP 在当前 TBO/DSpark 请求路径中的 dispatch collective
不同步。该版本没有继续跑 benchmark，也不把它作为性能结果。

#### Static fallback 复测

为验证问题是否只在动态求解阶段，使用已有的 rank-aware static fallback：
`SGLANG_LPLB_STATIC_FALLBACK=1`、`SGLANG_LPLB_REFRESH_INTERVAL=8`。此配置仍使用
`--ep-dispatch-algorithm lp` 并完成 LPLB solver 初始化，但热路径使用静态专家位置映射，
不执行每 batch 的 LP all-reduce/solve。`你是谁` 通过，8 组请求全部成功，服务日志没有
Traceback、RuntimeError、SIGQUIT 或 scheduler exception。

| ISL | OSL | C | Completed | Req/s | Out tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms | 相对 TBO-only Total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 10/10 | 0.19 | 195.72 | 391.44 | 196.74 | 4.92 | -11.07% |
| 1024 | 1024 | 16 | 160/160 | 2.52 | 2579.13 | 5158.26 | 336.71 | 5.58 | -5.54% |
| 1024 | 1024 | 256 | 2560/2560 | 25.27 | 25877.07 | 51754.14 | 728.23 | 8.64 | -4.25% |
| 1024 | 1024 | 512 | 5120/5120 | 31.82 | 32585.31 | 65170.63 | 1034.94 | 14.06 | +0.57% |
| 8192 | 1024 | 1 | 10/10 | 0.22 | 221.38 | 1992.39 | 202.40 | 4.32 | -2.77% |
| 8192 | 1024 | 16 | 160/160 | 2.62 | 2684.61 | 24161.48 | 556.70 | 5.12 | -4.09% |
| 8192 | 1024 | 256 | 2560/2560 | 7.99 | 8183.86 | 73654.70 | 24329.94 | 6.03 | -0.07% |
| 8192 | 1024 | 512 | 5120/5120 | 8.07 | 8262.40 | 74361.64 | 54211.71 | 6.09 | -0.24% |

这里的 TBO-only 对照是 10.4 的 v5 结果，而不是早期“原始 baseline+TBO”结果。结论是：
在当前随机均匀 workload 下，static fallback 只在 1024/C512 有约 0.57% 的微小收益，
其余 7 组均低于 TBO-only；动态 LPLB 还没有通过稳定性门槛，因此不能宣称 LPLB 已经带来
独立加速。要继续优化，应先修复 TBO 子 batch/DSpark verify 与 LPLB collective 的调用
次数和顺序一致性，再比较真正动态 LP；否则只能把 static fallback 作为稳定性兼容模式。

本轮日志和结果：

- 动态 LP 挂起尝试：`logs/runs/dspark_megamoe_tbo_lplb_20260825_v1/`
- static fallback 完整矩阵：`logs/runs/dspark_megamoe_tbo_lplb_20260825_v2/`
- 最终 static fallback `你是谁`：`logs/runs/dspark_megamoe_tbo_lplb_20260825_v2/validation/whoami.json`

### 10.4 Prefill MegaMoE + Decode DSpark/MegaMoE + Decode TBO 适配复测

本次按“当前 DSpark + MegaMoE 配置上单独加入 TBO”的要求重新适配并完成 8 组全量测试。Prefill
保持 MegaMoE，不启用 TBO；Decode 使用 DSpark、MegaMoE、Decode CUDA Graph 和
`--enable-two-batch-overlap`。本轮没有加入 Waterfill、LPLB 或其它联合优化。

部署要点：

- Decode：`CUDA_VISIBLE_DEVICES=4,5,6,7`，`--tp-size 4 --dp-size 4 --ep-size 4`，
  `--enable-dp-attention`，`--moe-a2a-backend megamoe`，`--moe-runner-backend auto`，
  `--speculative-algorithm DSPARK`，`--speculative-attention-mode decode`，
  `--deepep-mode low_latency` 不适用于 MegaMoE A2A，本配置的 A2A 是 MegaMoE；
  Decode 仍使用 `--cuda-graph-bs-decode 1 2 4 8 16 32 64 128` 和
  `--enable-two-batch-overlap`。
- 关键运行环境：`SGLANG_ENABLE_TBO=1`、`SGLANG_ENABLE_FP4_INDEXER=1`、
  `DS_MOE_A2A_BACKEND=megamoe`、`SGLANG_RAGGED_VERIFY_MODE=static`、
  `SGLANG_DEEPEP_LL_SPLIT_TOKENS=1024`，静态显存比例为 0.8，`--max-running-requests 2048`。
- Prefill 使用已有的 Prefill MegaMoE 服务和 1P1D Mooncake 通道；benchmark 统一经
  `http://127.0.0.1:13784` 路由，以确保 PD transfer 和请求完成状态都被计入。

源码适配：

1. 原 DSV4 `_can_run_tbo()` 只允许 `global_forward_mode.is_extend_without_speculative()`，
   因而 Decode TBO 只产生调度路径开销，模型层没有真正执行 Decode TBO。本次为 DSV4 增加
   Decode 操作编排，覆盖 attention、gate、expert dispatch/combine、shared expert 和输出阶段。
2. DSV4 TBO 的 Decode schedule 复用了原有 DeepSeek Decode TBO 的两阶段 dispatch/combine
   结构，并保留 MegaMoE EP 路径的 `op_dispatch_a/op_dispatch_b/op_combine_a/op_combine_b`。
3. 为保留 DSpark 所需的分层 hidden-state capture，TBO 两个子 batch 的 aux hidden state
   先分别采集，再按 parent token range 合并。
4. DSpark Target-Verify 的真实 `forward_batch.forward_mode` 是 `TARGET_VERIFY`，但
   `global_forward_mode` 会被 TBO 协调层改写为 `EXTEND`。若只检查 global mode，C512 长跑会
   把 Raw-Verify metadata 送入普通 TBO attention，触发
   `DSV4RawVerifyMetadata.core_attn_metadata` 异常。本次改为检查当前 batch 的实际 mode：
   普通 Decode 可进入 TBO，Target-Verify 强制回到原 DSpark eager/Graph 路径。

适配过程中已记录的失败：

- v1：DSV4 仍是 Prefill-only strategy，1024/C512 触发
  `NotImplementedError`，仅低并发结果有效。
- v2/v3：尝试用 `spec_info` 或 global mode 排除 Target-Verify，C512 长跑仍触发
  `AttributeError: DSV4RawVerifyMetadata has no attribute core_attn_metadata`，说明判断层级不对。
- v4：小批量预验证仍复现同一问题；随后定位到 per-batch mode，改为实际 mode 判断。
- v5：C512/512 预验证 512/512 成功，随后完成正式 8 组，服务日志无 Traceback、SIGQUIT
  或 scheduler exception。

最终完整结果（v5；所有行均为 `Completed/Num prompts` 全部成功）：

| ISL | OSL | Concurrency | Completed | Req/s | Out tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 10/10 | 0.21 | 220.09 | 440.18 | 190.50 | 4.36 |
| 1024 | 1024 | 16 | 160/160 | 2.67 | 2730.28 | 5460.56 | 321.68 | 5.31 |
| 1024 | 1024 | 256 | 2560/2560 | 26.39 | 27025.91 | 54051.82 | 669.84 | 8.30 |
| 1024 | 1024 | 512 | 5120/5120 | 31.64 | 32400.08 | 64800.15 | 960.86 | 14.24 |
| 8192 | 1024 | 1 | 10/10 | 0.22 | 227.68 | 2049.15 | 202.86 | 4.20 |
| 8192 | 1024 | 16 | 160/160 | 2.73 | 2799.23 | 25193.03 | 495.44 | 5.03 |
| 8192 | 1024 | 256 | 2560/2560 | 8.00 | 8189.33 | 73703.97 | 24436.54 | 5.89 |
| 8192 | 1024 | 512 | 5120/5120 | 8.09 | 8282.44 | 74541.94 | 54236.36 | 5.90 |

日志和结果：

- 服务日志：`logs/runs/dspark_megamoe_tbo_adapt_20260825_v5/services/decode/`
- `你是谁` 验证：`logs/runs/dspark_megamoe_tbo_adapt_20260825_v5/validation/whoami.json`
- 正式结果：`logs/runs/dspark_megamoe_tbo_adapt_20260825_v5/results/`
- 当前源码备份：`backups/stepwise_opt_20260825/dspark_megamoe_tbo_adapt_20260825/`，其中
  `deepseek_v4.py.after_actual_mode_guard` 是本次最终实际 mode 修复后的备份；适配前原文件和
  各阶段失败前文件也保存在同一目录。

本轮结果证明：Decode TBO 已能在普通 Decode batch 上实际运行，同时不再破坏 DSpark
Target-Verify；但这只是“DSpark + MegaMoE + TBO”配置的完整可运行结果，是否超过 baseline
仍应使用同一版本、同一随机 workload 的 baseline 表逐项比较，不能拿此前“原始 baseline+TBO”
结果直接替代本轮 DSpark+TBO 对照。
