# MTP + PD 分离整体实验记录（2026-08-24）

## 1. 实验目的

在当前原生 SGLang 0.5.16 派生源码上，固定 PD 分离、MTP/EAGLE、CUDA Graph 和 workload，对 Decode 侧技术做单变量消融。本文同时记录每个配置的部署参数、功能验证、日志位置和性能结果。

## 2. 公共环境与部署方式

- 源码：`/data/ssd2/sglang_v0.5.16_pr32281_fix7_repro_20260823`
- 模型：`/data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash`
- Prefill GPU：0,1,2,3；Decode GPU：4,5,6,7
- Prefill：TP=4、DP=1、EP=1、`flashinfer_mxfp4`、Mooncake、IB `mlx5_0..mlx5_3`
- Decode：TP=4、DP=4、EP=4、DP Attention、DP LM Head、Mooncake、IB `mlx5_4,mlx5_9,mlx5_10,mlx5_11`
- Prefill `mem_fraction_static=0.9`，Decode `mem_fraction_static=0.85`
- `--disable-flashinfer-autotune --disable-radix-cache --disable-overlap-schedule`
- MTP：`--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --speculative-attention-mode decode`
- Decode CUDA Graph：`--cuda-graph-backend-decode full --cuda-graph-bs-decode 1 2 4 8 16 32 64 128`
- 测试 workload：ISL/OSL 为 1024/1024 和 8192/1024；并发 1、16、256、512；请求数为并发的 10 倍；seed=1；随机输入，`tokenize-prompt` 开启。
- 每个服务日志均按变体单独保存，位于 `mtp_ablation_20260824/variants/<variant>/logs/services/`。

### 公共脚本

- Prefill：[`flash_prefill.sh`](../mtp_ablation_20260824/scripts/flash_prefill.sh)
- Decode/MTP：[`flash_decode_mtp.sh`](../mtp_ablation_20260824/scripts/flash_decode_mtp.sh)
- Router：[`flash_router.sh`](../mtp_ablation_20260824/scripts/flash_router.sh)
- 实验目录：[`mtp_ablation_20260824`](../mtp_ablation_20260824)
- 备份目录：[`backups/mtp_ablation_20260824`](../backups/mtp_ablation_20260824)

## 3. 变体与实际生效参数

| 变体 | Decode MoE A2A | Decode MoE Runner | 额外开关 | 矩阵状态 |
|---|---|---|---|---|
| MTP baseline | `megamoe` | `flashinfer_mxfp4` | 无 | 已完成 8/8 |
| MTP + DeepGEMM | `megamoe` | `deep_gemm` | 无 | 已完成 8/8 |
| MTP + Waterfill | `megamoe` | `flashinfer_mxfp4` | `--enable-waterfill` | 已完成 8/8 |
| MTP + LPLB | `megamoe` | `flashinfer_mxfp4` | `--ep-dispatch-algorithm lp` | 已完成 8/8 |
| MTP + TBO | `megamoe` | `flashinfer_mxfp4` | `--enable-two-batch-overlap` | 已完成 8/8 |
| MTP + FP4 indexer | `megamoe` | `flashinfer_mxfp4` | `--enable-deepseek-v4-fp4-indexer` | 7/8；8192/512 高并发失败 |

MTP baseline 的实际 Decode 参数还包括 `--speculative-moe-runner-backend` 未显式设置，因此 server args 中为默认值 `flashinfer_mxfp4`；其主 runner 为表中所列值。所有变体均重新启动服务并重新捕获 CUDA Graph。

## 4. 功能验证结果

每个变体都执行 `validate_pd_whoami.sh`，并执行 10 个随机请求、ISL=1024、OSL=128 的 smoke。

| 变体 | Prefill/Decode health | `你是谁` | Smoke 成功请求 | 关键证据 |
|---|---:|---:|---:|---|
| MTP baseline | 200/200 | `WHOAMI_VALID=True` | 10/10 | target verify、draft decode、draft extend Graph 均完成 |
| MTP + DeepGEMM | 200/200 | `WHOAMI_VALID=True` | 10/10 | `moe_runner_backend=deep_gemm`，Graph 均完成 |
| MTP + Waterfill | 200/200 | `WHOAMI_VALID=True` | 10/10 | `enable_waterfill=True`，Graph 均完成 |
| MTP + LPLB | 200/200 | `WHOAMI_VALID=True` | 10/10 | `ep_dispatch_algorithm='lp'`，Graph 均完成 |
| MTP + TBO | 200/200 | `WHOAMI_VALID=True` | 10/10 | `enable_two_batch_overlap=True` |
| MTP + FP4 indexer | 200/200 | `WHOAMI_VALID=True` | 10/10 | `enable_deepseek_v4_fp4_indexer=True` |

验证日志分别位于：`mtp_ablation_20260824/variants/<variant>/logs/validation/`；Smoke 结果位于对应的 `logs/results/smoke.log` 和 `smoke.jsonl`。

## 5. 完整矩阵结果

字段含义：`Out`=输出 token 吞吐，`Total`=输入+输出 token 吞吐，单位均为 tok/s；TTFT 和 TPOT 单位为 ms。每行均为成功请求的均值。

### 5.1 MTP baseline：MegaMoE + flashinfer_mxfp4

| ISL | OSL | Concurrency | Requests | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 10 | 198.63 | 397.25 | 273.93 | 4.77 |
| 1024 | 1024 | 16 | 160 | 2364.46 | 4728.93 | 605.62 | 5.96 |
| 1024 | 1024 | 256 | 2560 | 22689.10 | 45378.20 | 862.04 | 10.04 |
| 1024 | 1024 | 512 | 5120 | 27353.37 | 54709.55 | 1335.48 | 16.68 |
| 8192 | 1024 | 1 | 10 | 200.37 | 1803.36 | 253.20 | 4.75 |
| 8192 | 1024 | 16 | 160 | 2289.72 | 20607.44 | 638.94 | 6.15 |
| 8192 | 1024 | 256 | 2560 | 7446.73 | 67020.58 | 25740.67 | 7.69 |
| 8192 | 1024 | 512 | 5120 | 7467.32 | 67205.92 | 59008.25 | 7.70 |

### 5.2 MTP + DeepGEMM

| ISL | OSL | Concurrency | Requests | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 10 | 205.15 | 410.31 | 266.44 | 4.62 |
| 1024 | 1024 | 16 | 160 | 2314.89 | 4629.77 | 582.31 | 6.03 |
| 1024 | 1024 | 256 | 2560 | 22713.55 | 45427.11 | 740.71 | 10.13 |
| 1024 | 1024 | 512 | 5120 | 27386.39 | 54772.78 | 1287.71 | 16.76 |
| 8192 | 1024 | 1 | 10 | 203.91 | 1835.15 | 230.04 | 4.68 |
| 8192 | 1024 | 16 | 160 | 2299.26 | 20693.33 | 586.85 | 6.17 |
| 8192 | 1024 | 256 | 2560 | 7501.71 | 67515.37 | 25193.12 | 8.00 |
| 8192 | 1024 | 512 | 5120 | 7516.73 | 67650.60 | 58485.57 | 7.88 |

### 5.3 MTP + Waterfill

| ISL | OSL | Concurrency | Requests | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 10 | 182.51 | 365.02 | 258.56 | 5.23 |
| 1024 | 1024 | 16 | 160 | 2106.90 | 4213.81 | 604.38 | 6.74 |
| 1024 | 1024 | 256 | 2560 | 21712.11 | 43424.22 | 790.17 | 10.56 |
| 1024 | 1024 | 512 | 5120 | 23426.92 | 46853.84 | 1287.98 | 19.94 |
| 8192 | 1024 | 1 | 10 | 187.75 | 1689.79 | 233.80 | 5.10 |
| 8192 | 1024 | 16 | 160 | 2112.43 | 19011.84 | 622.83 | 6.75 |
| 8192 | 1024 | 256 | 2560 | 7470.99 | 67238.92 | 24792.20 | 8.46 |
| 8192 | 1024 | 512 | 5120 | 7549.61 | 67946.46 | 57548.65 | 8.46 |

### 5.4 MTP + LPLB

| ISL | OSL | Concurrency | Requests | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 10 | 155.17 | 310.34 | 263.87 | 6.19 |
| 1024 | 1024 | 16 | 160 | 1825.09 | 3650.18 | 564.08 | 7.88 |
| 1024 | 1024 | 256 | 2560 | 18291.50 | 36582.99 | 765.98 | 12.67 |
| 1024 | 1024 | 512 | 5120 | 19622.80 | 39245.61 | 1157.05 | 24.31 |
| 8192 | 1024 | 1 | 10 | 156.62 | 1409.56 | 231.14 | 6.16 |
| 8192 | 1024 | 16 | 160 | 1792.28 | 16130.55 | 666.21 | 7.97 |
| 8192 | 1024 | 256 | 2560 | 7317.43 | 65856.90 | 23333.83 | 10.56 |
| 8192 | 1024 | 512 | 5120 | 7493.77 | 67443.96 | 55872.34 | 10.59 |

## 6. 初步结论

1. DeepGEMM 在当前 MTP workload 下是最稳定的可替代 runner：相对 MTP baseline，高并发和长输入下 Total tok/s 小幅提升，且 TTFT 通常下降；低并发收益有限，TPOT没有一致性下降。
2. Waterfill 已经实际生效，不是 skip；但随机均匀路由下低/中并发明显增加开销，1024/512 的 TPOT 从 16.68 ms 增至 19.94 ms。8192 高并发的 Total tok/s 有提升，但不能据此断言 Waterfill 普遍有效；它更适合有共享专家热点或路由偏斜的 workload。
3. LPLB 在本次当前源码和 MegaMoE+MTP 路径下可以运行，但性能全面低于 MTP baseline，尤其 1024 输入下 TPOT 明显变差。因此“能启动”不等于“适合当前 workload”。之前的 `DeepseekV4ForCausalLM does not support --ep-dispatch-algorithm lp` 与本次不一致，必须以当前源码、完整 server args 和日志为准。
4. TBO 已完成完整 8 组矩阵；FP4 indexer 完成前 7 组，但 8192/512 高并发出现大量 Mooncake `KVTransferError`，因此该组不计入吞吐对比，必须按失败配置处理。
5. 本报告的完整矩阵是 MTP 技术消融，不是与原始 baseline 的直接最终排名；需要与对应 baseline 表按相同 workload 进一步计算变化百分比。

## 7. 日志、复现与备份

每个变体完整结果路径：

```text
mtp_ablation_20260824/variants/<variant>/logs/services/{prefill,decode,router}/
mtp_ablation_20260824/variants/<variant>/logs/validation/
mtp_ablation_20260824/variants/<variant>/logs/results/
```

完整矩阵由 `repro_runner_ab_20260824/run_matrix.sh` 执行；每组均在 `status.tsv` 中记录开始时间、结束时间、状态和退出码。源码/脚本备份在 [`backups/mtp_ablation_20260824`](../backups/mtp_ablation_20260824)，脚本校验文件为 `SHA256SUMS.scripts`。

### 5.5 MTP + TBO

| ISL | OSL | Concurrency | Requests | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 10 | 200.17 | 400.34 | 277.16 | 4.73 |
| 1024 | 1024 | 16 | 160 | 2339.17 | 4678.34 | 535.52 | 6.05 |
| 1024 | 1024 | 256 | 2560 | 22602.37 | 45204.74 | 746.18 | 10.18 |
| 1024 | 1024 | 512 | 5120 | 24231.09 | 48462.18 | 1176.47 | 19.34 |
| 8192 | 1024 | 1 | 10 | 200.15 | 1801.33 | 230.12 | 4.77 |
| 8192 | 1024 | 16 | 160 | 2286.06 | 20574.56 | 583.82 | 6.23 |
| 8192 | 1024 | 256 | 2560 | 7513.17 | 67618.53 | 25132.73 | 8.00 |
| 8192 | 1024 | 512 | 5120 | 7581.02 | 68229.16 | 57715.61 | 8.07 |

### 5.6 MTP + FP4 indexer

| ISL | OSL | Concurrency | Requests | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT | 状态 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1024 | 1024 | 1 | 10 | 202.54 | 405.08 | 273.97 | 4.67 | 成功 |
| 1024 | 1024 | 16 | 160 | 2343.62 | 4687.25 | 702.44 | 5.96 | 成功 |
| 1024 | 1024 | 256 | 2560 | 22652.39 | 45304.77 | 817.60 | 10.08 | 成功 |
| 1024 | 1024 | 512 | 5120 | 26264.18 | 52528.36 | 1333.77 | 17.51 | 成功 |
| 8192 | 1024 | 1 | 10 | 206.54 | 1858.90 | 234.65 | 4.62 | 成功 |
| 8192 | 1024 | 16 | 160 | 2256.60 | 20309.41 | 554.92 | 6.27 | 成功 |
| 8192 | 1024 | 256 | 2560 | 7647.13 | 68824.13 | 2903.95 | 29.06 | 成功 |
| 8192 | 1024 | 512 | 5120 | — | — | — | — | 失败 |

FP4 indexer 的失败证据：`variants/mtp_fp4_indexer/logs/results/status.tsv` 最后一行为 `8192 1024 512 5120 failed ... exit_code=1`；Prefill/Decode 日志出现 `KVTransferError`、`Decode instance could be dead` 和 `Failed to get kvcache from prefill instance`。虽然 HTTP health 仍为 200，但请求转移不完整，因此不能视为成功吞吐测试。

## 8. DSpark + MegaMoE + 其余 Decode 技术的组合结果

另行新建了 [`DSpark_MegaMoE_AllTech_20260824.md`](DSpark_MegaMoE_AllTech_20260824.md)，实际组合为 Prefill MegaMoE、Decode MegaMoE + DeepGEMM + DSpark + Waterfill + LPLB + TBO + FP4 indexer。该组合完成了服务启动和 CUDA Graph 捕获，但 `你是谁` 在 120 秒内超时，Smoke 未形成有效成功结果，因此没有进入正式吞吐矩阵。该失败边界及完整日志位于 [`dspark_alltech_megamoe_20260824`](../dspark_alltech_megamoe_20260824)。
