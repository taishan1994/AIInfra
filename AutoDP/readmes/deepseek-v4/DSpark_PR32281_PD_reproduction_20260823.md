# DSpark PR32281 PD 分离复现记录（2026-08-23）

## 目的

复现 `DSpark_PR32281_PD_优化实验报告_20260822.md` 中的最佳 DSpark 配置，作为后续逐项加入其他优化技术的唯一对照基线。本次没有启用 Waterfill、LPLB、TBO、FP4 indexer、MegaMoE，也没有把多个候选技术一次性叠加。

## 代码与备份

- 复现源码目录：`/data/ssd2/sglang_v0.5.16_pr32281_fix7_repro_20260823`
- 源码归档：`backups/dspark_pr32281_20260821_fix7/sglang_v0.5.16_after_pr32281_dspark_fix7.tar.gz`
- 归档 SHA256：`bdc843e5e88e015697cf1497f1f585a1e2c83d40dddbd4c48ad306bded3b93fe`
- 复现脚本目录：`repro_pr32281_20260823/`
- 结果目录：`logs/results/repro_pr32281_20260823/`
- 服务日志目录：`logs/services/repro_pr32281_20260823/`

源码中已确认包含 PR32281 DSpark 修复：`models/dspark.py` 的 `forward_embed`、DP metadata、`schedule_batch.py` 的 None-safe merge，以及 DSpark worker idle input 处理。

## 部署配置

### Prefill

- GPU 0–3，TP4、DP1、EP1
- `--disaggregation-mode prefill`
- Mooncake transfer，IB：`mlx5_0`–`mlx5_3`
- `--moe-runner-backend flashinfer_mxfp4`
- 禁用 FlashInfer autotune、Radix Cache 和 overlap schedule
- `mem-fraction-static=0.9`
- `max-running-requests=256`
- `max-prefill-tokens=16384`、`chunked-prefill-size=16384`
- DSpark hidden thread pool 为 2；PD hidden pool 为 131072

### Decode

- GPU 4–7，TP4、DP4、EP4，启用 DP attention 和 DP LM head
- `--disaggregation-mode decode`
- `--moe-runner-backend auto`
- `--moe-a2a-backend deepep`
- DeepEP `low_latency`，normal dispatch/combine `num_sms=96`
- DSpark draft model：`/data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash-dspark`
- DSpark SPS：`logs/flash_decode_dspark/archive_20260803_091434/dspark_sps.json`
- `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024`
- `SGLANG_DEEPEP_LL_SPLIT_TOKENS=1024`
- 静态 ragged verify
- target 和 draft 均启用完整 CUDA Graph，batch size 1、2、4、8、16、32、64、128
- 禁用 Waterfill、TBO、FP4 indexer、LPLB/冗余专家

完整启动参数保存在 `repro_pr32281_20260823/flash_prefill_baseline.sh` 和 `repro_pr32281_20260823/flash_decode_dspark.sh`。

## Whoami 与完整性验证

验证文件：`logs/services/repro_pr32281_20260823/whoami.log`

- HTTP：200
- `WHOAMI_VALID=True`
- 返回内容为 DeepSeek 中文身份回答
- 8 个测试组全部完成，成功请求数分别等于目标请求数：10、160、2560、5120、10、160、2560、5120
- 结果文件中的 retokenized output token 数与目标规模一致，没有大面积失败或空结果
- 服务日志未发现 Traceback、CUDA error、NCCL error、NotImplementedError、failed register

## 复现结果

| ISL | OSL | Concurrency | Num prompts | Req/s | Out tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 10 | 0.284 | 290.84 | 581.67 | 276.62 | 3.17 |
| 1024 | 1024 | 16 | 160 | 3.219 | 3295.97 | 6591.95 | 604.05 | 4.07 |
| 1024 | 1024 | 256 | 2560 | 30.937 | 31679.06 | 63358.12 | 788.76 | 6.88 |
| 1024 | 1024 | 512 | 5120 | 38.808 | 39739.25 | 79478.49 | 1151.18 | 11.10 |
| 8192 | 1024 | 1 | 10 | 0.295 | 302.04 | 2718.38 | 234.93 | 3.08 |
| 8192 | 1024 | 16 | 160 | 3.357 | 3437.58 | 30938.20 | 601.39 | 3.92 |
| 8192 | 1024 | 256 | 2560 | 7.396 | 7573.62 | 68162.62 | 28439.71 | 4.44 |
| 8192 | 1024 | 512 | 5120 | 7.428 | 7606.20 | 68455.80 | 60992.07 | 4.49 |

## 与报告历史 DSpark 结果的差异

本次结果与历史结果处于同一水平：

- 1024/C512：本次 79478.49，历史 77639.20，约 +2.37%
- 8192/C256：本次 68162.62，历史 67771.79，约 +0.58%
- 8192/C512：本次 68455.80，历史 67319.33，约 +1.69%

其余组也在正常运行波动范围内；没有出现此前“一股脑整合所有技术”后低于 DSpark 基线的情况。因此这套配置可以作为后续实验的可信起点。

## 后续实验方法

后续每一步只改变一个因素，并重新执行同一套 8 组测试：

1. 纯 DSpark（本文件，固定基线）。
2. 只加入 FP4 indexer。
3. 在基线或上一步最佳结果上只加入 Waterfill。
4. 只加入 LPLB。
5. 只加入 TBO。
6. 单独评估显式 DeepGEMM、MegaMoE 或其他 runner；runner 替换属于 A/B 分支，不与原 runner 同时宣称为“叠加收益”。

每一步都必须保留独立启动脚本、源码/参数快照、prefill/decode 日志、whoami 验证、完整请求数和 Total tok/s、TTFT、TPOT。只有单项技术在相同 workload 上稳定超过基线，才进入下一步组合实验。

## Step 1：仅加入 FP4 indexer（2026-08-23）

实验脚本和备份位于 `repro_pr32281_20260823/step1_fp4_indexer/` 与 `backups/dspark_incremental_20260823/step1_fp4_indexer/`。唯一新增参数是 decode 端的 `--enable-deepseek-v4-fp4-indexer`，其余配置与本报告的 DSpark 基线完全一致。

已完成的有效结果如下：

| ISL | OSL | Concurrency | Total tok/s | Mean TTFT ms | Mean TPOT ms | 相对 DSpark 基线 |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 576.15 | 275.13 | 3.20 | -0.95% |
| 1024 | 1024 | 16 | 6587.24 | 599.48 | 4.09 | -0.07% |
| 1024 | 1024 | 256 | 63420.51 | 767.26 | 6.96 | +0.10% |
| 1024 | 1024 | 512 | 76988.53 | 1249.71 | 10.60 | -3.13% |
| 8192 | 1024 | 1 | 2548.53 | 232.09 | 3.31 | -6.25% |
| 8192 | 1024 | 16 | 27191.46 | 505.94 | 4.65 | -12.11% |

8192/C256 的服务端返回了 2560 个请求，但结果完整性校验失败：`total_output_tokens_retokenized=39994`，而目标应约为 262 万；同时 TTFT 为 50.37 ms、TPOT 为 31.87 ms，明显不符合其他组。随后为避免继续浪费时间而中止尚未完成的 8192/C512 时，prefill 日志出现了对应的 `Decode instance could be dead` / `KVTransferError`；因此该组及 8192/C512 均不计入有效性能结果，不能据此断言 FP4 indexer 单独导致了 Mooncake 会话失效。

结论：FP4 indexer 在本 workload 上没有显示独立收益，低并发和长上下文反而明显下降，并在高并发长上下文触发 PD 传输会话异常。本项不进入后续组合；下一项应回到纯 DSpark 基线，只增加一个新的候选技术。

## Step 2：仅加入 Waterfill（2026-08-23）

实验脚本和备份位于 `repro_pr32281_20260823/step2_waterfill/` 与 `backups/dspark_incremental_20260823/step2_waterfill/`。唯一新增参数是 decode 端的 `--enable-waterfill`；DSpark、DeepEP `low_latency`、`--moe-runner-backend auto`、CUDA Graph、PD 参数和 workload 均保持不变。

Waterfill 已由启动日志确认实际生效：decode 端输出 `Waterfill is enabled with moe_a2a_backend='deepep'` 和 `Prepared 43 Waterfill TopK modules`。whoami 返回 HTTP 200 且 `WHOAMI_VALID=True`。

| ISL | OSL | Concurrency | Total tok/s | Mean TTFT ms | Mean TPOT ms | 相对 DSpark 基线 |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 498.44 | 273.54 | 3.75 | -14.30% |
| 1024 | 1024 | 16 | 5616.77 | 589.20 | 4.92 | -14.79% |
| 1024 | 1024 | 256 | 54213.84 | 825.93 | 8.06 | -14.43% |
| 1024 | 1024 | 512 | 66417.70 | 1343.26 | 13.33 | -16.43% |
| 8192 | 1024 | 1 | 2384.56 | 233.81 | 3.55 | -12.28% |
| 8192 | 1024 | 16 | 27017.46 | 623.66 | 4.53 | -12.66% |
| 8192 | 1024 | 256 | 66822.30 | 28435.20 | 5.10 | -1.97% |
| 8192 | 1024 | 512 | 66905.77 | 61758.78 | 5.17 | -2.26% |

8 组全部为 PASS；每组成功请求数等于目标数，retokenized output token 数与目标规模一致。服务日志未发现 Traceback、CUDA/NCCL error、illegal memory access 或 PD transfer failure。

结论：Waterfill 在该随机均匀路由 workload 上没有独立正收益，反而引入明显 dispatch/materialize 开销，因此不纳入后续组合。后续增量实验应继续回到纯 DSpark 基线，优先评估不会改变路由 metadata 的单项技术；Waterfill 只有在共享专家热点或偏斜路由 workload 上证明收益后，才值得重新组合。

## Step 3：仅加入 LPLB（2026-08-23）

实验脚本和备份位于 `repro_pr32281_20260823/step3_lplb/` 与 `backups/dspark_incremental_20260823/step3_lplb/`。相对纯 DSpark 基线只增加 LPLB 入口和其 DeepSeek-V4 适配所需参数：

```text
--ep-num-redundant-experts 0
--ep-dispatch-algorithm lp
SGLANG_LPLB_STATIC_FALLBACK=1
SGLANG_LPLB_IPM_ITERS=1
SGLANG_LPLB_REFRESH_INTERVAL=2
```

Waterfill、FP4 indexer、TBO、显式 DeepGEMM、MegaMoE 均未启用。LPLB 服务成功完成 target/draft CUDA Graph 捕获，Router whoami 返回 HTTP 200 且 `WHOAMI_VALID=True`。

| ISL | OSL | Concurrency | Total tok/s | Mean TTFT ms | Mean TPOT ms | 相对 DSpark 基线 |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 572.49 | 277.54 | 3.22 | -1.58% |
| 1024 | 1024 | 16 | 6555.01 | 579.73 | 4.11 | -0.56% |
| 1024 | 1024 | 256 | 62687.77 | 884.03 | 6.93 | -1.06% |
| 1024 | 1024 | 512 | 82474.11 | 1520.95 | 10.33 | +3.77% |
| 8192 | 1024 | 1 | 2659.79 | 231.43 | 3.16 | -2.15% |
| 8192 | 1024 | 16 | 31099.24 | 569.86 | 3.91 | +0.52% |
| 8192 | 1024 | 256 | 67129.67 | 28970.23 | 4.41 | -1.52% |
| 8192 | 1024 | 512 | 67449.12 | 61937.97 | 4.48 | -1.47% |

8 组全部为 PASS；每组成功请求数等于目标数，retokenized output token 数与目标规模一致，服务日志未发现 Traceback、CUDA/NCCL error、illegal memory access 或 PD transfer failure。

结论：LPLB 的 static fallback 有效消除了低并发动态 solver 的大部分固定开销，但在本次随机均匀 workload 上没有形成普遍提升。它在 1024/C512 获得 +3.77%，8192/C16 仅 +0.52%，其余组略低于 DSpark。LPLB 可以作为后续组合候选，但必须保留 `redundant experts=0` 和 static fallback；不能直接使用旧的 redundant-experts=16、IPM=5 全动态配置。

## Step 4：仅加入 TBO（2026-08-23，启动失败）

实验脚本和备份位于 `repro_pr32281_20260823/step4_tbo/` 与 `backups/dspark_incremental_20260823/step4_tbo/`。相对纯 DSpark 基线唯一新增的是 decode 参数：

```bash
--enable-two-batch-overlap
```

Prefill 服务可以正常启动并完成 PD warmup；decode 服务在初始化 decode CUDA Graph 时失败，未进入请求测试。关键错误为：

```text
AttributeError: 'DFlashVerifyInput' object has no attribute 'retrieve_index'
```

调用链为 `decode_cuda_graph_runner -> two_batch_overlap.capture_one_batch_size -> split_spec_info`。这说明当前 DSpark speculative verify 的 `DFlashVerifyInput` 元数据接口没有实现 TBO 所需的 `retrieve_index` 字段，TBO 与 DSpark 的 verify 输入结构尚未适配。该结果不是性能不如基线，而是配置不可运行，因此没有生成有效吞吐数据，也没有继续跑矩阵。

TBO 启动日志保存在 `logs/services/repro_pr32281_20260823_step4_tbo/prefill/`；decode 的失败堆栈见本次启动终端记录。TBO 不纳入后续组合，除非先完成 `DFlashVerifyInput` 与 TBO `split_spec_info` 的接口适配，并重新通过 target/draft CUDA Graph 和 whoami 验证。

配置记录：`repro_pr32281_20260823/step3_lplb/decode_config_record.txt`；结果目录：`logs/results/repro_pr32281_20260823_step3_lplb/`。

## Step 5：仅将 Decode MoE runner 切换为 DeepGEMM（2026-08-23）

该步骤从纯 DSpark 基线重新部署，只改变 Decode 端：

```bash
SGLANG_MOE_RUNNER_BACKEND=deep_gemm
--moe-runner-backend deep_gemm
```

Prefill 仍为 `flashinfer_mxfp4`；DeepEP `low_latency`、DSpark、PD Mooncake、Decode CUDA Graph、hidden transfer 和 workload 均未改变。脚本与备份分别为 `repro_pr32281_20260823/step5_deepgemm/` 和 `backups/dspark_incremental_20260823/step5_deepgemm/`。Prefill 日志保存于 `logs/services/repro_pr32281_20260823_step5_deepgemm/prefill/`，Router/whoami 日志保存在对应目录；Decode 的完整启动输出来自本次会话，关键有效配置和验证结果已固化在 `repro_pr32281_20260823/step5_deepgemm/decode_config_record.txt`。后续实验脚本会将 Decode stdout/stderr 同步 tee 到独立日志文件。

target verify 和 draft verify CUDA Graph 均成功捕获；8 组请求全部成功，结果目录为 `repro_pr32281_20260823/logs/results/repro_pr32281_20260823_step5_deepgemm/`。

| ISL | OSL | Concurrency | Out tok/s | Total tok/s | 相对纯 DSpark Total | Mean TTFT ms | Mean TPOT ms | 完成数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 291.31 | 582.62 | +0.16% | 310.87 | 3.13 | 10/10 |
| 1024 | 1024 | 16 | 3327.04 | 6654.07 | +0.94% | 518.16 | 4.12 | 160/160 |
| 1024 | 1024 | 256 | 30612.09 | 61224.19 | -3.37% | 746.98 | 7.06 | 2560/2560 |
| 1024 | 1024 | 512 | 40367.92 | 80735.85 | +1.58% | 1174.83 | 10.86 | 5120/5120 |
| 8192 | 1024 | 1 | 302.88 | 2725.90 | +0.28% | 229.45 | 3.08 | 10/10 |
| 8192 | 1024 | 16 | 3429.79 | 30868.14 | -0.23% | 630.80 | 3.89 | 160/160 |
| 8192 | 1024 | 256 | 7536.51 | 67828.56 | -0.49% | 28653.34 | 4.41 | 2560/2560 |
| 8192 | 1024 | 512 | 7551.35 | 67962.13 | -0.72% | 61476.11 | 4.50 | 5120/5120 |

结论：DeepGEMM runner 在当前纯 DSpark + DeepEP low-latency PD 配置下是可用的，但不是稳定的增益项。8 组中只有 4 组 Total 略高，最大提升为 1024/C512 的 +1.58%；1024/C256 反而下降 3.37%，8192 长输入的三组并发 16/256/512 也略低于纯 DSpark。也就是说，DeepGEMM 的 kernel 优势只在部分高并发短输入中抵消了 runner/调度差异，不能据此直接叠加到后续“最佳组合”。当前下一步应保留纯 DSpark 作为主线，并在需要时把 DeepGEMM 作为针对 1024/C512 的局部候选，而不是全 workload 默认替换。

## Step 6：LPLB + DeepGEMM（只覆盖高并发，2026-08-23）

这是在纯 DSpark 基线上继续增加的第二层组合：保留已验证的 LPLB 参数，并增加 Decode `deep_gemm` runner。相对纯 DSpark 新增：

```bash
SGLANG_ENABLE_LPLB=1
SGLANG_LPLB_STATIC_FALLBACK=1
SGLANG_LPLB_IPM_ITERS=1
SGLANG_LPLB_REFRESH_INTERVAL=2
--ep-num-redundant-experts 0
--ep-dispatch-algorithm lp
SGLANG_MOE_RUNNER_BACKEND=deep_gemm
--moe-runner-backend deep_gemm
```

脚本和备份位于 `repro_pr32281_20260823/step6_lplb_deepgemm/` 与 `backups/dspark_incremental_20260823/step6_lplb_deepgemm/`；配置记录为 `repro_pr32281_20260823/step6_lplb_deepgemm/decode_config_record.txt`。服务日志分别保存在 `logs/services/repro_pr32281_20260823_step6_lplb_deepgemm/{prefill,decode,router}/`。target/draft CUDA Graph 和 whoami 均通过，4 组请求全部完成。

为直接验证高并发瓶颈，这一步先覆盖 C256/C512；对照包括纯 DSpark 和 LPLB 单项：

| ISL | OSL | C | 组合 Out tok/s | 组合 Total tok/s | LPLB Total tok/s | 相对 LPLB | 相对纯 DSpark | TTFT ms | TPOT ms | 完成数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 256 | 30912.80 | 61825.61 | 62687.77 | -1.37% | -2.42% | 903.43 | 6.96 | 2560/2560 |
| 1024 | 1024 | 512 | 42273.22 | 84546.44 | 82474.11 | +2.51% | +6.38% | 1410.51 | 10.08 | 5120/5120 |
| 8192 | 1024 | 256 | 7490.39 | 67413.50 | 67129.67 | +0.42% | -1.10% | 28738.91 | 4.42 | 2560/2560 |
| 8192 | 1024 | 512 | 7524.36 | 67719.21 | 67449.12 | +0.40% | -1.08% | 61661.68 | 4.45 | 5120/5120 |

分析：LPLB + DeepGEMM 不是普遍叠加。1024/C512 中，LPLB 的静态负载均衡减少了专家路由不均衡，DeepGEMM 的高并发 grouped GEMM 又能摊薄固定 kernel 开销，因此出现协同收益；C256 的 batch 尚不足以摊平两套调度路径，反而比 LPLB 低 1.37%。8192 输入的组合虽然略高于 LPLB，但仍低于纯 DSpark，且 TTFT 仍为 28.7/61.7 秒，说明长输入主瓶颈在 Prefill 排队、PD hidden transfer 和首 token，而不是 Decode MoE GEMM。

因此当前可保留的策略不是“全场景整合”，而是按负载选择：1024/C512 可采用 LPLB + DeepGEMM；1024/C256 继续使用纯 DSpark 或 LPLB 单项；8192 长输入优先优化 Prefill/hidden transfer，暂不把 Decode runner 组合收益当作总吞吐提升。

## Step 8：LPLB + DeepGEMM 的刷新周期调优（2026-08-23）

Step6 中 C256 低于 LPLB 单项，日志显示该并发处于动态 LP dispatch 的固定开销与 DeepGEMM runner 开销都尚未充分摊平。针对这一瓶颈，只将 `SGLANG_LPLB_REFRESH_INTERVAL` 从 2 调到 8，其他配置完全不变：

```text
SGLANG_LPLB_STATIC_FALLBACK=1
SGLANG_LPLB_IPM_ITERS=1
SGLANG_LPLB_REFRESH_INTERVAL=8
SGLANG_MOE_RUNNER_BACKEND=deep_gemm
```

脚本和备份位于 `repro_pr32281_20260823/step8_lplb_deepgemm_refresh8/` 与 `backups/dspark_incremental_20260823/step8_lplb_deepgemm_refresh8/`，配置记录为 `repro_pr32281_20260823/step8_lplb_deepgemm_refresh8/decode_config_record.txt`。两组请求均完成，Graph/whoami 均通过。

| ISL | OSL | C | Refresh | Total tok/s | Step6 refresh=2 | 变化 | 纯 DSpark | 相对纯 DSpark | TTFT ms | TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 256 | 8 | 63094.73 | 61825.61 | +2.05% | 63358.12 | -0.42% | 892.99 | 6.82 |
| 1024 | 1024 | 512 | 8 | 83530.99 | 84546.44 | -1.20% | 79478.49 | +5.10% | 1492.32 | 10.10 |

结论：原先将该差异解释为“refresh=8 缓解 C256 的 solver/同步开销、但损害 C512 的动态负载适应性”是不严谨的，见下方的重要纠正。仅凭这两次结果不能证明 refresh 周期本身带来收益。

## 重要纠正：static fallback 下 refresh 参数没有进入请求热路径

复核当前源码调用链后发现，Step3、Step6、Step8 都设置了：

```text
SGLANG_LPLB_STATIC_FALLBACK=1
```

在 `python/sglang/srt/layers/moe/topk.py` 的 `select_experts` 路径中，`static_lplb_fallback` 为真时不会调用 `LPLBSolver.solve(topk_ids)`；代码直接走静态 logical-to-physical 映射。`LPLBSolver.solve()` 内部的 `SGLANG_LPLB_REFRESH_INTERVAL` 因而不会影响这些请求的 solver 调用频率，`SGLANG_LPLB_IPM_ITERS` 也不会成为请求路径的计算开销控制项。

因此，Step6 的 `refresh=2` 与 Step8 的 `refresh=8` 吞吐差异不能归因于 LPLB refresh 策略，更可能来自单次实验波动、CUDA/通信调度、PD 排队状态或 Decode runner 的运行时选择。Step8 的“按并发选择 refresh”结论撤销；后续若要研究 refresh，必须关闭 static fallback、确认日志中确实进入 `LPLBSolver.solve()`，并先完成无死锁和功能验证。

这也解释了为什么把多个技术“一股脑整合”后没有超过纯 DSpark：当前组合中部分开关并非独立增益，甚至互相绕开或改变了热路径。纯 DSpark 的 `moe-runner-backend=auto` 已经选择了适合该 workload 的路径；显式切到 DeepGEMM 只在 1024/C512 出现局部收益，在 C256 和长输入反而下降。LPLB static fallback 没有执行动态负载均衡，只增加了静态路由映射；Waterfill 在均匀随机路由上增加 dispatch/materialize 成本；TBO 与 DSpark verify metadata 不兼容；HiSparse 尚未通过 retokenized output 功能门槛。它们不能简单相加。

## Step 7：仅加入 HiSparse（功能验收失败，2026-08-23）

HiSparse 是针对长上下文 KV/稀疏注意力的候选技术。当前源码支持 DeepSeek-V4 HiSparse，并成功完成：

- HiSparse C4 host pool 初始化，配置 `top_k=1024、device_buffer_size=1024、host_to_device_ratio=8`；
- target verify 和 draft verify CUDA Graph 捕获；
- Router `whoami`：`HTTP=200`、`WHOAMI_VALID=True`。

但首个性能样例 1024/1024/C1 的 10 个请求虽然 HTTP 层显示 10/10 完成，retokenized output 只有 `8131/10240`，并且 TPOT 长尾明显异常。该结果与纯 DSpark 的固定输出规模不一致，不能作为性能数据；矩阵已立即停止，未继续测试 C16 或 8192。

结果和日志位于 `repro_pr32281_20260823/step7_hisparse/`、`repro_pr32281_20260823/logs/results/repro_pr32281_20260823_step7_hisparse/` 和 `logs/services/repro_pr32281_20260823_step7_hisparse/`，配置记录为 `repro_pr32281_20260823/step7_hisparse/decode_config_record.txt`。Decode 日志中的统计显示部分 DP rank 的 DSpark accept length 长时间为 1/0，说明当前 HiSparse KV 管理与 DSpark speculative verify 的状态/接受路径存在一致性问题。HiSparse 暂不进入任何组合，除非先修复并通过“retokenized output 等于目标规模”的功能门槛。
## Step 9：DeepEP low-latency graph256 容量适配（2026-08-23，未通过部署门槛）

本步骤回到纯 DSpark 基线，只针对高并发 graph256 的 DeepEP low-latency dispatch capacity 做适配；没有启用 Waterfill、LPLB、DeepGEMM、TBO、FP4 indexer 或 HiSparse。实验目录为 `repro_pr32281_20260823/step9_deepep_chunked_graph/`，源码修改备份位于 `backups/dspark_deepep_chunked_graph_20260823/`。

DSpark 的六 token verify window 在 graph256 下会产生约 `256 × 6 = 1536` 行，而当前 DeepEP low-latency 的单 rank capacity 为 1024。候选修改在 SGLang 的 FusedMoE/DeepEP low-latency 路径中，将超过 1024 行的输入拆成 `1024 + 512` 两个完整的 dispatch → MoE GEMM → combine transaction，再拼接输出。

retry2 日志显示 target verify graph256 已完成捕获（约 20.29 秒，显存使用约 24 GB），但 draft verify graph256 随后长时间停留在 CUDA allocator 和 DeepGEMM workspace 分配路径，剩余显存约 7.7–8.6 GB；没有完成服务 ready，也没有通过 whoami/请求功能验证。因此该分块方案只能证明绕过了 target dispatch 行数 assertion，不能称为可部署优化。

随后尝试将 `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` 和 `SGLANG_DEEPEP_LL_SPLIT_TOKENS` 都设为 2048，并只捕获 graph256，以避免 1536 行被拆分。Decode 在初始化阶段立即失败：

```text
AssertionError
deepep.py:382: assert self.num_max_dispatch_tokens_per_rank <= 1024
```

复核 DeepEP 实现后确认，1024 不是 SGLang 脚本默认值，而是 DeepEP internode low-latency 的协议约束：源码注释指出其使用 `FINISHED_SUM_TAG=1024`，并要求单 rank 发往另一 rank 的 token 数小于该上限；C++ low-latency dispatch/combine 也按这一 capacity 分配通信 buffer。因此不能只删除 Python assert 或把环境变量改成 2048，否则会绕过保护但不保证通信布局和 tag 语义正确。

Step 9 未通过“服务 ready + whoami + 固定输出规模”的功能门槛，不进入性能矩阵，也不纳入最佳组合。当前 graph256 的瓶颈已经从“单次 dispatch 行数超过上限”进一步定位为 DeepEP low-latency 协议的硬 capacity 以及 graph capture 的显存预算；正式 baseline 仍保持 graph128 的纯 DSpark 配置。

## Step 10：仅切换 DSpark draft MoE runner 到 flashinfer_mxfp4（2026-08-23）

用户要求回到已复现的纯 DSpark 配置，在此基础上逐步增加单项技术，避免把多个可能互相影响的开关一次性叠加。本步骤只增加：

```text
--speculative-moe-runner-backend flashinfer_mxfp4
```

Decode target 仍保持 `moe-runner-backend=auto`、`moe-a2a-backend=deepep`、`deepep-mode=low_latency`、Decode CUDA Graph、graph128、静态 verify 和原有 PD 配置；没有启用 Waterfill、LPLB、DeepGEMM target runner、TBO、FP4 indexer、HiSparse 或 graph256 适配。

服务启动、target/draft CUDA Graph、Router `whoami` 均通过。`whoami` 返回 HTTP 200 且 `WHOAMI_VALID=True`。C1/C16 的请求也全部完成，输出规模正常：

| ISL | OSL | Concurrency | Requests | 本轮 Total tok/s | 纯 DSpark baseline Total tok/s | Total 变化 | Mean TTFT ms | Mean TPOT ms | 完成数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 10 | 575.80 | 581.67 | -1.01% | 274.64 | 3.21 | 10/10 |
| 1024 | 1024 | 16 | 160 | 6422.30 | 6591.95 | -2.57% | 653.02 | 4.17 | 160/160 |

结果说明：显式使用 `flashinfer_mxfp4` 的 DSpark draft 路径是功能可用的，但在两个已测试低并发点都低于纯 DSpark baseline，因此不继续扩展到 C256/C512 或 8192。该结果也说明“draft runner 使用某个更快的 MoE backend”不能直接推导出端到端吞吐提升；DSpark 的 draft、target、CUDA Graph 和 PD hidden transfer 共同决定结果，draft 单点 kernel 优化可能被额外初始化、调度或与 target 的同步成本抵消。

配置、备份、结果和服务日志位于：

```text
repro_pr32281_20260823/step10_dspark_draft_flashinfer_mxfp4/
backups/dspark_draft_flashinfer_mxfp4_20260823/
```

本步骤的 Prefill 启动日志已单独保存；Decode/Router 当时通过持久 PTY 启动，启动输出没有在启动时重定向到独立文件，但请求结果、whoami 验证和配置备份均已保存。后续服务启动统一使用 `logs/services/{prefill,decode,router}/` 独立日志重定向。

目前该单项已排除。后续若继续优化，应优先针对 8192 输入下的 Prefill 排队、hidden transfer 和 Decode admission 做可观测性/瓶颈验证，而不是继续叠加 MoE runner 开关。

## Step 11：8192 长输入 Prefill 分块增大到 32768（2026-08-23）

在纯 DSpark 基线上继续只增加一个针对长输入瓶颈的改动：Prefill 的
`max-prefill-tokens` 和 `chunked-prefill-size` 同时从 `16384` 改为
`32768`。Decode 完全保持纯 DSpark 配置：`moe-runner-backend=auto`、DeepEP
`low_latency`、Decode CUDA Graph、graph128、静态 verify 和原有 PD/Mooncake
参数；没有加入 Waterfill、LPLB、DeepGEMM、TBO、FP4 indexer、HiSparse 或
draft runner 改动。

选择该变量的依据是：8192 高并发时 Decode TPOT 只有约 4.4 ms，而 Mean TTFT
达到 28 秒以上，Prefill 日志显示每个 batch 受 16384 token 分块限制；历史
A/B 也曾观察到 32768 对长输入有收益，但短输入可能退化。因此本步骤只验收
8192/1024/C256，不把它直接推广到所有输入长度。

服务、`whoami`、CUDA Graph 和请求完整性均通过。Prefill 日志确认运行参数为
`chunked_prefill_size=32768`、`max_prefill_tokens=32768`，Decode 日志确认
`cuda graph: True`，服务日志未发现 KV transfer failure、非法访问、NCCL 错误
或请求失败。

| ISL | OSL | C | Requests | 本轮 Out tok/s | 纯 DSpark Out tok/s | 本轮 Total tok/s | 纯 DSpark Total tok/s | Total 变化 | Mean TTFT ms | baseline TTFT ms | Mean TPOT ms | baseline TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 1024 | 256 | 2560/2560 | 7957.75 | 7573.62 | 71619.73 | 68162.62 | +5.07% | 27059.88 | 28439.71 | 4.20 | 4.44 |

本轮总输入为 20,971,520 token，总生成 2,621,440 token，retokenized 输出为
2,622,212，规模与目标一致。32768 分块使 Prefill 观测到约 65k token/s 的输入
处理速率，端到端总吞吐提升 5.07%，TTFT 降低 4.85%；这说明该收益来自
Prefill 分块/排队效率，而不是 Decode MoE runner。

该配置现在可以作为 **8192 长输入专用 profile** 保留，但不能称为全 workload
最佳配置。下一步若要纳入统一部署，需要至少补测 1024 输入，确认短输入是否
出现退化；在此之前，短输入继续使用 16384，长输入可单独使用 32768。

实验目录为 `repro_pr32281_20260823/step11_prefill_chunk32768/`，脚本和参数
备份为 `backups/dspark_prefill_chunk32768_20260823/`。

## Step 12：干净验证 32768 分块对 1024 短输入的影响（2026-08-23）

为了避免把 Step 11 的长输入收益误推广到全局，使用相同的纯 DSpark
Decode 配置和相同随机 workload，干净重启后只保留 Prefill
`max-prefill-tokens=32768`、`chunked-prefill-size=32768`，完成 1024/1024
的 C1、C16、C256、C512 矩阵。所有服务 ready、`whoami` 和请求完整性均通过，
服务日志没有发现 PD transfer、CUDA、NCCL 或非法访问错误。

| ISL | OSL | C | Requests | 32768 Total tok/s | 16384 pure DSpark Total tok/s | 变化 | 32768 TTFT ms | baseline TTFT ms | 32768 TPOT ms | baseline TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 10/10 | 576.70 | 581.67 | -0.85% | 277.14 | 276.62 | 3.20 | 3.17 |
| 1024 | 1024 | 16 | 160/160 | 6533.91 | 6591.95 | -0.88% | 632.61 | 604.05 | 4.10 | 4.07 |
| 1024 | 1024 | 256 | 2560/2560 | 62714.23 | 63358.12 | -1.02% | 870.97 | 788.76 | 6.88 | 6.88 |
| 1024 | 1024 | 512 | 5120/5120 | 76312.80 | 79478.49 | -3.98% | 1245.86 | 1151.18 | 11.47 | 11.10 |

结论很明确：32768 分块在 1024 短输入四组都低于纯 DSpark，C512 退化
3.98%。因此它不是全局优化，只能和 Step 11 的结果一起形成按输入长度分档
的配置：

```text
ISL=1024：Prefill 16384/16384
ISL=8192：Prefill 32768/32768（当前已验证 C256 +5.07%，仍需按目标并发继续确认）
```

Step 12 的脚本、结果、Prefill/Decode/Router 日志和验证记录位于
`repro_pr32281_20260823/step12_prefill32768_short_ab/`，备份位于
`backups/dspark_prefill32768_short_ab_20260823/`。

## Step 13：8192 长输入 C512 验证 Prefill 32768 profile（2026-08-23）

Step 12 已证明 Prefill 32768 对 1024 短输入不是全局优化，因此继续只在
8192 长输入 profile 上验证 C512。配置与 Step 11 完全一致：Prefill 使用
`max-prefill-tokens=32768`、`chunked-prefill-size=32768`；Decode 保持纯
DSpark baseline，即 `moe-runner-backend=auto`、DeepEP `low_latency`、Decode
CUDA Graph graph128、静态 verify 和原有 PD/Mooncake 参数。

服务启动、`whoami` 和 CUDA Graph 验证通过，benchmark 最终完成 5120/5120：

| ISL | OSL | C | Requests | 本轮 Out tok/s | 纯 DSpark Out tok/s | 本轮 Total tok/s | 纯 DSpark Total tok/s | Total 变化 | Mean TTFT ms | baseline TTFT ms | Mean TPOT ms | baseline TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 1024 | 512 | 5120/5120 | 8030.83 | 7606.20 | 72277.47 | 68455.80 | +5.58% | 57659.93 | 60992.07 | 4.29 | 4.49 |

相对纯 DSpark baseline，Total/Out tok/s 提升 5.58%，TTFT 降低 5.46%，TPOT
降低 4.48%。结合 Step 11 的 8192/C256 +5.07%，可以确认 Prefill 32768 是
当前 8192 长输入 profile 的有效单项优化；收益主要来自 Prefill 分块和排队
效率，而不是 Decode MoE runner。

可靠性方面，本轮 Decode 日志出现 1 次 `Decode transfer failed` / `AbortReq`，
但最终 5120 个请求全部成功，未出现持续性传输错误、CUDA/NCCL 错误或服务退出。
该异常已保留在原始日志中，不能表述为“零错误”运行。

最终按输入长度分档：

```text
ISL=1024：Prefill 16384/16384
ISL=8192：Prefill 32768/32768（C256、C512 均已验证优于纯 DSpark）
```

实验目录为 `repro_pr32281_20260823/step13_prefill32768_long_c512/`，启动脚本
及参数备份位于 `backups/dspark_prefill32768_long_c512/`。

## Step 14：8192/C512 Decode DSpark hidden pool 131072 A/B（2026-08-23）

Step 13 的 8192/C512 长输入 profile 已经优于纯 DSpark，但仍记录过一次
可恢复的 `AbortReq`。历史实验显示 C512 的 Decode hidden pool 在 131072
附近可能存在容量拐点，因此本步骤只改变一个变量：

```text
SGLANG_DSPARK_PD_HIDDEN_POOL_TOKENS: 65536 -> 131072
```

Prefill 仍使用 `max-prefill-tokens=32768`、`chunked-prefill-size=32768`，
Decode 仍为纯 DSpark 的 `moe-runner-backend=auto`、DeepEP `low_latency`、
Decode CUDA Graph graph128、静态 verify 和原有 PD/Mooncake 参数；没有加入
Waterfill、LPLB、DeepGEMM、TBO、FP4 indexer、HiSparse 或其他 runner。

`whoami` 返回 HTTP 200 且 `WHOAMI_VALID=True`，完整 benchmark 完成 5120/5120：

| ISL | OSL | C | Requests | 本轮 Out tok/s | Step13 Out tok/s | 本轮 Total tok/s | Step13 Total tok/s | Total 变化 | Mean TTFT ms | Step13 TTFT ms | Mean TPOT ms | Step13 TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 1024 | 512 | 5120/5120 | 8028.07 | 8030.83 | 72252.67 | 72277.47 | -0.03% | 57777.08 | 57659.93 | 4.2645 | 4.2890 |

本轮运行期间 Decode 日志没有出现 `Decode transfer failed`、`KVTransferError`
或 `AbortReq`；但相对于 Step13 吞吐基本不变，TTFT 还略高 0.20%。因此
131072 不能作为性能优化宣称，只能作为后续重复稳定性 A/B 的候选容量配置，
不能把“本轮没有 AbortReq”单独归因于 pool 增大。

实验目录为 `repro_pr32281_20260823/step14_dspark_hidden_pool131k_long_c512/`，
包含独立 Prefill/Decode/Router 日志、whoami、benchmark JSONL 和配置记录；
备份位于 `backups/dspark_hidden_pool131k_long_c512_20260823/`。

## Step 15：8192/C512 Prefill hidden thread pool 8 A/B（2026-08-23）

Step 14 表明把 Decode DSpark hidden pool 增大到 131072 没有带来吞吐收益，
因此回到 Step13 的性能配置，只针对 Prefill 侧的 PD hidden transfer 并行度
做单变量实验：

```text
SGLANG_DSPARK_HIDDEN_THREAD_POOL_SIZE: 2 -> 8
```

Prefill 的 `max-prefill-tokens=32768`、`chunked-prefill-size=32768` 不变；
Decode 恢复 Step13 的 DSpark hidden pool 65536，并保持 `moe-runner-backend=auto`、
DeepEP `low_latency`、Decode CUDA Graph graph128 和静态 verify。没有加入其他技术。

`whoami` 返回 HTTP 200 且 `WHOAMI_VALID=True`，完整 benchmark 完成 5120/5120：

| ISL | OSL | C | Requests | 本轮 Out tok/s | Step13 Out tok/s | 本轮 Total tok/s | Step13 Total tok/s | Total 变化 | Mean TTFT ms | Step13 TTFT ms | Mean TPOT ms | Step13 TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 1024 | 512 | 5120/5120 | 8038.38 | 8030.83 | 72345.44 | 72277.47 | +0.09% | 57646.18 | 57659.93 | 4.2804 | 4.2890 |

本轮运行期间没有 `Decode transfer failed`、`KVTransferError` 或 `AbortReq`，
输出 retokenized 数量为 5,242,784，接近目标 5,242,880。当前只完成一轮，
相对 Step13 的 +0.09% 很可能处于运行状态噪声范围，因此暂时标记为“候选”，
不能直接宣称稳定增益。若后续重复仍保持正收益，再将 hidden thread=8 纳入
8192 长输入 profile；1024 输入仍不能直接继承，需单独验证。

实验目录为 `repro_pr32281_20260823/step15_prefill_hidden_thread8_long_c512/`，
备份位于 `backups/dspark_prefill_hidden_thread8_long_c512_20260823/`。

### Step 15 repeat-2：hidden thread pool 8 复测

由于 Step15 首轮只有 +0.09%，按验收标准使用相同源码、参数、随机种子和
8192/1024/C512 workload 做第二次干净重启复测。第二轮同样通过 whoami，完成
5120/5120，且 Decode 日志没有运行期 `Decode transfer failed`、`KVTransferError`
或 `AbortReq`：

| 轮次 | Total tok/s | Out tok/s | Mean TTFT ms | Mean TPOT ms | 完成数 | 相对 Step13 Total |
|---|---:|---:|---:|---:|---:|---:|
| Step15-1 | 72345.44 | 8038.38 | 57646.18 | 4.2804 | 5120/5120 | +0.09% |
| Step15-2 | 72436.24 | 8048.47 | 57599.45 | 4.2946 | 5120/5120 | +0.22% |

两轮平均 Total tok/s 为 72390.84，较 Step13 提升约 0.16%。方向一致但幅度
很小，因此将 `SGLANG_DSPARK_HIDDEN_THREAD_POOL_SIZE=8` 保留为 8192/C512
的弱 workload-specific 候选；不能直接推广到 1024 输入，也不能把它描述成
主要性能突破。后续若继续优化，应优先针对 Prefill queue/hidden transfer
的实际等待时间做 profiling，而不是继续盲目扩大线程数。

复测目录为 `repro_pr32281_20260823/step15_prefill_hidden_thread8_long_c512_repeat2/`，
备份位于 `backups/dspark_prefill_hidden_thread8_long_c512_repeat2_20260823/`。

## Step 16：1024/C512 Prefill hidden thread pool 8 短输入边界验证（2026-08-23）

Step 15 的两轮复测只覆盖 8192/C512 长输入。为避免把长输入的微小收益错误推广到短输入，本步骤回到 1024 输入 profile，将 Prefill 的 `max-prefill-tokens` 和 `chunked-prefill-size` 恢复为 16384，只保留一个变量：

```text
SGLANG_DSPARK_HIDDEN_THREAD_POOL_SIZE: 2 -> 8
```

Decode 与纯 DSpark baseline 完全一致：`moe-runner-backend=auto`、DeepEP `low_latency`、Decode CUDA Graph graph128、静态 verify，以及原有 PD/Mooncake 参数。没有启用 Waterfill、LPLB、DeepGEMM、TBO、FP4 indexer、HiSparse 或其他 runner 变化。

服务启动后 `whoami` 返回 HTTP 200 且 `WHOAMI_VALID=True`，正式 benchmark 完成 5120/5120；Decode 日志没有运行期 `Decode transfer failed`、`KVTransferError` 或 `AbortReq`：

| ISL | OSL | C | Requests | 本轮 Out tok/s | baseline Out tok/s | 本轮 Total tok/s | baseline Total tok/s | Total 变化 | 本轮 TTFT ms | baseline TTFT ms | 本轮 TPOT ms | baseline TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 512 | 5120/5120 | 39727.34 | 39739.25 | 79454.68 | 79478.49 | -0.03% | 1336.97 | 1151.18 | 10.99 | 11.10 |

结论：线程池 8 在短输入 C512 下没有带来可证明的吞吐提升，Total/Out tok/s 基本持平；TTFT 反而高 16.14%，TPOT 小幅改善 0.99%。因此它不是全局优化，只能保留为 8192/C512 长输入的弱 workload-specific 候选。Step15 两轮长输入平均收益约 +0.16%，幅度很小，后续不应把该收益描述为主要突破。

本步骤目录为 `repro_pr32281_20260823/step16_prefill_thread8_short_c512/`，其中保存了 Prefill、Decode、Router 独立日志、whoami、benchmark JSONL 和配置记录；备份位于 `backups/dspark_prefill_thread8_short_c512_20260823/`。

## Step 17：8192/C512 Prefill 65536 上限验证（2026-08-23）

由于 Step 13 的日志显示 8192/C512 主要受 Prefill 分块和队列限制，本步骤
只把 Prefill `max-prefill-tokens` 与 `chunked-prefill-size` 从 32768 提到
65536；hidden thread pool 保持 2，Decode 继续使用纯 DSpark、DeepEP
`low_latency`、CUDA Graph graph128 和原有 PD/Mooncake 配置。

服务启动和 whoami 均通过，但第一批正式请求进入 Prefill 后，四个 Prefill
rank 都在 DeepSeek-V4 attention compressor 的
`plan_compress_prefill` 处失败：

```text
tvm.error.InternalError: ... c_plan.cuh:507
RuntimeCheck(batch_size <= num_q_tokens && num_q_tokens <= uint16_max)
```

65536 使 `num_q_tokens` 超过当前 compressor 的 uint16 metadata 上限 65535，
随后 Prefill scheduler 收到 SIGQUIT 并退出。benchmark 只记录到 158 个瞬时
HTTP-success 请求，输出 token 统计不完整；因此这轮不是性能结果，也不能与
baseline 比较。该失败同时说明 65536 不是当前源码下可部署的 Prefill 配置。

结论：拒绝 65536，保留 32768 作为已验证的长输入配置。下一候选必须低于
65535 metadata 上限，例如 49152，并先通过完整 5120/5120 功能门槛，再进行
吞吐比较；不能把这轮的 partial result 当作优化结果。

实验目录为 `repro_pr32281_20260823/step17_prefill65536_long_c512/`，其中保存
了完整失败日志、benchmark partial JSONL 和配置记录；备份位于
`backups/dspark_prefill65536_long_c512_20260823/`。

## Step 18：8192/C512 Prefill 49152 单变量优化（2026-08-23）

Step 17 将上限原因定位为 compressor 的 uint16 metadata 限制后，选择仍低于
65535 的 49152（8192 输入下每批 6 个请求），相对 Step 13 只改变 Prefill
`max-prefill-tokens` 和 `chunked-prefill-size`；hidden thread pool、Decode
DSpark、DeepEP `low_latency`、CUDA Graph graph128 和 PD 参数全部不变。

服务启动、whoami 和完整性检查通过，Prefill 日志确认每批为
`6 x 8192 = 49152` token，正式 benchmark 完成 5120/5120，运行期间没有
`Decode transfer failed`、`KVTransferError`、`AbortReq`、`InternalError`、
`Traceback` 或 `SIGQUIT`：

| ISL | OSL | C | Requests | 本轮 Out tok/s | Step13 Out tok/s | 本轮 Total tok/s | Step13 Total tok/s | 相对 Step13 | baseline Total tok/s | 相对 baseline | 本轮 TTFT ms | Step13 TTFT ms | 本轮 TPOT ms | Step13 TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 1024 | 512 | 5120/5120 | 8313.76 | 8030.83 | 74823.81 | 72277.47 | +3.52% | 68455.80 | +9.30% | 55582.32 | 57659.93 | 4.30 | 4.29 |

相对 Step 13，49152 将 Total/Out tok/s 提升 3.52%，TTFT 降低 3.60%；相对
纯 DSpark baseline，Total/Out tok/s 提升 9.30%，TTFT 降低 8.87%，TPOT
降低 4.23%。Prefill 局部输入速率约 67k token/s，说明收益确实来自更大的
Prefill 批和更少的批次调度，而非 Decode runner 替换。

结论：接受 `49152/49152` 作为当前 `8192/C512` 长输入的最佳单变量 profile；
`1024` 输入继续使用 `16384/16384`。这不是全局配置，后续若继续验证，只应
针对 8192 workload 的其他明确瓶颈做单变量实验，不能把 49152 直接推广到短输入。

实验目录为 `repro_pr32281_20260823/step18_prefill49152_long_c512/`，备份位于
`backups/dspark_prefill49152_long_c512_20260823/`。

## Step 19：在 Prefill 49152 基础上添加 hidden thread pool 8（2026-08-23）

Step 18 已得到当前长输入最佳 profile：Prefill 49152/49152、hidden thread
pool 2。Step 19 只增加一个已验证可运行的技术：

```text
SGLANG_DSPARK_HIDDEN_THREAD_POOL_SIZE: 2 -> 8
```

49152 分块、纯 DSpark Decode、DeepEP `low_latency`、Decode CUDA Graph
graph128 和 PD/Mooncake 参数全部保持不变。服务启动、whoami 和完整请求数
均通过，5120/5120 完成，且日志没有运行期错误：

| ISL | OSL | C | Requests | 本轮 Out tok/s | Step18 Out tok/s | 本轮 Total tok/s | Step18 Total tok/s | Total 变化 | 本轮 TTFT ms | Step18 TTFT ms | 本轮 TPOT ms | Step18 TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 1024 | 512 | 5120/5120 | 8269.64 | 8313.76 | 74426.79 | 74823.81 | -0.53% | 55884.03 | 55582.32 | 4.33 | 4.30 |

结论：hidden thread=8 在 49152 大分块下不能叠加收益，Total/Out tok/s
下降 0.53%，TTFT 和 TPOT 也变差。此前在 32768 下观察到的约 +0.16% 只属于
特定分块规模的弱收益，不能推广到当前最佳配置。当前接受配置回退为：

```text
8192/C512：Prefill 49152/49152 + hidden thread pool 2
1024 输入：Prefill 16384/16384
Decode：纯 DSpark + DeepEP low_latency + CUDA Graph graph128
```

实验目录为 `repro_pr32281_20260823/step19_prefill49152_thread8_long_c512/`，
备份位于 `backups/dspark_prefill49152_thread8_long_c512_20260823/`。

## Step 20：Prefill 57344 与 49152 的单变量比较（2026-08-23）

Step 20 在已接受的 Step 18（Prefill 49152/49152、hidden thread pool 2）上，
只将 Prefill `max-prefill-tokens` 和 `chunked-prefill-size` 改为 57344；纯
DSpark Decode、DeepEP `low_latency`、Decode CUDA Graph graph128、PD/Mooncake
参数和 Prefill hidden thread pool 均保持不变。57344 低于 compressor 的 uint16
metadata 上限 65535，服务启动、whoami 和完整性检查均通过。

| ISL | OSL | C | Requests | 本轮 Out tok/s | Step18 Out tok/s | 本轮 Total tok/s | Step18 Total tok/s | Total 变化 | 本轮 TTFT ms | Step18 TTFT ms | 本轮 TPOT ms | Step18 TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 1024 | 512 | 5120/5120 | 8336.36 | 8313.76 | 75027.28 | 74823.81 | +0.27% | 55385.04 | 55582.32 | 4.32 | 4.30 |

相对纯 DSpark baseline（Total 68455.80、Out 7606.20、TTFT 60992.07 ms、
TPOT 4.49 ms），57344 的 Total/Out tok/s 分别提升 9.60%，TTFT 降低 9.19%，
TPOT 降低 3.79%。本轮 5120/5120 成功，运行日志未发现 Decode transfer
failed、KVTransferError、AbortReq、InternalError、Traceback 或 SIGQUIT。

57344 比 Step 18 的 Total/Out tok/s 仅高 0.27%，TTFT 低 0.35%，TPOT 高
0.47%；这个差异不足以证明稳定收益。因此暂不替换已接受的 49152 配置，保留
57344 作为功能正常、值得后续重复验证的候选，不把单次微小波动写成优化收益。

实验目录为 `repro_pr32281_20260823/step20_prefill57344_long_c512/`，备份位于
`backups/dspark_prefill57344_long_c512/`。

## Step 21：Prefill Mooncake transfer worker 16 -> 32（2026-08-23）

Step 21 针对 8192/C512 日志中持续存在的 Prefill 排队和 PD hidden/KV transfer
背压，只改变 Prefill 侧的：

```text
SGLANG_DISAGGREGATION_THREAD_POOL_SIZE: 16 -> 32
```

Prefill 49152/49152、hidden thread pool 2、Decode 纯 DSpark、DeepEP
`low_latency`、CUDA Graph graph128、静态 verify、Mooncake 参数和 workload
全部不变。服务 health、Router `whoami` 均通过，`WHOAMI_VALID=True`。

| ISL | OSL | C | Requests | 本轮 Out tok/s | Step18 Out tok/s | 本轮 Total tok/s | Step18 Total tok/s | Total 变化 | 本轮 TTFT ms | Step18 TTFT ms | 本轮 TPOT ms | Step18 TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 1024 | 512 | 5120/5120 | 8337.21 | 8313.76 | 75034.89 | 74823.81 | +0.28% | 55640.98 | 55582.32 | 4.05 | 4.30 |

本轮完整完成 5120/5120，运行时间 628.85 s。相对 Step18，Total/Out tok/s
仅提升 0.28%，TTFT 反而增加 0.11%，因此不能据此宣称 transfer worker=32
带来稳定吞吐提升。TPOT 从 4.30 降到 4.05 ms，值得后续重复验证，但当前仍保留
worker=16 作为保守的接受配置，避免把单次 A/B 波动误判为收益。

本轮 Prefill 日志确认持续使用 49152-token 批次；保存的 Prefill 日志和
benchmark 输出未发现 `Decode transfer failed`、`KVTransferError`、`AbortReq`、
`InternalError`、`Traceback`、`CUDA error`、`NCCL error` 或 `SIGQUIT`。需要说明：
本轮 Decode/Router 启动脚本尚未将 stdout/stderr tee 到 Step21 目录，故独立目录中
保存的是 Prefill 日志、whoami 和 benchmark 结果；后续脚本已列为日志捕获修复项，
不能把缺少 Decode 文件描述成“没有启动日志”。

实验目录为 `repro_pr32281_20260823/step21_prefill_transfer_threads32_long_c512/`，
代码与配置备份位于 `backups/dspark_prefill_transfer_threads32_long_c512/`。

## Step 22：worker=32 的 8192/C512 重复验证（2026-08-23）

Step 22 完全复用 Step21 的配置，目的是判断 worker=32 的 TPOT 变化是否稳定，
不是新增技术。Prefill、Decode、Router 均使用独立日志文件；服务启动后首次
whoami 因 Router worker 尚未注册返回 503，等待注册后重试得到
`HTTP=200`、`WHOAMI_VALID=True`，因此正式 benchmark 在功能门槛通过后才开始。

| ISL | OSL | C | Requests | 本轮 Out tok/s | Step18 Out tok/s | 本轮 Total tok/s | Step18 Total tok/s | Total 变化 | 本轮 TTFT ms | Step18 TTFT ms | 本轮 TPOT ms | Step18 TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 1024 | 512 | 5120/5120 | 8304.42 | 8313.76 | 74739.77 | 74823.81 | -0.11% | 55916.19 | 55582.32 | 4.04 | 4.30 |

第二轮相对 Step18 的 Total/Out tok/s 下降 0.11%，TTFT 增加 0.60%，只有 TPOT
下降 6.05%。结合 Step21 首轮结果：

| 指标 | Step21 首轮 | Step22 复测 | 两轮均值相对 Step18 |
|---|---:|---:|---:|
| Total tok/s | +0.28% | -0.11% | +0.09% |
| Out tok/s | +0.28% | -0.11% | +0.09% |
| TTFT | +0.11% | +0.60% | +0.35% |
| TPOT | -5.81% | -6.05% | -5.93% |

结论：Prefill transfer worker=32 的 TPOT 改善在两轮中重复出现，但没有带来
端到端吞吐提升，且 TTFT 略有增加。它不能作为“进一步提升 Total tok/s”的已接受
优化，当前正式 profile 仍保持 worker=16；若目标是降低 Decode token 间隔，可将
worker=32 记录为有代价的 latency trade-off，不能和其他技术直接叠加宣称收益。

本轮 Prefill、Decode、Router 日志在
`repro_pr32281_20260823/step22_prefill_transfer_threads32_long_c512_repeat2/logs/`
下，备份位于 `backups/dspark_prefill_transfer_threads32_long_c512_repeat2/`。

## Step 23：worker=32 的 1024/C512 短输入边界验证（2026-08-23）

Step 23 将 Prefill max/chunk 恢复为 16384，只保留 Prefill transfer worker=32，
用于验证长输入观察到的 TPOT 改善是否适用于短输入。Decode 仍为纯 DSpark、
DeepEP `low_latency`、CUDA Graph graph128，其他参数不变；服务、whoami 和日志
完整性均通过。

| ISL | OSL | C | Requests | 本轮 Out tok/s | baseline Out tok/s | 本轮 Total tok/s | baseline Total tok/s | Total 变化 | 本轮 TTFT ms | baseline TTFT ms | 本轮 TPOT ms | baseline TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 512 | 5120/5120 | 39990.17 | 39739.25 | 79980.35 | 79478.49 | +0.63% | 1701.46 | 1151.18 | 10.45 | 11.10 |

worker=32 在短输入下 TPOT 下降 5.86%，但 TTFT 增加 47.80%，Total/Out
吞吐只有 0.63% 的变化，不能视为可接受的短输入优化。结合 Step21/22，当前
结论是 worker=32 只表现为“降低 TPOT、牺牲 TTFT”的 transfer 调度 trade-off，
不进入正式最佳配置；正式配置继续使用 worker=16。

本轮实验目录为 `repro_pr32281_20260823/step23_prefill_transfer_threads32_short_c512/`，
备份位于 `backups/dspark_prefill_transfer_threads32_short_c512/`。

## 当前阶段最终结论（截至 Step 23）

本轮没有把 Waterfill、LPLB、DeepGEMM、TBO、FP4 indexer、HiSparse、MegaMoE
等技术重新一股脑叠加。每个候选均以纯 DSpark 复现结果为起点，单独通过启动、
whoami、完整请求数和性能指标检查；不满足稳定收益或功能门槛的候选均未进入
后续组合。

当前接受的 workload-specific 配置为：

```text
ISL=1024：Prefill max/chunk=16384，transfer worker=16
ISL=8192：Prefill max/chunk=8192，transfer worker=16
Decode：纯 DSpark，DeepEP low_latency，CUDA Graph graph128，static verify
```

其中长输入最终采用 8192 Prefill 分块；49152 虽然减少了 Prefill 批次调度，
但在高并发下会使 attention/PD hidden 峰值 OOM，因此不能作为部署配置。8192
分块在本步骤通过了 8192/C16、C256、C512 的完整性门槛；性能收益来自稳定地
保持请求级批处理和 MegaMoE 路径，而不是更换 Decode MoE runner。transfer worker=32 在长输入两轮均稳定降低 TPOT 约
5.93%，但两轮平均 Total 只提升 0.09%、TTFT 增加 0.35%；在短输入中 TTFT
增加 47.80%。因此它是 latency trade-off，不是吞吐优化，正式 profile 保持 16。

后续若还要继续追求超过当前结果，应只针对新的、可观测的瓶颈设计一个变量并
复用同样的完整性门槛；不能用单次 TPOT 改善或多个技术的叠加替代 Total/TTFT
的独立证据。

### Step 24 有效结果（修复后）

修复后的配置为 Prefill MegaMoE、`moe-runner-backend=auto`、
`SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8192`、
`max-prefill-tokens=chunked-prefill-size=8192`；Decode 仍是纯 DSpark。
服务重启后 health、CUDA Graph 和 Router `whoami` 均通过。

| ISL | OSL | Concurrency | Completed | Retokenized output | Out tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 10/10 | 10236 | 299.41 | 598.82 | 185.71 | 3.16 |
| 1024 | 1024 | 16 | 160/160 | 164292 | 3445.61 | 6891.22 | 353.38 | 4.12 |
| 1024 | 1024 | 256 | 2560/2560 | 2621694 | 31842.97 | 63685.94 | 714.44 | 6.95 |
| 1024 | 1024 | 512 | 5120/5120 | 5240595 | 36572.46 | 73144.91 | 1101.05 | 12.36 |
| 8192 | 1024 | 1 | 10/10 | 10238 | 302.15 | 2719.31 | 205.19 | 3.11 |
| 8192 | 1024 | 16 | 160/160 | 164302 | 3411.87 | 30706.81 | 548.56 | 3.98 |
| 8192 | 1024 | 256 | 2560/2560 | 2622363 | 5452.67 | 49074.07 | 41318.77 | 4.27 |
| 8192 | 1024 | 512 | 5120/5120 | 5241539 | 5486.57 | 49379.13 | 86588.80 | 4.34 |

原先的 49152 分块 A/B 结果均不纳入正式表：`auto`/普通 FP8 路径曾触发
shape assertion，显式 DeepGEMM 在高并发下 OOM，显式 MxFP4 曾触发缺失
`output1_scale_scalar`；即使 benchmark 客户端显示 completed，也存在
`retokenized=0` 或 Prefill 断连。最终 8192 配置通过请求完整性后才重新测量。

### Step 25：TBO 与 DSpark/DFlash 兼容性修复及功能/性能复验（2026-08-23）

针对历史 Step 4 的 TBO 启动失败，按运行时堆栈分三层修复：

1. `split_spec_info` 对 `DFlashVerifyInput` 使用可选的 Eagle-only 字段，避免访问不存在的 `retrieve_*` 和 `seq_lens_cpu`。
2. `dataclasses.replace` 只传入具体 dataclass 实际拥有的字段。
3. DSpark/DFlash draft worker 没有 `TboAttnBackend` wrapper，因此 Decode CUDA Graph runner 在 draft worker 跳过 TBO capture/replay；target worker 仍保留 TBO 路径。

修复后的 TBO 服务完成 target/draft CUDA Graph capture，Router `whoami` 返回 `HTTP=200`、`WHOAMI_VALID=True`，1024/1024/C1 的 10 个请求全部成功且 retokenized output 为 10236。性能为：Out tok/s=182.91、Total tok/s=365.81、Mean TTFT=194.09 ms、Mean TPOT=5.28 ms。

相比同一 workload 的纯 DSpark 结果（Out=295.63、Total=591.27、TTFT=195.03 ms、TPOT=3.19 ms），TBO Total 下降约 38.1%，TPOT 增加约 65.5%。因此 TBO 当前结论是“功能可用，但在 DSpark PD 路径上不是收益项”，不进入最终部署；修复源码和日志仍保留用于后续专项优化。

证据：`repro_pr32281_20260823/step24_prefill_megamoe_decode_dspark/logs/services/decode_tbo_fix4/`、`logs/validation/whoami_tbo_fix4_20260823.log`、`logs/results/tbo_fix4/`。

### Step 26：HiSparse 独立路径与 DSpark 兼容性边界（2026-08-23）

Step 26 将当前 PD 配置中的 Decode DSpark 关闭，只保留 HiSparse，用于区分
“HiSparse 自身是否能工作”和“HiSparse+DSpark 是否兼容”。Prefill 仍使用 MegaMoE，
Decode 使用 DeepEP `low_latency`、CUDA Graph `1/2/4/8/16/32/64/128`，HiSparse
参数为 `top_k=1024`、`device_buffer_size=1024`、`host_to_device_ratio=8`。

HiSparse 独立路径成功完成初始化，日志显示 C4 host pool 已建立（每个 Decode DP
rank 约 60.54 GB host pool），目标 Decode CUDA Graph 全部 capture 成功；Router
`whoami` 返回 `HTTP=200`、`WHOAMI_VALID=True`。1024/1024/C1 的 10 个请求全部
成功，retokenized output 为 10236/10240：

| ISL | OSL | C | Requests | Out tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 10/10 | 72.31 | 144.63 | 201.97 | 13.64 |

相同输入下，纯 DSpark 的 Out/Total 为 295.63/591.27 tok/s、TPOT 为 3.19 ms，
因此 HiSparse 独立路径当前主要验证了功能和分层 KV 的实际生效，并没有带来性能
收益。它不进入最终 DSpark 部署。

此前 HiSparse+DSpark 能启动但出现 8131/10240 retokenized output；这属于静默
正确性风险，不再作为有效结果。现在 `hisparse_hook.py` 在启动阶段明确拒绝
`--enable-hisparse` 与 speculative decoding 的组合，避免服务进入不安全状态。
Decode 脚本增加了 `SGLANG_ENABLE_DSPARK=0` 和 `SGLANG_ENABLE_HISPARSE=1` 开关，
并修复 HiSparse 默认 JSON 参数多余 `}` 的脚本错误。

实验目录：`repro_pr32281_20260823/step26_hisparse_no_spec/`；源码和脚本备份：
`backups/prefill_megamoe_decode_dspark_20260823/source_patches/`。配置记录、服务
日志、whoami 和 JSONL 结果均保存在实验目录中。

### Step 27：最终配置恢复与 8192 路径闭环审计（2026-08-23）

本步骤不是新的性能 A/B，而是对“Prefill MegaMoE + Decode 仅 DSpark”最终方案的
运行态复核，避免把早期 49152 分块失败记录误当成当前配置。当前正式配置明确为：

```text
Prefill：MegaMoE，ISL=8192 时 max-prefill-tokens=chunked-prefill-size=8192，
         transfer worker=16
Decode：DSpark + DeepEP low_latency，moe-runner-backend=auto，
        static verify，CUDA Graph batch=1/2/4/8/16/32/64/128
关闭：HiSparse、TBO、Waterfill、LPLB、FP4 indexer、DeepGEMM target runner
```

最终恢复后的 Prefill 和 Decode 服务均输出 `The server is fired up and ready to
roll!`。Decode 日志明确出现 `Initialized DSpark draft runner`，并且 target/draft
verify CUDA Graph 均完成 capture；没有启用 HiSparse。Router 验证为
`HTTP=200`、`WHOAMI_VALID=True`，因此服务启动、PD 注册和 DSpark 路径均通过功能门槛。

8192/1024 的完整矩阵证据为：

| Concurrency | Requests | Retokenized output | Out tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10/10 | 10238 | 302.15 | 2719.31 | 205.19 | 3.11 |
| 16 | 160/160 | 164302 | 3411.87 | 30706.81 | 548.56 | 3.98 |
| 256 | 2560/2560 | 2622363 | 5452.67 | 49074.07 | 41318.77 | 4.27 |
| 512 | 5120/5120 | 5241539 | 5486.57 | 49379.13 | 86588.80 | 4.34 |

其中 retokenized 数量与目标输出规模一致到 DSpark 接受/补偿造成的正常小幅差异，
没有出现早期失败实验中的 `retokenized=0`、Prefill 断连或服务退出。原始 49152
profile 仍保留在 Step 18--20 作为失败/不适用的历史证据，但已被 Step 24 的
8192 分块正式替代，不能用于最终部署。

本次最终恢复日志位于：
`repro_pr32281_20260823/step24_prefill_megamoe_decode_dspark/logs/services/`
下的 `prefill_final_restore2/` 和 `decode_final_restore2/`；whoami 记录为
`logs/validation/whoami_final_restore2_20260823.log`。当前启动脚本和 HiSparse
兼容性修复的最终副本已备份到：
`backups/prefill_megamoe_decode_dspark_20260823/source_patches/`。

### Step 28--32：Prefill runner 与 chunk size 的针对性优化（2026-08-23）

为定位高并发长输入吞吐瓶颈，保持 Decode 完全不变，仅替换 Prefill runner 和
chunk size。Decode 始终为 DeepEP `low_latency`、pure DSpark、static verify 和
CUDA Graph batch `1/2/4/8/16/32/64/128`。

Step 28 验证了 `MegaMoE + chunk32768` 不可用：8192/C16 时 Prefill 在
`sparse_prefill_fwd` 因额外 2 GiB attention 分配 OOM，随后 Decode 收到
`KVTransferError/Lost connection with prefill instance`。因此不能把客户端显示的
completed 当作有效结果。

切换到 `flashinfer_mxfp4` 后，全部请求均通过完整性检查：

| Prefill profile | ISL | C1 Total | C16 Total | C256 Total | C512 Total | 结论 |
|---|---:|---:|---:|---:|---:|---|
| MegaMoE/chunk8192 | 1024 | 598.82 | 6891.22 | 63685.94 | 73144.91 | 低/中并发最佳 |
| flashinfer_mxfp4/chunk16384 | 1024 | 564.21 | 6687.43 | 61401.04 | **84766.97** | C512 最佳 |
| MegaMoE/chunk8192 | 8192 | **2719.31** | 30706.81 | 49074.07 | 49379.13 | C1 最佳 |
| flashinfer_mxfp4/chunk32768 | 8192 | 2649.45 | **30895.17** | **72049.21** | 72362.83 | C16/C256/C512 显著更佳 |
| flashinfer_mxfp4/chunk49152 | 8192 | — | — | — | **73989.76** | C512 当前最佳 |

其中 Step 29 的 8192/C16/C256/C512 分别为 160/160、2560/2560、5120/5120
成功；Step 30 的 1024 四组也全部成功；Step 31 的 8192/C1 为 10/10；Step 32
的 8192/C512 为 5120/5120，retokenized=5241532。Step 32 的结果相比
chunk32768 再提升约 2.25%。完整 JSONL、启动参数、whoami 和独立服务日志位于：

```text
repro_pr32281_20260823/step29_prefill_flashinfer_mxfp4_chunk32768/
repro_pr32281_20260823/step30_prefill_flashinfer_mxfp4_short/
repro_pr32281_20260823/step31_prefill_flashinfer_mxfp4_chunk32768_c1/
repro_pr32281_20260823/step32_prefill_flashinfer_mxfp4_chunk49152/
```

当前结论不是“一个 runner 在所有 workload 都最优”，而是 Prefill attention
峰值和调度粒度随 ISL/并发变化：短输入低并发保留 MegaMoE/chunk8192；1024 高并发
使用 flashinfer_mxfp4/chunk16384；8192 中高并发使用
flashinfer_mxfp4/chunk32768，8192/C512 可用 chunk49152。Decode 侧 DSpark
配置已证明稳定，主要瓶颈已转移到 Prefill 的 attention 峰值、chunk 调度和 PD
KV 传输排队。
