# 单机 1P1D DSpark 实验报告（2026-08-17 整理版）

> 范围：同一台机器内的 1P1D PD 分离。Prefill 使用 GPU 0–3，Decode 使用 GPU 4–7；不包含 2026-08-16 之后两台机器的 8P/8D RDMA 结果。所有吞吐均按 `Total tok/s = Input tok/s + Output tok/s` 记录。

## 结论摘要

- DSpark 正常完整 PD 结果已覆盖 ISL/OSL=`1024/1024` 与 `8192/1024`，并发 `1/16/256/512`。
- 该单机 1P1D 轮次的纯 MTP/EAGLE 和 MTP+MegaMoE/EAGLE 均已完成同一组 8 条件；完整对照见本文后半部分。
- 纯 MTP 是 `EAGLE`，不是普通 decode；有效参数为 `--speculative-algo EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`。
- 后续两机实验中曾出现没有 EAGLE 参数却标为 MTP 的结果，那些数据不属于本报告，也不用于本报告的比较。

## 部署

Prefill（GPU 0--3，端口 30000）：

```bash
bash ./flash_prefill_baseline.sh > logs/flash_decode_dspark/pr31466_smoke/prefill_hidden_source_debug.log 2>&1
```

Decode/DSpark（GPU 4--7，DP=4、TP=4、EP=4，端口 30001）：

```bash
env CUDA_GRAPH_BS_DECODE='1 2 4 8 16 32 64' \
  bash ./flash_decode_dspark.sh \
  logs/flash_decode_dspark/archive_20260803_091434/dspark_sps.json \
  > logs/flash_decode_dspark/pr31466_smoke/decode_graph64_tpot_20260804.log 2>&1
```

上述脚本实际展开的 `sglang.launch_server`（graph64、原 DSpark/none 轮次）为：

```bash
python3 -m sglang.launch_server \
  --model-path /data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash \
  --served-model-name deepseek-ai/DeepSeek-V4-Flash --trust-remote-code \
  --tool-call-parser deepseekv4 --host 0.0.0.0 --port 30001 \
  --base-gpu-id 4 --skip-server-warmup --tp-size 4 --dp-size 4 --ep-size 4 \
  --enable-dp-attention --enable-dp-lm-head --disable-flashinfer-autotune \
  --mem-fraction-static 0.8 --swa-full-tokens-ratio 0.1 \
  --chunked-prefill-size 1024 --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device '{"4":"mlx5_4","5":"mlx5_9","6":"mlx5_10","7":"mlx5_11"}' \
  --moe-a2a-backend none --moe-runner-backend flashinfer_mxfp4 \
  --max-running-requests 1024 \
  --cuda-graph-bs-decode 1 2 4 8 16 32 64 \
  --speculative-algorithm DSPARK --speculative-attention-mode decode \
  --speculative-draft-model-path /data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash-dspark \
  --speculative-dspark-sps-table-path logs/flash_decode_dspark/archive_20260803_091434/dspark_sps.json \
  --disable-radix-cache
```

当前正在验证的 DeePEP 变体只替换下面三项（其余参数完全相同）：

```bash
--moe-a2a-backend deepep \
--deepep-mode low_latency \
--deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'
```

路由端口为 `13784`。当前 DSpark 脚本使用 `moe-a2a-backend none`、`flashinfer_mxfp4`，并通过 SPS table 启用 DSPARK；graph64 这一轮的服务信息确认 `DP=4, EP=4`，decode graph 为 `[1,2,4,8,16,32,64]`。

## 实际环境参数

### Prefill 环境变量

| 参数 | 值 |
|---|---|
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` |
| `SGLANG_DEFAULT_THINKING` | `1` |
| `SGLANG_DSV4_REASONING_EFFORT` | `max` |
| `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` / `WAITING_TIMEOUT` | `100000 / 100000` |
| `SGLANG_PD_HIDDEN_POOL_TOKENS` | 正式 graph64 轮 `524288`（早期稳定轮 `131072`） |
| `SGLANG_DSPARK_PD_HIDDEN_BUFFER_POOL_LIMIT` | `512` |
| `SGLANG_DSPARK_PD_HIDDEN_TRANSFER_QUEUE_LIMIT` | `64` |
| `SGLANG_DSPARK_PD_HIDDEN_TRANSFER_QUEUE_BYTES` | `2147483648` |
| `SGLANG_RADIX_FORCE_MISS` | `1` |
| `SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN` | `1` |
| `SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW` | `1` |
| `SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT` | `1` |
| `SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2` | `0` |

Prefill 服务固定为 TP=4、DP=1、EP=1，`flashinfer_mxfp4`，`--mem-fraction-static 0.9`，关闭 radix cache 和 overlap schedule。Decode 服务固定为 TP=4、DP=4、EP=4，开启 DP attention/LM head，DeepEP 变体使用 `low_latency`，`normal_dispatch/normal_combine.num_sms=96`，`--mem-fraction-static 0.8`，`--chunked-prefill-size 4096`，关闭 radix cache 和 overlap schedule。Decode 的 DSpark draft 模型为 `/data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash-dspark`，SPS table 为 `logs/flash_decode_dspark/archive_20260803_091434/dspark_sps.json`。

## 压测命令

```bash
python3 -m sglang.benchmark.serving \
  --backend sglang --base-url http://127.0.0.1:13784 \
  --dataset-name random \
  --tokenizer /data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash \
  --num-prompts $((10 * CONCURRENCY)) \
  --random-input-len 1024 --random-output-len 1024 \
  --random-range-ratio 1 --max-concurrency "$CONCURRENCY" \
  --output-file <result>.jsonl --seed 1 --disable-tqdm \
  --warmup-requests 1 --tokenize-prompt
```

## 已完成结果

| CUDA Graph | 并发 | 请求数 | 输入/输出 tok/s | Total tok/s | Mean TTFT | Mean TPOT | 结果文件 |
|---|---:|---:|---:|---:|---:|---:|---|
| max-bs=16（原配置） | 16 | 160 | 4,073.04 / 4,073.04 | 8,146.07 | 962.90 ms | 2.82 ms | `clean_restart_isl1024_osl1024_c16_n160.jsonl` |
| max-bs=16（原配置） | 256 | 2,560 | 5,145.76 / 5,145.76 | 10,291.51 | 1,874.49 ms | 46.24 ms | `clean_restart_isl1024_osl1024_c256_n2560.jsonl` |
| max-bs=16（原配置） | 512 | 5,120 | 9,561.26 / 9,561.26 | 19,122.52 | 2,888.57 ms | 48.44 ms | `clean_restart_isl1024_osl1024_c512_n5120.jsonl` |
| sparse graph `[1,2,4,8,16,32,64]` | 256 | 2,560 | 15,875.23 / 15,875.23 | **31,750.45** | 9,349.06 ms | **6.26 ms** | `graph64_isl1024_osl1024_c256_n2560.jsonl` |
| graph64 + hidden pool/queue 512/64（decode侧） | 256 | 2,560 | 20,319.82 / 20,319.82 | 40,639.64 | 4,362.86 ms | 7.76 ms | `graph64_hidden_pool512_isl1024_osl1024_c256_n2560.jsonl` |
| graph64 + hidden pool/queue 512/64（prefill+decode侧） | 256 | 2,560 | 20,997.22 / 20,997.22 | 41,994.43 | 4,031.20 ms | 7.84 ms | `graph64_hidden_pool512_both_isl1024_osl1024_c256_n2560.jsonl` |
| graph128 + hidden pool/queue 512/64 + chunk4096 | 256 | 2,560 | 20,218.37 / 20,218.37 | 40,436.74 | 4,458.60 ms | 7.81 ms | `graph128_pool512_chunk4096_isl1024_osl1024_c256_n2560.jsonl` |

对照：已有 MTP（MegaMoE + EAGLE，非 DSpark）结果为 c256 `48,112.78 tok/s / 9.07 ms TPOT`、c512 `68,561.52 tok/s / 11.76 ms TPOT`。因此 DSpark graph64 的 TPOT 已低于 MTP，但总吞吐仍低于 MTP，主要差距来自 TTFT/输入阶段（DSpark c256 平均 TTFT 9.35 s，而 MTP 为 1.12 s），不能把 DSpark 当前结果视为已达到 MTP。

## MTP 对照结果（全部条件）

以下两组均为此前已完成的 1P1D decode 实验，指标来自对应目录中的原始 JSONL；`Total tok/s = input + output`。

### MTP only（无 MegaMoE，EAGLE）

配置与说明见 `readmes/PD_COMPLETED_REPORT_20260803.md`；原始结果目录：`logs/flash_decode_mtp_only/results_20260802/`。

| ISL | OSL | 并发 | Req/s | Input tok/s | Output tok/s | Total tok/s | Mean TTFT | Mean TPOT | 结果文件 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1024 | 1024 | 1 | 0.228 | 233.36 | 233.36 | 466.73 | 335.21 ms | 3.96 ms | `isl1024_osl1024_c1_n10.jsonl` |
| 1024 | 1024 | 16 | 2.445 | 2,503.65 | 2,503.65 | 5,007.31 | 818.52 ms | 5.48 ms | `isl1024_osl1024_c16_n160.jsonl` |
| 1024 | 1024 | 256 | 20.338 | 20,826.03 | 20,826.03 | 41,652.06 | 1,057.71 ms | 10.79 ms | `isl1024_osl1024_c256_n2560.jsonl` |
| 1024 | 1024 | 512 | 27.591 | 28,253.34 | 28,253.34 | 56,506.67 | 1,657.06 ms | 15.69 ms | `isl1024_osl1024_c512_n5120.jsonl` |
| 8192 | 1024 | 1 | 0.228 | 1,869.11 | 233.64 | 2,102.75 | 280.20 ms | 4.01 ms | `isl8192_osl1024_c1_n10.jsonl` |
| 8192 | 1024 | 16 | 2.294 | 18,794.44 | 2,349.31 | 21,143.75 | 922.71 ms | 5.62 ms | `isl8192_osl1024_c16_n160.jsonl` |
| 8192 | 1024 | 256 | 7.198 | 58,965.99 | 7,370.75 | 66,336.74 | 26,160.52 ms | 7.45 ms | `isl8192_osl1024_c256_n2560.jsonl` |
| 8192 | 1024 | 512 | 7.243 | 59,336.35 | 7,417.04 | 66,753.39 | 59,777.21 ms | 7.43 ms | `isl8192_osl1024_c512_n5120.jsonl` |

### MTP + MegaMoE（EAGLE）

实际 decode 配置见 `logs/flash_decode_mtp/report.md`（`--moe-a2a-backend megamoe`）；原始结果目录：`logs/flash_decode_mtp/results/`。

| ISL | OSL | 并发 | Req/s | Input tok/s | Output tok/s | Total tok/s | Mean TTFT | Mean TPOT | 结果文件 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1024 | 1024 | 1 | 0.263 | 269.44 | 269.44 | 538.87 | 334.79 ms | 3.39 ms | `isl1024_osl1024_c1_n10.jsonl` |
| 1024 | 1024 | 16 | 2.761 | 2,827.61 | 2,827.61 | 5,655.22 | 852.90 ms | 4.69 ms | `isl1024_osl1024_c16_n160.jsonl` |
| 1024 | 1024 | 256 | 23.493 | 24,056.39 | 24,056.39 | 48,112.78 | 1,115.38 ms | 9.07 ms | `isl1024_osl1024_c256_n2560.jsonl` |
| 1024 | 1024 | 512 | 33.477 | 34,280.76 | 34,280.76 | 68,561.52 | 2,488.64 ms | 11.76 ms | `isl1024_osl1024_c512_n5120.jsonl` |
| 8192 | 1024 | 1 | 0.254 | 2,081.28 | 260.16 | 2,341.44 | 291.94 ms | 3.56 ms | `isl8192_osl1024_c1_n10.jsonl` |
| 8192 | 1024 | 16 | 2.682 | 21,973.45 | 2,746.68 | 24,720.13 | 781.37 ms | 4.87 ms | `isl8192_osl1024_c16_n160.jsonl` |
| 8192 | 1024 | 256 | 7.268 | 59,537.46 | 7,442.18 | 66,979.64 | 27,353.09 ms | 6.07 ms | `isl8192_osl1024_c256_n2560.jsonl` |
| 8192 | 1024 | 512 | 7.253 | 59,412.81 | 7,426.60 | 66,839.41 | 61,097.70 ms | 6.06 ms | `isl8192_osl1024_c512_n5120.jsonl` |

说明：`logs/flash_decode_mtp/retest_fixed_20260802/` 还保留了一组复测值；上表统一采用 `report.md` 引用的 `results/` 主结果，避免混用不同轮次。

## 启动脚本与原始日志索引

- Prefill 脚本：`flash_prefill_baseline.sh`
- DSpark Decode 脚本：`flash_decode_dspark.sh`
- 纯 MTP/EAGLE Decode 脚本：`flash_decode_mtp_only.sh`
- MTP+MegaMoE/EAGLE Decode 脚本：`flash_decode_mtp.sh`
- Router：单机端口 `13784`，Prefill `30000`，Decode `30001`
- DSpark 原始结果：`logs/flash_decode_dspark/results/`
- 纯 MTP 原始结果：`logs/flash_decode_mtp_only/results_20260802/`
- MTP+MegaMoE 原始结果：`logs/flash_decode_mtp/results/`
- Prefill/Decode benchmark 日志：各结果目录中同名 `.log`，以及 `logs/flash_decode_dspark/pr31466_smoke/`

本文中的 MTP 表只采用带 EAGLE 参数的单机 1P1D 结果；后续两机目录中缺少 EAGLE 参数的所谓 MTP 结果已明确排除。

### 单机 1P1D Total throughput 对比摘要

下表把本文前面的 DSpark 正式结果与同条件纯 MTP/EAGLE、MTP+MegaMoE/EAGLE 直接并列；百分比为 DSpark 相对对应 MTP 的变化。

| ISL/OSL | 并发 | DSpark | 纯 MTP/EAGLE | DSpark vs 纯 MTP | MTP+MegaMoE/EAGLE | DSpark vs MTP+MegaMoE |
|---|---:|---:|---:|---:|---:|---:|
|1024/1024|1|578.99|466.73|+24.1%|538.87|+7.4%|
|1024/1024|16|6,294.22|5,007.31|+25.7%|5,655.22|+11.3%|
|1024/1024|256|47,361.84|41,652.06|+13.7%|48,112.78|-1.6%|
|1024/1024|512|51,323.91|56,506.67|-9.2%|68,561.52|-25.1%|
|8192/1024|1|2,266.68|2,102.75|+7.8%|2,341.44|-3.2%|
|8192/1024|16|15,675.56|21,143.75|-25.9%|24,720.13|-36.6%|
|8192/1024|256|18,992.51|66,336.74|-71.4%|66,979.64|-71.6%|
|8192/1024|512|19,760.36|66,753.39|-70.4%|66,839.41|-70.4%|

结论：单机短输入低/中并发时 DSpark 可接近或超过纯 MTP；长输入高并发时主要受 PD hidden/KV 传输和 Prefill 排队限制，明显落后 MTP。MTP+MegaMoE 是组合基线，不能误称为“纯 MTP”。

### MTP 结果索引与已测条件说明

两组 MTP 结果的原始文件位置如下：

- 单独 MTP（无 MegaMoE）：`logs/flash_decode_mtp_only/results_20260802/`
- MTP + MegaMoE：`logs/flash_decode_mtp/results/`
- MTP + MegaMoE 的部署参数和运行说明：`logs/flash_decode_mtp/report.md`
- 单独 MTP 的运行状态记录：`logs/flash_decode_mtp_only/results_20260802/status.tsv`

当前磁盘上两组实验都实际完成了同一组 8 个条件：
`(ISL, OSL, 并发) = (1024,1024,{1,16,256,512})` 和
`(8192,1024,{1,16,256,512})`。因此上面两张表已经覆盖当前已完成的全部 MTP / MTP+MegaMoE 数据；`c8/c32/c64/c128` 没有对应 JSONL 原始结果，不能从已有日志补出数值，后续若需要这些并发档位应单独补测。

每轮对应的 benchmark 日志位于同目录下的同名 `.log` 文件；Prefill 日志为 `pr31466_smoke/prefill_hidden_source_debug.log`，Decode 日志为 `pr31466_smoke/decode_graph64_tpot_20260804.log`。

### DeePEP low_latency 回归（shape 修复后）

在 `deepep.py` 的 low-latency dispatch 入口对 DSpark CUDA-Graph 产生的 leading padded 维度做 token-major 展平，使 activation、top-k id 和 top-k weight 的首维一致。shape 诊断示例为 `hidden=(96,4096)`、`topk_ids=(96,6)`、`topk_weights=(96,6)`，DeepEP 的 `x.size(0)==topk_idx.size(0)` 断言通过。

| CUDA Graph | 并发 | 输入/输出 tok/s | Total tok/s | Mean TTFT | Mean TPOT | 结果文件 |
|---|---:|---:|---:|---:|---:|---|
| DeePEP low_latency，graph16 | 16 | 3,870.01 / 3,870.01 | 7,740.02 | 918.55 ms | 3.09 ms | `deepep_flatten_graph16_diag_isl1024_osl1024_c16_n160.jsonl` |

graph64 DeePEP 的高并发回归已完成隔离；正常 hidden 注入和关闭注入的结果分别见下文，不能把关闭注入结果当作完整 PD DSpark 结果。

graph64 DeePEP 已完成 CUDA-Graph capture，但 c256 压测过程中在 decode scheduler 的 `prepare_for_prebuilt()` 处触发 `CUDA error: an illegal memory access was encountered`，导致全部请求返回 502，未形成有效吞吐数据。失败日志为 `pr31466_smoke/decode_deepep_graph64_exportfix_20260804.log`；该轮不计入性能表。

进一步隔离：关闭 PD hidden 注入时 graph64/c256 可完整完成（Total 33,230.32 tok/s，Mean TPOT 10.64 ms）；恢复注入后，即使增加 full→SWA 映射边界校验，仍在异步 fused hidden-KV 写入后报同一 illegal memory access，且没有 invalid/unmapped location 记录。该结果将问题范围收敛到 DeePEP low_latency 与 DSpark hidden-KV 注入/overlap 调度的竞态，不能以简单 cache index 修复解决。对应隔离结果：`deepep_graph64_nohidden_isl1024_osl1024_c256_n2560.jsonl`；带注入失败日志：`pr31466_smoke/decode_deepep_graph64_swa_boundsfix_20260804.log`。

追加同步探针：在 hidden-KV 写入后执行 `torch.cuda.current_stream().synchronize()`，c256 仍复现 illegal memory access（`pr31466_smoke/decode_deepep_graph64_sync_hidden2_20260804.log`）。因此不是单纯 stream 未同步，更可能是 DeePEP 路径与 DSpark target-hidden KV 写入 kernel/缓存布局本身不兼容。

另行尝试 `--disable-overlap-schedule`，c256 仍失败且无有效结果；该参数不能规避问题，未作为默认配置（脚本支持通过 `SGLANG_DISABLE_OVERLAP_SCHEDULE=1` 进行复现）。

运行时比较主模型与 draft 模型的 req→token pool：四个 DP rank 均 `same_storage=True`、shape `(257,1048843)`、`max_delta=0`，排除“主/draft 使用不同 KV 页地址”这一假设。诊断日志：`pr31466_smoke/decode_deepep_pool_diag_20260804.log`。

关闭 DSpark draft CUDA Graph（`SGLANG_DSPARK_DISABLE_DRAFT_CUDA_GRAPH=1`）后，target graph64 + DeePEP 的 c256 仍失败，进一步排除 draft graph 状态复用；失败日志：`pr31466_smoke/decode_deepep_graph64_no_draft_graph_20260804.log`。

强制关闭 DSpark fused `CommitKvProj`、改用 PyTorch WKV 投影后，graph64/c256 仍失败（`pr31466_smoke/decode_deepep_graph64_no_fused_commit_20260804.log`），排除单个 fused WKV 投影 kernel；该开关 `SGLANG_DSPARK_DISABLE_FUSED_COMMIT_KV=1` 仅用于诊断。

按上游 PR #31466 示例将 `--swa-full-tokens-ratio` 调为 `0.8`，graph64/c256 仍失败（`pr31466_smoke/decode_deepep_graph64_swa08_20260804.log`），排除 SWA/full 页面比例差异。

## 当前结论

原 max-bs=16 在高并发时大量请求无法命中 decode CUDA Graph，TPOT 升至 46--48 ms。扩展 graph、同步扩大 prefill/decode hidden transfer pool（buffer 512、queue 64、queue bytes 2 GiB）后，built-in TP MoE c256 的 Total throughput 达到 41,994 tok/s，Mean TPOT 约 7.8 ms。MTP（MegaMoE + EAGLE）仍为 48,113 tok/s，因此 DSpark 仍有约 13% 总吞吐差距；差距主要在 PD 首 token/输入阶段，而非 decode TPOT。DeePEP low_latency 经 shape 展平修复后 graph16/c16 可用，graph64/c256 仅在关闭 hidden 注入时稳定，完整 PD hidden 注入仍存在异步 illegal-memory 竞态，稳定默认继续使用 built-in TP MoE。

## 2026-08-05 回退基线：8192/1024 DSpark + DeepEP（可复现部署）

本节记录后续优化使用的干净基线。该轮在容器重启后重新启动，prefill 使用 131072 hidden pool，decode 使用 DSpark、DeepEP low-latency、decode CUDA Graph max-bs=16，完成 8192/1024、并发16、160请求，160/160 成功。注意：此前更早的“8192 完全跑通”日志中 `moe_a2a_backend=None`，本节才是恢复 DeepEP 后的有效基线。

### 实际启动指令

工作目录：`/data/ssd2/gongoubo/single_node`。

Prefill（GPU 0--3，端口 30000；以下为实际 Python 启动命令）：

```bash
setsid -f env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  PYTHONUNBUFFERED=1 SGLANG_DEFAULT_THINKING=1 \
  SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
  SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
  SGLANG_DSV4_REASONING_EFFORT=max \
  SGLANG_PD_HIDDEN_POOL_TOKENS=131072 \
  SGLANG_DSPARK_PD_HIDDEN_BUFFER_POOL_LIMIT=512 \
  SGLANG_DSPARK_PD_HIDDEN_TRANSFER_QUEUE_LIMIT=64 \
  SGLANG_DSPARK_PD_HIDDEN_TRANSFER_QUEUE_BYTES=2147483648 \
  python3 -m sglang.launch_server \
  --model-path /data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash \
  --served-model-name deepseek-ai/DeepSeek-V4-Flash \
  --trust-remote-code --tool-call-parser deepseekv4 \
  --host 0.0.0.0 --port 30000 --skip-server-warmup \
  --tp-size 4 --dp-size 1 --ep-size 1 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device '{"0":"mlx5_0","1":"mlx5_1","2":"mlx5_2","3":"mlx5_3"}' \
  --moe-runner-backend flashinfer_mxfp4 \
  --disable-flashinfer-autotune --mem-fraction-static 0.9 \
  --swa-full-tokens-ratio 0.1 --disable-radix-cache \
  --disable-overlap-schedule \
  > logs/flash_decode_dspark/prefill_rollback8192_20260805.log 2>&1
```

Decode（GPU 4--7，端口 30001；EP=4，DeepEP low-latency，Graph 最大16；以下为实际 Python 启动命令）：

```bash
setsid -f env SGLANG_DEEPEP_LL_SPLIT_TOKENS=0 \
  SGLANG_DSPARK_PD_HIDDEN_POOL_TOKENS=65536 \
  SGLANG_DSPARK_PD_HIDDEN_BUFFER_POOL_LIMIT=512 \
  SGLANG_DSPARK_PD_HIDDEN_TRANSFER_QUEUE_LIMIT=64 \
  SGLANG_DSPARK_PD_HIDDEN_TRANSFER_QUEUE_BYTES=2147483648 \
  python3 -m sglang.launch_server \
  --model-path /data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash \
  --served-model-name deepseek-ai/DeepSeek-V4-Flash \
  --trust-remote-code --tool-call-parser deepseekv4 \
  --host 0.0.0.0 --port 30001 --base-gpu-id 4 --skip-server-warmup \
  --tp-size 4 --dp-size 4 --ep-size 4 \
  --enable-dp-attention --enable-dp-lm-head \
  --disable-flashinfer-autotune --mem-fraction-static 0.8 \
  --swa-full-tokens-ratio 0.1 --chunked-prefill-size 4096 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device '{"4":"mlx5_4","5":"mlx5_9","6":"mlx5_10","7":"mlx5_11"}' \
  --moe-a2a-backend deepep --max-running-requests 1024 \
  --deepep-mode low_latency \
  --deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}' \
  --cuda-graph-max-bs-decode 16 --disable-overlap-schedule \
  --speculative-algorithm DSPARK --speculative-attention-mode decode \
  --speculative-draft-model-path /data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash-dspark \
  --speculative-dspark-sps-table-path /data/ssd2/gongoubo/single_node/logs/flash_decode_dspark/dspark_sps.json \
  --disable-radix-cache \
  > logs/flash_decode_dspark/decode_rollback8192_deepep16_20260805.log 2>&1
```

该脚本展开后的关键参数为：
`--tp-size 4 --dp-size 4 --ep-size 4 --enable-dp-attention --enable-dp-lm-head --moe-a2a-backend deepep --deepep-mode low_latency --cuda-graph-max-bs-decode 16 --speculative-algorithm DSPARK --speculative-attention-mode decode --disable-overlap-schedule`。

Router（端口 13784；实际 Python 启动命令）：

```bash
setsid -f python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 \
  --decode http://127.0.0.1:30001 \
  --host 0.0.0.0 --port 13784 \
  --disable-circuit-breaker --disable-health-check \
  --health-check-interval-secs 999999 \
  > logs/flash_decode_dspark/router_rollback8192_20260805.log 2>&1
```

### 实际测试指令与结果

```bash
python3 -m sglang.benchmark.serving \
  --backend sglang --base-url http://127.0.0.1:13784 \
  --dataset-name random --model deepseek-ai/DeepSeek-V4-Flash \
  --tokenizer /data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash \
  --random-input-len 8192 --random-output-len 1024 \
  --random-range-ratio 1 --num-prompts 160 \
  --max-concurrency 16 --request-rate inf --tokenize-prompt \
  --disable-tqdm --seed 5 \
  --output-file logs/flash_decode_dspark/results/dspark_isl8192_osl1024_c16_p160_rollback_deepep_20260805.jsonl
```

| ISL | OSL | 并发 | 请求数 | Req/s | Input tok/s | Output tok/s | Total tok/s | Mean TTFT | Mean TPOT | 完成数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 1024 | 16 | 160 | 1.2997 | 10647.08 | 1330.88 | 11977.96 | 3185.93 ms | 8.74 ms | 160/160 |

原始结果：[dspark_isl8192_osl1024_c16_p160_rollback_deepep_20260805.jsonl](dspark_isl8192_osl1024_c16_p160_rollback_deepep_20260805.jsonl)；Decode 日志：[decode_rollback8192_deepep16_20260805.log](../decode_rollback8192_deepep16_20260805.log)；Prefill 日志：[prefill_rollback8192_20260805.log](../prefill_rollback8192_20260805.log)；Router 日志：[router_rollback8192_20260805.log](../router_rollback8192_20260805.log)。

这组数据是后续优化的起点，不是最终性能结论。下一轮应保持 EP=4、DeepEP low-latency 和 CUDA Graph，优先将 decode hidden pool 从65536恢复至131072并复测，再逐步扩大 Graph bucket（目标256），同时避免连续强杀 worker 造成 NCCL/端口残留。

### 已完成的 8 个 DSpark 样例总表

下表汇总当前磁盘上已完成且请求全部完成的 8 个正式样例。8192/c16 采用 pool131072 的 clean 轮次作为该条件的最佳有效结果；本节上一表的 rollback 轮次用于验证当前可复现启动链路，吞吐较低，不覆盖该最佳值。

| ISL | OSL | 并发 | 请求数 | Req/s | Input tok/s | Output tok/s | Total tok/s | Mean TTFT | Mean TPOT | 完成数 | 原始结果 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1024 | 1024 | 1 | 10 | 0.283 | 289.49 | 289.49 | 578.99 | 348.68 ms | 3.11 ms | 10/10 | `dspark_isl1024_osl1024_c1_p10_20260805.jsonl` |
| 1024 | 1024 | 16 | 160 | 3.073 | 3147.11 | 3147.11 | 6294.22 | 669.35 ms | 4.20 ms | 160/160 | `dspark_isl1024_osl1024_c16_p160_20260805.jsonl` |
| 1024 | 1024 | 256 | 2560 | 23.126 | 23680.92 | 23680.92 | 47361.84 | 1674.10 ms | 8.73 ms | 2560/2560 | `dspark_isl1024_osl1024_c256_p2560_20260805.jsonl` |
| 1024 | 1024 | 512 | 5120 | 25.061 | 25661.95 | 25661.95 | 51323.91 | 7509.92 ms | 11.82 ms | 5120/5120 | `dspark_isl1024_osl1024_c512_p5120_20260805.jsonl` |
| 8192 | 1024 | 1 | 10 | 0.246 | 2014.83 | 251.85 | 2266.68 | 759.12 ms | 3.23 ms | 10/10 | `dspark_isl8192_osl1024_c1_p10_20260805.jsonl` |
| 8192 | 1024 | 16 | 160 | 1.701 | 13933.83 | 1741.73 | 15675.56 | 1732.91 ms | 7.23 ms | 160/160 | `dspark_isl8192_osl1024_c16_p160_pool131k_clean_20260805.jsonl` |
| 8192 | 1024 | 256 | 2560 | 2.061 | 16882.23 | 2110.28 | 18992.51 | 79947.74 ms | 41.77 ms | 2560/2560 | `dspark_isl8192_osl1024_c256_p2560_syncfix_20260805.jsonl` |
| 8192 | 1024 | 512 | 5120 | 2.144 | 17564.76 | 2195.60 | 19760.36 | 164131.12 ms | 64.79 ms | 5120/5120 | `dspark_isl8192_osl1024_c512_p5120_syncfix_20260805.jsonl` |

其中 8192/c256 和 c512 是 hidden-transfer sync-fix 后的完整轮次，虽然已完成但明显低于 MTP，不能作为最终目标结果；后续优化从上面的可复现 DeepEP low-latency 部署开始，优先重测更大 hidden pool/queue 和 Graph 256。

### 2026-08-15 流式 hidden transfer 与两张 IB 对照

本轮在 decode 端显式启用 `SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=true`，确保 `CUDA_VISIBLE_DEVICES=4,5,6,7` 映射到物理 GPU 4--7；并使用 `--moe-runner-backend flashinfer_mxfp4`、DeepEP low-latency 和 CUDA Graph。单请求及 160 请求均成功完成。

| 配置 | ISL/OSL | 并发 | Req/s | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---|---:|---:|---:|---:|---:|---:|---:| 
| DSpark 流式 hidden | 8192/1024 | 16 | 5.18 | 2747.14 | 24105.30 | 972.64 ms | 3.32 ms |
| MTP（两张 IB） | 8192/1024 | 16 | -- | 2492.26 | 22430.32 | 573.50 ms | 5.65 ms |

DSpark 相比两张 IB 的 MTP：Total tok/s 提升约 7.5%，Output tok/s 提升约 10.2%，TPOT 降低约 41.2%；TTFT 仍高约 399 ms。原始结果：`../aug15/results/current_stream_isl8192_osl1024_c16_n160.jsonl`。
