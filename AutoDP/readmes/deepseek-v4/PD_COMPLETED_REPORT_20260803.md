# DeepSeek-V4-Flash 单机 1P1D PD 分离已完成实验报告

生成日期：2026-08-03（UTC）

## 1. 验收标准

- 单机 8×B200：Prefill 使用 GPU 0–3，Decode 使用 GPU 4–7。
- 测试矩阵固定为 ISL/OSL=`1024/1024`、`8192/1024`，并发=`1/16/256/512`。
- `num_prompts=concurrency×10`。
- “完成”要求 8 个配置均 exit code 0、成功请求数等于 num_prompts、总输出 token 等于 `num_prompts×1024`。
- TBO 额外要求服务日志确认 `cuda graph: True`；本轮满足。
- Total tok/s 同时包含输入与输出；Decode 性能比较应优先使用 Out tok/s。

## 2. 已完成项目

| 类别 | 项目 | 8/8 | 结果目录 | 备注 |
|---|---|---:|---|---|
| 基准 | PD baseline | 是 | `logs/flash_baseline_ibfix/results` | IB 修复后的基准 |
| Prefill 单技术 | MegaMoE | 是 | `logs/flash_prefill_megamoe_retry2/results` | Decode 保持基准 |
| Prefill 单技术 | DeepEP EP4 | 是 | `logs/flash_prefill_deepep_ep4/results` | TP4/EP4；Decode 保持基准 |
| Decode 单技术 | MegaMoE | 是 | `logs/flash_decode_megamoe/results` | 启用 DP LM Head |
| Decode 单技术 | MTP（纯） | 是 | `logs/flash_decode_mtp_only/results_20260802` | 未启用 MegaMoE；EAGLE；CUDA Graph bs128 |
| Decode 单技术 | FP4 Indexer（纯） | 是 | `logs/flash_decode_fp4_indexer_only/results_20260802`；`logs/runs/fp4_indexer_retest_8192_c512_stride_and_64k_registration_fix_20260803` | 8 个点有效；`8192/c512` 修复 Mooncake KV 元数据与 64 KiB 注册边界后重测通过 |
| Decode 单技术 | TBO（纯） | 是 | `logs/runs/tbo_20260802_150500/results` | DeepEP low_latency；CUDA Graph bs256 |
| Decode 组合 | Waterfill + MegaMoE | 是 | `logs/flash_decode_waterfill_megamoe/results` | 组合结果，不代表纯 Waterfill |
| Decode 组合 | HiSparse + MegaMoE | 是 | `logs/flash_decode_hisparse/results` | 组合结果，不代表纯 HiSparse |

## 3. 公共部署与测试命令

公共模型：`/data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash`。

Prefill 公共参数：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server \
  --model-path /data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash \
  --served-model-name deepseek-ai/DeepSeek-V4-Flash \
  --host 0.0.0.0 --port 30000 \
  --tp-size 4 --dp-size 1 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device '{"0":"mlx5_0","1":"mlx5_1","2":"mlx5_2","3":"mlx5_3"}' \
  --mem-fraction-static 0.9 --disable-radix-cache
```

Decode 公共参数：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m sglang.launch_server \
  --model-path /data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash \
  --served-model-name deepseek-ai/DeepSeek-V4-Flash \
  --host 0.0.0.0 --port 30001 \
  --tp-size 4 --dp-size 4 --ep-size 4 --enable-dp-attention \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device '{"0":"mlx5_4","1":"mlx5_9","2":"mlx5_10","3":"mlx5_11"}' \
  --max-running-requests 1024 --disable-radix-cache
```

Router：

```bash
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 \
  --decode http://127.0.0.1:30001 \
  --host 0.0.0.0 --port 13784 \
  --disable-circuit-breaker --health-check-interval-secs 999999
```

测试：

```bash
TOKENIZER=/data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash \
RESULT_DIR=<result-dir> ./run_flash_baseline_bench.sh
```

技术增量参数：

| 项目 | 增量参数/脚本 |
|---|---|
| Prefill MegaMoE | `--moe-a2a-backend megamoe`；`flash_prefill_megamoe.sh` |
| Prefill DeepEP EP4 | `--ep-size 4 --moe-a2a-backend deepep --deepep-mode normal`；`flash_prefill_deepep.sh` |
| Decode MegaMoE | `--moe-a2a-backend megamoe --enable-dp-lm-head`；`flash_decode_megamoe.sh` |
| MTP only | `--speculative-algo EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --cuda-graph-max-bs-decode 128`；`mem-fraction-static=0.78`；`flash_decode_mtp_only.sh` |
| FP4 Indexer only | `--enable-deepseek-v4-fp4-indexer --moe-a2a-backend deepep --deepep-mode low_latency`；`flash_decode_fp4_indexer_only.sh` |
| TBO only | `--enable-two-batch-overlap --cuda-graph-max-bs-decode 256 --moe-a2a-backend deepep --deepep-mode low_latency`；`flash_decode_tbo_only.sh` |
| Waterfill + MegaMoE | `--enable-waterfill --moe-a2a-backend megamoe --enable-dp-lm-head`；`flash_decode_waterfill_megamoe.sh` |
| HiSparse + MegaMoE | `--enable-hisparse --moe-a2a-backend megamoe --enable-dp-lm-head`；`flash_decode_hisparse.sh` |

## 4. 完整测试结果

### 4.1 Baseline

| ISL | OSL | C | Req/s | Out tok/s | Total tok/s | TTFT ms | TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 0.11 | 107.55 | 215.09 | 332.51 | 8.98 |
| 1024 | 1024 | 16 | 1.32 | 1348.95 | 2697.90 | 631.91 | 11.25 |
| 1024 | 1024 | 256 | 12.80 | 13106.67 | 26213.34 | 2284.50 | 16.82 |
| 1024 | 1024 | 512 | 12.93 | 13243.63 | 26487.25 | 16554.06 | 21.32 |
| 8192 | 1024 | 1 | 0.10 | 106.52 | 958.65 | 348.65 | 9.05 |
| 8192 | 1024 | 16 | 1.30 | 1328.45 | 11956.04 | 607.99 | 11.23 |
| 8192 | 1024 | 256 | 6.70 | 6861.28 | 61751.53 | 21913.55 | 14.34 |
| 8192 | 1024 | 512 | 6.94 | 7110.57 | 63995.17 | 55535.20 | 14.42 |

### 4.2 Prefill MegaMoE

| ISL | OSL | C | Req/s | Out tok/s | Total tok/s | TTFT ms | TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 0.11 | 108.22 | 216.44 | 266.64 | 8.99 |
| 1024 | 1024 | 16 | 1.26 | 1292.49 | 2584.97 | 1103.23 | 11.31 |
| 1024 | 1024 | 256 | 12.02 | 12308.21 | 24616.43 | 2956.59 | 17.36 |
| 1024 | 1024 | 512 | 12.81 | 13115.63 | 26231.25 | 18108.87 | 20.27 |
| 8192 | 1024 | 1 | 0.10 | 106.97 | 962.69 | 308.72 | 9.05 |
| 8192 | 1024 | 16 | 1.30 | 1335.21 | 12016.88 | 634.57 | 11.21 |
| 8192 | 1024 | 256 | 7.18 | 7350.65 | 66155.86 | 19325.25 | 14.51 |
| 8192 | 1024 | 512 | 7.55 | 7727.70 | 69549.27 | 49756.10 | 14.62 |

### 4.3 Prefill DeepEP EP4

| ISL | OSL | C | Req/s | Out tok/s | Total tok/s | TTFT ms | TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 0.11 | 108.11 | 216.22 | 272.82 | 8.99 |
| 1024 | 1024 | 16 | 1.31 | 1339.67 | 2679.34 | 654.53 | 11.31 |
| 1024 | 1024 | 256 | 13.13 | 13446.49 | 26892.99 | 1798.08 | 16.83 |
| 1024 | 1024 | 512 | 16.33 | 16723.75 | 33447.51 | 9800.63 | 20.06 |
| 8192 | 1024 | 1 | 0.10 | 106.63 | 959.70 | 343.21 | 9.05 |
| 8192 | 1024 | 16 | 1.29 | 1316.47 | 11848.19 | 877.17 | 11.15 |
| 8192 | 1024 | 256 | 6.08 | 6229.43 | 56064.87 | 25827.99 | 14.04 |
| 8192 | 1024 | 512 | 6.11 | 6260.04 | 56340.36 | 65544.71 | 14.04 |

### 4.4 Decode MegaMoE

| ISL | OSL | C | Req/s | Out tok/s | Total tok/s | TTFT ms | TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 0.12 | 121.42 | 242.84 | 370.10 | 7.88 |
| 1024 | 1024 | 16 | 1.55 | 1584.71 | 3169.42 | 592.20 | 9.50 |
| 1024 | 1024 | 256 | 13.82 | 14147.84 | 28295.69 | 3360.49 | 14.30 |
| 1024 | 1024 | 512 | 14.13 | 14466.35 | 28932.71 | 18266.58 | 16.34 |
| 8192 | 1024 | 1 | 0.12 | 121.50 | 1093.47 | 312.88 | 7.93 |
| 8192 | 1024 | 16 | 1.52 | 1552.37 | 13971.33 | 735.34 | 9.41 |
| 8192 | 1024 | 256 | 7.06 | 7233.62 | 65102.56 | 22728.68 | 11.64 |
| 8192 | 1024 | 512 | 7.25 | 7422.27 | 66800.47 | 55416.20 | 11.68 |

### 4.5 Decode Waterfill + MegaMoE

| ISL | OSL | C | Req/s | Out tok/s | Total tok/s | TTFT ms | TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 0.12 | 121.18 | 242.35 | 339.84 | 7.93 |
| 1024 | 1024 | 16 | 1.51 | 1545.24 | 3090.48 | 651.41 | 9.72 |
| 1024 | 1024 | 256 | 12.59 | 12889.40 | 25778.81 | 3499.84 | 15.65 |
| 1024 | 1024 | 512 | 12.51 | 12809.61 | 25619.21 | 21251.47 | 17.81 |
| 8192 | 1024 | 1 | 0.12 | 121.13 | 1090.13 | 285.48 | 7.98 |
| 8192 | 1024 | 16 | 1.48 | 1519.61 | 13676.53 | 739.12 | 9.59 |
| 8192 | 1024 | 256 | 7.14 | 7310.72 | 65796.49 | 21758.37 | 12.23 |
| 8192 | 1024 | 512 | 7.18 | 7347.90 | 66131.14 | 55566.04 | 12.25 |

### 4.6 Decode HiSparse + MegaMoE

| ISL | OSL | C | Req/s | Out tok/s | Total tok/s | TTFT ms | TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 0.12 | 124.26 | 248.52 | 347.49 | 7.71 |
| 1024 | 1024 | 16 | 1.57 | 1611.07 | 3222.15 | 573.57 | 9.37 |
| 1024 | 1024 | 256 | 14.07 | 14406.64 | 28813.29 | 2366.00 | 15.01 |
| 1024 | 1024 | 512 | 13.99 | 14324.96 | 28649.92 | 15719.77 | 19.18 |
| 8192 | 1024 | 1 | 0.12 | 123.89 | 1114.99 | 282.47 | 7.80 |
| 8192 | 1024 | 16 | 1.50 | 1537.94 | 13841.47 | 703.31 | 9.58 |
| 8192 | 1024 | 256 | 6.89 | 7058.77 | 63528.93 | 22083.26 | 13.15 |
| 8192 | 1024 | 512 | 7.28 | 7452.15 | 67069.32 | 53691.60 | 13.12 |

### 4.7 Decode MTP only

| ISL | OSL | C | Req/s | Out tok/s | Total tok/s | TTFT ms | TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 0.23 | 233.36 | 466.73 | 335.21 | 3.96 |
| 1024 | 1024 | 16 | 2.44 | 2503.65 | 5007.31 | 818.52 | 5.48 |
| 1024 | 1024 | 256 | 20.34 | 20826.03 | 41652.06 | 1057.71 | 10.79 |
| 1024 | 1024 | 512 | 27.59 | 28253.34 | 56506.67 | 1657.06 | 15.69 |
| 8192 | 1024 | 1 | 0.23 | 233.64 | 2102.75 | 280.20 | 4.01 |
| 8192 | 1024 | 16 | 2.29 | 2349.31 | 21143.75 | 922.71 | 5.62 |
| 8192 | 1024 | 256 | 7.20 | 7370.75 | 66336.74 | 26160.52 | 7.45 |
| 8192 | 1024 | 512 | 7.24 | 7417.04 | 66753.39 | 59777.21 | 7.43 |

服务日志显示 EAGLE accept length 约 3、accept rate 约 0.67，并持续显示 `cuda graph: True`。因此 `1024/1024/c512` 的 28,253 Out tok/s 是 MTP 输出吞吐；56,507 是输入与输出相加后的 Total tok/s。

### 4.8 Decode FP4 Indexer only

| ISL | OSL | C | Req/s | Out tok/s | Total tok/s | TTFT ms | TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 0.10 | 107.41 | 214.82 | 346.29 | 8.98 |
| 1024 | 1024 | 16 | 1.32 | 1353.72 | 2707.44 | 584.11 | 11.25 |
| 1024 | 1024 | 256 | 12.73 | 13040.45 | 26080.90 | 2127.89 | 16.99 |
| 1024 | 1024 | 512 | 13.75 | 14079.76 | 28159.51 | 14356.11 | 21.10 |
| 8192 | 1024 | 1 | 0.10 | 106.55 | 958.94 | 297.13 | 9.10 |
| 8192 | 1024 | 16 | 1.27 | 1304.51 | 11740.62 | 665.48 | 11.41 |
| 8192 | 1024 | 256 | 6.71 | 6873.48 | 61861.32 | 20947.57 | 15.19 |
| 8192 | 1024 | 512 | 7.15 | 7326.22 | 65935.96 | 53638.28 | 14.37 |

原始 `8192/1024/c256` 和 `c512` 结果因 HTTP 200 空流/部分传输失败被 benchmark 误计成功，已经废弃。修复后 c256 重测为 2560/2560 成功，c512 重测为 5120/5120 成功；两者每请求 `output_lens=1024`、生成文本全部非空、TTFT 全部大于 0，三类服务日志均无 descriptor/session/transfer 错误。因此 FP4 Indexer 现为 8/8 有效、完成。

修复后日志与原始 JSONL：

- `logs/runs/fp4_indexer_retest_8192_c512_stride_and_64k_registration_fix_20260803/prefill.log`
- `logs/runs/fp4_indexer_retest_8192_c512_stride_and_64k_registration_fix_20260803/decode.log`
- `logs/runs/fp4_indexer_retest_8192_c512_stride_and_64k_registration_fix_20260803/router.log`
- `logs/runs/fp4_indexer_retest_8192_c512_stride_and_64k_registration_fix_20260803/benchmark.log`
- `logs/runs/fp4_indexer_retest_8192_c512_stride_and_64k_registration_fix_20260803/isl8192_osl1024_c512_n5120.jsonl`
- `logs/runs/fp4_indexer_retest_8192_c256_fixed_20260803/isl8192_osl1024_c256_n2560.jsonl`

### 4.9 Decode TBO only

| ISL | OSL | C | Req/s | Out tok/s | Total tok/s | TTFT ms | TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 0.10 | 106.30 | 212.61 | 332.67 | 9.09 |
| 1024 | 1024 | 16 | 1.30 | 1334.33 | 2668.66 | 636.82 | 11.37 |
| 1024 | 1024 | 256 | 13.37 | 13695.50 | 27391.01 | 1758.10 | 16.58 |
| 1024 | 1024 | 512 | 16.91 | 17320.93 | 34641.87 | 8107.83 | 20.42 |
| 8192 | 1024 | 1 | 0.10 | 105.79 | 952.13 | 289.92 | 9.18 |
| 8192 | 1024 | 16 | 1.27 | 1299.28 | 11693.49 | 800.10 | 11.29 |
| 8192 | 1024 | 256 | 7.05 | 7223.33 | 65009.98 | 19751.98 | 14.59 |
| 8192 | 1024 | 512 | 7.16 | 7335.12 | 66016.04 | 53259.60 | 14.62 |

TBO 同批次服务日志：

- `logs/runs/tbo_20260802_150500/prefill.log`
- `logs/runs/tbo_20260802_150500/decode.log`
- `logs/runs/tbo_20260802_150500/router.log`
- `logs/runs/tbo_20260802_150500/benchmark.log`

Decode 日志同时确认 CUDA Graph 配置为 `backend=full, max_bs=256`，运行期间为 `cuda graph: True`。

## 5. 暂不计为完成

| 项目 | 状态/原因 |
|---|---|
| SBO | 已创建纯 SBO + CUDA Graph 配置，但尚无完整 8 组结果 |
| 纯 Waterfill | 旧 `results_ag_flash_20260731` 虽有 8 个客户端文件，但耗时/吞吐明显异常，且后续定位到服务兼容问题，不纳入有效完成项 |
| 纯 HiSparse | 当前完整结果实际为 HiSparse+MegaMoE，尚不能作为纯 HiSparse 消融 |
| DSpark | SPS table 已生成；最终 8 组矩阵未完成 |
| LPLB | DeepSeek-V4 不支持 `--ep-dispatch-algorithm lp`，已记录为不支持，不属于性能测试完成 |
| HiCache | 按要求最后执行；需重复前缀专用 workload，尚未完成 |
| 旧 MTP/MegaMoE+MTP | 8 组文件存在，但属于混合配置；已被纯 MTP 结果替代，不纳入单技术报告 |

## 6. 快速结论

- 纯 MTP 在短输入高并发下提升最大：`1024/1024/c512` Out tok/s 从 baseline 13,243.63 提升到 28,253.34（约 2.13×）。
- 纯 TBO 在 `1024/1024/c512` 达到 17,320.93 Out tok/s（约比 baseline 高 30.8%），且已确认使用 CUDA Graph。
- FP4 Indexer 已完成 8/8；修复后的 `8192/c512` 为 7.15 req/s、7,326.22 Out tok/s、65,935.96 Total tok/s，性能与长输入 baseline 同量级。
- Prefill DeepEP EP4 对 `1024/c512` 有提升，但在 8192 长输入高并发下低于 baseline，说明 Prefill/Decode 瓶颈随输入长度发生转移。
- Waterfill+MegaMoE 与 HiSparse+MegaMoE 是组合结果，不能替代对应技术的独立消融。
