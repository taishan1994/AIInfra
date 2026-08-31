# Prefill MegaMoE + Decode MegaMoE/DSpark/全技术组合实验（2026-08-24）

## 1. 目标

验证如下完整组合是否能在当前源码、PD 分离和 Decode CUDA Graph 下正常工作：

- Prefill：MegaMoE
- Decode：MegaMoE A2A、DeepGEMM、DSpark
- Decode 额外技术：Waterfill、LPLB、TBO、DeepSeek V4 FP4 indexer
- Decode Graph：full，`1 2 4 8 16 32 64 128`

## 2. 实际部署参数

源码为 `/data/ssd2/sglang_v0.5.16_pr32281_fix7_repro_20260823`，模型为 `DeepSeek-V4-Flash`。

Prefill 使用 GPU 0–3，TP4/DP1/EP1：

```text
--moe-a2a-backend megamoe
--moe-runner-backend auto
--disaggregation-mode prefill
--disaggregation-transfer-backend mooncake
--disaggregation-ib-device '{"0":"mlx5_0","1":"mlx5_1","2":"mlx5_2","3":"mlx5_3"}'
--mem-fraction-static 0.9
--max-prefill-tokens 16384
--chunked-prefill-size 16384
```

Decode 使用 GPU 4–7，TP4/DP4/EP4、DP Attention、DP LM Head：

```text
--moe-a2a-backend megamoe
--moe-runner-backend deep_gemm
--speculative-algorithm DSPARK
--speculative-attention-mode decode
--speculative-draft-model-path /data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash-dspark
--speculative-dspark-sps-table-path /data/ssd2/gongoubo/single_node/logs/flash_decode_dspark/dspark_sps.json
--enable-waterfill
--ep-dispatch-algorithm lp
--enable-two-batch-overlap
--enable-deepseek-v4-fp4-indexer
--cuda-graph-bs-decode 1 2 4 8 16 32 64 128
--mem-fraction-static 0.8
--chunked-prefill-size 1024
--disable-overlap-schedule
```

其他关键环境变量：

```text
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024
SGLANG_DEEPEP_LL_SPLIT_TOKENS=1024
SGLANG_RAGGED_VERIFY_MODE=static
SGLANG_DSPARK_PD_HIDDEN_POOL_TOKENS=65536
SGLANG_PD_HIDDEN_RECV_POOL_TOKENS=131072
SGLANG_DSPARK_PD_HIDDEN_BUFFER_POOL_LIMIT=512
SGLANG_DSPARK_PD_HIDDEN_TRANSFER_QUEUE_LIMIT=64
SGLANG_DSPARK_PD_HIDDEN_TRANSFER_QUEUE_BYTES=2147483648
```

脚本：[`flash_prefill_megamoe.sh`](../dspark_alltech_megamoe_20260824/scripts/flash_prefill_megamoe.sh)、[`flash_decode_dspark_megamoe.sh`](../dspark_alltech_megamoe_20260824/scripts/flash_decode_dspark_megamoe.sh)、[`flash_router.sh`](../dspark_alltech_megamoe_20260824/scripts/flash_router.sh)。

## 3. 启动验证

服务层面启动成功：

- Prefill health：200
- Decode health：200
- `server_args` 明确记录：`speculative_algorithm='DSPARK'`
- `moe_a2a_backend='megamoe'`
- `moe_runner_backend='deep_gemm'`
- `enable_waterfill=True`
- `ep_dispatch_algorithm='lp'`
- `enable_two_batch_overlap=True`
- `enable_deepseek_v4_fp4_indexer=True`
- target/draft CUDA Graph 均完成捕获

但启动成功不代表请求路径正确。

## 4. 请求正确性结果

`你是谁` 请求没有在 120 秒内返回，验证日志为：

```text
TimeoutError: timed out
```

因此没有得到 `WHOAMI_VALID=True`。随后 10 请求 Smoke 也未形成有效的成功结果文件，不能进入正式性能矩阵。

Decode 日志在请求阶段进入错误响应路径，Prefill 日志虽然出现 HTTP 200，但 PD hidden/KV 请求没有形成完整的端到端返回。该组合因此判定为“服务可启动，但请求功能验证失败”。

## 5. 与已成功组合的边界

此前已完成并成功的组合是：

- Prefill MegaMoE
- Decode DSpark + MegaMoE
- 不开启 Waterfill/LPLB/TBO/FP4 indexer/DeepGEMM 全组合

结果位于 [`repro_dspark_megamoe_20260824`](../repro_dspark_megamoe_20260824)，8 组矩阵全部成功。

本轮新增的失败组合比成功组合多同时打开了 Waterfill、LPLB、TBO、FP4 indexer，并把 Decode runner 固定为 DeepGEMM；因此当前证据只能说明“全技术叠加路径存在组合兼容性问题”，不能把问题归因到某一个开关。下一步应按以下顺序做最小增量定位：

1. DSpark + MegaMoE + DeepGEMM；
2. 再增加 FP4 indexer；
3. 再增加 Waterfill；
4. 再增加 LPLB；
5. 最后增加 TBO。

每一步都必须先通过 `你是谁` 和 10/10 请求，再进入性能矩阵。

## 6. 日志与备份

- 实验目录：[`dspark_alltech_megamoe_20260824`](../dspark_alltech_megamoe_20260824)
- Decode 日志：`dspark_alltech_megamoe_20260824/logs/services/decode/`
- Prefill 日志：`dspark_alltech_megamoe_20260824/logs/services/prefill/`
- Router 日志：`dspark_alltech_megamoe_20260824/logs/services/router/`
- Whoami 失败日志：`dspark_alltech_megamoe_20260824/logs/validation/whoami.log`
- Smoke 日志：`dspark_alltech_megamoe_20260824/logs/results/smoke.log`

