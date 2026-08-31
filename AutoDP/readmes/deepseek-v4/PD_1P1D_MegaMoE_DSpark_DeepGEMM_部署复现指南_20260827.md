# 1P1D PD 分离：Prefill MegaMoE + Decode MegaMoE + DSpark + DeepGEMM

更新时间：2026-08-27

本文记录已经验证过的 PD 分离部署方式，作为后续两张 B200、DeepSeek-V4-Pro
1P1D 实验的启动模板。当前已验证模型为 DeepSeek-V4-Flash；V4-Pro 实验必须替换
模型路径，并重新完成启动、`你是谁`、单请求和并发 smoke 验证后，才能开始性能测试。

## 1. 已验证配置

| 项目 | Prefill | Decode |
|---|---|---|
| GPU | 物理 GPU 0--3 | 物理 GPU 4--7 |
| TP/DP/EP | TP4 / DP1 / EP1 | TP4 / DP4 / EP4 |
| A2A backend | MegaMoE | MegaMoE |
| MoE runner | DeepGEMM | DeepGEMM |
| DSpark | 不启用 | 启用 |
| CUDA Graph | Prefill 不启用 | Decode 启用 |
| FP4 indexer | 关闭 | 关闭 |
| Waterfill/LPLB/TBO | 关闭 | 关闭 |
| PD backend | Mooncake | Mooncake |
| Router | 30000 | 30001 |
| 对外入口 | 13784 | 13784 |

当前实验使用的源码：

```text
/data/ssd2/sglang_v0.5.16_pr32281_fix7_repro_20260823/python
```

当前模型：

```text
/data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash
```

DSpark draft 模型：

```text
/data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash-dspark
```

## 2. 目录和日志

推荐每次实验使用独立目录，避免覆盖服务日志：

```bash
cd /data/ssd2/gongoubo/single_node

export RUN_DIR=/data/ssd2/gongoubo/single_node/logs/runs/v4pro_1p1d_$(date -u +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR/services/prefill" "$RUN_DIR/services/decode" \
  "$RUN_DIR/services/router" "$RUN_DIR/validation" "$RUN_DIR/results"
```

服务启动脚本会按时间和 PID 生成独立日志，例如：

```text
$RUN_DIR/services/prefill/prefill_YYYYMMDD_HHMMSS_pid*.log
$RUN_DIR/services/decode/decode_YYYYMMDD_HHMMSS_pid*.log
$RUN_DIR/services/router/router_YYYYMMDD_HHMMSS_pid*.log
```

## 3. Prefill 服务

当前已验证脚本：

```text
dspark_stepwise_ablation_20260824/variants/dspark_final_valid/flash_prefill_megamoe.sh
```

启动命令如下。注意：当前脚本中的主模型路径是硬编码的，`MODEL_PATH` 只是下面命令中
用于说明目标路径的变量，脚本不会自动读取它；正式跑 V4-Pro 前必须复制脚本并将其中
的 `--model-path /data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash` 改为 V4-Pro
路径。还要根据显存容量调整 `SGLANG_PREFILL_MEM_FRACTION_STATIC`、
`SGLANG_MAX_PREFILL_TOKENS` 和 GPU 映射。

```bash
MODEL_PATH=/data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Pro

CUDA_VISIBLE_DEVICES=0,1,2,3 \
SGLANG_SERVICE_LOG_DIR="$RUN_DIR/services/prefill" \
SGLANG_PREFILL_MOE_RUNNER_BACKEND=deep_gemm \
SGLANG_ENABLE_FP4_INDEXER=0 \
SGLANG_DSV4_FIX_TP_ATTN_A2A_SCATTER=0 \
SGLANG_DEFAULT_THINKING=1 \
bash dspark_stepwise_ablation_20260824/variants/dspark_final_valid/flash_prefill_megamoe.sh
```

该脚本实际使用的关键参数：

```text
--model-path $MODEL_PATH
--tp-size 4
--dp-size 1
--ep-size 1
--disaggregation-mode prefill
--disaggregation-transfer-backend mooncake
--disaggregation-ib-device {"0":"mlx5_0","1":"mlx5_1","2":"mlx5_2","3":"mlx5_3"}
--moe-a2a-backend megamoe
--moe-runner-backend deep_gemm
--disable-flashinfer-autotune
--mem-fraction-static 0.9
--max-running-requests 256
--max-prefill-tokens 16384
--chunked-prefill-size 16384
--swa-full-tokens-ratio 0.1
--disable-radix-cache
--disable-overlap-schedule
```

Prefill 侧不需要增加 TBO、Waterfill、LPLB、FP4 indexer 或 DSpark 参数。

## 4. Decode 服务

当前已验证脚本：

```text
dspark_stepwise_ablation_20260824/variants/dspark_final_valid/flash_decode_dspark_megamoe.sh
```

启动命令如下。当前 Decode 脚本也将主模型和 draft 模型路径写在启动命令中，因此
V4-Pro 实验必须复制脚本并把对应的 `--model-path`、`--speculative-draft-model-path`
替换为 V4-Pro 的实际路径；仅设置下面的 shell 变量不会改变脚本内置路径。

```bash
MODEL_PATH=/data/ssd1/checkpoints/DeepSeek-V4-Pro
DRAFT_MODEL_PATH=/data/ssd1/checkpoints/DeepSeek-V4-Pro-dspark

CUDA_VISIBLE_DEVICES=4,5,6,7 \
SGLANG_SERVICE_LOG_DIR="$RUN_DIR/services/decode" \
DS_MOE_A2A_BACKEND=megamoe \
SGLANG_MOE_RUNNER_BACKEND=deep_gemm \
SGLANG_ENABLE_DSPARK=1 \
SGLANG_ENABLE_FP4_INDEXER=0 \
SGLANG_ENABLE_WATERFILL=0 \
SGLANG_ENABLE_LPLB=0 \
SGLANG_ENABLE_TBO=0 \
SGLANG_DSV4_FIX_TP_ATTN_A2A_SCATTER=0 \
SGLANG_RAGGED_VERIFY_MODE=static \
SGLANG_DSPARK_DRAFT_MODEL_PATH="$DRAFT_MODEL_PATH" \
bash dspark_stepwise_ablation_20260824/variants/dspark_final_valid/flash_decode_dspark_megamoe.sh
```

注意：当前脚本中的 draft model 路径是脚本内置值。如果 V4-Pro 的 draft 模型路径
不同，需要直接修改脚本，或者确认脚本已支持该环境变量后再启动；启动日志中的
`--speculative-draft-model-path` 必须核对为 V4-Pro 对应路径。

Decode 实际使用的关键参数：

```text
--model-path $MODEL_PATH
--tp-size 4
--dp-size 4
--ep-size 4
--enable-dp-attention
--enable-dp-lm-head
--moe-a2a-backend megamoe
--moe-runner-backend deep_gemm
--disaggregation-mode decode
--disaggregation-transfer-backend mooncake
--disaggregation-ib-device {"0":"mlx5_4","1":"mlx5_9","2":"mlx5_10","3":"mlx5_11"}
--max-running-requests 2048
--cuda-graph-bs-decode 1 2 4 8 16 32 64 128
--disable-overlap-schedule
--speculative-algorithm DSPARK
--speculative-attention-mode decode
--speculative-dspark-sps-table-path /data/ssd2/gongoubo/single_node/logs/flash_decode_dspark/dspark_sps.json
--disable-radix-cache
```

Decode 必须保留：

```text
DeepGEMM
MegaMoE A2A
DSpark
DeepEP low_latency 相关兼容环境
Decode CUDA Graph
SGLANG_RAGGED_VERIFY_MODE=static
```

当前 DSpark + MegaMoE 路径不应直接打开 TBO、Waterfill 或 LPLB；这些技术需要单独
做 A/B，不能混入 baseline。

## 5. Router

Prefill 和 Decode 都 ready 后，再启动 router：

```bash
SGLANG_SERVICE_LOG_DIR="$RUN_DIR/services/router" \
nohup bash dspark_stepwise_ablation_20260824/scripts/flash_router.sh \
  > "$RUN_DIR/services/router/launcher.out" 2>&1 &
echo $! > "$RUN_DIR/services/router/router.pid"
```

Router 的实际参数：

```text
--pd-disaggregation
--prefill http://127.0.0.1:30000
--decode http://127.0.0.1:30001
--host 0.0.0.0
--port 13784
--disable-circuit-breaker
--disable-health-check
--health-check-interval-secs 999999
```

## 6. 启动顺序和检查

先确认两侧服务 ready：

```bash
curl -f http://127.0.0.1:30000/health
curl -f http://127.0.0.1:30001/health
curl -f http://127.0.0.1:13784/v1/models
```

启动日志必须确认：

```text
Prefill: disaggregation_mode='prefill'
Decode: disaggregation_mode='decode'
Prefill/Decode: moe_a2a_backend='megamoe'
Prefill/Decode: moe_runner_backend='deep_gemm'
Prefill/Decode: enable_deepseek_v4_fp4_indexer=False
Decode: speculative_algorithm='DSPARK'
Decode: cuda graph capture completed
Decode: cuda graph: True
```

如果 V4-Pro 的 HCA 映射不同，必须修改：

```text
Prefill --disaggregation-ib-device
Decode  --disaggregation-ib-device
```

不能只改 `CUDA_VISIBLE_DEVICES` 而保留错误的 HCA 映射；否则可能出现 bootstrap
失败、RDMA 注册失败或 Mooncake transfer timeout。

## 7. 每次重启后的正确性验收

每次服务重启后，性能测试前必须先运行原生 DSV4 格式的“你是谁”验证：

```bash
WHOAMI_RESULT_FILE="$RUN_DIR/validation/whoami.json" \
bash dspark_stepwise_ablation_20260824/validate_pd_whoami.sh
```

必须满足：

```text
HTTP=200
WHOAMI_VALID=True
Prefill 和 Decode 均有请求完成日志
返回内容正常，没有重复输出或异常思维内容
```

该验证脚本会使用 SGLang 原生 DSV4 message encoder 拼接正确 prompt，不使用裸文本
冒充 chat template。当前 tokenizer 没有标准 HF `chat_template`，因此不要自行把
`apply_chat_template` 的结果当作已验证格式。

## 8. 单请求 smoke test

建议先使用一个短请求确认 PD transfer 和 Decode Graph：

```bash
PYTHONPATH=/data/ssd2/sglang_v0.5.16_pr32281_fix7_repro_20260823/python \
python3 -m sglang.bench_serving \
  --backend sglang \
  --base-url http://127.0.0.1:13784 \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 256 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --request-rate inf \
  --seed 1 \
  --disable-tqdm \
  --pd-separated \
  --output-file "$RUN_DIR/results/smoke.json"
```

确认 `Successful requests=1` 后，再进行正式 benchmark。

## 9. 性能测试模板

例如测试 `ISL=8192、OSL=1024、Concurrency=512`：

```bash
PYTHONPATH=/data/ssd2/sglang_v0.5.16_pr32281_fix7_repro_20260823/python \
python3 -m sglang.bench_serving \
  --backend sglang \
  --base-url http://127.0.0.1:13784 \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --num-prompts 5120 \
  --max-concurrency 512 \
  --request-rate inf \
  --seed 1 \
  --disable-tqdm \
  --pd-separated \
  --output-file "$RUN_DIR/results/isl8192_osl1024_c512.json"
```

正式结果至少记录：

```text
Completed / Num prompts
Req/s
Out tok/s
Total tok/s
Mean TTFT
Mean TPOT
```

## 10. 常见错误和处理

### 10.1 Prefill 退出或 Decode bootstrap 失败

优先检查：

1. Prefill 和 Decode 是否使用同一模型版本。
2. 两侧 `--disaggregation-ib-device` 是否与实际 HCA 对应。
3. `nvidia_peermem`、IB 链路和 RDMA 注册是否正常。
4. 是否存在旧 router 仍指向已经退出的服务。
5. Prefill/Decode 是否都使用了相同的源码目录和 checkpoint。

### 10.2 FP4 indexer 报错

本 baseline 明确关闭 FP4 indexer。若后续单独测试 FP4，必须 Prefill/Decode 两端
同时增加 `--enable-deepseek-v4-fp4-indexer`，不能只在 Decode 开启。

### 10.3 TBO、Waterfill、LPLB 结果混入 baseline

本配置的 baseline 必须满足：

```text
SGLANG_ENABLE_TBO=0
SGLANG_ENABLE_WATERFILL=0
SGLANG_ENABLE_LPLB=0
SGLANG_ENABLE_FP4_INDEXER=0
```

每次只能增加一项技术，并重新做“你是谁”和请求完整性验证。

### 10.4 CUDA Graph 未命中

Decode 必须检查：

```text
--cuda-graph-bs-decode 1 2 4 8 16 32 64 128
SGLANG_RAGGED_VERIFY_MODE=static
--disable-overlap-schedule
```

如果并发 batch 超出 Graph bucket，可能回退 eager；如果修改 graph bucket，必须重新
完成 warmup 和 smoke，不要只根据启动成功判断 Graph 已生效。

## 11. V4-Pro 两张 B200 迁移清单

1. 替换 Prefill 和 Decode 的 `--model-path`。
2. 替换 DSpark draft model 和 SPS table，确认 draft 模型与 V4-Pro 主模型匹配。
3. 确认两张 B200 是同机两张卡，还是两个节点；两种拓扑的 TP/DP/EP 和 Mooncake
   HCA 配置不同，不能直接复用本文件的 GPU/HCA 映射。
4. 如果是两张卡组成 1P1D，通常需要重新评估 TP、EP、DP 的可行组合，不能直接使用
   当前验证的 TP4/DP4/EP4。
5. 重新确认模型 context length、KV dtype、显存比例和 Graph bucket。
6. 重新运行“你是谁”、单请求 smoke 和短并发 smoke。
7. 只有在两侧请求全部成功且 Decode 日志确认 `cuda graph: True` 后，才开始正式测试。

## 12. 本次已验证记录

当前 Flash 版本 baseline 恢复记录：

```text
logs/runs/sglang_profile_20260826/restored_base_final/
```

其中已确认：

```text
Prefill health=200
Decode health=200
Router /v1/models=200
WHOAMI_VALID=True
Prefill FP4=False
Decode FP4=False
Prefill/Decode moe_runner_backend=deep_gemm
Decode moe_a2a_backend=megamoe
```

相关启动脚本：

```text
dspark_stepwise_ablation_20260824/variants/dspark_final_valid/flash_prefill_megamoe.sh
dspark_stepwise_ablation_20260824/variants/dspark_final_valid/flash_decode_dspark_megamoe.sh
dspark_stepwise_ablation_20260824/scripts/flash_router.sh
```

# Deepseek-v4-pro

decode机器：xxxx 密码：xxxx

权重路径：/data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Pro-dspark

prefill机器：xxxx 密码：xxxx

权重路径：/data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Pro 

容器都是sgl0516，prefill和decode部署都使用8卡以及IB

先跑prefill不使用megamoe，decode不使用megamoe、dspark、deepgemm作为baseline。

然后再测试prefill+megamoe+decode+megamoe+dspark+deepgemm
