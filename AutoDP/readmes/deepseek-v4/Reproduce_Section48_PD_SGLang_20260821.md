# 第 48 节 PD 分离结果复现记录（2026-08-21）

## 1. 已验证结果

配置：原生 SGLang v0.5.16、FlashInfer MxFP4、DeepEP `low_latency`、Waterfill、decode full CUDA Graph、PD 分离。

| ISL | OSL | Concurrency | Requests | Output tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 512 | 5120/5120 | 16400.43 | **32800.85** | 5438.08 ms | 23.98 ms |

历史第 48 节 Total tok/s 为 31076.22。本次达到历史值的 105.55%。输入和输出 token 均为 5242880。

结果文件：

`logs/flash_decode_waterfill/results_native_flashinfer_20260821/section48_historical_mxfp4_runtime128/isl1024_osl1024_c512_n5120.jsonl`

## 2. 源码快照

完整源码快照：

`backups/sglang_section48_20260821/sglang_v0.5.16_section48_runtime128.tar.gz`

SHA256：

`8a3faaa490f9573c56e5c67b3af8e3d55c91f43816444313ae4df5748c204d37`

对应 SGLang 基线 commit：`fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1`。

恢复方式（会覆盖目标目录，执行前请确认目标路径）：

```bash
mv /data/ssd2/sglang_v0.5.16 /data/ssd2/sglang_v0.5.16.before_restore
tar -xzf /data/ssd2/gongoubo/single_node/backups/sglang_section48_20260821/sglang_v0.5.16_section48_runtime128.tar.gz \
  -C /data/ssd2
sha256sum -c /data/ssd2/gongoubo/single_node/backups/sglang_section48_20260821/SHA256SUMS
```

源码中的关键修改文件包括：

`deepep.py`、`waterfill.py`、`fp8.py`、`mxfp4_flashinfer_trtllm_moe.py`、`mxfp4_flashinfer.py`、`load_inquirer.py`、`metrics_reporter.py`、`full_cuda_graph_backend.py`。

## 3. 服务启动

工作目录：

```bash
cd /data/ssd2/gongoubo/single_node
```

### Prefill

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
SGLANG_SERVICE_LOG_DIR=/data/ssd2/gongoubo/single_node/logs/services/prefill \
bash flash_prefill_baseline.sh
```

实际关键参数：

```text
port=30000
tp_size=4, dp_size=1, ep_size=1
disaggregation_mode=prefill
disaggregation_transfer_backend=mooncake
IB={"0":"mlx5_0","1":"mlx5_1","2":"mlx5_2","3":"mlx5_3"}
moe_runner_backend=flashinfer_mxfp4
disable_flashinfer_autotune=true
mem_fraction_static=0.9
max_running_requests=256
max_prefill_tokens=16384
chunked_prefill_size=16384
disable_overlap_schedule=true
disable_radix_cache=true
```

### Decode

```bash
SGLANG_SERVICE_LOG_DIR=/data/ssd2/gongoubo/single_node/logs/services/decode \
bash flash_decode_waterfill.sh
```

实际关键参数：

```text
port=30001, base_gpu_id=4
tp_size=4, dp_size=4, ep_size=4
enable_dp_attention=true, enable_dp_lm_head=true
disaggregation_mode=decode
disaggregation_transfer_backend=mooncake
IB={"4":"mlx5_4","5":"mlx5_9","6":"mlx5_10","7":"mlx5_11"}
moe_a2a_backend=deepep
moe_runner_backend=flashinfer_mxfp4
max_running_requests=1024
deepep_mode=low_latency
deepep_dispatcher_output_dtype=bf16
deepep_config={"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}
enable_waterfill=true
disable_custom_all_reduce=true
cuda_graph_backend_decode=full
cuda_graph_bs_decode=1 2 4 8 16 32 64 128 256 512
disable_radix_cache=true
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512
SGLANG_DEEPEP_LL_SPLIT_TOKENS=0
SGLANG_DEEPEP_MXFP8_DISPATCH=0
SGLANG_FLASHINFER_NATIVE_EXPECTED_M=1
SGLANG_MXFP4_NATIVE_TUNE_ACTUAL_M=auto
```

重要：不要设置 `SGLANG_POST_CAPTURE_MAX_RUNNING_REQUESTS=64`。历史正式 C512 的实际 decode batch 为每个 DP 128；该临时 cap 会把 batch 限制为 64 并降低吞吐。当前运行由内存池自动得到 `max_running_requests=256`，每个 DP 实际 batch=128。

### Router

```bash
bash flash_router_baseline.sh
```

Router 参数：`--pd-disaggregation`、prefill `127.0.0.1:30000`、decode `127.0.0.1:30001`、监听 `0.0.0.0:13784`。

## 4. 启动后校验

```bash
bash validate_pd_whoami.sh
```

应看到 `HTTP=200` 和 `WHOAMI_VALID=True`。随后确认 decode 日志包含：

```text
cuda graph: True
#running-req: 128
```

服务日志分别保存在：

```text
logs/services/prefill/
logs/services/decode/
```

## 5. C512 复测命令

```bash
RESULT_DIR=logs/flash_decode_waterfill/results_native_flashinfer_20260821/reproduce_section48 \
BASE_URL=http://127.0.0.1:13784 \
bash -c '
mkdir -p "$RESULT_DIR"
python3 -m sglang.benchmark.serving \
  --backend sglang --base-url "$BASE_URL" --dataset-name random \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --tokenizer /data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash \
  --random-input-len 1024 --random-output-len 1024 \
  --random-range-ratio 1 --num-prompts 5120 --max-concurrency 512 \
  --request-rate inf --tokenize-prompt --disable-tqdm --seed 1 \
  --output-file "$RESULT_DIR/isl1024_osl1024_c512_n5120.jsonl" \
  2>&1 | tee "$RESULT_DIR/c512.log"
'
```

## 6. 日志和备份脚本

本次使用的脚本副本也保存在：

`backups/sglang_section48_20260821/`

包括 `flash_prefill_baseline.sh`、`flash_decode_waterfill.sh` 和 `validate_pd_whoami.sh`。服务启动日志必须使用按时间和 PID 单独命名的文件，不要覆盖历史日志。
