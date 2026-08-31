# mtp
    ISL     OSL    并发    请求数    Req/s    Out tok/s    Total tok/s      Mean TTFT    Mean TPOT
  ━━━━━━  ━━━━━━  ━━━━━━  ━━━━━━━━  ━━━━━━━  ━━━━━━━━━━━  ━━━━━━━━━━━━━  ━━━━━━━━━━━━━  ━━━━━━━━━━━
   1024    1024       1        10     0.23       233.36         466.73      335.21 ms      3.96 ms
  ──────  ──────  ──────  ────────  ───────  ───────────  ─────────────  ─────────────  ───────────
   1024    1024      16       160     2.44      2503.65        5007.31      818.52 ms      5.48 ms
  ──────  ──────  ──────  ────────  ───────  ───────────  ─────────────  ─────────────  ───────────
   1024    1024     256      2560    20.34     20826.03       41652.06     1057.71 ms     10.79 ms
  ──────  ──────  ──────  ────────  ───────  ───────────  ─────────────  ─────────────  ───────────
   1024    1024     512      5120    27.59     28253.34       56506.67     1657.06 ms     15.69 ms
  ──────  ──────  ──────  ────────  ───────  ───────────  ─────────────  ─────────────  ───────────
   8192    1024       1        10     0.23       233.64        2102.75      280.20 ms      4.01 ms
  ──────  ──────  ──────  ────────  ───────  ───────────  ─────────────  ─────────────  ───────────
   8192    1024      16       160     2.29      2349.31       21143.75      922.71 ms      5.62 ms
  ──────  ──────  ──────  ────────  ───────  ───────────  ─────────────  ─────────────  ───────────
   8192    1024     256      2560     7.20      7370.75       66336.74    26160.52 ms      7.45 ms
  ──────  ──────  ──────  ────────  ───────  ───────────  ─────────────  ─────────────  ───────────
   8192    1024     512      5120     7.24      7417.04       66753.39    59777.21 ms      7.43 ms

  其中：

  Total tok/s = Input tok/s + Out tok/s

  这组实验使用的核心配置是：

  TP=4
  DP=4
  EP=4
  DeepEP low_latency
  EAGLE
  speculative-num-steps=3
  speculative-eagle-topk=1
  speculative-num-draft-tokens=4
  Decode CUDA Graph 最大 batch=128
  无 MegaMoE
  无 DSpark draft

  纯 MTP Decode 实际命令：

  CUDA_VISIBLE_DEVICES=4,5,6,7 \
  PYTHONUNBUFFERED=1 \
  SGLANG_DEFAULT_THINKING=1 \
  SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
  SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
  SGLANG_DSV4_REASONING_EFFORT=max \
  SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1 \
  SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN=1 \
  SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW=1 \
  SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1 \
  SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0 \
  SGLANG_RADIX_FORCE_MISS=1 \
  SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024 \
  python3 -m sglang.launch_server \
    --model-path /data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash \
    --served-model-name deepseek-ai/DeepSeek-V4-Flash \
    --trust-remote-code \
    --tool-call-parser deepseekv4 \
    --host 0.0.0.0 \
    --port 30001 \
    --tp-size 4 \
    --dp-size 4 \
    --ep-size 4 \
    --enable-dp-attention \
    --disable-flashinfer-autotune \
    --mem-fraction-static 0.78 \
    --swa-full-tokens-ratio 0.1 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-ib-device '{"0":"mlx5_4","1":"mlx5_9","2":"mlx5_10","3":"mlx5_11"}' \
    --moe-a2a-backend deepep \
    --max-running-requests 1024 \
    --deepep-mode low_latency \
    --deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}' \
    --cuda-graph-max-bs-decode 128 \
    --speculative-algo EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --disable-radix-cache

  Prefill 端使用同样的 TP4/DP1/EP1 单机 Prefill 配置；MTP 对照时应额外关闭 DSpark hidden pool：

  SGLANG_DISABLE_DSPARK_PD_HIDDEN=1

  该组原始结果目录：

  logs/flash_decode_mtp_only/results_20260802/

  对应脚本：

  flash_decode_mtp_only.sh


# dspark

ISL	OSL	Concurrency	Num prompts	Req/s	Out tok/s	Total tok/s	Mean TTFT ms	Mean TPOT ms
1024	1024	1	10	0.283	289.49	578.99	348.68 ms	3.11 ms
1024	1024	16	160	3.073	3147.11	6294.22	669.35 ms	4.20 ms
1024	1024	256	2560	23.126	23680.92	47361.84	1674.10 ms	8.73 ms
1024	1024	512	5120	25.061	25661.95	51323.91	7509.92 ms	11.82 ms
8192	1024	1	10	0.246	251.85	2266.68	759.12 ms	3.23 ms
8192	1024	16	160	1.701	1741.73	15675.56	1732.91 ms	7.23 ms
8192	1024	256	2560	2.061	2110.28	18992.51	79947.74 ms	41.77 ms
8192	1024	512	5120	2.144	2195.6	19760.36	164131.12 ms	64.79 ms

  logs/flash_decode_dspark/results/dspark_isl*_osl*_20260805.jsonl

  当时主要做了以下修复。

  1. 修复 Prefill 长输入高并发卡死

  原问题：

  prefill.py::_write_dspark_hidden_rows_for_batch
  torch.cuda.synchronize()

  会同步整个设备，等待 Mooncake/DeepEP 的无关 stream，导致 8192 输入高并发时卡死。

  修复：

  - 备份：backups/sglang_prefill_before_forward_stream_sync_20260805.py
  - 只同步 scheduler 的 forward_stream
  - 不再执行全设备 torch.cuda.synchronize()

  这使得 8192/C256 从卡在约 512 或 2429 个请求，恢复为 2560/2560 完成。

  2. 修复 DSpark hidden tail 的高开销合并

  原问题：

  torch.cat([...])

  每个请求都复制大规模 [batch, 1024, 12288] hidden tensor，导致 c256/c512 长尾和 TPOT 飙升。

  修复：

  - spec_info.py 不再构造整批 hidden tensor
  - hidden tail 保留在 request 上
  - 在 dspark_worker_v2.py forward 前按请求、分块注入
  - 去除 CPU→GPU→CPU 往返
  - 备份：
      - backups/sglang_spec_info_before_lazy_tail_20260805.py
      - backups/sglang_dspark_worker_before_lazy_tail_20260805.py
      - backups/sglang_spec_info_before_cpu_roundtrip_fix_20260805.py
      - backups/sglang_dspark_worker_before_cpu_roundtrip_fix_20260805.py

  3. 扩展 Decode CUDA Graph

  最终使用：

  --cuda-graph-bs-decode 1 2 4 8 16 32 64 128 256

  并确认日志持续出现：

  cuda graph: True

  4. 使用 DeepEP low-latency

  --moe-a2a-backend deepep
  --deepep-mode low_latency
  --deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'

  同时设置：

  SGLANG_DEEPEP_LL_SPLIT_TOKENS=1024
  SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024

  5. 修复 hidden buffer 连续运行耗尽

  之前设置了：

  SGLANG_DSPARK_PD_HIDDEN_DISABLE_REUSE=1

  导致注册页无法复用，连续高并发时耗尽。

  最终改为：

  SGLANG_DSPARK_PD_HIDDEN_DISABLE_REUSE=0

  并使用：

  SGLANG_DSPARK_PD_HIDDEN_BUFFER_POOL_LIMIT=512
  SGLANG_DSPARK_PD_HIDDEN_TRANSFER_QUEUE_LIMIT=64
  SGLANG_DSPARK_PD_HIDDEN_TRANSFER_QUEUE_BYTES=2147483648

  6. 关闭调试输出

  SGLANG_DSPARK_DEBUG_MAIN_OUTPUT=0

  此前开启 debug 输出会显著拖慢 c16 及以上并发。

  Prefill 实际启动命令：

  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  PYTHONUNBUFFERED=1 \
  SGLANG_DEFAULT_THINKING=1 \
  SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
  SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
  SGLANG_DSV4_REASONING_EFFORT=max \
  SGLANG_PD_HIDDEN_POOL_TOKENS=524288 \
  SGLANG_DSPARK_PD_HIDDEN_BUFFER_POOL_LIMIT=512 \
  SGLANG_DSPARK_PD_HIDDEN_TRANSFER_QUEUE_LIMIT=64 \
  SGLANG_DSPARK_PD_HIDDEN_TRANSFER_QUEUE_BYTES=2147483648 \
  SGLANG_RADIX_FORCE_MISS=1 \
  SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN=1 \
  SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW=1 \
  SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1 \
  SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0 \
  python3 -m sglang.launch_server \
    --model-path /data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash \
    --served-model-name deepseek-ai/DeepSeek-V4-Flash \
    --trust-remote-code \
    --tool-call-parser deepseekv4 \
    --host 0.0.0.0 \
    --port 30000 \
    --tp-size 4 \
    --dp-size 1 \
    --ep-size 1 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-ib-device '{"0":"mlx5_0","1":"mlx5_1","2":"mlx5_2","3":"mlx5_3"}' \
    --moe-runner-backend flashinfer_mxfp4 \
    --disable-flashinfer-autotune \
    --mem-fraction-static 0.9 \
    --swa-full-tokens-ratio 0.1 \
    --disable-radix-cache \
    --disable-overlap-schedule

  Decode 实际启动命令：

  SGLANG_DEEPEP_LL_SPLIT_TOKENS=1024 \
  SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024 \
  SGLANG_DSPARK_PD_HIDDEN_INJECT_REQS=64 \
  SGLANG_PD_HIDDEN_RECV_POOL_TOKENS=65536 \
  SGLANG_DSPARK_PD_HIDDEN_POOL_TOKENS=65536 \
  SGLANG_DSPARK_PD_HIDDEN_BUFFER_POOL_LIMIT=512 \
  SGLANG_DSPARK_PD_HIDDEN_TRANSFER_QUEUE_LIMIT=64 \
  SGLANG_DSPARK_PD_HIDDEN_TRANSFER_QUEUE_BYTES=2147483648 \
  SGLANG_DSPARK_PD_HIDDEN_DISABLE_REUSE=0 \
  SGLANG_DSPARK_DEBUG_MAIN_OUTPUT=0 \
  SGLANG_DSPARK_SYNC_PD_HIDDEN_INJECT=0 \
  SGLANG_DSPARK_DISABLE_DRAFT_CUDA_GRAPH=0 \
  SGLANG_RAGGED_VERIFY_MODE=static \
  SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
  SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
  SGLANG_RADIX_FORCE_MISS=1 \
  python3 -m sglang.launch_server \
    --model-path /data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash \
    --served-model-name deepseek-ai/DeepSeek-V4-Flash \
    --trust-remote-code \
    --tool-call-parser deepseekv4 \
    --host 0.0.0.0 \
    --port 30001 \
    --base-gpu-id 4 \
    --skip-server-warmup \
    --tp-size 4 \
    --dp-size 4 \
    --ep-size 4 \
    --enable-dp-attention \
    --enable-dp-lm-head \
    --disable-flashinfer-autotune \
    --mem-fraction-static 0.8 \
    --swa-full-tokens-ratio 0.1 \
    --chunked-prefill-size 4096 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-ib-device '{"4":"mlx5_4","5":"mlx5_9","6":"mlx5_10","7":"mlx5_11"}' \
    --moe-a2a-backend deepep \
    --deepep-mode low_latency \
    --deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}' \
    --max-running-requests 1024 \
    --cuda-graph-bs-decode 1 2 4 8 16 32 64 128 256 \
    --disable-overlap-schedule \
    --speculative-algorithm DSPARK \
    --speculative-attention-mode decode \
    --speculative-draft-model-path /data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash-dspark \
    --speculative-dspark-sps-table-path /data/ssd2/gongoubo/single_node/logs/flash_decode_dspark/dspark_sps.json \
    --disable-radix-cache

  Router：

  python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://127.0.0.1:30000 \
    --decode http://127.0.0.1:30001 \
    --host 0.0.0.0 \
    --port 13784 \
    --disable-circuit-breaker \
    --disable-health-check \
    --health-check-interval-secs 999999

  压测命令：

  python3 -m sglang.benchmark.serving \
    --backend sglang \
    --base-url http://127.0.0.1:13784 \
    --dataset-name random \
    --tokenizer /data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash \
    --random-input-len "$ISL" \
    --random-output-len "$OSL" \
    --random-range-ratio 1 \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$CONCURRENCY" \
    --request-rate inf \
    --tokenize-prompt \
    --disable-tqdm \
    --seed 5 \
    --output-file "$RESULT_JSON"

  Router、Prefill、Decode 日志分别保存在：

  logs/flash_decode_dspark/router_*.log
  logs/flash_decode_dspark/prefill_*.log
  logs/flash_decode_dspark/decode_*.log

  这组结果和修复说明已经整理在：

  readmes/DSpark_1P1D_single_node_report_20260817.md

  以及：

  readmes/SOLUTION.MD的 6.34–6.46 节。
