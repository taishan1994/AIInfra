# HiCache / HiSparse PD 分离可复用测试方案

> 初版日期：2026-08-22
>
> 目标：为 HiCache 与后续 HiSparse 建立一套不依赖随机 prompt、能够验证缓存命中和性能收益的统一测试方法。

## 1. 为什么原始随机 benchmark 不适用

普通 serving benchmark 每个请求都使用独立随机 token。请求之间没有稳定的 token 前缀，HiCache 没有可复用的 KV，因此测到的主要是：

- 首次 Prefill 计算；
- Decode 生成吞吐；
- PD hidden/KV transfer；
- 缓存管理额外开销。

它无法回答 HiCache 的核心问题：历史 KV 是否被保存、是否在 GPU 淘汰后从 CPU/外部存储恢复，以及恢复是否降低 TTFT。

HiCache 的正确测试必须让相同 session 的后续请求携带完全相同的历史 token，并控制 round barrier、cache flush、工作集大小和请求顺序。

## 2. 测试对象和公平 A/B

### 2.1 Baseline

Baseline 不是原来用于随机请求的 `--disable-radix-cache` 配置。对于 HiCache A/B，baseline 应该是：

```text
相同模型、TP/DP/EP、PD 拓扑、DeepEP/DSpark、CUDA Graph、SWA 参数
开启普通 GPU radix cache
关闭 hierarchical cache
```

原因是 HiCache 是 GPU radix cache 的层级扩展；若 baseline 也关闭 radix cache，会把“是否有缓存”与“HiCache 是否有效”混为一项。

### 2.2 HiCache

在 baseline 上只增加：

```text
--enable-hierarchical-cache
--hicache-ratio <固定值>
--hicache-write-policy write_through
--hicache-io-backend kernel
--hicache-mem-layout page_first
```

第一阶段只测 GPU+CPU HiCache，不接外部存储；第二阶段再单独测 file/Mooncake/3FS 等 storage backend，避免把 CPU cache 和远端存储收益混在一起。

### 2.3 HiSparse 复用方式

后续 HiSparse 只替换服务开关和必要 attention backend，保持以下 workload 完全不变：

- 相同 deterministic token workload；
- 相同 client、round、prefix/question/output 长度；
- 相同 seed 和 round 顺序；
- 相同 cold/warm/eviction 阶段；
- 相同 completed、cached_tokens、TTFT、TPOT 统计。

## 3. 确定性 workload 设计

测试工具：

```text
benchmark_hicache_replay.py
```

它不从随机文本生成 prompt，而是根据 tokenizer 的 vocabulary 构造稳定 token-id 序列。每个 client 的初始 session prefix 唯一但可重复；每轮 question 由 `(client, round, seed)` 唯一确定。

每个 client 的请求历史为：

```text
session_prefix
+ question_0
+ model_output_0
+ question_1
+ model_output_1
+ ...
```

所有 client 在同一轮完成后才进入下一轮。这样：

- Round 0 测冷启动；
- Round 1 及以后测同一 session 的 prefix reuse；
- 增加 client 数和 prefix 长度可以制造 GPU cache eviction；
- round 顺序、token 内容和 reuse distance 每次完全一致。

工具同时记录每个请求的 `cached_tokens`、`prompt_len`、TTFT、latency、output length 和错误信息，并生成逐请求 JSONL 与汇总 JSON。

## 4. 分层测试矩阵

### 阶段 A：功能与命中正确性

```bash
PYTHONPATH=/data/ssd2/sglang_v0.5.16/python:$PYTHONPATH \
python3 benchmark_hicache_replay.py \
  --base-url http://127.0.0.1:13784 \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --tokenizer /data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash \
  --clients 2 --rounds 3 \
  --prefix-len 4096 --question-len 128 --output-len 16 \
  --max-concurrency 2 --clear-storage \
  --output results/hicache_smoke.jsonl \
  --summary results/hicache_smoke.summary.json
```

验收：

- `completed == clients * rounds`；
- Round 0 的 `cached_tokens` 应接近 0；
- 后续 round 的 `cached_tokens` 应随历史 prefix 增长；
- 不允许出现请求失败、PD transfer error 或服务退出。

### 阶段 B：热缓存收益

目标是测“缓存已经存在且 GPU 仍可容纳”时的收益，主要比较缓存命中带来的 TTFT 降低：

```text
clients=8/16
rounds=5
prefix_len=8192
question_len=256
output_len=32
max_concurrency=8/16
```

Round 0 单独报告，不与 warm round 平均混合。重点指标：Round 1--4 的 cached tokens、cache hit rate、Mean/P95 TTFT。

### 阶段 C：GPU eviction / HiCache 收益

这是 HiCache 的核心测试。增加 session 工作集直到：

```text
clients * prefix_len > GPU KV cache capacity
```

推荐逐步扫描：

```text
clients=16, 32, 64, 128, 256
prefix_len=8192 或 16384
rounds=3
```

在每个 round 使用固定 client permutation，使部分旧 session 被访问、部分 session 暂不访问。不能每轮 `/flush_cache`，因为那会同时清除 GPU radix tree，无法测出 HiCache 的层级恢复能力。若要验证 L3 storage，再单独调用 `/hicache/storage-backend/clear` 做 cold run。

对每个工作集记录：

- GPU-only baseline 的 cache hit rate；
- HiCache 的 cache hit rate；
- HiCache host restore/load 的日志或 metrics；
- TTFT、P95 TTFT；
- 吞吐和显存/host memory 占用。

### 阶段 D：长上下文 PD 压力

在 C 阶段确认 HiCache 确实发生 host restore 后，再测试 PD：

```text
prefix_len=8192/32768/65536
clients=16/64/128
rounds=3
output_len=32 或 128
```

这一阶段同时记录 Prefill、Decode、Router 日志。HiCache 的收益应主要体现为后续轮次 TTFT 和 Prefill 重算下降，而不是把 Decode speculative throughput 与缓存收益混为一个数字。

## 5. Cold / warm / eviction 操作定义

| 阶段 | GPU radix cache | HiCache host/storage | 目的 |
|---|---|---|---|
| cold | 清空 | `--clear-storage` 时清空 | 测首次计算成本 |
| warm | 不清空 | 不清空 | 测 GPU/host 命中 |
| eviction | 不主动清空 | 不清空 | 测 GPU 淘汰后层级恢复 |
| storage-cold | 清空 | `/hicache/storage-backend/clear` | 测外部存储首次写入/读取 |

`/flush_cache` 只能在阶段边界使用。它不是 warm-cache 测试步骤；否则会把要观察的 GPU/host 状态一起清掉。

## 6. 指标与判定标准

每个 workload 至少输出：

```text
completed / total requests
cached_tokens / prompt_tokens
cache_hit_rate
Mean/P50/P95 TTFT
Mean/P95 TPOT
request throughput
output tok/s
total tok/s
```

HiCache 是否有效必须同时满足：

1. 后续轮次出现可重复的 cached tokens；
2. eviction workload 中 HiCache 命中率高于 GPU-only baseline；
3. 命中轮次 TTFT 明显下降，或在相同 TTFT 下支持更高并发；
4. completed 等于计划请求数；
5. 日志无 cache restore failure、PD transfer timeout、CUDA illegal access 或服务退出。

只看到吞吐上升但没有 cache hit/restore 证据，不能归因于 HiCache；只看到 cached tokens 增加但 TTFT 变差，则应记录为缓存生效但当前 IO/调度开销抵消收益。

## 7. 当前源码约束

当前 SGLang v0.5.16 实现支持：

- `--enable-hierarchical-cache`；
- host pool：`--hicache-ratio` 或 `--hicache-size`；
- `write_back`、`write_through`、`write_through_selective`；
- `kernel` / `direct` IO；
- `page_first`、`page_first_direct` 等布局；
- file、Mooncake、hf3fs、NIXL 等 storage backend；
- `/flush_cache`；
- `/hicache/storage-backend/clear`。

对于 PD/DSpark：

- 两侧必须使用相同 workload 和明确的 cache 边界；
- Decode radix cache 与 speculative decoding 的组合参数需要逐项确认；
- 不能把原来 `--disable-radix-cache` 的随机 DSpark baseline 直接作为 HiCache cache-hit baseline；
- HiCache host restore、Mooncake PD hidden transfer、DSpark draft/verify 三条链路必须分别看日志。

## 8. 日志与结果目录约定

每次运行使用独立目录：

```text
logs/hicache/<experiment>/<run-id>/prefill.log
logs/hicache/<experiment>/<run-id>/decode.log
logs/hicache/<experiment>/<run-id>/router.log
logs/hicache/<experiment>/<run-id>/replay.jsonl
logs/hicache/<experiment>/<run-id>/summary.json
```

每次服务重启后先执行：

```bash
bash validate_pd_whoami.sh
```

并把验证输出写入同一 run 目录。脚本、源码和服务参数也要随 run 保存，避免把不同 cache 配置的结果混在一起。

## 9. 推荐执行顺序

1. 用阶段 A 验证 deterministic replay、`cached_tokens` 和服务功能；
2. 以 baseline 运行阶段 B/C，确认 GPU radix cache 的命中行为；
3. 重启 HiCache，运行完全相同的阶段 B/C；
4. 只有确认 host eviction/restore 后，才运行阶段 D；
5. 对 HiSparse 复用阶段 A--D，仅替换 `--enable-hisparse` 和 attention backend，并保持 workload/统计脚本不变。

## 10. 首次 PD A/B 实测结果（2026-08-22）

本次先完成了同一台机器上的最小确定性 PD A/B，避免将随机请求吞吐误当成 HiCache 收益。两组均使用：原始 SGLang v0.5.16、Prefill TP4、`flashinfer_mxfp4`、Decode DP4/EP4、DeepEP `low_latency`、Decode CUDA Graph batch 1/2/4/8/16/32/64/128、DSpark、相同 Router 和相同 seed `20260822`。唯一 cache 差异是：baseline 只启用 GPU UnifiedRadixCache；HiCache 额外启用 `--enable-hierarchical-cache --hicache-ratio 0.01 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first`。

工作负载为 2 个固定 session、3 轮 barrier replay，每个 session 4096 token 稳定前缀，每轮追加 128 token 问题和上一轮真实输出，输出上限 16 token。两组均 6/6 请求成功，cache 行为完全一致：冷轮 cached tokens=0，第二轮命中 8192/8736，第三轮命中 8704/9024。

| 配置 | 总耗时 s | 总 cache hit rate | Round 0 Mean TTFT ms | Round 1 Mean TTFT ms | Round 2 Mean TTFT ms |
|---|---:|---:|---:|---:|---:|
| GPU radix baseline | 2.8792 | 64.47% | 1488.77 | 450.04 | 457.58 |
| HiCache ratio=0.01（最新重跑） | 1.9653 | 64.47% | 334.62 | 472.14 | 464.98 |
| HiCache 相对 baseline（单次观测） | -31.74% | 0 | -77.53% | +4.91% | +1.62% |

结论：HiCache 在本次小规模 workload 中已确认真实生效，且没有请求丢失；但由于 4096 token 前缀能够完全留在 GPU radix cache，后续轮次没有触发 host eviction/restore，因此不能据此证明 HiCache 的 host-tier 加速。最新重跑的冷轮 TTFT 明显低于 baseline，但命中轮略高；由于每个配置当前只有一次测量，冷轮差异暂不归因于 HiCache。当前结论应限定为“功能生效、GPU-only 与 HiCache 都能正确命中，尚未完成 host-tier 优势验证”。下一步必须用更大 session 数/更长前缀制造 GPU eviction，再比较 host restore 的命中率和 TTFT。

本轮证据文件：

- `logs/hicache/hicache_smoke_20260822/replay_baseline/baseline.summary.json`
- `logs/hicache/hicache_smoke_20260822/replay_hicache/hicache.summary.json`
- `logs/hicache/hicache_smoke_20260822/router_baseline_whoami.log`
- `logs/hicache/hicache_smoke_20260822/router_hicache_whoami.log`
- `logs/hicache/hicache_smoke_20260822/prefill_baseline_normal2/`
- `logs/hicache/hicache_smoke_20260822/prefill_hicache_normal/`

## 11. 三层部署命令与参数归档

以下命令均基于 `/data/ssd2/sglang_v0.5.16/python` 的原始 SGLang 0.5.16 工作树。Prefill 与 Decode 日志必须分别保存；每次重启后先执行 `bash validate_pd_whoami.sh`，验证通过后才允许开始 benchmark。

### 11.1 GPU-only baseline Prefill

```bash
HICACHE_AB_MODE=baseline \
SGLANG_SERVICE_LOG_DIR="$PWD/logs/hicache/<run-id>/prefill" \
SGLANG_PREFILL_MEM_FRACTION_STATIC=0.35 \
PYTHONPATH=/data/ssd2/sglang_v0.5.16/python:$PYTHONPATH \
bash flash_prefill_hicache_ab.sh
```

等价的关键启动参数为：

```text
--model-path /data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash
--port 30000 --tp-size 4 --dp-size 1 --ep-size 1
--disaggregation-mode prefill --disaggregation-transfer-backend mooncake
--disaggregation-ib-device {"0":"mlx5_0","1":"mlx5_1","2":"mlx5_2","3":"mlx5_3"}
--moe-runner-backend flashinfer_mxfp4 --disable-flashinfer-autotune
--mem-fraction-static 0.35 --max-running-requests 256
--max-prefill-tokens 16384 --chunked-prefill-size 16384
--swa-full-tokens-ratio 0.1 --disable-overlap-schedule
```

Baseline 不带 `--enable-hierarchical-cache`，但保留普通 GPU radix cache。

### 11.2 GPU+Host HiCache

```bash
HICACHE_AB_MODE=hicache \
SGLANG_HICACHE_RATIO=0.01 \
SGLANG_SERVICE_LOG_DIR="$PWD/logs/hicache/<run-id>/prefill" \
SGLANG_PREFILL_MEM_FRACTION_STATIC=0.35 \
PYTHONPATH=/data/ssd2/sglang_v0.5.16/python:$PYTHONPATH \
bash flash_prefill_hicache_ab.sh
```

在 baseline 参数上增加：

```text
--enable-hierarchical-cache
--hicache-ratio 0.01
--hicache-write-policy write_through
--hicache-io-backend kernel
--hicache-mem-layout page_first
```

`hicache-ratio` 是 host pool 相对 device pool 的比例。DeepSeek-V4 当前不能用 `--hicache-size`，必须使用 ratio。

### 11.3 GPU+Host+Storage（三层）

本地 file backend 是本机可复现的第三层，不依赖额外 Mooncake metadata 服务：

```bash
RUN_DIR="$PWD/logs/hicache/<run-id>"
HICACHE_AB_MODE=hicache_file \
SGLANG_HICACHE_RATIO=0.01 \
SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="$RUN_DIR/file_storage" \
SGLANG_HICACHE_STORAGE_PREFETCH_POLICY=wait_complete \
SGLANG_HICACHE_STORAGE_EXTRA_CONFIG='{"prefetch_threshold":256,"max_size":"200G","eviction_ratio":0.9}' \
SGLANG_SERVICE_LOG_DIR="$RUN_DIR/prefill" \
SGLANG_PREFILL_MEM_FRACTION_STATIC=0.35 \
PYTHONPATH=/data/ssd2/sglang_v0.5.16/python:$PYTHONPATH \
bash flash_prefill_hicache_ab.sh
```

新增参数为：

```text
--hicache-storage-backend file
--hicache-storage-prefetch-policy wait_complete
--hicache-storage-backend-extra-config {"prefetch_threshold":256,"max_size":"200G","eviction_ratio":0.9}
```

这里的三层分别是 GPU radix/KV、CPU host HiCache、file storage。`/hicache/storage-backend/clear` 只清理第三层；`/flush_cache` 用于阶段边界清理服务内 cache，不能在 warm/eviction 阶段每轮调用。

### 11.4 当前 Decode 与 Router

Decode 继续使用已验证的 DSpark/DeepEP low-latency/CUDA Graph 配置，不能为了 HiCache A/B 改动：

```text
# Decode 必须保持全 GPU 可见，并用 base-gpu-id 4 选择物理 GPU 4--7；
# 不要设置 CUDA_VISIBLE_DEVICES=4,5,6,7，否则进程内设备只剩 0--3，base-gpu-id 4 会失败。
source: /data/ssd2/sglang_v0.5.16/python
--port 30001 --base-gpu-id 4 --tp-size 4 --dp-size 4 --ep-size 4
--enable-dp-attention --enable-dp-lm-head
--disaggregation-mode decode --disaggregation-transfer-backend mooncake
--disaggregation-ib-device {"4":"mlx5_4","5":"mlx5_9","6":"mlx5_10","7":"mlx5_11"}
--moe-a2a-backend deepep --moe-runner-backend auto --deepep-mode low_latency
--deepep-config {"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}
--cuda-graph-bs-decode 1 2 4 8 16 32 64 128
--speculative-algorithm DSPARK --speculative-attention-mode decode
--disable-radix-cache
```

Router：

```bash
PYTHONPATH=/data/ssd2/sglang_v0.5.16/python:$PYTHONPATH \
python3 -m sglang_router.launch_router --pd-disaggregation \
  --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 \
  --host 0.0.0.0 --port 13784 --disable-circuit-breaker \
  --disable-health-check --health-check-interval-secs 999999
```

## 12. 不同长度的三层验证协议

每种 cache 配置都必须使用完全相同的 seed、client 数和顺序，至少覆盖以下长度：

| profile | prefix_len | question_len | output_len | clients | rounds |
|---|---:|---:|---:|---:|---:|
| short | 4096 | 128 | 16 | 8 | 3 |
| medium | 8192 | 256 | 32 | 16 | 3 |
| long | 32768 | 256 | 32 | 16 | 3 |
| eviction | 65536 | 256 | 32 | 32+ | 3 |

短、中、长三组用于确认长度变化下的正常 cache 命中；eviction 组必须使 `clients * prefix_len` 超过 Prefill 日志中的 `DSV4 pool sizes: full=...`，本轮通过统一降低 `--mem-fraction-static` 到 0.35 控制实验成本，并以启动日志中的实际 pool size 为准调整 client 数。

每个 profile 依次运行 GPU-only、GPU+Host、GPU+Host+Storage。首次轮次后不再 `/flush_cache`；Storage 组还要保留 file storage 目录大小、`.bin` 文件数量和 `HiCacheFile` 的 reserve/evict 日志。只有出现“GPU cache 已淘汰、Host restore/prefetch 成功”，或 storage backend 的明确 get/put/clear 证据，才算验证到对应层。仅有 `cached_tokens` 相同不能证明三层都被使用。

统一 benchmark 模板：

```bash
PYTHONPATH=/data/ssd2/sglang_v0.5.16/python:$PYTHONPATH \
python3 benchmark_hicache_replay.py \
  --base-url http://127.0.0.1:13784 \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --tokenizer /data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash \
  --clients <clients> --rounds 3 --prefix-len <prefix_len> \
  --question-len <question_len> --output-len <output_len> \
  --max-concurrency <max_concurrency> --seed 20260822 \
  --output logs/hicache/<run-id>/replay.jsonl \
  --summary logs/hicache/<run-id>/summary.json
```

File Storage 组通过 Router 发业务请求，但 Router 不转发管理接口；清理第三层必须直接访问 Prefill：

```bash
curl -fsS -X POST http://127.0.0.1:30000/hicache/storage-backend/clear
python3 benchmark_hicache_replay.py ...
```

也可以使用脚本参数自动走 Prefill 清理：

```bash
python3 benchmark_hicache_replay.py ... \
  --clear-storage --storage-url http://127.0.0.1:30000
```

每组必须记录 total requests、completed、cached/prompt tokens、Mean/P95 TTFT、Mean/P95 TPOT、吞吐、Prefill/Decode/Router 日志和 whoami 验证。三层结果未完成前，不得把当前第 10 节的小规模 GPU 命中结果表述为 HiCache 加速。

## 13. 低 pool eviction 实验记录（2026-08-22）

为实际制造 GPU eviction，本轮首先将三层 A/B 的 Prefill `--mem-fraction-static` 统一设置为 `0.35`。Baseline 服务启动日志确认：

```text
DSV4 pool sizes: full=2782720, swa=278272, c4=695680, c128=21740, c4_state=17392
```

因此 eviction 工作集不能再按“几百个 8K session”盲跑，而应以 `2,782,720` 个 full-cache token 为阈值；例如 16 个 32K session 只有约 524K token，仍不足以淘汰 full pool，至少应使用 64 个 64K session（约 4.2M token），并在资源允许时再提高到 128 个 session。short/medium/long 仍需分别运行，不能只测 eviction 长样例。

本轮重启和验证证据：

- Baseline Prefill：`logs/hicache/three_tier_20260822/baseline/prefill/prefill_20260822_022045_pid1802663.log`；已记录 ready 和上述 pool size。
- Decode：`logs/hicache/three_tier_20260822/decode/`；原始 DSpark、DeepEP low_latency、CUDA Graph 1/2/4/8/16/32/64/128/256 参数已重新启动并 ready。
- 第一次 whoami 失败原因：旧 Decode 在 Prefill 重启后仍使用失效 bootstrap room，Prefill 记录 `KVTransferError ... Aborted by AbortReq`；随后 Decode 已重启。
- 第二次 whoami/短 benchmark：Prefill 和 Router 看到请求返回 200，但客户端流未在预期时间结束，`summary.json` 尚未生成，因此本轮结果判定为“服务链路未通过可复现验收”，不纳入性能表。

该现象说明：降低 Prefill pool 后，PD bootstrap 重注册和长流结束必须独立验收，不能只看 HTTP 200。后续正式三层测试的硬门槛为：whoami 必须拿到非空语义结果；每个 benchmark 必须生成 summary；`completed == requests`；否则先修复 bootstrap/stream 生命周期，再继续不同长度和三层对比。

## 14. 三层不同长度实测结果（2026-08-22）

本轮使用原生 `/data/ssd2/sglang_v0.5.16/python`、Decode DSpark + DeepEP `low_latency` + CUDA Graph 1/2/4/8/16/32/64/128、PD Mooncake；每次 Prefill 切换均单独保存日志，并在业务测试前执行 `whoami`。三种配置均通过 HTTP 200、语义校验，且所有 benchmark 请求成功。

| 层级 | prefix | clients×rounds | completed | 冷轮 Mean TTFT ms | 第2轮 cached/prompt | 第3轮 cached/prompt | 结果目录 |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline GPU | 4096 | 8×3 | 24/24 | 1690.14 | 32768/34944 | 34816/36096 | `pd_restart6/replay/baseline_short` |
| Host HiCache | 4096 | 8×3 | 24/24 | 2144.38 | 32768/34944 | 34816/36096 | `pd_restart7/replay/hicache_short` |
| File Storage | 4096 | 8×3 | 24/24 | 5881.72 | 32768/34944 | 34816/36096 | `pd_restart10/replay/file_short_retry` |
| baseline GPU | 8192 | 16×3 | 48/48 | 2967.58 | 135168/139520 | 139264/143872 | `pd_restart8/replay/baseline_medium` |
| Host HiCache | 8192 | 16×3 | 48/48 | 1756.49 | 135168/139520 | 139264/143872 | `pd_restart7/replay/hicache_medium` |
| File Storage | 8192 | 16×3 | 48/48 | 1833.63 | 135168/139520 | 139264/143872 | `pd_restart10/replay/file_medium_retry` |
| baseline GPU | 32768 | 16×3 | 48/48 | 5264.55 | 528384/532736 | 532480/537088 | `pd_restart8/replay/baseline_long` |
| Host HiCache | 32768 | 16×3 | 48/48 | 5564.46 | 528384/532736 | 532480/537088 | `pd_restart7/replay/hicache_long` |
| File Storage | 32768 | 16×3 | 48/48 | 5528.56 | 528384/532736 | 532480/537088 | `pd_restart10/replay/file_long_retry` |

结论边界：三种配置在这三种长度下的 `cached_tokens` 完全一致，说明工作集仍停留在 GPU radix cache，不能把这些结果解释为 Host/File restore 的加速。File Storage 已确认实际生效：Prefill 日志出现 `HiCacheFile storage` 清理成功，目录产生多个 `.bin` 文件；但本轮没有 GPU eviction，因此尚不能从 TTFT 证明第三层收益。File Storage 短轮冷启动 TTFT 较高，符合文件层初始化/写入额外代价，不能与 Host 优势混为一谈。

本轮关键日志：

- Decode：`logs/hicache/three_tier_20260822/pd_restart6/decode/launcher.log`。
- Baseline Prefill：`logs/hicache/three_tier_20260822/pd_restart8/prefill/`。
- Host Prefill：`logs/hicache/three_tier_20260822/pd_restart7/prefill/`，含 `hierarchical=True`、host pool 分配和 `UnifiedRadixCache` 证据。
- File Prefill：`logs/hicache/three_tier_20260822/pd_restart10/prefill/`，含 `HiCacheFile storage backend cleared successfully`、`.bin` 文件落盘证据。
- whoami：各 `pd_restart6/7/10/router/whoami.log`。

下一步的真正优势验证仍需 eviction 工作集：以 pool size 为阈值，至少运行 `64×65536` 的独立 session 或降低 GPU pool 后重新按正确顺序启动 PD。只有日志同时出现 GPU eviction 与 Host restore/prefetch，或 File backend 的明确 get/put 命中，才能声称对应层带来收益。

## 15. 低 pool 三层 eviction A/B 结果与部署复现（2026-08-22）

为验证 Host/File 层是否真正解决 GPU cache 淘汰，本轮将 Prefill 的 GPU cache pool 降到约 2.8M full tokens，并使用完全相同的 `64` 个独立前缀、每个前缀 `65536` tokens、连续 `3` 轮 replay。单轮工作集约 4.19M tokens，超过 GPU pool；不在轮次之间执行 `/flush_cache`。三组均使用原生 SGLang 0.5.16、相同 Decode、相同 Router、相同 seed 和相同请求顺序。

### 15.1 复现实验命令

Baseline：

```bash
RUN_DIR="$PWD/logs/hicache/three_tier_20260822/<run-id>"
HICACHE_AB_MODE=baseline \
SGLANG_PREFILL_MEM_FRACTION_STATIC=0.35 \
SGLANG_SERVICE_LOG_DIR="$RUN_DIR/prefill" \
PYTHONPATH=/data/ssd2/sglang_v0.5.16/python:$PYTHONPATH \
bash flash_prefill_hicache_ab.sh
```

Host HiCache：

```bash
HICACHE_AB_MODE=hicache \
SGLANG_HICACHE_RATIO=2.0 \
SGLANG_PREFILL_MEM_FRACTION_STATIC=0.35 \
SGLANG_SERVICE_LOG_DIR="$RUN_DIR/prefill" \
PYTHONPATH=/data/ssd2/sglang_v0.5.16/python:$PYTHONPATH \
bash flash_prefill_hicache_ab.sh
```

本组把 Host ratio 设为 `2.0`，使 Host pool 约为 GPU pool 的两倍，足以容纳本次 4.19M token 工作集；此前 `0.01` 只适合功能验证，不能用于验证 Host eviction 优势。

File Storage：

```bash
HICACHE_AB_MODE=hicache_file \
SGLANG_HICACHE_RATIO=0.01 \
SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="$RUN_DIR/file_storage" \
SGLANG_PREFILL_MEM_FRACTION_STATIC=0.35 \
SGLANG_SERVICE_LOG_DIR="$RUN_DIR/prefill" \
PYTHONPATH=/data/ssd2/sglang_v0.5.16/python:$PYTHONPATH \
bash flash_prefill_hicache_ab.sh
curl -fsS -X POST http://127.0.0.1:30000/hicache/storage-backend/clear
```

三组均使用以下业务回放命令；File 组的清理必须直接访问 Prefill，Router 不转发管理接口：

```bash
python3 benchmark_hicache_replay.py \
  --base-url http://127.0.0.1:13784 \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --clients 64 --rounds 3 --prefix-len 65536 \
  --question-len 256 --output-len 16 --max-concurrency 64 \
  --seed 20260822 \
  --output "$RUN_DIR/replay/<case>/records.jsonl" \
  --summary "$RUN_DIR/replay/<case>/summary.json"
```

Decode 保持全 GPU 可见，使用 `--base-gpu-id 4` 选择 GPU 4--7；配置不变：DeepEP `low_latency`、CUDA Graph batch `1 2 4 8 16 32 64 128`、DSpark、PD Mooncake。每次 Prefill 重启后先重启 Decode/Router，再执行 `validate_pd_whoami.sh`。

### 15.2 真实淘汰结果

| 配置 | 实际 GPU full pool | 请求 | Round 0 cached/prompt | Round 1 cached/prompt | Round 2 cached/prompt | Round TTFT ms | 证据目录 |
|---|---:|---:|---:|---:|---:|---|---|
| GPU baseline | 2,782,720 | 192/192 | 0/4,210,688 | 0/4,228,096 | 0/4,245,504 | 41302.18 / 40556.49 / 40405.44 | `pd_restart11/replay/baseline_eviction` |
| GPU+Host HiCache, ratio=2.0 | 2,791,424 | 192/192 | 0/4,210,688 | 592,128/4,228,096 | 4,227,072/4,245,504 | 43014.68 / 37397.70 / 2578.62 | `pd_restart12/replay/hicache_eviction` |
| GPU+Host+File, ratio=0.01 | 2,791,424 | 192/192 | 0/4,210,688 | 65,792/4,228,096 | 0/4,245,504 | 42616.00 / 40807.48 / 41780.12 | `pd_restart13/replay/file_eviction` |

Baseline 的三轮 `cached_tokens=0`，证明本工作集确实持续超过 GPU pool；Host 组第三轮恢复 `4,227,072` tokens，命中率约 `99.57%`，Mean TTFT 降至 `2.58s`，这是本轮唯一明确证明 Host tier 带来 revisit 加速的结果。Host 第一轮略慢属于冷启动/写入代价，不能只比较第一轮。

File 组所有 `192/192` 请求均成功，Prefill 日志确认创建了 `HiCacheFile` 并成功 clear；本轮目录产生 `914` 个文件、约 `924,756,864` bytes，证明第三层确实有数据落盘。但 cached tokens 只在第二轮短暂出现 `65,792`，第三轮又回到 `0`，TTFT 也没有改善。因此当前可以确认“File backend 已启用且发生写盘”，不能确认“File tier 已成功恢复完整工作集并带来性能优势”。可能原因包括：`ratio=0.01` 的 Host 中间层过小、文件层淘汰/异步写入与本次高并发随机前缀访问不匹配，以及当前 replay 没有形成可复用的文件层命中路径；需要后续用更低 Host ratio、固定可回放前缀和明确 storage get/restore 计数继续拆分验证。

### 15.3 结论边界

这组实验修正了此前“Baseline 和 HiCache 的 cached tokens 一样，因此 HiCache 没有优势”的误判：此前工作集没有超过 GPU pool，只能验证功能；本轮超过 GPU pool 后，Host tier 的恢复优势已经出现。与此同时，不能把 Host 的收益外推到 File Storage。当前三层结论是：GPU baseline 可复现；Host HiCache 在真实淘汰后有效；File Storage 已落盘但尚未证明有效恢复。

本轮证据：

- Baseline：`logs/hicache/three_tier_20260822/pd_restart11/`
- Host HiCache：`logs/hicache/three_tier_20260822/pd_restart12/`
- File Storage：`logs/hicache/three_tier_20260822/pd_restart13/`
- File Prefill 明确日志：`pd_restart13/prefill/launcher.log` 中的 `Creating storage backend 'file'`、`HiCacheFile` 和 `Cleared all entries`。
