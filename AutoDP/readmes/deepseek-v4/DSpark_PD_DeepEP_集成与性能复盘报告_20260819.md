# DSpark 集成 PD 分离与 DeepEP 适配报告

> 整理日期：2026-08-19。范围：单机 1P1D，Prefill GPU 0--3，Decode GPU 4--7。
> 只有请求集完整完成且无 IB/GPU 故障的轮次才作为正式性能结果。

## 1. 总结

DSpark 集成到 PD 分离包含三条链路：

1. Decode 加载 DSpark draft 模型，执行 draft/verify；
2. Prefill 计算长上下文，并保存 target hidden states；
3. Prefill 通过 Mooncake/RDMA 将 hidden states 发送到 Decode，Decode 写入 target-hidden KV/cache，再执行 DSpark verify 和 MoE dispatch。

已确认：`1024/1024/C256` 历史完整结果为 `23680.92 output tok/s`、`47361.84 total tok/s`；当前源码 + DeepEP 的一轮完整复现为 `23792.61 output tok/s`、`47585.22 total tok/s`；优化阶段最好稳定 C256 为 `24766.66 output tok/s`，超过 MTP 的 `20826.03 output tok/s`。历史 C512 曾达到 `27889.71 output tok/s`、`55779.41 total tok/s`，但当前未稳定复现。

`8192/1024` 的 C1/C16 可以超过 MTP output throughput，C256/C512 则明显落后。高并发长输入的主要瓶颈是 Prefill 排队、hidden/KV 传输和 TTFT，不是 DSpark 单步 decode 接受率。

## 2. DSpark 如何集成到 PD 分离

### 2.1 服务拓扑

```text
Client -> Router :13784
             |-- Prefill :30000, GPU 0--3, TP4/DP1/EP1
             |       主模型 + hidden source
             |       Mooncake/RDMA hidden transfer
             `-- Decode :30001, GPU 4--7, TP4/DP4/EP4
                     主模型 + DSpark draft
                     DSpark verify + MoE dispatch/combine
```

Prefill、Decode 是两个独立 SGLang server。Router 负责请求路由和 bootstrap room；Mooncake/RDMA 负责 PD 之间的 KV/hidden 传输。DSpark 不改变 Router 协议，主要扩展 Decode speculative worker 和 PD hidden transfer 数据路径。

### 2.2 Decode 侧参数

```bash
--disaggregation-mode decode
--disaggregation-transfer-backend mooncake
--enable-dp-attention --enable-dp-lm-head
--tp-size 4 --dp-size 4 --ep-size 4
--speculative-algorithm DSPARK
--speculative-attention-mode decode
--speculative-draft-model-path /data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash-dspark
--speculative-dspark-sps-table-path logs/flash_decode_dspark/dspark_sps.json
--disable-radix-cache
```

SPS table 描述不同 batch/token 档位的验证预算；`enable-dp-attention` 让四个 DP rank 并行处理请求；当前实验关闭 radix cache，避免 cache 复用影响 A/B。

### 2.3 Prefill hidden source

当前 DSpark target layers 为 `[40, 41, 42]`，hidden size 为 `12288`。典型参数：

```bash
SGLANG_PD_HIDDEN_POOL_TOKENS=131072
SGLANG_DSPARK_PD_HIDDEN_BUFFER_POOL_LIMIT=512
SGLANG_DSPARK_PD_HIDDEN_TRANSFER_QUEUE_LIMIT=64
SGLANG_DSPARK_HIDDEN_THREAD_POOL_SIZE=8
```

8192 输入时，一个请求约占用 8192 hidden rows。早期 1M pool 在高并发时出现 `rows=8192, free_rows=4218, pool_rows=1048576`。因此 C256/C512 必须按每个 DP rank 的实际并发重新估算 pool。2M pool 可以消除 allocation blocking，但会把更多压力推到 RDMA/硬件链路，不能把 pool 不阻塞等同于系统稳定。

### 2.4 首 batch/首请求修复

原实现中 DSpark 首个 Decode batch 可能在 `batch.spec_info` 还是 None 时调用 `prepare_for_decode()`。

已做修复：

- `spec_utils.spec_prepare_for_decode()` 对 DFlash/DSpark family 的 None 状态做处理；
- overlap PD decode 创建 idle `DFlashDraftInputV2`，绑定当前 `req_pool_indices`；
- `overlap_utils` 对 `future_indices=None` 做 None-safe 处理；
- DSpark worker 首 active step 使用 `batch.input_ids` 初始化 draft anchor；
- 当前稳定诊断默认使用 `--disable-overlap-schedule`，隔离 overlap 与 hidden-KV 注入的竞态。

这些修改解决的是启动/首请求功能问题，不等于解决高并发性能问题。

## 3. DSpark 与 DeepEP 的适配

### 3.1 DeepEP 配置

```bash
--moe-a2a-backend deepep
--deepep-mode low_latency
--deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'
```

对照路径为 `--moe-a2a-backend none --moe-runner-backend flashinfer_mxfp4`。DeepEP 可以降低 EP MoE 的通信成本和 TPOT，但要求 activation shape、top-k 元数据、CUDA Graph、异步 stream 和 DSpark hidden-KV 生命周期同时正确。

### 3.2 leading padded shape 修复

DSpark CUDA Graph verify 会产生 leading padded 维度，而 DeepEP low-latency 要求 activation、top-k ids、top-k weights 的 token 首维一致。已在 `DeepEPMoE` low-latency 入口增加可控 flatten/split：

1. 计算 `flat_rows = hidden_states.numel() / hidden_size`；
2. hidden reshape 为 `[flat_rows, hidden_size]`；
3. 对相同 leading shape 的 top-k ids/weights 同步 flatten；
4. 按 `SGLANG_DEEPEP_LL_SPLIT_TOKENS` 分块调用 DeepEP；
5. 拼接 chunk 输出。

当前使用过 `SGLANG_DEEPEP_LL_SPLIT_TOKENS=1024`。该修复解决了 `x.size(0)==topk_idx.size(0)` 和 dispatch capacity 问题。但已经按 expert 分组的 DeepEP staging buffer 不是普通 token 输入，不能简单再次 reshape 后交给 routed-MoE kernel。

### 3.3 DeepEP API 版本适配

当前 DeepEP `low_latency_combine()` 比部分 SGLang overlap 代码旧。已用 `inspect.signature()` 过滤当前 Buffer 实际支持的参数，再调用 combine，避免 graph capture 因旧版本参数不兼容而失败。

`DeepEPMoE._maybe_split_low_latency()` 对 oversized verify rows 分块，避免 `x.size(0) <= num_max_dispatch_tokens_per_rank`。split 会增加 dispatch/combine 次数，必须用完整请求集验证。

### 3.4 DeepEP 与 hidden-KV 注入边界

```text
Prefill hidden source -> Mooncake/RDMA -> Decode hidden receive
-> target hidden-KV write -> DSpark verify -> DeepEP dispatch/combine
```

已确认 DeepEP low-latency + graph16/C16 在 shape 修复后可用；graph64/C256 关闭 PD hidden 注入时曾完整完成，得到 `33230.32 total tok/s`；恢复完整 hidden 注入后，即使增加 full/SWA 边界检查、stream synchronize、关闭 draft graph、关闭 fused CommitKvProj、调整 SWA ratio，仍出现 illegal memory access。

因此当前更像是 DeepEP async staging/dispatch 与 DSpark target-hidden KV 写入生命周期的竞态，而不是单一 index 越界。稳定默认优先使用 built-in TP MoE，DeepEP 作为仍需继续收敛的实验路径。

## 4. 早期完整 DSpark 阶段

| ISL | OSL | Concurrency | Num prompts | Req/s | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 10 | 0.283 | 289.49 | 578.99 | 348.68 ms | 3.11 ms |
| 1024 | 1024 | 16 | 160 | 3.073 | 3147.11 | 6294.22 | 669.35 ms | 4.20 ms |
| 1024 | 1024 | 256 | 2560 | 23.126 | 23680.92 | 47361.84 | 1674.10 ms | 8.73 ms |
| 1024 | 1024 | 512 | 5120 | 25.061 | 25661.95 | 51323.91 | 7509.92 ms | 11.82 ms |
| 8192 | 1024 | 1 | 10 | 0.246 | 251.85 | 2266.68 | 759.12 ms | 3.23 ms |
| 8192 | 1024 | 16 | 160 | 1.701 | 1741.73 | 15675.56 | 1732.91 ms | 7.23 ms |
| 8192 | 1024 | 256 | 2560 | 2.061 | 2110.28 | 18992.51 | 79947.74 ms | 41.77 ms |
| 8192 | 1024 | 512 | 5120 | 2.144 | 2195.60 | 19760.36 | 164131.12 ms | 64.79 ms |

这一阶段完成了 DSpark 首 batch、overlap 首 step、draft anchor、SPS table、CUDA Graph bucket、hidden pool/queue、DeepEP shape flatten、DeepEP combine API 兼容、oversized rows split，以及 DP/TP/EP、HCA、seed 固定。该阶段可以完成完整矩阵，但长输入高并发 TTFT 和 hidden transfer 仍是瓶颈。

## 5. 1024/1024/C512 阶段

| ISL | OSL | Concurrency | Num prompts | Req/s | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 512 | 5120 | — | 27889.71 | 55779.41 | 10866.71 ms | 6.76 ms |

主要修复：扩大 CUDA Graph bucket；增加 Decode receive/Prefill source、buffer、transfer queue；调整 Prefill chunk/max token；固定 worker、Mooncake queue/thread、HCA、seed；修正 DP attention 对 `schedule_conservativeness` 的二次乘 0.3；对比 DeepEP、built-in TP MoE 和官方 none 路径；排查 capture synchronize、Prefill-first、固定 seed、thread pool、NUMA、`inject_reqs=64`。

该历史高分当前重放通常只有约 19k--21k output tok/s，说明性能对 IB/NVLink/worker 状态高度敏感，不能直接当作当前稳定默认值。

## 6. 8192/1024 C1/C16 阶段

| ISL | OSL | Concurrency | Num prompts | Req/s | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 1024 | 1 | 10 | — | 300.41 | 2703.67 | 369.66 ms | 2.97 ms |
| 8192 | 1024 | 16 | 160 | — | 2975.57 | 26780.09 | 1106.26 ms | 4.10 ms |

该阶段按 `8192 rows/request` 重新计算两侧 pool，增大 hidden transfer queue，保留 DeepEP flatten/token split 和 Decode graph，关闭 radix/overlap，校验 DP/HCA/bootstrap 映射及 full/SWA page 布局。C1/C16 完整完成并超过 MTP，说明基本 PD/DeepEP 组合可用；但不能外推到 C256/C512。

## 7. 成功、失败经验

### 成功经验

- 扩大 CUDA Graph bucket 能降低高并发 TPOT；
- hidden thread=8、buffer=512、transfer queue=64、Prefill chunk/max=16384 是当前短输入较好的组合；
- DeepEP token-major flatten 解决 low-latency shape 断言，graph16/C16 可用；
- 固定服务身份、seed、SPS、HCA 映射是公平 A/B 的必要条件；
- 压测前确认 `nvidia_peermem`、IB Active/400G、GPU 显存归零，可以减少环境故障误判。

### 失败经验

- 1M hidden pool + 8192/C256：约 128 请求/rank 时 allocation failed；
- 2M pool + 8192/C256：pool 阻塞消除后，曾在 1703/2560 触发 `cudaErrorNvlinkUncorrectable`，部分结果无效；
- IB `transport retry counter exceeded`、`Sync batch data transfer timeout` 会使 Prefill/Decode 失联；
- DeepEP + 完整 hidden 注入 + graph64/C256 多次 illegal memory access；
- stream synchronize、关闭 draft graph、关闭 fused CommitKvProj、改变 SWA ratio 均未解决竞态；
- 盲目增大 queue/thread/pool 并未稳定提升；NUMA 绑定反而下降；
- 关闭 hidden 注入的 DeepEP 结果不能当完整 PD DSpark 结果；
- 不能只看 TPOT，必须同时看 output、total、TTFT、TPOT 和 completed。

## 8. 为什么高并发长输入超过不了 MTP

MTP 主要在同一个 Decode 服务内完成 draft/verify；DSpark PD 额外经过长 Prefill、hidden source 分配、RDMA transfer、Decode receive 排队、target hidden-KV 写入和 DeepEP dispatch/combine。高并发时任何一个队列或链路不足都会变成 TTFT。

历史完整矩阵中：

- `8192/C1`：DSpark 300.41，MTP 233.64；
- `8192/C16`：DSpark 2975.57，MTP 2349.31；
- `8192/C256`：DSpark 2110.28，MTP 7370.75；
- `8192/C512`：DSpark 2195.60，MTP 7417.04。

TTFT 从 C1 的 369.66 ms 增长到 C256 的 79.95 s、C512 的 164.13 s；因此主瓶颈是 Prefill/hidden transfer/排队，而不是 DSpark 接受率。IB retry、transfer timeout、NVLink/GPU lost 又会进一步放大波动。

## 9. 后续建议

### 9.1 先建立硬件/IB 基线

- 每次确认 `nvidia_peermem` 已加载；
- `ibstat` 检查端口 Active、LinkUp、400G，并记录 HCA 映射；
- 清理 server、router、benchmark 和 orphan CUDA worker；
- 确认 8 张 GPU 显存归零后启动；
- 先跑 1024/C256、1024/C512、8192/C16，再跑 8192/C256；
- 出现 IB retry、timeout、GPU lost 的轮次直接作废。

### 9.2 把 hidden-KV 与 DeepEP 做 request-scoped 同步

建议让 hidden receive 返回 `(request_id, target_layer, buffer, ready_event)`；target-KV writer 消费 ready event 并返回 write-complete event；DeepEP 只消费 write-complete 后的 rows；请求完成或 retraction 后再释放 buffer；CUDA Graph 只捕获地址稳定的 staging buffer；idle DP rank 只接收一致 metadata，不访问无效 hidden rows。

### 9.3 按容量模型扫描

```text
hidden rows >= 每个 DP rank 的并发请求数
               × 每请求 hidden window rows
               × transfer/prealloc 保留系数
```

建议扫描 hidden pool `131072/262144/524288/1048576/2097152`、hidden thread `4/8/16`、transfer queue `32/64/128`、buffer `256/512`、Prefill chunk `16384/24576/32768`、Decode graph max-bs `64/128/256`。每组必须完整完成 C256，并记录 completed、output、total、TTFT、TPOT、free rows、transfer queue 和 IB retry。

## 10. 验收标准

- `completed == Num prompts`；
- 无 memory registration、IB retry、transfer timeout、NVLink/GPU lost、illegal memory access；
- Prefill/Decode/Router 全程存活；
- 记录 DSpark、DeepEP、graph、hidden pool、queue/thread、HCA 和源码 commit；
- 至少重复两轮；
- 同时比较 output tok/s、total tok/s、Mean TTFT、Mean TPOT；
- 高并发长输入至少完成 C256，最好补齐 C512；
- 失败轮次只记录失败原因，不使用 partial result 推导吞吐。

## 11. 相关文件

- 总问题记录：`readmes/SOLUTION.MD`
- 单机历史报告：`readmes/DSpark_1P1D_single_node_report_20260817.md`
- 优化记录：`readmes/DSpark_优化实验_20260817.md`
- Prefill：`flash_prefill_baseline.sh`
- Decode：`flash_decode_dspark.sh`
- Router：`flash_router_baseline.sh`
- SPS table：`logs/flash_decode_dspark/dspark_sps.json`
- 当前长输入复测：`logs/flash_decode_dspark/aug19_8192_pool2m_retry3/`
- 代码备份：`backups/sglang_20260731_064316/`

