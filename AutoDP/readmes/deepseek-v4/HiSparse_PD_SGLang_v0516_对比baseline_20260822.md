# HiSparse PD 分离对比原始 baseline（2026-08-22）

## 1. 实验目标

在原生 `/data/ssd2/sglang_v0.5.16`、DeepSeek-V4-Flash、PD 分离环境中启用 HiSparse，并与原始 baseline 对比。Prefill 使用 GPU-only baseline；Decode 保持 DeepEP `low_latency`、PD Mooncake 和 Decode CUDA Graph，额外启用 HiSparse。

本轮重点验证不同输入长度。原始 baseline 数据来自 `readmes/LPLB_PD_SGLang_v0516_baseline_20260821.md`。

## 2. 实际部署配置

### 2.1 Prefill

```bash
HICACHE_AB_MODE=baseline \\
SGLANG_PREFILL_MEM_FRACTION_STATIC=0.9 \\
SGLANG_SERVICE_LOG_DIR="$PWD/logs/hisparse_pd_20260822/baseline_decode_hisparse/prefill" \\
PYTHONPATH=/data/ssd2/sglang_v0.5.16/python:$PYTHONPATH \\
bash flash_prefill_hicache_ab.sh
```

Prefill 关键参数：`tp=4`、`dp=1`、`ep=1`、`flashinfer_mxfp4`、Mooncake、GPU-only radix cache。实际 full pool 约 `16.32M tokens`。

### 2.2 Decode HiSparse

```bash
unset CUDA_VISIBLE_DEVICES
export PYTHONPATH=/data/ssd2/sglang_v0.5.16/python:$PYTHONPATH
export CUDA_GRAPH_BS_DECODE="1 2 4 8 16 32 64"
export CUDA_GRAPH_MAX_BS=64
export SGLANG_MEM_FRACTION_STATIC=0.8
export SGLANG_CHUNKED_PREFILL_SIZE=64
export SGLANG_HISPARSE_CONFIG='{"top_k":1024,"device_buffer_size":1024,"host_to_device_ratio":8}'
bash flash_decode_hisparse_pd.sh
```

Decode 关键参数：`base-gpu-id=4`、`tp=4`、`dp=4`、`ep=4`、DeepEP `low_latency`、96-SMS dispatch/combine、`--disable-radix-cache`、DSpark、CUDA Graph batch `1/2/4/8/16/32/64`。HiSparse 日志确认 `enable_hisparse=True` 和 `HiSparse c4 host-to-device ratio = 8`。

### 2.3 Router 和 whoami

```bash
PYTHONPATH=/data/ssd2/sglang_v0.5.16/python:$PYTHONPATH \\
python3 -m sglang_router.launch_router --pd-disaggregation \\
  --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 \\
  --host 0.0.0.0 --port 13784 --disable-circuit-breaker \\
  --disable-health-check --health-check-interval-secs 999999

ROUTER_URL=http://127.0.0.1:13784 bash validate_pd_whoami.sh
```

本轮有效服务均通过 `HTTP=200`、`WHOAMI_VALID=True`。Decode 不能同时设置 `CUDA_VISIBLE_DEVICES=4,5,6,7` 和 `--base-gpu-id 4`，否则进程内设备编号不匹配。

## 3. 结果对比

| ISL | OSL | C | 成功请求 | HiSparse Out tok/s | baseline Out tok/s | Out 变化 | HiSparse Total tok/s | baseline Total tok/s | Total 变化 | HiSparse TTFT ms | baseline TTFT ms | HiSparse TPOT ms | baseline TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 10/10 | 97.20 | 107.55 | -9.62% | 194.40 | 215.09 | -9.62% | 241.22 | 332.51 | 10.06 | 8.98 |
| 1024 | 1024 | 16 | 160/160 | 2585.96 | 1348.95 | **+91.70%** | 5171.93 | 2697.90 | **+91.70%** | 586.75 | 631.91 | 5.39 | 11.25 |
| 1024 | 1024 | 256 | 2560/2560 | 5313.31 | 13106.67 | -59.46% | 10626.62 | 26213.34 | -59.46% | 963.78 | 2284.50 | 45.96 | 16.82 |
| 8192 | 1024 | 1 | 10/10 | 181.78 | 106.52 | **+70.65%** | 1635.99 | 958.65 | **+70.65%** | 250.25 | 348.65 | 5.26 | 9.05 |
| 8192 | 1024 | 16 | 160/160 | 2104.92 | 1328.45 | **+58.45%** | 18944.30 | 11956.04 | **+58.45%** | 492.65 | 607.99 | 6.15 | 11.23 |

有效结果目录：

- `logs/hisparse_pd_20260822/baseline_decode_hisparse/results/isl1024_osl1024_c1_n10.log`
- `logs/hisparse_pd_20260822/baseline_decode_hisparse/results/isl1024_osl1024_c16_n160.log`
- `logs/hisparse_pd_20260822/baseline_decode_hisparse/results/isl1024_osl1024_c256_n2560_retry3.log`
- `logs/hisparse_pd_20260822/baseline_decode_hisparse/results/isl8192_osl1024_c1_n10_hisparse.log`
- `logs/hisparse_pd_20260822/baseline_decode_hisparse/results/isl8192_osl1024_c16_n160_hisparse.log`

### 3.1 统一 native SGLang 0.5.16 后的复测

此前 Decode 脚本没有显式设置 `PYTHONPATH`，实际可能加载 `/sgl-workspace/sglang/python`，而 Prefill 使用 `/data/ssd2/sglang_v0.5.16/python`。两棵树的 `schedule_batch.py`、`disaggregation/decode.py` 和 `disaggregation/prefill.py` 哈希不同。已在 `flash_decode_hisparse_pd.sh` 中固定使用 `/data/ssd2/sglang_v0.5.16/python`，并重新通过 whoami 验证。

以下是统一源码后的正式复测结果，baseline 仍取第 3 节原始 baseline：

| ISL | OSL | C | 成功请求 | HiSparse Out tok/s | baseline Out tok/s | Out 变化 | HiSparse Total tok/s | baseline Total tok/s | Total 变化 | HiSparse TTFT ms | baseline TTFT ms | HiSparse TPOT ms | baseline TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 10/10 | 202.58 | 107.55 | **+88.37%** | 405.16 | 215.09 | **+88.37%** | 267.47 | 332.51 | 4.68 | 8.98 |
| 1024 | 1024 | 16 | 160/160 | 2621.92 | 1348.95 | **+94.37%** | 5243.83 | 2697.90 | **+94.37%** | 581.97 | 631.91 | 5.29 | 11.25 |
| 1024 | 1024 | 256 | 2560/2560 | 2866.82 | 13106.67 | -78.14% | 5733.64 | 26213.34 | -78.13% | 801.83 | 2284.50 | 85.34 | 16.82 |
| 8192 | 1024 | 1 | 10/10 | 141.80 | 106.52 | **+33.12%** | 1276.22 | 958.65 | **+33.12%** | 229.85 | 348.65 | 6.83 | 9.05 |
| 8192 | 1024 | 16 | 160/160 | 2375.56 | 1328.45 | **+78.82%** | 21380.06 | 11956.04 | **+78.82%** | 450.67 | 607.99 | 5.97 | 11.23 |

复测结果目录：`logs/hisparse_pd_20260822/native0516_path_c1/results/`。四组低/中并发均通过 whoami 且超过 baseline；1024/C256 虽然请求全部成功，但 CUDA Graph 在高并发 transfer 压力下频繁 fallback，故吞吐显著低于 baseline。

### 3.2 针对 C4 buffer 瓶颈的 ratio=6 复测

保持 native 0.5.16、DeepEP low-latency、Decode CUDA Graph `1/2/4/8/16/32/64`、PD 和请求参数不变，仅将 `host_to_device_ratio` 从 8 调整为 6。这样每 rank 的 C4 GPU pool 从约 `433,856` 增加到约 `565,098` tokens，Host pool 约 `41.65 GB/rank`。

`8192/C256` 完整完成 `2560/2560`，并首次通过高并发容量边界：

| ISL | OSL | C | ratio | HiSparse Out tok/s | baseline Out tok/s | Out 变化 | HiSparse Total tok/s | baseline Total tok/s | Total 变化 | HiSparse TTFT ms | baseline TTFT ms | HiSparse TPOT ms | baseline TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 1024 | 256 | 6 | 7138.53 | 7110.57 | **+0.39%** | 64246.81 | 63995.17 | **+0.39%** | 27943.15 | 55535.20 | 6.80 | 14.42 |

结果目录：`logs/hisparse_pd_20260822/native0516_ratio6_c256/results/`。ratio8 的同一轮在 `alloc_device_buffer` 处失败；ratio6 后请求全部成功，证明主要瓶颈是 C4 device buffer 容量，而不是 PD 传输协议或模型计算错误。

## 4. 失败样例与边界

`8192/C256` 在请求进行到约两千余个请求后触发：

```text
AssertionError: Hisparse allocation failed in alloc_device_buffer
```

该结果不计入性能表。原因是长前缀的压缩 C4 页和每请求 device buffer 同时占用 Decode pool；`mem_fraction_static=0.8` 下容量不足。尝试提高到 `0.9` 时，保留较大 CUDA Graph 会触发 NVSHMEM `cuMemCreate failed`；缩小 Graph 后仍受 GPU/NVSHMEM 资源约束。因此当前配置尚未通过 8192 高并发验收，不能声称 8/8 全部超过 baseline。

统一 native 0.5.16 的 `8192/C256` 复测同样触发 `Hisparse allocation failed in alloc_device_buffer`，随后 Decode scheduler 崩溃并被清理；benchmark 侧出现 `ClientPayloadError: Response payload is not completed`，无有效吞吐数据。失败日志为 `logs/hisparse_pd_20260822/native0516_path_c1/decode/console.log`，该轮不计入性能表。

此前还发现 HiSparse C4 pool 未实现通用 `get_cpu_copy()`。高并发 retraction 会因此触发 `NotImplementedError`；本轮已在 `schedule_batch.py` 中让 HiSparse retraction 跳过不支持的 device-to-CPU KV snapshot，改为释放后重新计算。该修复已通过 1024/C256 的完整 2560/2560 测试。

### 4.1 历史成功配置的重新启动验收

2026-08-22 06:11--06:20 按历史有效 restart7 参数重新启动：`ratio=8`、`top_k=1024`、`device_buffer_size=1024`、`chunked_prefill_size=64`、CUDA Graph `1/2/4/8/16/32/64`。Prefill、Decode、Router 均完成启动和 Graph capture，HiSparse pool 日志确认 `c4=433856`、Host C4 pool 约 `42.64 GB/rank`。

但本轮 `validate_pd_whoami.sh` 未完成，约 120 秒后 Decode 报：

```text
KVTransferError(...): Aborted by AbortReq
```

因此“服务 ready”不等于 PD 请求链路可用；本次没有产生新的性能数据。完整 prefill 日志在 `logs/hisparse_pd_20260822/repro_restart7_c256/prefill/`，whoami 日志在同目录 `router/whoami.log`。该现象属于 Mooncake/RDMA bootstrap 失败，不能归因于 HiSparse C4 pool。

## 5. 结论

HiSparse 已在原生 SGLang 0.5.16 PD 链路中正确生效，并在长上下文低/中并发明显超过原始 baseline：8192/C1、8192/C16 分别提升约 70.65% 和 58.45%。1024/C16 也提升约 91.70%，但 1024/C1 和 1024/C256 低于 baseline，说明 HiSparse 的收益依赖稀疏注意力节省是否能覆盖索引、host/device 管理和调度开销。

当前不能把结果表述为“所有并发均超过 baseline”：ratio8 下的 8192/C256 因 HiSparse buffer 容量失败，ratio6 已使该样例通过并略超 baseline；8192/C512 尚未测试。后续若要覆盖更高并发，应继续优化 HiSparse C4 pool 的容量规划/host swap 路径，或让高并发请求按 DP rank 正确分摊 device buffer；单纯增大 Graph 或 `mem_fraction_static` 会与 NVSHMEM 产生显存冲突。

### 5.1 针对 8192/C256 的 ratio 扫描

为判断失败是否仅由 `host_to_device_ratio` 引起，另外保留了三次独立启动日志：

| ratio | Decode 启动阶段 | whoami/Graph 结果 | 判断 |
|---:|---|---|---|
| 1 | CUDA Graph 完成 | Router→Decode 首个 whoami 长时间不返回 | 未通过服务验收，不计性能 |
| 2 | CUDA Graph 完成 | Router→Decode 首个 whoami 长时间不返回 | 未通过服务验收，不计性能 |
| 4 | target Graph 完成；draft Graph 在 bs=128 停滞约 4 分钟 | GPU 4/7 持续满载，无 traceback，未 ready | 启动边界失败，不计性能 |

日志分别位于 `logs/hisparse_pd_20260822/ratio1_c256/`、`ratio2_c256/` 和 `ratio4_c256/`。这说明降低 ratio 虽然扩大了 C4 host-backed pool（ratio=4 时约 40.08 GB/rank），但会显著增加启动和运行时资源压力，不能简单推断为能解决 8192/C256 的分配失败。当前唯一通过完整 whoami 和请求成功率验收的性能数据仍是第 3 节表格。

## 6. 代码备份

本轮备份目录：`backups/hisparse_pd_20260822/`。

其中包括：

- `schedule_batch.py.before_hisparse_retraction`：源码修复前版本
- `hisparse_allocator.py.after_offload_fix`：HiSparse allocator 相关源码快照
- `flash_decode_hisparse.sh.orig`：原始 HiSparse 启动脚本
- `flash_decode_dspark.sh.baseline`：对照用 DSpark 启动脚本
- `flash_decode_hisparse_pd.sh.before_pythonpath_fix`：显式固定 native 0.5.16 PYTHONPATH 前版本

当前工作树中的 HiSparse 启动脚本为 `flash_decode_hisparse_pd.sh`。
