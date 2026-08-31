# 原始 baseline、纯 DeepGEMM 与纯 flashinfer_mxfp4 的 PD 分离 A/B

日期：2026-08-24

## 1. 实验目的

本轮严格按照“原始 baseline 不叠加其它优化技巧”的要求，比较 Decode MoE runner：

1. 原始 baseline：Decode `--moe-runner-backend` 使用默认 `auto`。
2. 纯 DeepGEMM：Decode 显式使用 `--moe-runner-backend deep_gemm`。
3. 纯 flashinfer_mxfp4：Decode 显式使用 `--moe-runner-backend flashinfer_mxfp4`。

三套配置均使用同一套 PD 分离拓扑、模型、请求矩阵、随机种子和服务参数。为了隔离 Decode runner，Prefill 三套均固定为共同的 `flashinfer_mxfp4` Prefill 配置；因此本报告结论是 Decode runner A/B，不是 Prefill runner A/B。

## 2. 共同配置与边界

- 源码：`/data/ssd2/sglang_v0.5.16_pr32281_fix7_repro_20260823`
- Git HEAD：`fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1`
- 模型：`deepseek-ai/DeepSeek-V4-Flash`
- PD：Prefill GPU 0–3，Decode GPU 4–7，Mooncake transfer，Router `13784`
- Decode：TP4/DP4/EP4，DeepEP `low_latency`，`normal_dispatch/normal_combine num_sms=96`
- Decode CUDA Graph：开启 `full`，batch `1 2 4 8 16 32 64 128`
- 输入/输出：1024 或 8192 / 1024，`random_range_ratio=1`，seed=1，tokenize-prompt，warmup=1
- 每组请求数：`10 × concurrency`
- 三套均关闭：Waterfill、LPLB、DSpark、TBO、HiSparse、FP4 indexer、speculative decoding、DP LM head、radix cache。
- MxFP4 额外保留其 runner 必需的 `--deepep-dispatcher-output-dtype bf16` 和 MxFP4 native dispatch 环境变量；这不属于 Waterfill 等额外优化。

说明：当前可用源码是 v0.5.16 派生的复现实验 worktree，包含此前修复；本轮所有可选优化均通过参数关闭。Decode 使用 `--base-gpu-id 0` 是为适配 `CUDA_VISIBLE_DEVICES=4,5,6,7` 后的逻辑设备映射，不改变算法。第一次旧启动产生过一次 `device 4 is not visible` 日志，随后已修正并重新启动；正式矩阵使用修正后的实例。

## 3. 功能与完整性验证

三套服务均分别通过 PD `你是谁`：HTTP 200、`WHOAMI_VALID=True`，返回内容包含“我是DeepSeek”。所有八组矩阵的成功请求数均完整：

`10/10、160/160、2560/2560、5120/5120`，两种输入长度各四组。

服务日志未发现正式运行中的 Traceback、CUDA error、NCCL error、OutOfMemory、KVTransferError 或 Decode transfer failed。

## 4. 实测结果

表中百分比均相对同一行的原始 baseline；吞吐越高越好，TTFT/TPOT 越低越好。

| ISL | Concurrency | baseline Out / Total | DeepGEMM Out / Total | DeepGEMM Total变化 | MxFP4 Out / Total | MxFP4 Total变化 | baseline TTFT / TPOT ms | DeepGEMM TTFT / TPOT ms | MxFP4 TTFT / TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1 | 108.25 / 216.50 | 108.46 / 216.92 | +0.19% | 98.36 / 196.71 | -9.14% | 283.07 / 8.97 | 273.08 / 8.96 | 273.81 / 9.91 |
| 1024 | 16 | 1362.33 / 2724.66 | 1366.40 / 2732.80 | +0.30% | 1297.20 / 2594.41 | -4.78% | 564.41 / 11.14 | 524.98 / 11.15 | 515.10 / 11.78 |
| 1024 | 256 | 14055.68 / 28111.36 | 14095.35 / 28190.70 | +0.28% | 12889.04 / 25778.09 | -8.30% | 1455.11 / 16.45 | 1328.89 / 16.56 | 2087.61 / 17.39 |
| 1024 | 512 | 17167.58 / 34335.16 | 17331.50 / 34662.99 | +0.96% | 16639.40 / 33278.81 | -3.08% | 7812.24 / 21.00 | 8076.06 / 20.48 | 5830.46 / 22.91 |
| 8192 | 1 | 112.80 / 1015.24 | 112.84 / 1015.58 | +0.03% | 103.66 / 932.94 | -8.11% | 236.11 / 8.64 | 228.56 / 8.64 | 243.48 / 9.42 |
| 8192 | 16 | 1398.11 / 12583.01 | 1405.47 / 12649.24 | +0.53% | 1331.05 / 11979.49 | -4.80% | 624.14 / 10.69 | 589.21 / 10.68 | 711.14 / 11.17 |
| 8192 | 256 | 7378.04 / 66402.32 | 7269.03 / 65421.23 | -1.48% | 7191.09 / 64719.78 | -2.53% | 19725.02 / 13.95 | 20247.63 / 13.93 | 19407.54 / 15.06 |
| 8192 | 512 | 7429.88 / 66868.91 | 7366.81 / 66301.31 | -0.85% | 7431.83 / 66886.51 | +0.03% | 53097.58 / 13.96 | 53686.33 / 13.94 | 51879.57 / 15.08 |

## 5. 结论

### 5.1 纯 DeepGEMM

纯 DeepGEMM 在短输入 1024 下四个并发档均略高于 baseline，Total 提升约 `+0.19%～+0.96%`；8192 输入低并发 C1/C16 也略高，但 C256/C512 分别下降 `1.48%` 和 `0.85%`。整体说明在没有其它优化叠加时，DeepGEMM 与原始 baseline 基本同量级，优势主要出现在短输入和部分低/中并发，不能宣称普遍提升。

### 5.2 纯 flashinfer_mxfp4

纯 MxFP4 在 8 组中只有 8192/C512 的 Total 略高 baseline `+0.03%`；其余 7 组下降，1024 输入下降约 `3.08%～9.14%`，8192/C1/C16/C256 分别下降 `8.11%/4.80%/2.53%`。低并发 TPOT 普遍变差，说明单请求或小 batch 下 MxFP4 的 dispatch、dtype 转换、kernel 选择和固定开销没有被足够大的矩阵规模摊薄。

### 5.3 直接回答“基于 baseline 比较两者效果”

在本次严格控制变量的实验中，纯 DeepGEMM 明显优于纯 flashinfer_mxfp4：DeepGEMM 的 1024 四档全部超过 baseline，而 MxFP4 四档全部低于 baseline；8192 下 DeepGEMM 在 C1/C16 超过 baseline，MxFP4 仅 C512 基本持平。若目标是以原始 baseline 为基准稳定获得收益，应优先保留 DeepGEMM；MxFP4 需要进一步针对低并发 tactic、dispatch dtype 和 Graph batch 映射优化，不能仅凭高并发单点结果判断整体更优。

## 6. 文件与复现入口

- 实验根目录：[repro_runner_ab_20260824](/data/ssd2/gongoubo/single_node/repro_runner_ab_20260824)
- baseline 日志与结果：[baseline](/data/ssd2/gongoubo/single_node/repro_runner_ab_20260824/baseline)
- DeepGEMM 日志与结果：[deep_gemm](/data/ssd2/gongoubo/single_node/repro_runner_ab_20260824/deep_gemm)
- flashinfer_mxfp4 日志与结果：[flashinfer_mxfp4](/data/ssd2/gongoubo/single_node/repro_runner_ab_20260824/flashinfer_mxfp4)
- 复现实验脚本：[run_matrix.sh](/data/ssd2/gongoubo/single_node/repro_runner_ab_20260824/run_matrix.sh)
- 代码和脚本备份：[repro_runner_ab_20260824 backup](/data/ssd2/gongoubo/single_node/backups/repro_runner_ab_20260824)

每套配置均保存了独立的 Prefill/Decode/Router 服务日志、whoami 日志、每组 benchmark 日志和 JSONL 输出；备份目录包含启动脚本、矩阵脚本和 `SHA256SUMS`。
