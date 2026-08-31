# Prefill MegaMoE + Decode DSpark/MegaMoE 实验记录

日期：2026-08-24

## 1. 实验目标

验证之前没有单独验证过的精确组合：

- Prefill：MegaMoE
- Decode：DSpark speculative decoding + MegaMoE A2A
- Decode MoE 参数：`--moe-a2a-backend megamoe --moe-runner-backend auto`
- Decode CUDA Graph：target verify 和 draft verify 均为 `1/2/4/8/16/32/64/128`

这不是之前的“Prefill MegaMoE + Decode DSpark/DeepEP”实验；本轮把 Decode 的 A2A backend 明确切换成了 MegaMoE。

## 2. 部署配置

- 源码：`/data/ssd2/sglang_v0.5.16_pr32281_fix7_repro_20260823`
- Prefill GPU 0–3，Decode GPU 4–7
- Decode 使用 `CUDA_VISIBLE_DEVICES=4,5,6,7`、`--base-gpu-id 0`
- PD transfer：Mooncake/RDMA
- Decode：TP4/DP4/EP4，DP attention，DP LM head
- DSpark draft：`/data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash-dspark`
- DSpark SPS：`/data/ssd2/gongoubo/single_node/logs/flash_decode_dspark/dspark_sps.json`
- Waterfill、LPLB、TBO、HiSparse、FP4 indexer、radix cache 均未启用
- Prefill 使用 `max-prefill-tokens=chunked-prefill-size=16384`
- 正式矩阵：ISL `1024/8192`，OSL `1024`，并发 `1/16/256/512`，每组 `10×并发` 请求

## 3. 启动过程与问题

第一次启动 Decode 时没有设置 `CUDA_VISIBLE_DEVICES=4,5,6,7`，Decode 错误地与 Prefill 共用了 GPU 0–3，随后出现 KV pool OOM。第二次启动使用了正确 GPU，但前一次 Prefill 进程的显存占用被误判为 Decode 可用显存，启动探针再次失败。

之后清理并确认 Decode GPU 4–7 显存为 0，再以正确映射干净启动。正式成功配置使用 `mem-fraction-static=0.8`，不需要提高到 0.93/0.98。

正式实例验证：

- target verify Graph：四个 Decode rank 全部 capture end
- draft verify Graph：四个 Decode rank 全部 capture end
- Router `whoami`：HTTP 200，`WHOAMI_VALID=True`
- 额外 smoke：1024 输入、128 输出、10/10 成功，TTFT 189.06 ms，TPOT 3.16 ms

## 4. 正式结果

以下 baseline 是同日同请求矩阵下的原始 baseline。需要特别注意：本实验 Prefill 使用 MegaMoE，而此前原始 baseline A/B 的 Prefill 使用共同的 `flashinfer_mxfp4` 配置，因此这里是“原始 baseline vs Prefill MegaMoE + Decode DSpark/MegaMoE”的组合对比，不是只隔离 Decode 的单变量 A/B。百分比为本组合相对 baseline 的 Total tok/s 变化。

| ISL | OSL | Concurrency | Requests | Out tok/s | Total tok/s | baseline Total | Total 变化 | TTFT ms | TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 10/10 | 344.09 | 688.18 | 216.50 | +217.9% | 183.97 | 2.73 |
| 1024 | 1024 | 16 | 160/160 | 3933.39 | 7866.77 | 2724.66 | +188.7% | 324.25 | 3.59 |
| 1024 | 1024 | 256 | 2560/2560 | 35508.11 | 71016.22 | 28111.36 | +152.6% | 658.82 | 6.20 |
| 1024 | 1024 | 512 | 5120/5120 | 46216.09 | 92432.17 | 34335.16 | +169.2% | 1378.47 | 8.70 |
| 8192 | 1024 | 1 | 10/10 | 358.91 | 3230.18 | 1015.24 | +218.1% | 204.16 | 2.59 |
| 8192 | 1024 | 16 | 160/160 | 4008.20 | 36073.81 | 12583.01 | +186.8% | 495.21 | 3.36 |
| 8192 | 1024 | 256 | 2560/2560 | 8269.72 | 74427.49 | 66402.32 | +12.1% | 26291.04 | 3.82 |
| 8192 | 1024 | 512 | 5120/5120 | 8302.53 | 74722.79 | 66868.91 | +11.7% | 56097.16 | 3.91 |

## 5. 结论

这套组合可以运行，而且不是只通过启动：target/draft CUDA Graph、PD whoami、smoke 和完整 8 组矩阵均通过，成功请求数为 10/160/2560/5120（两种输入长度各四组）。

相对原始 baseline，这个完整组合的 TPOT 从 baseline 的约 `8.64–21.00 ms` 降到 `2.59–8.70 ms`，说明 Decode DSpark+MegaMoE 路径确实有效；但由于 Prefill 同时从 baseline 的 flashinfer_mxfp4 切换为 MegaMoE，Total 吞吐提升不能全部归因于 Decode。1024 输入四档 Total 提升约 `+152.6%～+217.9%`；8192 输入低/中并发提升约 `+186.8%～+218.1%`，高并发仍提升约 `+11.7%～+12.1%`。

8192 高并发的 TTFT 仍然很高（C256 26.29 s，C512 56.10 s），说明 PD Prefill 排队和 hidden/KV transfer 仍是长输入高并发瓶颈；MegaMoE 主要改善 Decode token generation，不能消除 Prefill 侧的首 token 等待。

因此，结论是：Prefill MegaMoE + Decode DSpark/MegaMoE 在当前环境下功能可用，并且相对原始 baseline 有显著组合收益；其中 TPOT 结果证明 Decode 组合本身有效，但要精确拆分 Prefill MegaMoE 与 Decode DSpark/MegaMoE 各自贡献，还需要在同一 Prefill MegaMoE 下补一个“Decode 仅 DSpark/DeepEP”或“Decode 原始 runner”的对照。高并发长输入的端到端瓶颈仍在 TTFT/PD 输入路径，后续应重点分析 Prefill 排队、hidden transfer 和 Decode 接收队列。

## 6. 文件位置

- 独立实验目录：[repro_dspark_megamoe_20260824](/data/ssd2/gongoubo/single_node/repro_dspark_megamoe_20260824)
- Decode 服务日志：[decode logs](/data/ssd2/gongoubo/single_node/repro_dspark_megamoe_20260824/logs/services/decode)
- Prefill 服务日志：[prefill logs](/data/ssd2/gongoubo/single_node/repro_dspark_megamoe_20260824/logs/services/prefill)
- Router/whoami 日志：[validation logs](/data/ssd2/gongoubo/single_node/repro_dspark_megamoe_20260824/logs/validation)
- 矩阵结果：[results](/data/ssd2/gongoubo/single_node/repro_dspark_megamoe_20260824/logs/results)
- 启动脚本备份：[backup](/data/ssd2/gongoubo/single_node/backups/repro_dspark_megamoe_20260824)
