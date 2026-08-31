# PD 分离 + MegaMoE + DSpark 单项技术优化与瓶颈报告

更新时间：2026-08-26

## 1. 目标与实验边界

固定主路径为：Prefill MegaMoE + Decode MegaMoE + DSpark + Decode CUDA Graph。
在此基础上分别评估 DeepSeek-V4 FP4 indexer、TBO、Waterfill 和 LPLB；每次只增加一项，
不把多个技术的收益混在一起。所有正式性能结论使用 ISL/OSL=`1024/1024` 和
`8192/1024`、并发 `1/16/256/512`、每组 `10 × concurrency` 请求。

当前复现实验使用 `/data/ssd2/sglang_v0.5.16_pr32281_fix7_repro_20260823`，Prefill 使用
GPU 0--3、TP4/DP1/EP1，Decode 使用 GPU 4--7、TP4/DP4/EP4，MoE A2A 为 MegaMoE，
Decode runner 为 DeepGEMM，DSpark draft 模型为
`/data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Flash-dspark`。

## 2. 当前无 indexer 基线复测

本轮重启后两侧均明确关闭 FP4 indexer，并重新完成 8 组矩阵；所有请求成功。

| ISL | OSL | C | Requests | Req/s | Out tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 10 | 0.341 | 348.87 | 697.73 | 205.69 | 2.67 |
| 1024 | 1024 | 16 | 160 | 3.905 | 3998.71 | 7997.42 | 395.03 | 3.46 |
| 1024 | 1024 | 256 | 2560 | 34.459 | 35285.81 | 70571.62 | 694.54 | 6.24 |
| 1024 | 1024 | 512 | 5120 | 45.439 | 46529.73 | 93059.46 | 1145.89 | 9.36 |
| 8192 | 1024 | 1 | 10 | 0.356 | 364.73 | 3282.60 | 203.88 | 2.54 |
| 8192 | 1024 | 16 | 160 | 3.959 | 4053.80 | 36484.23 | 502.92 | 3.30 |
| 8192 | 1024 | 256 | 2560 | 8.019 | 8211.25 | 73901.21 | 26603.53 | 3.72 |
| 8192 | 1024 | 512 | 5120 | 8.062 | 8255.94 | 74303.49 | 56534.52 | 3.80 |

原始数据：
`logs/runs/prefill_megamoe_decode_megamoe_dspark_deepgemm_20260826_no_fp4_indexer/perf_baseline_matrix/`。
两侧启动日志确认 `enable_deepseek_v4_fp4_indexer=False`，并确认 Decode Graph
捕获了 `1/2/4/8/16/32/64/128` 桶。

根据后续实验约定，逐项优化阶段只采用两个代表点：`ISL=8192, OSL=1024` 的
`Concurrency=1` 和 `512`。在 2026-08-26 重启后的当前主路径上，重新测得：

| ISL | OSL | Concurrency | Completed | Duration s | Out tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 1024 | 1 | 10 | 28.61 | 357.89 | 3221.03 | 208.03 | 2.59 |
| 8192 | 1024 | 512 | 5120 | 637.17 | 8228.39 | 74055.52 | 56766.57 | 3.78 |

这两个点作为后续 FP4 indexer、Waterfill、LPLB、TBO A/B 的聚焦基线；结果位于
`logs/runs/prefill_megamoe_decode_megamoe_dspark_deepgemm_20260826_no_fp4_indexer/focused_baseline_8192_20260826/`。

### 2.1 聚焦 TBO A/B

在完全相同的 Prefill、PD、MegaMoE、DeepGEMM、DSpark、CUDA Graph 和随机 workload
下，仅在 Decode 端增加 `--enable-two-batch-overlap`，Prefill 不加 TBO。两组请求均
全部成功，服务日志确认 CUDA Graph 持续启用且无 retraction。

| ISL | OSL | Concurrency | Completed | Duration s | Out tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms | Total 相对基线 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 1024 | 1 | 10/10 | 45.02 | 227.44 | 2046.95 | 216.10 | 4.19 | -36.45% |
| 8192 | 1024 | 512 | 5120/5120 | 635.20 | 8253.96 | 74285.62 | 54471.65 | 5.86 | +0.31% |

相对聚焦基线（C1/C512 Total tok/s 为 3221.03/74055.52），TBO 的收益只在 C512
出现且幅度接近测量噪声；C1 明显下降，C512 的 TPOT 也从 3.78 ms 增至 5.86 ms。
这支持当前瓶颈判断：DSpark target-verify 在该配置下没有获得足够的 TBO 重叠收益，
但仍承担 TBO 的 batch split、同步和调度成本。原始日志位于
`logs/runs/prefill_megamoe_decode_megamoe_dspark_deepgemm_20260826_no_fp4_indexer/focused_tbo_8192_20260826/`，
服务日志位于同目录下的 `services/`。

### 2.2 聚焦 Waterfill A/B

仅在 Decode 端启用 `--enable-waterfill`，其余配置与聚焦基线一致。两组请求均全部
成功，服务启动日志明确记录 `Waterfill is enabled with moe_a2a_backend='megamoe'`，
并记录 `Prepared 43 Waterfill TopK modules`。

| ISL | OSL | Concurrency | Completed | Duration s | Out tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms | Total 相对基线 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 1024 | 1 | 10/10 | 32.25 | 317.54 | 2857.86 | 215.78 | 2.94 | -11.27% |
| 8192 | 1024 | 512 | 5120/5120 | 632.58 | 8288.09 | 74592.81 | 55551.19 | 4.54 | +0.73% |

Waterfill 在 C512 的净收益仍小于 1%，且 TPOT 从 3.78 ms 增至 4.54 ms；C1 则下降
11.27%。在均匀随机路由下没有可供 Waterfill 利用的共享专家热点，额外的 routed
slot、count/materialize 和 dispatch 路径开销无法被负载均衡收益覆盖。因此目前只能
证明 Waterfill 生效，不能证明它在该 workload 上带来有效加速；要验证设计收益还需
使用共享专家热点/偏斜路由 workload。原始日志位于
`logs/runs/prefill_megamoe_decode_megamoe_dspark_deepgemm_20260826_no_fp4_indexer/focused_waterfill_8192_20260826/`。

### 2.3 聚焦 LPLB A/B

仅在 Decode 端启用 `--ep-dispatch-algorithm lp`，并使用
`SGLANG_LPLB_REFRESH_INTERVAL=1`、`SGLANG_LPLB_STATIC_FALLBACK=0`；其余配置与聚焦
基线完全一致。两组请求均全部成功，服务端 CUDA Graph=True、retracted=0。

| ISL | OSL | Concurrency | Completed | Duration s | Out tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms | Total 相对基线 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 1024 | 1 | 10/10 | 37.91 | 270.14 | 2431.22 | 214.32 | 3.49 | -24.52% |
| 8192 | 1024 | 512 | 5120/5120 | 633.19 | 8280.10 | 74520.91 | 54742.15 | 5.41 | +0.63% |

服务启动阶段明确记录 `Initialized LPLB solvers for 43 layers`，并对
`(NC=4, NV=6)` 预热 CUDA IPM solver，证明本轮不是参数未生效。C1 的损失与 C512
接近基线但 TPOT 上升，说明主要代价是每轮 dispatch 的 expert count、EP collective
和 LP/IPM 求解/映射维护；在当前均匀随机路由 workload 下没有足够的负载偏斜可以
回收这些成本。原始结果和独立服务日志位于
`logs/runs/prefill_megamoe_decode_megamoe_dspark_deepgemm_20260826_no_fp4_indexer/focused_lplb_8192_20260826/`。

### 2.4 LPLB 静态回退优化 A/B

针对上述瓶颈，保持 `--ep-dispatch-algorithm lp`，但设置
`SGLANG_LPLB_STATIC_FALLBACK=1`。该路径在服务初始化时缓存 rank-aware static map，
并在 MoE forward 中跳过 LPLB solver 的动态概率求解和 EP all-reduce；因此它是针对
低并发的调度优化，不等同于完整动态 LPLB。两组请求均全部成功。

| ISL | OSL | Concurrency | Completed | Duration s | Out tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms | Total 相对基线 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 1024 | 1 | 10/10 | 29.23 | 350.37 | 3153.35 | 203.39 | 2.66 | -2.10% |
| 8192 | 1024 | 512 | 5120/5120 | 633.28 | 8278.98 | 74510.79 | 56258.93 | 3.92 | +0.61% |

相对动态 LPLB，C1 总吞吐提升 29.7%，TPOT 从 3.49 ms 降至 2.66 ms；这说明主要
瓶颈确实是动态 solver/collective，而不是显存或 CUDA Graph。C1 仍比基线低 2.1%，
差距已接近单次测量波动，可能来自 LP 配置导致的静态 expert placement 与 baseline
placement 不同；若要进一步超过基线，应比较相同 static map 并测量 rank 间 token
偏斜，而不是继续盲目增加 solver 刷新频率。原始结果和服务日志位于
`logs/runs/prefill_megamoe_decode_megamoe_dspark_deepgemm_20260826_no_fp4_indexer/focused_lplb_static_fallback_8192_20260826/`。

### 2.5 FP4 indexer 双端适配与聚焦 A/B

FP4 indexer 必须在 Prefill 和 Decode 两端同时开启；只在 Decode 开启会造成 KV/indexer
layout 不一致，不能作为性能实验。第一次双端启动使用默认
`SGLANG_DSV4_FIX_TP_ATTN_A2A_SCATTER=1`，请求进入 Prefill 后两端均报：
`ValueError: Tensor input and output of _broadcast_oop must have the same number of elements`，
随后 Prefill 退出、Decode 丢失 bootstrap connection，因此该轮结果作废。日志保存在
`logs/runs/prefill_megamoe_decode_megamoe_dspark_deepgemm_20260826_fp4_indexer_dual/services/`。

针对该已定位的 scatter shape 冲突，双端均设置
`SGLANG_DSV4_FIX_TP_ATTN_A2A_SCATTER=0` 后重新启动。两端启动参数均明确包含
`--enable-deepseek-v4-fp4-indexer`，Decode 完成 target/draft CUDA Graph 捕获且运行期
`cuda graph: True`。在新脚本中，“你是谁”不再使用裸文本请求：DeepSeek-V4 tokenizer
没有 HF `chat_template`，验证脚本改为调用 SGLang 原生 `encoding_dsv4.encode_messages`
拼接 `<｜begin▁of▁sentence｜><｜User｜>...<｜Assistant｜></think>` 格式，再发送到
`/generate`。本轮验证结果为 `HTTP=200`、`WHOAMI_VALID=True`，返回“我是DeepSeek……”。
脚本旧版已备份为
`dspark_stepwise_ablation_20260824/validate_pd_whoami.sh.bak_20260826_before_apply_chat_template`。

聚焦 A/B 结果如下；基线为同一轮的无 indexer 聚焦基线（C1/C512 Total tok/s 为
3221.03/74055.52）：

| ISL | OSL | Concurrency | Completed | Duration s | Out tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms | Total 相对基线 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 1024 | 1 | 10/10 | 28.81 | 355.42 | 3198.74 | 240.62 | 2.58 | -0.69% |
| 8192 | 1024 | 512 | 5120/5120 | 715.59 | 7326.63 | 65939.70 | 64242.57 | 3.78 | -10.96% |

C1 基本持平但 TTFT 增加约 32.6 ms；C512 在请求全部成功、无 retraction、CUDA Graph
持续开启的前提下仍下降 10.96%，说明当前 FP4 indexer 的额外 indexer 计算/显存占用及
共享专家融合被禁用的代价超过收益。该结论与旧矩阵中 C256/C512 的 TTFT=0/KV transfer
异常结果不同，本轮 C512 是完整、可计入的双端结果，但仍未超过基线。原始结果、服务
日志和正确性验证均位于：
`logs/runs/prefill_megamoe_decode_megamoe_dspark_deepgemm_20260826_fp4_indexer_dual_scatter0_8192_20260826/`。

随后在不重启服务的情况下补齐正式 8 组矩阵。下表中的基线是第 2 节同轮完整矩阵，
FP4 的 8192/C1、C512 沿用已完成的聚焦结果；因此这些结果均保持
`seed=1`、`OSL=1024`、`N=10×C`，且每组都核对了成功请求数。

| ISL | OSL | C | Completed | Req/s | Out tok/s | Total tok/s | Baseline Total tok/s | Total 变化 | Mean TTFT ms | Mean TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 10/10 | 0.326 | 334.12 | 668.24 | 697.73 | -4.23% | 170.33 | 2.83 |
| 1024 | 1024 | 16 | 160/160 | 3.740 | 3829.94 | 7659.89 | 7997.42 | -4.22% | 414.09 | 3.65 |
| 1024 | 1024 | 256 | 2560/2560 | 33.214 | 34010.74 | 68021.48 | 70571.62 | -3.62% | 751.27 | 6.33 |
| 1024 | 1024 | 512 | 5120/5120 | 48.925 | 50101.21 | 100202.42 | 93059.46 | **+7.68%** | 1446.49 | 8.34 |
| 8192 | 1024 | 1 | 10/10 | 0.347 | 355.42 | 3198.74 | 3282.60 | -2.55% | 240.62 | 2.58 |
| 8192 | 1024 | 16 | 160/160 | 3.737 | 3827.09 | 34443.79 | 36484.23 | -5.59% | 564.89 | 3.48 |
| 8192 | 1024 | 256 | 2560/2560 | 7.131 | 7301.18 | 65710.65 | 73901.21 | -11.08% | 30410.40 | 3.68 |
| 8192 | 1024 | 512 | 5120/5120 | 7.151 | 7326.63 | 65939.70 | 74303.49 | -11.27% | 64242.57 | 3.78 |

FP4 只有 1024/C512 超过完整基线（+7.68%），其余 7 组未超过；8192 长输入的
高并发反而下降约 11%。因此不能用单个 1024/C512 的收益宣称 FP4 全面加速。8 组
原始结果位于该目录的 `formal_matrix/`，C1/C512 原始聚焦结果和服务日志也保留在
同一目录根下。

### 2.6 正确性验收规则更新

每次服务重启后，必须先用 native DSV4 prompt 验证“你是谁”，再运行性能测试。仅有
HTTP 200、裸 `/generate` 或非空输出都不算通过；必须记录 `WHOAMI_VALID=True`，并核对
性能 benchmark 的 `Successful requests == Num prompts`。随机吞吐 benchmark 仍按原始
 random prompt 生成，因此不把 chat prompt 的额外 token 混入性能样本。验证脚本现在还
支持 `WHOAMI_RESULT_FILE`，会保存实际发送的完整 `text`、`sampling_params` 和响应；本轮
真实验证文件为
`logs/runs/validation_native_prompt_20260826/validation/whoami_request_response.json`，其中
`whoami_valid=true`，实际 prompt 为：
`<｜begin▁of▁sentence｜><｜User｜>你是谁？请只用中文一句话回答，不要展开推理。<｜Assistant｜></think>`。
由于该 tokenizer 没有 HF `chat_template`，这里采用 SGLang 原生 DSV4 encoder 是模型正确
格式，不使用裸文本，也不把不存在的 `apply_chat_template` 结果冒充为实际请求。

### 2.7 TBO batch gate 尝试及约束结论

为消除 TBO 在低并发下的固定 split/sync 成本，新增了可选的
`--tbo-min-batch-size` 参数，并将其从 shell 环境改为 `ServerArgs` 字段，确保多进程
scheduler 能继承；源码和启动脚本均已备份在
`backups/tbo_batch_gate_20260826/`。参数语法和 Python 编译检查通过，Decode 启动日志
确认 `enable_two_batch_overlap=True, tbo_min_batch_size=16`，target/draft Graph capture
也完成，native “你是谁”验证通过。

但第一次 C1 请求的运行日志显示 `cuda graph: False`，同时吞吐下降到约 50 tok/s；
这是因为 gate 将 `forward_batch.can_run_tbo` 置为 false，而当前 Decode Graph runner
把该条件作为 Graph 可运行性条件，最终回退 eager。该结果没有计入性能表，也说明
“直接以 `can_run_tbo=False` 绕过 TBO”不能满足 Decode CUDA Graph 硬约束。历史 TBO
路径在不使用 gate 时运行期 `cuda graph: True`，但仍有 C1/C16 固定调度损失。因此后续
若继续优化 TBO，必须同时实现普通 decode Graph 与 TBO Graph 的双路 capture/replay，
不能只加 batch-size 判断；在此适配完成前，TBO 按失败技术记录。

## 3. 历史单项结果与基线差分

下表来自同一主路径的历史 8 组单项矩阵，基准为当轮 `DSpark + MegaMoE`。百分比为
单项 Total tok/s 相对基准的变化。

| 技术 | 1024/C1 | 1024/C16 | 1024/C256 | 1024/C512 | 8192/C1 | 8192/C16 | 8192/C256 | 8192/C512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepGEMM | +1.33% | +2.11% | -0.56% | **+10.65%** | +0.67% | -0.40% | -0.04% | +0.39% |
| FP4 indexer（旧矩阵） | +0.80% | +1.62% | +1.72% | +9.45% | -9.58% | +4.23% | +0.96%* | +0.86%* |
| Waterfill | -12.41% | -14.19% | -8.64% | -12.40% | -12.91% | -11.76% | -0.45% | -0.68% |
| LPLB | -23.50% | -22.50% | -17.63% | -29.78% | -21.73% | -21.05% | -0.43% | -0.11% |
| TBO | -35.64% | -29.13% | -22.59% | -30.66% | -36.31% | -28.97% | -0.73% | -0.18% |

`*` FP4 indexer 的 8192/C256、C512 历史结果 Mean TTFT=0，并伴随 KV transfer
异常；后续双端 layout 修复复测才可作为有效 FP4 结果，不能只看表面 Total tok/s。

原始矩阵目录：`dspark_stepwise_ablation_20260824/variants/{base,dspark_deepgemm,dspark_fp4_indexer,dspark_waterfill,dspark_lplb,dspark_tbo}/logs/results/`。

### 3.1 历史单项 8 组原始指标核对

为避免只展示百分比，下面保留历史单项目录中可直接复核的完整指标。每行均为
`N=10×C`，`Completed` 为成功请求数；这些数据用于第 3 节的差分计算。FP4 的
8192/C256、C512 因历史 TTFT=0/KV transfer 异常不在此处重复计入，修复后的有效
FP4 8 组见第 2.5 节。

**Waterfill**（`dspark_stepwise_ablation_20260824/variants/dspark_waterfill/logs/results/`）

| ISL | C | Completed | Req/s | Out tok/s | Total tok/s | TTFT ms | TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1 | 10/10 | 0.294 | 301.38 | 602.76 | 185.73 | 3.14 |
| 1024 | 16 | 160/160 | 3.296 | 3375.36 | 6750.73 | 360.86 | 4.24 |
| 1024 | 256 | 2560/2560 | 31.680 | 32440.37 | 64880.73 | 753.30 | 6.72 |
| 1024 | 512 | 5120/5120 | 39.535 | 40483.73 | 80967.47 | 1174.30 | 10.88 |
| 8192 | 1 | 10/10 | 0.305 | 312.58 | 2813.18 | 207.73 | 3.00 |
| 8192 | 16 | 160/160 | 3.454 | 3536.64 | 31829.74 | 475.99 | 3.89 |
| 8192 | 256 | 2560/2560 | 8.039 | 8232.20 | 74089.84 | 25909.40 | 4.30 |
| 8192 | 512 | 5120/5120 | 8.053 | 8245.96 | 74213.67 | 56071.15 | 4.33 |

**LPLB 动态路径**（`dspark_stepwise_ablation_20260824/variants/dspark_lplb/logs/results/`）

| ISL | C | Completed | Req/s | Out tok/s | Total tok/s | TTFT ms | TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1 | 10/10 | 0.257 | 263.22 | 526.43 | 178.67 | 3.63 |
| 1024 | 16 | 160/160 | 2.977 | 3048.45 | 6096.91 | 352.08 | 4.68 |
| 1024 | 256 | 2560/2560 | 28.564 | 29249.59 | 58499.18 | 611.53 | 7.73 |
| 1024 | 512 | 5120/5120 | 31.693 | 32453.50 | 64907.01 | 1021.75 | 14.20 |
| 8192 | 1 | 10/10 | 0.274 | 280.91 | 2528.22 | 205.51 | 3.36 |
| 8192 | 16 | 160/160 | 3.090 | 3164.67 | 28482.02 | 493.63 | 4.37 |
| 8192 | 256 | 2560/2560 | 8.041 | 8233.90 | 74105.11 | 24972.68 | 5.25 |
| 8192 | 512 | 5120/5120 | 8.099 | 8293.17 | 74638.53 | 54688.27 | 5.34 |

**TBO**（`dspark_stepwise_ablation_20260824/variants/dspark_tbo/logs/results/`）

| ISL | C | Completed | Req/s | Out tok/s | Total tok/s | TTFT ms | TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1 | 10/10 | 0.216 | 221.47 | 442.93 | 184.62 | 4.34 |
| 1024 | 16 | 160/160 | 2.722 | 2787.43 | 5574.86 | 320.30 | 5.18 |
| 1024 | 256 | 2560/2560 | 26.842 | 27486.69 | 54973.39 | 639.57 | 8.18 |
| 1024 | 512 | 5120/5120 | 31.296 | 32047.08 | 64094.17 | 933.58 | 14.28 |
| 8192 | 1 | 10/10 | 0.223 | 228.58 | 2057.20 | 205.34 | 4.18 |
| 8192 | 16 | 160/160 | 2.780 | 2847.06 | 25623.57 | 490.88 | 4.93 |
| 8192 | 256 | 2560/2560 | 8.017 | 8209.19 | 73882.75 | 24362.10 | 5.88 |
| 8192 | 512 | 5120/5120 | 8.094 | 8287.87 | 74590.85 | 54089.90 | 5.93 |

## 4. 已确认的适配和证据

### 4.1 DeepGEMM

Decode 通过 `--moe-runner-backend deep_gemm` 使用 DeepEP/MegaMoE 输出对应的 masked
DeepGEMM 路径；服务日志显示 `moe_runner_backend=deep_gemm`，并完成 DeepGEMM warmup
和 CUDA Graph 捕获。它在 1024/C512 的收益最大，说明高并发 decode expert GEMM
是可被计算 kernel 优化的部分。

### 4.2 FP4 indexer

FP4 indexer 必须 Prefill/Decode 双端一致。历史问题是仅 Decode 开启时产生 indexer/KV
layout 不一致，出现 TTFT=0 和 Mooncake transfer 异常。双端修复版本已保存在
`backups/fp4_indexer_full_retest_20260825/` 和 `backups/fp4_indexer_kv_transfer_fix_20260825/`。
当前为严格基线复测，Prefill/Decode 两侧均未开启 indexer；后续 FP4 A/B 必须使用双端
一致配置和有效性审计。

### 4.3 Waterfill

Waterfill 通过 routed-count、rank load 计算和 shared-expert expansion 将共享专家纳入
dispatch 负载平衡。当前源码中 Waterfill 会把 fused shared expert 改成额外 routed slot；
即使随机 workload 没有共享专家热点，也会改变 routed expert 的调度和计算路径。
`SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE` 只能跳过小 batch 的 balance/materialize 部分，
不能消除初始化时的额外 routed slot 和路径开销。

历史 A/B 已证明 Waterfill 真实执行，但均匀随机 workload 下收益不足以抵消额外开销；
它只有在共享专家存在真实热点或专家负载偏斜时才有机会产生净收益。

### 4.4 LPLB

LPLB 的动态路径每次刷新包含本地 expert count、EP all-reduce 和 IPM/LP solve。当前实现
明确要求所有 EP rank（包括空 token rank）参与 solve，以避免 DP attention 下 collective
死锁。`SGLANG_LPLB_REFRESH_INTERVAL` 可降低刷新频率，`SGLANG_LPLB_STATIC_FALLBACK=1`
可绕过热路径动态求解。

历史结果显示 LPLB 低并发损失最大，长输入高并发接近基线；这与低 token batch 下
all-reduce/solver 固定成本占比高、高并发下该成本被 expert GEMM 摊薄一致。

### 4.5 TBO

当前 DeepSeek-V4 TBO 适配受 forward mode 和 DSpark target-verify metadata 约束。
源码 `_can_run_tbo()` 对 DSpark target-verify 保持现有 decode eager/graph 路径；TBO
不能简单通过 decode 端开关获得真正的 decode 层级重叠。当前 decode 仍承担 TBO 调度、
dispatcher 或检查路径的固定开销，因此低并发明显变慢。

## 5. 当前瓶颈判断

| 技术 | 主要瓶颈 | 证据 | 优先级 |
|---|---|---|---:|
| FP4 indexer | 双端 indexer/KV layout 一致性、DSA/indexer 计算与 transfer 语义 | 旧矩阵 TTFT=0；双端修复记录；启动参数审计 | 高 |
| Waterfill | 额外 routed slot、count/materialize、均匀 workload 无可平衡热点 | Waterfill A/B；源码 Waterfill slot 改写；低 batch 阈值实验 | 中 |
| LPLB | 每次刷新固定的 count + EP all-reduce + LP/IPM solve | solver 源码；低并发 -17%~-30%；refresh/static fallback 结果 | 高 |
| TBO | DSV4 Decode/DSpark verify 不真正进入 TBO overlap，仍支付调度成本 | `_can_run_tbo()`；Decode Graph 与 TBO gate 实验 | 高 |

显存不是当前四项技术低并发损失的首要证据：服务均能完成 Graph capture，且当前基线
GPU 显存与利用率正常。Nsight Systems 只在采集到运行期请求后，才用于区分 GPU kernel、
NCCL/RDMA transfer 和 scheduler queue 时间；没有运行期覆盖的 profile 不作为吞吐证据。

### 5.1 当前基线运行期硬件采样

Nsight Systems 配套 Python 的 `duckdb`、`pyarrow` 和 `pandas` 已安装完成（版本分别为
`1.5.5`、`25.0.1`、`3.0.5`），并成功解析
首份可审计报告 `logs/runs/validation_native_prompt_20260826/nsys/base_c1/pd_base_c1.nsys-rep`。
该 profile 的服务日志显示采集窗口主要落在启动/预热阶段，`timeline_summary` 没有 CUDA
Graph 事件，也没有覆盖 benchmark 请求，因此不能用于解释吞吐；对应的标准化 fact 输出
保存在同目录 `facts/`。另一次运行期采集确实完成了 8192/C512 的 128/128 benchmark，
但 profiler 在服务保持存活时未自动落盘，结束后没有生成第二份 `.nsys-rep`，所以只保留
请求结果和服务日志，不把它当作运行期 Nsight 证据。改用同一 `/generate` benchmark 期间的
`nvidia-smi dmon -s pucm` 采样，结果保存于
`logs/runs/validation_native_prompt_20260826/hw_sampling/`。这是短采样，不替代正式
8 组吞吐数据，但可以排除“服务没有使用 GPU”这一类误判：

| 诊断样本 | 完成请求 | Total tok/s | Mean TTFT ms | Mean TPOT ms | Decode GPU 平均 SM（4--7） | Decode GPU 平均功耗 W（4--7） | 采样显存 MB（4--7） |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8192/C1，10 请求 | 10/10 | 3054.19 | 213.23 | 2.82 | 40.2% | 295.9 | 154,909--155,005 |
| 8192/C512，128 请求 | 128/128 | 52455.10 | 4261.01 | 3.34 | 35.0% | 341.3 | 154,682--155,002 |

C1/C512 期间 Decode 4 张卡均保持约 155 GB 显存占用，且有持续功耗和 SM 活动；因此
当前低并发损失更符合固定 dispatch/collective、Graph replay 和 PD 调度开销占比过高，
而不是显存未释放或服务落到 CPU。dmon 原始日志同时保留了 Prefill GPU 0--3 的
对应采样，可用于复核 PD 两侧活动。要精确拆出 NCCL/RDMA 与 DeepGEMM kernel 的
时间比例；依赖问题已经排除，但还需要一个能在服务保持存活时正确收尾的 profiler
采集流程来生成完整运行期 trace，本报告不对未生成的运行期 trace 作推断。

为验证 Waterfill 是否只是改变显存占用，随后在同一当前服务脚本上切换到
`--enable-waterfill`，先通过 native “你是谁”，再运行相同的 8192/C512、128 请求短样本。
Waterfill 样本为 `Total=47594.76 tok/s`、`Mean TPOT=4.30 ms`，而基线短样本为
`52455.10 tok/s`、`3.34 ms`（短样本仅用于硬件对照，正式结论仍使用 5120 请求矩阵）。
Decode 4 卡平均 SM 约 `38.1%`、平均功耗约 `328.7 W`，显存约 `154556--154878 MB`；
基线对应约 `35.0%`、`341.3 W`、`154682--155002 MB`。这说明 Waterfill 并未带来更高
的硬件利用率或显著显存节省，反而在相近显存占用下拉长 token 间隔；结合启动时
`Prepared 43 Waterfill TopK modules` 和正式 A/B，当前最可信的瓶颈是 count/materialize
与调度路径增加了等待，且随机路由没有足够偏斜可供其回收。两轮原始 benchmark 和
`nvidia-smi dmon` 文件分别位于：
`logs/runs/validation_native_prompt_20260826/hw_sampling/{baseline_8192_c512_short,waterfill_8192_c512_short}/`。

### 5.2 运行期基线短测与服务正确性

补采的目标基线为 Prefill+MegaMoE / Decode+MegaMoE+DSpark+DeepGEMM，Decode 未启用
FP4 indexer、Waterfill、LPLB 或 TBO。8192/1024/C512 短测使用 128 个请求，结果为：
`128/128` 成功、Total tok/s=`51631.99`、Out tok/s=`5991.81`、Mean TTFT=`4116.69 ms`、
Mean TPOT=`3.89 ms`。Decode 服务日志在请求期间持续记录 `cuda graph: True`，没有发现
Traceback、NotImplementedError 或 4xx/5xx；native-format “你是谁”验证返回 HTTP 200，
`WHOAMI_VALID=True`。结果、验证 JSON 和独立 Decode 日志位于：
`logs/runs/validation_native_prompt_20260826/nsys/base_c512_runtime/`。

### 5.3 Nsight Systems 运行期证据

为避免将启动期 kernel 当作推理期瓶颈，重新启动同一目标基线并在服务 ready 后执行
native-format whoami 校验和 8192/C512 短测。运行期 profile 位于：
`logs/runs/validation_native_prompt_20260826/nsys/base_c512_runtime2/pd_base_c512_runtime2.nsys-rep`。
该 profile 的采集窗口覆盖服务运行期，标准化 fact 输出位于同目录的
`report_context.json`、`activity_summary.json`、`kernel_summary.json`、
`cuda_api_summary.json`、`memcpy_summary.json`、`timeline_summary.json`、
`kernel_variance.json` 和 `nccl_distribution.json`；关键事实如下：

| 事实 | 结果 | 解释边界 |
|---|---:|---|
| 活跃 GPU | 物理 GPU 4--7（4 张 B200） | 与 Decode 的 `CUDA_VISIBLE_DEVICES=4,5,6,7` 一致 |
| CUDA Graph trace events | 1196 | 证明采集窗口内实际发生 Graph trace；服务日志同时记录 `cuda graph: True` |
| CUDA runtime rows / kernel rows | 88342 / 25335 | 证明窗口内存在实际 CUDA 工作，不是空闲服务 |
| kernel timeline span | 约 1.69 s/卡 | 运行期窗口的 kernel 首尾覆盖范围 |
| kernel duration upper-bound coverage | 1.88%--6.02%/卡 | 仅是 kernel 时间覆盖上界，不等同 SM 利用率 |
| NCCL kernel | `ncclDevKernel_AllGather_RING_LL`，264 次 | Nsight 没有 NCCL event table；这里只能证明 NCCL kernel 执行，不能推导完整通信占比 |
| CUDA Graph launch | 528 次，API 聚合 413.26 ms | host API 聚合时间，可能与设备执行重叠，不等同端到端耗时 |
| CUDA stream synchronize | 3216 次，API 聚合 198.02 ms | 支持“同步/调度固定成本”假设，但不能单独归因于某一个技术 |

该 profile 没有 GPU Metrics，因此不从 Nsight 推断硬件 SM utilization；GPU 利用率仍以
dmon 样本为准。它给出的可审计结论是：基线运行期确实进入 CUDA Graph，并同时存在大量
Graph launch、stream synchronize 和跨卡 AllGather；低并发优化应优先减少固定调度/同步及
通信等待，而不是继续增加额外的 route/materialize 或 solver 工作。第二次 profile 的
benchmark 因采集窗口自动收尾而只完成 32/128，吞吐数字作废，仅保留上述时间线证据。

### 5.4 SGLang 内置 profiler：逐项技术对比

使用原生 SGLang `/start_profile` 接口和 `sglang.benchmark.serving --profile --profile-by-stage --profile-steps 8 --profile-activities CPU GPU` 采集 Decode 端 Chrome trace。固定 workload 为 `ISL=8192、OSL=1024`，分别采集 C1/C512；Prefill 保持 MegaMoE+DeepGEMM，Decode 保持 MegaMoE+DeepGEMM+DSpark+CUDA Graph，仅切换被测技术。trace 和解析结果在：

`logs/runs/sglang_profile_20260826/{base,tbo,waterfill,lplb,fp4_indexer}/{c1,c512}/`

profile 会引入 instrumentation overhead，benchmark 吞吐仅用于确认请求完成，不作为正式性能表；瓶颈结论以相同 rank（TP0）的结构性事件为依据。

| 配置 | 场景 | 成功请求 | TARGET_VERIFY 事件/总耗时 | scheduler.run_batch GPU | AllGather GPU | cudaMemcpyAsync | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| Base | C1 | 10/10 | 88 / 544.04 ms | 101.31 ms | 0.13 ms | 54.40 ms | 稳定基线 |
| Waterfill | C1 | 10/10 | 88 / 547.48 ms | 100.64 ms | 0.07 ms | 56.12 ms | 与基线基本相同 |
| TBO | C1 | 10/10 | 80 / 674.94 ms | 约216.87 ms | 0.23 ms | 59.76 ms | 调度路径变重 |
| LPLB | C1 | 10/10 | 88 / 817.33 ms | 139.27 ms | 0.07 ms | 89.32 ms | solver/调度开销明显 |
| FP4 indexer | C1 | 10/10 | 88 / 542.51 ms | 101.07 ms | 0.07 ms | 54.04 ms | C1 未见额外代价 |
| Base | C512 | 128/128 | bs=7/8/9，合计 903.44 ms | 157.28 ms | 8.74 ms | 78.95 ms | 稳定 Graph 批次 |
| Waterfill | C512 | 128/128 | bs=8，961.66 ms | 154.11 ms | 14.66 ms | 103.28 ms | 通信/拷贝代价增加 |
| TBO | C512 | 128/128 | 出现 bs=1/15/16/32 拆分 | 约3349 ms | 约8.97 ms | 约3112 ms | rank imbalance |
| LPLB | C512 | 128/128 | 11×bs=1 + bs=8/9 | 219.23 ms | 8.25 ms | 136.44 ms；GraphLaunch 约1729 ms | 破坏 Graph 复用 |
| FP4 indexer | C512 | 12/128 | 88×bs=1，partial trace | 1484.45 ms | 3.78 ms | 1442.96 ms | CUDA illegal memory access 后退出 |

Base C512 的 `TARGET_VERIFY` 为 bs=8/9/7 三类事件之和（447.91+340.20+115.33=903.44 ms）。Waterfill 将批次集中为 bs=8，但设备执行和拷贝时间仍上升；TBO、LPLB 的主要问题是批次形态、Graph 复用和 handoff 同时恶化，而不是单纯 GEMM 变慢。FP4 indexer C512 的首个致命事件是 Decode scheduler 的 `CUDA error: an illegal memory access was encountered`，之后 Prefill 才出现 Mooncake `transport retry counter exceeded`，后者是 Decode 退出后的连带错误；FP4 C1 trace 可用，C512 仅作为稳定性失败证据。

#### 5.4.1 FP4 indexer 8192/C512 修复后重新 profile（2026-08-26）

上一次 profile 报错的直接原因不是 FP4 indexer 修复失效，而是启动命令误用了不带 FP4 参数的
`flash_prefill_megamoe.sh`：Decode 开启了 FP4 indexer，Prefill 实际为
`enable_deepseek_v4_fp4_indexer=False`。历史修复脚本为
`backups/fp4_indexer_kv_transfer_fix_20260825/flash_prefill_megamoe.sh.with_fp4_both_sides`，
本轮改用该脚本，并确认两端日志均为 `enable_deepseek_v4_fp4_indexer=True`。同时保留
`SGLANG_DSV4_FIX_TP_ATTN_A2A_SCATTER=1`、Decode full CUDA Graph bucket=`1,2,4,8,16,32,64,128`、
`SGLANG_RAGGED_VERIFY_MODE=static` 和关闭 overlap。

历史能跑完 8192/C512 的 FP4 记录使用 `moe-runner-backend=auto`，因此本次修复 profile 也使用
auto；此前显式 `deep_gemm` 的 profile 属于另一配置，不能混写。

| ISL | OSL | Concurrency | Profile requests | Completed | Total tok/s | Mean TTFT ms | Mean TPOT ms | 结果 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8192 | 1024 | 1 | 10 | 10/10 | 2938.17 | 224.12 | 3.79 | 通过 |
| 8192 | 1024 | 512 | 128（短 profile） | 128/128 | 49897.62 | 4302.81 | 4.16 | 通过 |

8192/C512 profile trace 的 TP0 结构性指标为：`scheduler.get_next_batch_to_run` GPU 151.14 ms、
`nccl:_all_gather_base` GPU 109.16 ms、`scheduler.run_batch` GPU 14.54 ms；profile 期间没有
Decode CUDA illegal memory access，CUDA Graph 正常执行。trace、结果和服务日志位于
`logs/runs/sglang_profile_20260826/fp4_indexer_fixed_scatter1/c512/` 及其 `services/` 目录；
C1 对应同目录下的 `c1/`。因此，之前的 `fp4_indexer/c512` partial trace 应标记为“Prefill 未开启
FP4 的配置错误导致的无效 profile”，不能作为 FP4 indexer 在 8192/C512 上的性能或稳定性结论。

#### 5.4.2 FP4 indexer 与主线 DeepGEMM 基线同口径 profile

为避免把 `auto` 与主线的 `deep_gemm` 混在一起，随后又使用 Prefill/Decode 两端均开启
FP4 indexer、Prefill `deep_gemm`、Decode `deep_gemm + MegaMoE + DSpark` 的正式口径重新采样。
启动脚本为 `backups/fp4_indexer_kv_transfer_fix_20260825/flash_prefill_megamoe.sh.with_fp4_both_sides`
和 `flash_decode_dspark_megamoe.sh`；两端日志均确认 `enable_deepseek_v4_fp4_indexer=True`，
Decode CUDA Graph 与 `SGLANG_RAGGED_VERIFY_MODE=static` 保持开启。

| ISL | OSL | Concurrency | Profile requests | Completed | Total tok/s | Mean TTFT ms | Mean TPOT ms | 结果 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8192 | 1024 | 1 | 10 | 10/10 | 2964.56 | 209.44 | 3.83 | 通过 |
| 8192 | 1024 | 512 | 128（短 profile） | 128/128 | 52110.60 | 4303.47 | 4.13 | 通过 |

其中 C512 还完成了 128/128 的独立 smoke test，Total tok/s=51337.19；两次均无 Decode
illegal memory access。C512 TP0 trace 的主要 GPU 时间为：`step[IDLE bs=0]` 262.13 ms、
`scheduler.get_next_batch_to_run` 131.33 ms、NCCL AllGather 87.34 ms、
`scheduler.run_batch` 15.12 ms；C1 则为 TARGET_VERIFY 136.61 ms、scheduler.get_next 69.79 ms、
AllGather 11.36 ms、run_batch 30.38 ms。由此可见，修复后 FP4 indexer 本身不再触发崩溃，
但 8192/C512 的主要剩余成本仍是 Decode 空闲/批次等待、调度和跨卡 AllGather，而不是
`scheduler.run_batch` 内的 GEMM。profile 结果和 trace 位于
`logs/runs/sglang_profile_20260826/fp4_indexer_fixed_deepgemm/{c1,c512}/`；该组用于与
主线 DeepGEMM baseline 做结构性对比，不能直接替代无 profiler 的正式吞吐矩阵。

### 5.5 Prefill 端 profile 补充

为确认 PD 分离中的阶段归因，又在同一组 `ISL=8192、OSL=1024` workload 下对 Prefill 端采集了
C1/C512。结果文件位于 `logs/runs/sglang_profile_20260826/base_prefill/{c1,c512}/`，
解析结果为 `analysis/base_prefill_c1.json` 和 `analysis/base_prefill_c512.json`；两组分别完成
10/10、128/128 请求。

| Prefill 场景 | EXTEND 主体 | GPU scheduler.run_batch | AllGather/AllReduce | 主要 CUDA runtime 特征 | 归因 |
|---|---|---:|---:|---|---|
| C1 | 单请求长 token extend，约 228--248 ms/步 | 1659.31 ms | AllGather 2813.14 ms | stream synchronize 2495.10 ms | 首 token 和通信/同步固定成本占主导 |
| C512 | 多请求 extend，约 222--239 ms/步 | 1654.87 ms | AllReduce 164.52 ms | memcpy 53.69 ms，stream synchronize 6.04 ms | 长输入批处理吞吐稳定，排队/批次等待影响 TTFT |

Prefill 的 profile 没有启用单项 Decode 优化，因此它是所有 Decode A/B 的共同背景。结合 Decode
profile 可将瓶颈边界明确为：低并发主要受 Prefill/PD handoff 与 Decode 固定调度成本影响；高并发
时 Prefill 批处理摊薄固定成本，Decode 的 batch 形态、Graph 复用、AllGather 和 handoff 才决定
吞吐差异。因此不能用 Decode 单项技术的高并发收益推断它会改善低并发 TTFT。

### 5.6 LPLB 静态回退 profile

动态 LPLB 之外，单独 profile 了 `SGLANG_ENABLE_LPLB=1`、
`SGLANG_LPLB_STATIC_FALLBACK=1` 路径。该路径仍使用 LPLB 的静态 rank-aware map，但跳过运行期
solver、概率求解和 EP all-reduce。结果位于：
`logs/runs/sglang_profile_20260826/lplb_static_fallback/{c1,c512}/`。

| 配置 | 场景 | 成功请求 | TARGET_VERIFY | scheduler.run_batch GPU | AllGather GPU | cudaMemcpyAsync | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| LPLB static fallback | C1 | 10/10 | 88 / 550.90 ms | 103.15 ms | 0.23 ms | 52.85 ms | 与 Base 接近，未引入动态 solver |
| LPLB static fallback | C512 | 128/128 | 88×bs=5 / 827.69 ms | 138.79 ms | 5.57 ms | 82.79 ms | 避免动态 LPLB 的 GraphLaunch 异常，但仍受 batch/通信影响 |

静态回退 profile 证明动态 LPLB 的主要额外开销确实来自运行期求解与调度，而不是 LPLB map
本身；不过静态 map 也没有在该随机 workload 上带来独立的设备执行收益。因此它更适合作为
低并发的安全候选实现，是否超过原始基线仍需用正式（非 profile）吞吐复测确认。

## 6. 复现与日志

- 当前无 indexer 服务日志：
  `logs/runs/prefill_megamoe_decode_megamoe_dspark_deepgemm_20260826_no_fp4_indexer/services/`
- 当前基线原始结果：
  `logs/runs/prefill_megamoe_decode_megamoe_dspark_deepgemm_20260826_no_fp4_indexer/perf_baseline_matrix/`
- 启动脚本备份：`backups/no_fp4_indexer_20260826/`
- 历史单项原始结果：`dspark_stepwise_ablation_20260824/variants/*/logs/results/`
- 历史单项报告：`readmes/DSpark_MegaMoE_Stepwise_Ablation_20260824.md`
- 本轮 Waterfill 硬件短采样：`logs/runs/validation_native_prompt_20260826/hw_sampling/waterfill_8192_c512_short/`
- 采样后恢复的无额外优化 Decode 服务日志与验收：
  `logs/runs/validation_native_prompt_20260826/restored_base2/`
- 本轮 SGLang profiler 原始 trace、benchmark 结果、独立服务日志和 whoami 验收：
  `logs/runs/sglang_profile_20260826/`
- 本轮 trace 解析工具：
  `tools/analyze_sglang_profile.py`

## 7. 后续可选优化（不影响本轮结论）

1. FP4 双端 8 组已完成且通过 native prompt；仍需针对其 8192 长输入下降定位 indexer kernel、显存带宽和 KV/attention 时间线。
2. 对 Waterfill 做真实共享专家热点/偏斜路由 workload；随机均匀 workload 只能证明额外成本，不能证明其设计收益。
3. 对 LPLB 做 refresh interval、static map 和 solver kernel 的定向 A/B，并采集 solver/all-reduce 时间。
4. 对 TBO 分离 Prefill-only 与 Decode request path，确认是否能让 Decode 未命中 TBO 时完全绕过额外调度成本。
5. 若需要进一步做硬件时间占比归因，可对各单项补采同口径 Nsight Systems profile；本轮已完成基线启动期 profile、基线运行期 `.nsys-rep`、运行期短测和 dmon 采样，四个单项已有完整 8 组 A/B、服务日志和源码路径证据，但没有将单项精确 kernel/通信占比冒充为已测事实。

本报告结论已闭合：四个单项均已在 8 组样例上完成有效性对比；未超过基线的技术给出
了可复核的 A/B、日志、源码和硬件证据边界。后续工作属于针对特定 workload 的进一步
优化，不改变当前推荐基线的结论。

## 8. OSL=8192 长输出压力补测记录

为检查长输出下当前主路径的稳定性，使用与 8 组矩阵相同的随机输入、并发和
`prefill+MegaMoE / decode+MegaMoE+DSpark+DeepGEMM` 配置，额外将 OSL 设为 8192。
该压力矩阵不是替代 OSL=1024 的正式 8 组 A/B，而是用于暴露长请求下的内存和
请求生命周期问题。

当前已完成且可用的三组如下：

| ISL | OSL | Concurrency | Completed | Req/s | Out tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 8192 | 1 | 10 | 0.042976 | 352.062 | 396.070 | 202.170 | 2.816 |
| 1024 | 8192 | 16 | 160 | 0.569005 | 4661.293 | 5243.954 | 273.467 | 3.252 |
| 1024 | 8192 | 256 | 2560 | 1.850915 | 15162.692 | 17058.028 | 3978.198 | 15.735 |

对应原始结果位于：
`logs/runs/prefill_megamoe_decode_megamoe_dspark_deepgemm_20260826_no_fp4_indexer/perf_osl8192/`。

### 8.1 C512 长输出失败证据

`ISL=1024, OSL=8192, C=512, N=5120` 首次运行在旧 decode 服务上发生大量
`ClientPayloadError`，服务端最终进入：

```text
token_to_kv_pool_allocator.get_cpu_copy
NotImplementedError: HiSparseC4DevicePool does not support get_cpu_copy
```

该结果无效，不计入吞吐表。重启 decode 并将 `mem_fraction_static` 从 0.8 调到
0.9 后，重跑达到 5120 个 HTTP 200 响应，整个过程中：

- `#retracted-req=0`；
- CUDA Graph 持续为 `True`；
- 没有 `Traceback`、`NotImplementedError`、400/500 或断连日志；
- 但最后一个请求停留在 `#running-req: 1`，从 05:28:37 起超过 5 分钟没有
  新的 decode 进度，GPU 利用率降至约 3%--26%，客户端 CPU 保持 100%。

因此该重跑也不能作为完整 C512 吞吐结果。已终止卡住的 benchmark 客户端，保留
日志；随后 decode 已使用同一配置重新启动，服务日志单独保存于：
`perf_osl8192/recovery_decode/services/decode_20260826_053437_pid1051095.log`。

重启后的冒烟验证已通过：向 Router `127.0.0.1:13784/generate` 发送“你是谁？”
得到正常 JSON 响应，`prompt_tokens=2`、`completion_tokens=64`，并返回
`spec_accept_rate=0.22` 等 DSpark 元数据；该请求的 `finish_reason=length` 是因为
主动设置了 64 个输出 token 上限，不是服务错误。

这组证据将问题从“单纯显存不足”进一步收敛为：长输出、高并发下的请求收尾/传输
生命周期路径存在死锁或未完成状态；mem0.9 可以避免原先的 KV offload 异常，但还
不能证明 C512 长输出已经稳定。后续正式对比仍以 OSL=1024 的 8 组矩阵为准，长
输出 C512 只作为稳定性缺陷记录，不能与有效 baseline 比较。

## 9. 当前完成度审计与领导汇报结论

### 9.1 数据覆盖审计

- 基线、Waterfill、LPLB 动态路径和 TBO 的原始结果目录各包含 8 个正式样例文件，
  每组的 `completed` 均等于 `10×concurrency`；FP4 修复版本的 8 组由 6 个正式矩阵
  文件和 2 个聚焦文件组成，8 组均为完整成功请求。
- 所有有效对比均固定在 Prefill+MegaMoE / Decode+MegaMoE+DSpark+DeepGEMM，Decode
  使用 CUDA Graph；单项开关只在 Decode 端增加，FP4 indexer 是必须双端一致的例外。
- 每次重启后的服务验收均使用原生 DSV4 prompt，不使用裸 `/generate` 文本；最近一次
  恢复服务的 whoami 结果为 HTTP 200、`WHOAMI_VALID=True`。

### 9.2 可用于决策的结论

1. FP4 indexer 在 `1024/C512` 有 +7.68% 的局部收益，但完整 8 组中只有这一组超过
   基线；`8192/C256` 和 `8192/C512` 分别下降 11.08% 和 11.27%，不能作为默认全局
   开启项。主要风险是 indexer/KV layout 适配和额外 indexer 计算、显存访问路径。
2. Waterfill 在当前随机均匀路由下没有可回收的共享专家热点；完整矩阵和 dmon A/B
   均显示其额外 count/materialize 路径未转化为吞吐收益。只有在真实共享专家热点或
   偏斜路由 workload 上，才有充分依据继续评估它。
3. LPLB 动态路径的主要代价是每轮 count、EP collective 和 LP/IPM solve；静态回退已
   将 `8192/C1` 从 -24.52% 改善到 -2.10%，证明优化方向正确，但尚未稳定超过基线。
4. TBO 在 C512 只获得接近噪声的收益，低并发因 batch split、同步和 DSpark target-
   verify 不能完全进入 TBO overlap 而明显下降；仅设置 batch gate 会破坏 Decode
   CUDA Graph 约束，不能作为最终方案。
5. 基线运行期 Nsight 证据确认 Graph replay、跨卡 AllGather 和大量 stream synchronize
   同时存在；dmon 证据确认 GPU 显存并非主要瓶颈。因各单项尚未完成同口径运行期
   `.nsys-rep`，本报告对单项的精确 kernel/通信时间占比保持保守表述，不把源码路径
   推断写成硬件测量结论。

因此当前推荐用于生产对比的仍是无额外单项开关的基线；若要继续争取收益，应按
“先构造偏斜路由验证 Waterfill，再做 LPLB 静态 map 与通信/solver 融合，再实现 TBO
普通 Graph/TBO Graph 双路 capture”顺序推进。FP4 indexer 则应先解决长输入高并发的
indexer/KV 带宽与 layout 成本，再决定是否按输入长度和并发做条件启用。
