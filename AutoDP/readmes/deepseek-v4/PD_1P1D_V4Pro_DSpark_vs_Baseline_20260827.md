# DeepSeek-V4-Pro 旧版适配 PD 1P1D：DSpark 与原生 baseline 对比

日期：2026-08-27  
源码：`/data/ssd2/sglang_v0.5.16_pr32281_fix7_repro_20260823`（不是最新 SGLang）  
模型：`/data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Pro`  
拓扑：b200-4 prefill（TP8/DP1/EP1）+ b200-3 decode（TP8/DP8/EP8）+ Mooncake PD 分离。

## 1. 本轮配置

DSpark 配置：prefill/decode 均使用 `--moe-a2a-backend megamoe --moe-runner-backend deep_gemm`；prefill 和 decode 均开启 DSpark，`dspark_block_size=5`、target layers=`58,59,60`、Markov rank=`512`。decode 使用 CUDA Graph buckets `1 2 4 8 16 32 64 128`。

baseline 配置：不使用 DSpark、不使用 MegaMoE；prefill 使用 `--moe-runner-backend flashinfer_mxfp4`，decode 使用 `--moe-a2a-backend none --moe-runner-backend flashinfer_mxfp4`，decode 同样开启 CUDA Graph。两侧其余 PD、模型、TP/DP、输入输出长度、seed 和请求速率保持一致。

本轮发现旧版 `forward_mla.py` 直接导入 `sgl_kernel.bmm_fp8`，当前环境不存在该符号，已改为导入失败时 fallback 到 `flashinfer.bmm_fp8`。原文件备份为：

`/data/ssd2/sglang_v0.5.16_pr32281_fix7_repro_20260823/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py.before_pd_1p1d_bmm_fallback_20260827`

远端 b200-4 已同步 DSpark 专用 shard 64/65/66，并校验 SHA256；基础 shard 仍通过原模型路径映射。部署脚本：

- [dspark prefill](../logs/runs/v4pro_1p1d_oldadapt_20260827/prefill.sh)
- [dspark decode](../logs/runs/v4pro_1p1d_oldadapt_20260827/decode.sh)
- [baseline prefill](../logs/runs/v4pro_1p1d_oldadapt_20260827/baseline_prefill.sh)
- [baseline decode](../logs/runs/v4pro_1p1d_oldadapt_20260827/baseline_decode.sh)
- [DSpark 测试矩阵](../logs/runs/v4pro_1p1d_oldadapt_20260827/run_final_dspark_matrix.sh)
- [baseline 测试矩阵](../logs/runs/v4pro_1p1d_oldadapt_20260827/run_baseline_matrix.sh)

## 2. 正确性与完整性

两套服务均通过 `/generate` 的“你是谁”验证，HTTP=200，`WHOAMI_VALID=True`。DSpark 8 组均达到目标 completed 数、benchmark errors=0。baseline 前 7 组均完整且 errors=0；8192/C512 的 baseline 两次 rate=4 测试出现响应流截断和 `KVTransferError: Aborted by AbortReq`，不能宣称全请求成功。

## 3. DSpark 8 组结果

8192 高并发采用 rate=4，以避免无限速压测下的传输截断；其余组为无限速。单位：tok/s、ms。

| ISL | OSL | C | Rate | Completed | Req/s | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:| 
|1024|1024|1|inf|10/10|0.210|215.36|430.73|411.47|4.24|0|
|1024|1024|16|inf|160/160|2.298|2353.57|4707.14|681.22|5.86|0|
|1024|1024|256|inf|2560/2560|22.502|23042.32|46084.65|1144.24|9.41|0|
|1024|1024|512|inf|5120/5120|30.513|31245.59|62491.18|4863.46|10.62|0|
|8192|1024|1|inf|10/10|0.209|213.95|1925.54|383.61|4.30|0|
|8192|1024|16|inf|160/160|2.287|2341.69|21075.17|976.68|5.62|0|
|8192|1024|256|4|2560/2560|3.880|3972.92|35756.31|3721.92|5.94|0|
|8192|1024|512|4|5120/5120|3.969|4064.27|36578.45|7461.37|5.96|0|

原始 JSONL 位于：`../logs/runs/v4pro_1p1d_oldadapt_20260827/results/dspark_megamoe_deepgemm_sync_hidden_fix/`。同名旧轮次或中断轮次不作为最终值，最终值取完整 rerun 文件。

## 4. 原生 baseline 8 组结果

| ISL | OSL | C | Rate | Completed | Req/s | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1024|1024|1|inf|10/10|0.048|49.39|98.77|479.66|19.80|0|
|1024|1024|16|inf|160/160|0.653|669.14|1338.27|792.05|23.02|0|
|1024|1024|256|inf|2560/2560|5.698|5834.69|11669.39|1380.87|42.14|0|
|1024|1024|512|inf|5120/5120|9.835|10070.99|20141.99|2175.48|47.92|0|
|8192|1024|1|inf|10/10|0.048|49.40|444.61|396.56|19.87|0|
|8192|1024|16|inf|160/160|0.641|656.35|5907.13|1309.28|22.80|0|
|8192|1024|256|4|2560/2560|3.414|3495.94|31463.46|29114.91|31.97|0|
|8192|1024|512|4|2409/5120|1.796|1838.66|16547.96|33798.79|31.80|2711，失败|

baseline 原始 JSONL 位于：`../logs/runs/v4pro_1p1d_oldadapt_20260827/results/baseline_flashinfer_none_ab/`。8192/C512 原始失败日志保留在对应 `.log`；第二次重测使用 `isl8192_osl1024_c512_n5120_rerun2_rate4`，在确认同类错误持续后中止，不能作为性能结果。

## 5. 有效性能对比（仅成功的 7 组）

相对 baseline，DSpark 的 Total tok/s 变化如下：1024/C1 **+336.1%**、C16 **+251.7%**、C256 **+294.9%**、C512 **+210.3%**；8192/C1 **+333.1%**、C16 **+256.8%**、C256 **+13.6%**。DSpark 的 TPOT 在所有可比组更低，但 baseline 使用原生 FlashInfer-MxFP4、无 MegaMoE，本表反映的是完整配置差异，不是 DSpark 单项增益。

8192/C512 不做性能结论：baseline 原生路径在该压力下出现大规模客户端响应截断/AbortReq，而 DSpark 同条件完成 5120/5120；这应作为 baseline PD 稳定性瓶颈单独修复后再做严格 A/B。

## 6. 日志与复现

- DSpark decode 日志：`../logs/runs/v4pro_1p1d_oldadapt_20260827/decode/server.log`
- DSpark prefill 日志：远端容器内 `../logs/runs/v4pro_1p1d_oldadapt_20260827/prefill/server.log`
- baseline decode 日志：`../logs/runs/v4pro_1p1d_oldadapt_20260827/baseline/services/decode/server.log`
- baseline prefill 日志：远端容器内 `../logs/runs/v4pro_1p1d_oldadapt_20260827/baseline/services/prefill/server.log`
- DSpark whoami：`../logs/runs/v4pro_1p1d_oldadapt_20260827/validation/whoami_dspark_20260827.json`
- baseline whoami：`../logs/runs/v4pro_1p1d_oldadapt_20260827/baseline/whoami_native_prompt_rerun.json`

复现时必须使用上述旧版源码、两侧对应启动脚本、同一 router、相同 tokenizer 参数；请求 prompt 使用旧版 SGLang 的 DSV4 native encoder 组装，不能直接假设 HF `chat_template`。

## 7. 结论

本轮已完成“旧版适配 DSpark 8 组 + 原生 baseline 8 组”的部署与测试闭环。DSpark 8 组功能正确且完整；baseline 7 组可用于性能对比，第 8 组暴露出原生 baseline 在 8192/C512 PD 高并发下的传输稳定性问题，应在修复响应截断后补测，不能用不完整数据支持性能结论。

## 8. 1024/C512 的 TTFT 说明

DSpark 的 1024/C512 结果中 Mean TTFT=4863.46 ms，Median=4429.98 ms，P90=5253.62 ms，P99=14333.82 ms；因此高 TTFT 不是少数离群请求单独造成的，而是 `request-rate=inf` 下 C512 突发提交后，prefill 排队、PD hidden-state/KV transfer 和 decode admission 等等待时间的总和。该组 Mean TPOT=10.62 ms、P99 TPOT=17.01 ms，decode 逐 token 计算本身没有同量级异常。

该组 `concurrency=480`、`max_concurrent_requests=559` 也说明无限速调度下存在明显的在途请求峰值/调度 overshoot；C256 的 TTFT=1144.24 ms，而 C512 上升到 4863.46 ms，表现为高并发排队的非线性增长。服务端无 Decode transfer failed、Client disconnected 或模型错误。若需要评估纯首 token 延迟，应使用固定 request rate、预热后重复多轮，并同时报告 median/P90/P99，不能只看 Mean TTFT。

baseline 的 1024/C512 Mean TTFT=2175.48 ms 确实低于 DSpark 的 4863.46 ms，但不能简单归因于“DSpark 更快所以排队更严重”。两者都是 `request-rate=inf`，实际请求到达时间线不同：baseline 总运行 520.59 s、request throughput=9.835 req/s；DSpark 总运行 167.80 s、request throughput=30.513 req/s。更重要的是，DSpark 的首 token 路径还包含 draft hidden-state 传输/同步和 speculative decode admission，而 baseline 是直接 prefill→KV transfer→decode；这些路径差异在 C512 阈值下可能放大首 token 等待。

分位数也表明 baseline 的低均值主要来自中位数：baseline Median/P90/P99 TTFT=1138/3096/17093 ms，DSpark 为 4430/5254/14334 ms；baseline 的 P99 并不更好。现有日志只能证明两套路径在 `inf+C512` 下的调度和同步行为不同，不能单凭吞吐量证明某一方的排队一定更严重。要严格归因，应让两套配置使用相同固定 request rate（例如 rate=4）重新测试 1024/C512，并分别采集 prefill 排队、KV transfer、hidden-state sync 和 decode admission 的阶段时间。
