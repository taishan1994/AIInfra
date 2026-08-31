# PROBLEM 排查与解决记录

基线源码：`/sgl-workspace/sglang`

基线 commit：`fdebc93`（2026-07-31 开始排查）

独立备份根目录：
`/data/ssd2/gongoubo/single_node/backups/sglang_20260731_064316`

每项技术在对应子目录保存了修改前的 commit、工作树状态和 binary diff：

- `dspark/`
- `tbo_sbo/`
- `mtp/`
- `fp4_indexer/`
- `hicache/`
- `waterfill/`

## 问题清单

按 `readmes/PROBLEM.MD` 顺序逐项处理：

1. prefill + HiCache 初始化被 SIGKILL/OOM；
2. decode + DSpark 的 `batch.spec_info is None` 崩溃；
3. HiSparse 缺少 decode baseline 对照；
4. LPLB 与 DeepSeek-V4 的能力不兼容；
5. MegaMOE + Waterfill 未体现收益；
6. TBO/SBO 使用了 DeepEP normal dispatch，需要验证 low-latency；
7. MTP 指标异常且日志有错误；
8. FP4-indexer 在 ISL=8192 的高并发 case 未完成。

## 处理状态

- [x] HiCache（ratio=0.01 初始化验证通过；原 ratio=2.0 为 cgroup OOM）
- [x] DSpark（首个 decode batch 空 spec_info 已修复；PD decode 服务已成功启动）
- [ ] HiSparse baseline 对照
- [x] LPLB 兼容性记录
- [ ] Waterfill
- [x] TBO（纯 TBO + DeepEP low-latency + CUDA Graph，8/8 完成；成功方案见 6.2）
- [ ] SBO（low-latency 的 MXFP4 masked-runner/layout 兼容性仍未解决，见 6.1）
- [ ] MTP（原始结果需重新确认收尾日志）
- [x] FP4-indexer（8/8 完成；`8192/1024/c512` 已修复并重测通过，详见 5.1）

## 已完成实验结果索引

以下是此前已经实际跑完、并已纳入 `PD_COMPLETED_REPORT_20260803.md` 的结果；本文件
记录对应问题、配置约束和有效性结论：

- PD baseline：`logs/flash_baseline_ibfix/results`
- Prefill MegaMoE：`logs/flash_prefill_megamoe_retry2/results`
- Prefill DeepEP EP4：`logs/flash_prefill_deepep_ep4/results`
- Decode MegaMoE：`logs/flash_decode_megamoe/results`
- Decode MTP（已有矩阵；纯 MTP 的 DeepEP-LL 兼容性结论见第 4 节）：
  `logs/flash_decode_mtp_only/results_20260802`
- Decode 纯 TBO：`logs/runs/tbo_20260802_150500/results`
- Decode 纯 FP4 Indexer（8/8，含本次补跑）：
  `logs/flash_decode_fp4_indexer_only/results_20260802` 和
  `logs/runs/fp4_indexer_retest_8192_c512_stride_and_64k_registration_fix_20260803`

组合实验和未通过/未完成项目仍按下文单独标注，不能与纯技术消融结果混用：
Waterfill、HiSparse baseline、TBO/SBO 的未通过 low-latency 变体、DSpark 正式 PD
矩阵、LPLB 以及 HiCache 正式 workload。

## 1. HiCache

### 定位

容器 cgroup 的 `memory.max` 为 `9,119,289,344` bytes（约 8.5GiB），而宿主机
`MemAvailable` 约 2.1TiB。原配置 `--hicache-ratio 2.0` 按每个 TP rank
分配约 81.93GB SWA + 100.03GB C4 host pool，4 个 rank 合计约 727.84GB。
因此 `psutil.virtual_memory()` 看到的宿主机可用内存不能代表当前容器可用内存，实际分配过程中触发了 cgroup OOM，scheduler 收到 SIGKILL(-9)。

内核日志确认：`constraint=CONSTRAINT_MEMORY_POLICY`，`task_memcg=docker-...scope`，
并记录了 `Out of memory: Killed process ... sglang::schedul`。

### 验证

未修改 SGLang 源码；新增独立重试脚本：
`flash_prefill_hicache_retry.sh`，仅将 `--hicache-ratio` 改为 `0.01`，日志在
`logs/flash_prefill_hicache_retry/prefill.log`。

该配置已完成：权重加载、DeepSeek V4 host pool 分配、Unified RadixTree 初始化、
PD warmup，并输出 `The server is fired up and ready to roll!`。实际分配约为每 rank
0.41GB SWA + 0.50GB C4 + 其他 sidecar，未再触发 OOM。

结论：HiCache 代码路径本身可用，失败根因是运行容器的内存上限；后续实验应使用
不超过容器预算的 ratio，不能直接恢复 `2.0`。

## 2. DSpark

### 定位

失败栈位于 `spec_utils.spec_prepare_for_decode()`：DSpark 首个 decode batch
还没有 draft result，因此 `batch.spec_info is None`，但公共 dispatcher 直接调用
`batch.spec_info.prepare_for_decode(batch)`。DSpark worker 的 `_forward_decode()`
本来已经支持 None 并创建 idle `DFlashDraftInputV2`，说明这是 scheduler 调用时机
不一致造成的初始化漏洞，不是 SPS table 缺失。

### 修改与备份

修改文件：`/sgl-workspace/sglang/python/sglang/srt/speculative/spec_utils.py`

修改前备份：
`backups/sglang_20260731_064316/dspark/spec_utils.py.before`

PD 重测又发现首个 overlap decode step 的 `future_indices=None` 被
`overlap_utils._resolve_spec_extras()` 无条件解引用；该文件也已单独备份：
`backups/sglang_20260731_064316/dspark/overlap_utils.py.before`，并加入 None-safe
返回逻辑。

随后确认非 overlap PD 首步没有 FutureMap relay，DSpark worker 仍拿到空
`bonus_tokens`。`dspark_worker_v2.py` 已单独备份到
`backups/sglang_20260731_064316/dspark/dspark_worker_v2.py.before`，现会在 active
首步用 `batch.input_ids` 初始化 draft anchor；idle DP rank 仍使用空输入。相关脚本
临时采用 `--disable-overlap-schedule`，以隔离已知 overlap-only idle padding 问题。

修复后重测中，PD router 已改为 loopback 并能发现 prefill/decode；但 DSpark 首个
PD 请求仍触发 DP idle participation 的负 padding（开启 DP attention），而关闭 DP
attention 后服务长时间停在初始化，均未产生有效性能数据。该轮不覆盖旧 matrix；
问题栈、启动参数和 SPS table 均保留，后续继续从兼容的 DSpark 控制配置排查。

补丁在 DFlash-family dispatcher 中检测 None，创建
`DFlashDraftInputV2.create_idle_input(device=batch.device)` 后再调用
`prepare_for_decode()`。

### 验证

- `python3 -m py_compile .../spec_utils.py` 通过；
- `test/registered/unit/spec/test_dflash_overlap_hostsync.py`：7 passed。

集成 smoke：`flash_decode_dspark.sh` 在 07:08:50 输出
`The server is fired up and ready to roll!`，四个 DP/EP rank 均完成主模型和 DSpark
draft 权重加载；随后 `/health` 返回 200。直接调用 decode `/generate` 被服务以 400
拒绝是预期行为，因为 PD 分离请求缺少 bootstrap room id，并非 spec_info 崩溃。
随后按官方 `sglang.benchmark.dspark_sps_profiler` 新增独立 profiling server，使用
固定 tokenized prompt、`SGLANG_RAGGED_VERIFY_MODE=static`、
`SGLANG_DSPARK_ENABLE_SPS_RECORD=1` 和 `SGLANG_SIMULATE_ACC_LEN=1.0` 完成 5 档
per-rank sweep（1/8/32/64/128）。拟合结果已写入：
`logs/flash_decode_dspark/dspark_sps.json`，self-check passed；正式脚本已加入
`--speculative-dspark-sps-table-path` 指向该文件。profiling server 与正式 PD
server 分开，避免 PD bootstrap 请求影响 SPS 测量。

## 3. HiSparse baseline

新增独立脚本 `flash_decode_hisparse_baseline.sh`，使用 DeepEP low-latency + HiSparse，
不使用 MegaMOE，不覆盖原结果。启动时发现 HiSparse C4 host pool 按当前配置需要约
67.2GB；容器 `memory.max` 约 8.5GiB，因此在 CUDA Graph 和关闭 CUDA Graph 两种
配置下都无法完成 ready。已将独立脚本的 ratio 降为 1 作为下一次容量验证；该限制
属于 host cgroup 预算，不能把未启动的 baseline 标成通过。

补充对照：`logs/flash_decode_hisparse/decode.log`（MegaMOE）并没有跳过该分配，
同样记录了每个 rank 约 66.63GB 的 `dsv4_hisparse_c4` host pool，并最终 ready；
baseline 日志则记录约 67.21GB，随后未 ready。两者差异不是“MegaMOE 不需要 host
pool”，而是后端路径和当时的主机/cgroup 内存状态不同；baseline 还额外经过
DeepEP+MxFP4 路径（此前 CUDA Graph 捕获曾触发 3D scale-layout assertion）。因此
MegaMOE 的一次成功不能证明 baseline 的 host pool 需求较小，需用相同 ratio、相同
图开关和清空进程后的环境重新做公平对照。

ratio=1 重测时 host pool 降为约 43.81GB/rank，但 warmup 仍未 ready。对
`mxfp4_flashinfer_trtllm_moe.py` 做过一次独立备份的前导维度展平尝试：原 3D
`hidden_states_scale` assertion 消失，但随后 DeepEP routing 报
`Routing kernel requires at least one input parameter`，说明 top-k dispatch 元数据
不能这样直接展平；该尝试已恢复备份，不作为正式修复。

## 4. LPLB

`flash_decode_lplb.sh` 已启动验证。DeepSeek-V4 在初始化时明确报：
`DeepseekV4ForCausalLM does not support --ep-dispatch-algorithm lp`；当前
LPLB 仅验证了 DeepSeek-V2/V3/V3.2 等模型，V4 因 DP-attention 下空 token
路径可能破坏 EP all-reduce 而被主动拒绝。因此该配置属于模型能力不兼容，
不是内存或启动参数问题，未继续强行修改源码绕过。

## 5. Waterfill

通过 router `13784` 做 PD benchmark 时，原 Waterfill decode 在第一个真实请求的
idle DP batch 中调用 FlashInfer `mxfp8_quantize`，SM100 返回
`cudaErrorInvalidValue`，导致 scheduler_0 退出、decode 端口消失；之前直连
`30001` 的结果也因缺少 bootstrap room id 而无效。已独立备份并增加空 batch
快速返回，重启后 `/health=200`，`ISL=1024/OSL=1024/concurrency=1` smoke
通过。完整矩阵需在该修复上重新跑，旧结果不计入。

Waterfill decode 已按 PD 约束改为 `--deepep-mode low_latency`。此模式能恢复
CUDA Graph，但 warmup 暴露 DeepEP LL + MxFP4 的 3D scale-layout 错误：
`hidden_states_scale (65,2048,128)`，期望首维 `133120`。因此之前 normal 模式
下约 4 tok/s 只是 eager（normal 会主动关闭 CUDA Graph），不能作为正式性能结果；
LL 版本需先完成该 MxFP4 layout 兼容修复。

进一步尝试了“仅对 MxFP4 kernel 输入 flatten、combine 前恢复形状”，并修正了
DeepEP combine 所需的 int64 top-k ids。CUDA Graph 捕获阶段可绕过 layout/dtype
错误，但真实请求在 TRT-LLM routing 报 `Routing kernel requires at least one
input parameter`；关闭 graph 后同样复现。因此当前 LL+Waterfill+MxFP4 仍未通过，
相关源码改动保留在独立备份 `backups/sglang_20260731_064316/waterfill_idle/`，
不能把这轮结果计入性能数据。

2026-08-02 继续定位确认，DeepEP low-latency dispatch 的激活是按本地 expert
分组的 `[65, 2048, hidden]` staging buffer；DeepEP combine 也明确要求输出首维等于
`num_experts / num_ranks`。因此把它展平或裁成 active-token 行再交给
`trtllm_fp4_block_scale_routed_moe` 在语义上不成立：该 kernel 期望 token + top-k
routing 输入，而 LL buffer 已完成 expert dispatch。源码中已有正确形态的
`deepep + flashinfer_cutedsl` masked-MoE 路径（消费 `masked_m`），但对当前
DeepSeek-V4-Flash checkpoint 做独立启动 smoke 时在参数校验阶段被拒绝：模型整体
量化配置为 `fp8`、routed experts 为自定义 MXFP4，且没有 `nvfp4_moe_meta`；现有
CuteDSL 权重准备仅支持 modelopt/hybrid NVFP4。因此 Waterfill+DeepEP-LL 当前需要
新增“DSV4 MXFP4 权重 -> masked MoE runner”的后端适配，不能靠 reshape 修复。
独立验证脚本为 `flash_decode_waterfill_cutedsl.sh`，原 Waterfill 脚本未被覆盖。

## 6. TBO/SBO

原始脚本 `flash_decode_tbo.sh` / `flash_decode_sbo.sh` 固定使用
`--deepep-mode normal`。已新增 low-latency 版本，不覆盖原脚本。

第一次 TBO low-latency 启动在 CUDA graph 捕获 bs=256 时失败，DeepEP 报：
`x.size(0) <= num_max_dispatch_tokens_per_rank`，因为当前环境变量默认值为 128。
这发生在 graph capture 阶段，不是 TBO overlap 计算本身。

新增 `flash_decode_tbo_low_latency_1024.sh`，设置
`SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024` 后重试；SBO 同样准备了
`flash_decode_sbo_low_latency.sh`，待 TBO smoke 结果后按同一容量配置验证。

随后确认 MxFP4 量化方法对 DeepEP dispatch format 也缺少适配：它只读取
`dispatch_output.topk_output`，在 DeepEP LL/normal 中该字段不存在。已分别备份并
修改 `mxfp4_flashinfer_trtllm_moe.py`：兼容直接携带的 `topk_ids/topk_weights`，
并按 dispatch format 返回 `DeepEPLLCombineInput` / `DeepEPNormalCombineInput`，
保留 DeepEP combine 所需元数据。修改前文件位于 TBO/SBO 和 Waterfill 各自备份目录。

补丁后的连续验证又暴露了两个输入约束：LL 的 expert id 为 int64（已转换为
int32），部分 LL 激活为 FP8（已在量化前转 BF16）。再进一步将三维 staging buffer
直接展平会使 graph capture 按 262,144 token 申请约 12GiB 临时显存并 OOM，因此该
展平尝试已撤回；当前剩余问题是 LL staging/layout 与 TRT-LLM MxFP4 临时 workspace
的容量协调；TBO 已用稳定的 low-latency + CUDA Graph 方案完成。

### 6.2 TBO 已成功方案（2026-08-02）

纯 TBO 使用 `flash_decode_tbo_only.sh`，不启用 MegaMoE，关键配置为：

- `--moe-a2a-backend deepep`
- `--deepep-mode low_latency`
- `--enable-two-batch-overlap`
- `--cuda-graph-max-bs-decode 256`
- `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512`
- `--disable-radix-cache`
- Decode：GPU 4–7，TP4/DP4/EP4；Prefill：GPU 0–3，TP4/DP1/EP1

关键是将 DeepEP LL capacity 设为 512、CUDA Graph 最大 batch 设为 256；默认
capacity=128 会在 graph capture 报 `x.size(0) <= num_max_dispatch_tokens_per_rank`。
该方案完整跑完 `1024/1024`、`8192/1024` 的并发 `1/16/256/512` 八组矩阵，
decode 日志持续显示 `cuda graph: True`，结果目录为
`logs/runs/tbo_20260802_150500/results`。

代表性结果：`1024/1024/c512` 为 17,320.93 Out tok/s、20.42 ms TPOT；
`8192/1024/c512` 为 7,335.12 Out tok/s、14.62 ms TPOT。对应 prefill/decode/router/
benchmark 日志均保存在 `logs/runs/tbo_20260802_150500/`。

注意：该成功方案是 TBO（two-batch-overlap），不能直接迁移为 SBO。SBO 的
single-batch-overlap hook 要求 DeepEP LL 三维 expert-padded layout，而当前
自定义 MXFP4 runner 尚不支持该 layout。

## 4. MTP 初步日志分析

原 MTP benchmark 的 `Traceback` 出现在完整 matrix 完成、手工停止 server 的
收尾阶段，栈停在各 DP rank 等待 broadcast/ZeroMQ，不是请求 forward 错误。
`accept_len` 日志也显示 EAGLE/MTP 正常工作（约 0.6 的 acceptance）。
不过由于用户要求重新尝试，正式结论暂不提前标记完成；后续将用可正常收尾的
重跑记录覆盖原报告。

2026-08-02 确认原 `flash_decode_mtp.sh` 实际是 `MegaMOE + MTP` 组合实验，不能
作为 MTP 单项消融。已新增 `flash_decode_mtp_only.sh`，严格基于 DeepEP
low-latency baseline，仅增加 EAGLE 参数且保持 decode CUDA Graph。MTP verify
每请求 4 token，因此将 DeepEP capacity/graph max bs 配成 1024/256 后可通过容量
校验，但 graph 首次 MoE forward 随即失败：`DeepEPLLDispatchOutput` 没有
`topk_output`。这与 TBO/SBO/Waterfill 暴露的是同一 DSV4 MXFP4 runner 对
DeepEP-LL dispatch format 的缺失。故当前纯 MTP 未通过；MegaMOE+MTP 的 8 组
结果只保留为组合实验，不计为纯 MTP 成绩。

## 5. FP4-indexer 初步定位

8192/c256 的失败是 PD KV transfer：decode 日志为
`Failed to get kvcache from prefill instance, it might be dead`，随后大量请求被
`AbortReq` 取消。对应时间点 decode 侧并非 FP4 kernel traceback；同一轮的 prefill
日志显示服务后来在 18:27 才 ready，说明该轮存在 prefill 未 ready/被重启的协调
问题。后续补跑必须先确认 prefill `/health` ready，再启动 decode 和该 case。

### 5.1 8192/c512 空流与 Mooncake descriptor 越界：已修复并通过

后续稳定复现确认，TTFT=0 并不是 FP4 Indexer 的真实性能，而是 Mooncake KV
传输失败后 router 返回 HTTP 200 空流，benchmark 仍按固定 `output_len=1024` 计数。
失败发生在负载约 456 秒、KV allocator 使用到高位页索引时。

根因包含两个相互独立的 Mooncake 元数据/注册问题：

1. DSV4 decode 的 62 个 KV buffer 并非同一 page stride，实际 item length 为
   `1728/4352/37440`。Mooncake 协议原来只发送一个标量 `dst_kv_item_len`，导致其他
   buffer 用错误步长计算目标地址。现已与 NIXL 路径一致，逐 buffer 序列化
   `dst_kv_item_lens`，并在 K/V、PP 切分后分别使用对应的源/目标步长。
2. Mooncake 以 64 KiB 粒度解析 CUDA segment descriptor。KV allocation 的精确尾部
   不一定为 64 KiB 对齐，最后一个合法 token slot 的 descriptor 查询因此可能越过
   注册尾部。现将 KV buffer 的注册长度向上对齐到 64 KiB。

修改文件：
`/sgl-workspace/sglang/python/sglang/srt/disaggregation/mooncake/conn.py`。
独立备份：
`conn.py.backup_per_buffer_item_lens_20260803`、
`conn.py.backup_descriptor_64k_alignment_20260803`。

修复后 `8192/1024/c512/n5120` 完整通过：5120/5120 请求成功，总生成 token
5,242,880，可重分词 token 5,242,637，Mean TTFT 53,638.28 ms，Mean TPOT
14.37 ms，Out tok/s 7,326.22，Total tok/s 65,935.96。5120 个生成文本全部非空，
三类服务日志无 descriptor/session/transfer 错误。结果目录：
`logs/runs/fp4_indexer_retest_8192_c512_stride_and_64k_registration_fix_20260803`。

随后补跑此前无效的 `8192/1024/c256/n2560`：2560/2560 请求成功，逐请求
`output_lens/ttfts/generated_texts` 均为 2560 条且文本全部非空，Mean TTFT
20,947.57 ms、Mean TPOT 15.19 ms、Out tok/s 6,873.48、Total tok/s 61,861.32，
服务日志无 descriptor/session/transfer 错误。结果目录：
`logs/runs/fp4_indexer_retest_8192_c256_fixed_20260803`。

## 6.1 SBO 继续排查记录（2026-08-03）

按“必须使用 CUDA Graph”的要求继续验证了纯 SBO：

1. `flash_decode_sbo_low_latency.sh` 首次在 graph capture 失败，原因是本机
   DeepEP `Buffer.low_latency_combine()` 不接受 SGLang 传入的 `overlap` 参数。
   已备份并加入按运行时签名过滤不兼容参数的适配：
   `deepep.py.backup_sbo_overlap_api_20260803`。
2. 补齐 `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024` 后通过容量断言。
3. MXFP4 TRT-LLM runner 原先只读取 normal dispatch 的 `topk_output`；LL dispatch
   提供 `topk_ids/topk_weights`，已加入 carrier 适配，备份为
   `mxfp4_flashinfer_trtllm_moe.py.backup_sbo_ll_dispatch_20260803`。
4. 继续修复 LL 的 int64 top-k ids（转 int32）和 FP8 activation（转 BF16），备份分别为
   `mxfp4_flashinfer_trtllm_moe.py.backup_sbo_ll_topk_int32_20260803`、
   `mxfp4_flashinfer_trtllm_moe.py.backup_sbo_ll_fp8_20260803`。随后 graph capture
   明确暴露 LL expert-padded layout：`hidden_states` 为
   `[experts, capacity, hidden]`，而现有 TRT-LLM MXFP4 runner 需要 token-major
   routing；简单 flatten 会放大临时 workspace，且缺少按 `masked_m` 重建 top-k 的
   语义，不能作为正确修复。
5. 切回 normal dispatch 并强制 `--cuda-graph-backend-decode full` 也不能实现 SBO：
   SBO hook 要求三维 LL layout，normal 的二维 hidden state 触发
   `compute_overlap_args: not enough values to unpack`。
6. 最后验证 `flashinfer_cutedsl` masked runner，服务初始化直接拒绝：
   `Invalid quantization 'None'`；当前 DeepSeek-V4 checkpoint 是自定义 MXFP4/FP8，
   不满足 CuteDSL 只支持 `modelopt_fp4/modelopt_mixed` 的前置条件。

本轮日志目录：
`logs/runs/sbo_retry_overlap_compat_20260803`、
`logs/runs/sbo_retry_overlap_compat_capacity_20260803`、
`logs/runs/sbo_retry_ll_topk_compat_20260803`、
`logs/runs/sbo_retry_ll_topk_int32_20260803`、
`logs/runs/sbo_retry_ll_fp8_bf16_20260803`、
`logs/runs/sbo_normal_graph_20260803`、
`logs/runs/sbo_cutedsl_smoke_20260803`。

结论：SBO 调度/overlap 本身已定位，当前阻塞是 DeepSeek-V4 自定义 MXFP4 权重没有
兼容 DeepEP-LL expert-padded layout 的 masked MoE runner；不能把 normal dispatch
或关闭 CUDA Graph 的结果冒充 SBO 完成。待补齐该 runner 后即可沿用现有脚本继续跑
完整 8 组矩阵。

## 6.2 DSpark 单机 1P1D 适配记录（2026-08-03）

DSpark draft CUDA-graph 首次请求曾在 `cuda_graph_buffer_registry.py` 中因
`ForwardBatch.num_token_non_padded=None` 崩溃。根因是 draft runner 的 EP=1 不会
初始化该 device tensor，但 graph registry 仍启用了该 slot。已在
`_num_token_non_padded_post_fill` 中回退到 `num_token_non_padded_cpu`，该字段也缺失
时再使用本次 replay 的 `ctx.padded_num_tokens`，然后按 attention-TP rank 做切分。
修改前备份：
`backups/sglang_20260803_0930_dspark_num_token_non_padded_fix/`。

修复后小批量 SPS profile（全局 bs=4，每 rank 1）完成 408 个稳定步骤，匹配率 1.00，
未再出现 `NoneType` 或 scheduler 崩溃；后续高批量 profile 曾被外部服务生命周期
SIGTERM 中断，日志中的 detokenizer `exit -15` 不是源码异常。正式 table 需使用不受
15 分钟 timeout 影响的 detached 服务重新生成后，再启动完整 1P1D 矩阵。

## 6.3 DSpark graph 稳定性与 draft graph 对照（2026-08-04）

1. `num_token_non_padded` post-fill 不再读取 GPU tensor 的 `.item()`；优先使用
   `num_token_non_padded_cpu`，缺失时使用 graph replay 的 padded token 数，避免在
   replay 前同步并暴露异步 CUDA 错误。为规避本机 CUDA 13/PyTorch 的
   `torch._foreach_copy_` 首请求非法访问，registry 当前使用逐 tensor `copy_`。
2. 修复 `get_parallel` 导入路径为 `sglang.srt.runtime_context`。修改前均保存在
   `backups/sglang_20260804_1p1d_dspark_*` 对应目录。
3. 实测 graph 上限 256 时，即使关闭 draft graph，首请求仍在 DSV4 draft metadata
   Triton kernel 报 `illegal memory access`；因此问题不是 draft graph 开关，而是大
   graph bucket 捕获后的共享 CUDA 状态/缓冲区路径。graph 上限 1 和 16 均可完成
   1024/1024/c1/n10，target 与 draft 日志均显示 `cuda graph: True`。
4. 上限 16 的实测结果：Out tok/s 307.20、Total tok/s 614.40、Mean TTFT
   337.23 ms、Mean TPOT 2.93 ms；无 scheduler/Traceback/非法访问。故
   `flash_decode_dspark.sh` 默认 `CUDA_GRAPH_MAX_BS` 暂设为 16；超过 16 的 batch
   走 eager，优先保证 256/512 并发矩阵可持续运行。若后续升级 CUDA/PyTorch，可再
   逐级验证 32/64/128/256。

5. 并发 16 首次运行暴露 `ScheduleBatch.merge_batch()` 将 `spec_info=None` 合入
   DFlash 状态的问题，报 `AttributeError: ... reserved_seq_lens_cpu`。现已在
   `schedule_batch.py` 对 speculative 状态不一致时清空旧状态，由 DSpark worker
   对合并后的 batch 重新 seed。备份：
   `backups/sglang_20260804_1p1d_dspark_spec_merge/`。修复后
   1024/1024/c16/n160 成功，Out tok/s 1819.70、Total tok/s 3639.40、Mean TTFT
   714.48 ms、Mean TPOT 7.87 ms。

## 6.4 DSpark PD hidden transfer（PR #31466 对齐，2026-08-04）

1. 按 PR #31466 的首个核心提交迁移 `DSPARK_HIDDEN` 状态类型、prefill hidden
   capture、Mooncake 元数据/动态接收 buffer、decode hidden 组装，以及 DSpark
   draft 的 `prefill_tail_hidden_states` 注入链路；涉及文件均先备份到
   `backups/sglang_20260804_1p1d_dspark_pr31466_commit001/` 和
   `backups/sglang_20260804_1p1d_dspark_pr31466_followup/`。
2. 补齐 forward/PP proxy 的 aux hidden 回传和
   `SGLANG_DSPARK_DEBUG_MAIN_OUTPUT` 默认环境项；否则会分别出现 hidden rows
   不完整或 decode scheduler 属性错误。
3. 本机 launch_server 子进程未可靠继承 `CUDA_VISIBLE_DEVICES=4,5,6,7`，因此
   decode 显式使用 `--base-gpu-id 4`，Mooncake IB 映射同步改为 GPU 4–7；否则
   prefill/decode 会落到同一组 GPU 并 OOM。
4. 端到端冒烟已成功：router + 1P1D DSpark，1024 输入、16 输出、并发 1，
   Mean TTFT 412.53 ms、Mean TPOT 6.41 ms、Total tok/s 2005.95；日志确认
   prefill 捕获 `[40,41,42]` hidden，decode 为该请求分配 1024 行动态接收 buffer，
   未再出现 `missing_offsets` 或 scheduler 崩溃。

## 6.5 DSpark PD 合批状态修复（PR #33411 / #33098，对齐验证，2026-08-04）

1. 根因：PD 的 `process_prebuilt()` 为新请求构造 DSpark `spec_info` 时没有设置
   `future_indices`，也没有向 overlap `FutureMap` 做 `publish/stash`。已有 decode
   batch 携带 future state，而新 batch 不携带，`ScheduleBatch.merge_batch()` 每次都
   reset speculative state，随后接受率从约 0.4–0.9 降至 0.01–0.08。
2. 修复：DSpark PD draft input 构造完成后，在 overlap 模式下执行：
   `spec_info.future_indices = batch.req_pool_indices`、
   `future_map.publish(spec_info.future_indices, batch.seq_lens)`，以及
   `future_map.stash(... RelayPayload(bonus_tokens=last_tokens_tensor))`。该路径与
   PR #33411 的 DFLASH-family PD bootstrap 实现一致。修改前备份：
   `backups/sglang_20260804_1p1d_dspark_before_pr33411_future_map.py`。
3. 同步应用 PR #33098 的 DP/EP metadata 修复：draft `ForwardBatch` 补齐
   `original_global_num_tokens_cpu`、`num_token_non_padded` 和
   `num_token_non_padded_cpu`，修改前备份：
   `backups/sglang_20260804_1p1d_dspark_before_pr33098_metadata.py`。
4. 验证配置：1P1D、prefill TP4/DP1、decode TP4/DP4/EP4、DP attention、
   DeepSeek-V4-Flash、SPS archive table、CUDA graph 上限 16、输入 1024、输出
   256、并发 16。修复前 `Total tok/s=2732.54` 且接受率持续约 0.01；修复后
   `Output tok/s=1006.92`、`Total tok/s=5398.55`、Mean TTFT 1154.86 ms、
   Mean TPOT 11.31 ms，接受率保持约 0.4–0.9，日志不再出现
   `DSPARK_SPEC_MERGE_RESET` 或 scheduler 崩溃。
5. 关闭 overlap 的对照虽能暂时保持高接受率，但随后触发 DP attention CUDA
   index 越界，因此不能作为最终方案；正式配置保持 overlap 开启。

## 6.6 DSpark 语义 smoke 与 DeepSeek-V4 编码格式（2026-08-04）

直接向 `/v1/chat/completions` 发送“你是谁”时，返回了与问题无关的数学题；
这不是 DSpark hidden/acceptance 的证据。检查 checkpoint tokenizer 后确认
`chat_template=None`，而 DeepSeek-V4 的编码规范要求显式使用特殊 token：
`<｜begin▁of▁sentence｜><｜User｜>...<｜Assistant｜></think>`（chat/non-think）。

使用该规范向当前 1P1D DSpark 服务发送同一问题后，返回正常的 DeepSeek 自我介绍，
并且 `spec_accept_rate` 约 0.73，`cuda graph: True`。结果保存于：
`logs/flash_decode_dspark/results/who_are_you_dspark_proper_encoding.json`。

清理并行 benchmark 后的独立复测仍正常，`spec_accept_rate=0.76`，最终结果保存于：
`logs/flash_decode_dspark/results/who_are_you_dspark_proper_encoding_final.json`。

因此，之前的异常是请求没有使用 DeepSeek-V4 编码格式，不能据此判定 DSpark
生成链路错误。`who_are_you_dspark_finalcheck.json` 仅保留作错误请求对照。

## 6.7 DSpark 稀疏 CUDA Graph 与并发 smoke 结论（2026-08-04）

decode 端使用稀疏 graph bucket `1 2 4 8 16 32 64` 可正常启动，启动日志确认
target/draft 两侧均完成 capture，在线 decode 日志持续显示 `cuda graph: True`。
单请求复测成功（1024 输入、64 输出，Mean TTFT 341.69 ms、Mean TPOT 2.72 ms）。

并发 smoke 中 router、prefill、decode 均曾对所有请求返回 HTTP 200，且 hidden
receive buffer 与 prefill hidden source 成对出现；但 benchmark 客户端在部分批次
完成后未退出，最终由外层 timeout 断开连接，服务端随后记录
`Request is disconnected from the client side`/`AbortReq`。这不是 CUDA Graph 或
scheduler 崩溃；应将该类客户端超时结果视为无效，不写入吞吐汇总。正式矩阵仍需用
能够正常收集所有 response 的压测客户端重跑。
### 6.8 DSpark PD DP0 偶发卡住的排查结论（2026-08-04）

- `--base-gpu-id 4` 时 Mooncake 的 IB 映射必须使用物理 GPU ID：`{"4":"mlx5_4","5":"mlx5_9","6":"mlx5_10","7":"mlx5_11"}`；源码通过 `scheduler.ps.gpu_id` 查表，不能改成 0–3。
- `mlx5_4/9/10/11` 均为 `ACTIVE/LinkUp`，且拓扑分别对应 decode GPU4–7。
- 异常表现为 DP0 的 bootstrap/KV transfer 长时间不完成，客户端断开后 prefill/decode 才出现 `Aborted by AbortReq`；不是服务主动崩溃。
- 重新干净启动 decode 后，4 并发和 8 并发 `stream=false` 请求均全部 HTTP 200，未复现 DP0 abort。结果保存于 `logs/flash_decode_dspark/results/dspark_dp0_clean_restart_concurrency_20260804.txt`。
- 因此当前结论是“异常状态由旧 decode/Mooncake 进程状态触发的偶发卡住”，尚无证据支持修改 IB 映射；后续正式压测前必须使用干净重启并同步保存 prefill/decode/router 日志。
### 6.9 DSpark 1024 输入高并发 TPOT/TTFT 调优（2026-08-04）

- 原 DSpark decode 只捕获 graph16；c256/c512 大量 batch 走 eager，c256 Mean TPOT 为 46.24 ms。graph64 后降到 6.26 ms。
- PD hidden transfer 的默认 `SGLANG_DSPARK_PD_HIDDEN_BUFFER_POOL_LIMIT=16`、`...QUEUE_LIMIT=2` 会把高并发隐藏态发送限制在很小窗口。prefill 和 decode 两侧统一设置 buffer pool=512、queue=64、queue bytes=2 GiB 后，1024/1024/c256 Total throughput 从 31,750 提升到 41,994 tok/s，TTFT 从 9.35 s 降至 4.03 s。
- DSpark decode 显式 `chunked-prefill-size=1024` 会在 DP attention 下被降为 256；改为 4096 后实际为 1024，但本轮 c256 未带来额外收益。
- MTP 对照使用 MegaMoE + EAGLE + graph512，c256 为 48,113 tok/s、TPOT 9.07 ms；DSpark 当前 TPOT 已不差，但 PD/输入阶段仍约低 13%。
- DeePEP `low_latency` 启动前的 DSpark DP guard 已备份并解除；运行时发现 DSpark verify CUDA Graph 的 hidden tensor shape 不满足 DeepEP low-latency dispatch 上限/布局，graph capture 失败。DeePEP eager 可启动但 c1 仅约 5.37 tok/s，因此稳定默认仍为 built-in TP MoE (`moe_a2a_backend=none`)；后续若适配 DeePEP，应单独修复 verify graph 的 token flatten/layout。

### 6.10 DeePEP low_latency 的 DSpark graph 修复与高并发边界（2026-08-04）

- 在 `deepep.py` 的 low-latency dispatch 入口，将 DSpark verify CUDA Graph 产生的 leading padded 维度展平为 token-major，并同步展平 FP8 scale；shape 诊断显示 `hidden=(96,4096)`、`topk_ids=(96,6)`，graph16 capture/c16 回归通过（Total 7,740 tok/s，TPOT 3.09 ms）。源码备份：`backups/sglang_20260804_deepep_before_dspark_flatten.py`。
- 另修复 `flash_decode_dspark.sh` 的环境变量作用域：原赋值在 `GRAPH_ARGS=(...)` 后未传给 launch process，DeepEP 实际退回默认 `128` token 上限；显式 `export` 后进程环境确认 `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024`。脚本备份：`backups/flash_decode_dspark.after_env_export_fix_20260804`。
- graph64 capture 在上限修复后可以完成，但 c256 高并发压测触发 `CUDA error: an illegal memory access`，服务退出；失败日志及实验记录见 `logs/flash_decode_dspark/pr31466_smoke/decode_deepep_graph64_exportfix_20260804.log` 和 `isl1024_dspark_summary_20260804.md`。因此 DeePEP 目前仅证明 graph16/c16 可用，尚不能作为 1P1D 高并发稳定路径。
- 隔离实验关闭 `SGLANG_DSPARK_DISABLE_PD_HIDDEN_INJECT=1` 后，graph64/c256 可完成（Total 33,230.32 tok/s，TPOT 10.64 ms）；恢复 hidden 注入后，即使校验 full→SWA 映射，仍复现 illegal memory access，且无 invalid location 告警。问题范围收敛为 DeePEP low_latency 与 DSpark fused hidden-KV 注入/overlap 的异步竞态，不能用 cache index 边界过滤解决；因此当前稳定默认继续使用 built-in TP MoE。
- 在注入后增加 `torch.cuda.current_stream().synchronize()` 仍复现 c256 illegal memory access，排除单纯 stream 未同步；更可能是 DeePEP 路径与 DSpark target-hidden KV 写入 kernel/缓存布局不兼容。同步探针日志：`logs/flash_decode_dspark/pr31466_smoke/decode_deepep_graph64_sync_hidden2_20260804.log`。
- 额外尝试 `--disable-overlap-schedule` 仍不能通过 c256；脚本现支持 `SGLANG_DISABLE_OVERLAP_SCHEDULE=1` 作为诊断开关，但不作为默认配置。
- 运行时比较主模型/draft 模型 req→token pool，四个 DP rank 均同 storage、shape `(257,1048843)`、`max_delta=0`，排除两套 KV 页地址不一致。诊断备份：`backups/sglang_20260804_dspark_worker_before_pool_diag.py`，日志：`logs/flash_decode_dspark/pr31466_smoke/decode_deepep_pool_diag_20260804.log`。
- 关闭 DSpark draft CUDA Graph 后 target graph64 + DeePEP c256 仍失败，排除 draft graph 状态复用；日志：`logs/flash_decode_dspark/pr31466_smoke/decode_deepep_graph64_no_draft_graph_20260804.log`。
- 强制关闭 DSpark fused `CommitKvProj`、改用 PyTorch WKV 投影后仍失败，排除单个 fused WKV 投影 kernel；诊断日志：`logs/flash_decode_dspark/pr31466_smoke/decode_deepep_graph64_no_fused_commit_20260804.log`。
- 按上游 PR #31466 将 `--swa-full-tokens-ratio` 调为 `0.8` 后仍失败，排除 SWA/full 页面比例差异；日志：`logs/flash_decode_dspark/pr31466_smoke/decode_deepep_graph64_swa08_20260804.log`。

### 6.11 DeePEP c512 卡死时 GPU7 利用率为 0（2026-08-04）

- 在 `DS_MOE_A2A_BACKEND=deepep`、`SGLANG_DSPARK_DISABLE_PD_HIDDEN_INJECT=1`、graph64、1024/1024/c512 压测中，decode GPU4/5/6 利用率为 100%，GPU7 为 0%。GPU7 的 `DP3/TP3/EP3` 进程仍存在并占用约 177GB 显存，但 `/proc/<pid>/wchan` 为 `futex_wait_queue`；因此不是 GPU7 未启动或显存映射错误，而是 DP3 在等待同步。
- 四个 rank 的 hidden-summary 数量接近（32/33/34/32），日志在 13:48:35–13:48:36 后全部停止，随后 decode HTTP 接口超时，benchmark 未产生有效 JSONL 结果。该实例已按精确 PID 清理，GPU4–7 显存释放。
- 当前证据指向 DeePEP/DP collective 在 c512 高并发下的 rank 不一致或前序异步 CUDA 错误导致的同步卡死；不能把 GPU7 的 0% 利用率单独解释成硬件问题。该 c512 结果作废，后续应先用 c16/c256 验证，并在每个 rank 记录 collective/DeePEP batch shape 后再扩大并发。

### 6.12 Prefill hidden 异步读取修复与 c256 复测（2026-08-04）

- 在 `disaggregation/prefill.py::_write_dspark_hidden_rows_for_batch` 读取
  `logits_output.hidden_states`/proxy 前增加 DSpark 请求条件下的
  `torch.cuda.synchronize()`，并保留 `DSparkHiddenRowPool.write` 的同步 GPU→CPU
  拷贝；备份为 `backups/sglang_20260804_prefill_before_sync_dspark_hidden_capture.py`、
  `backups/sglang_20260804_disagg_utils_before_sync_dspark_hidden_write.py`。
- 修复后 c16（1024/1024，160 请求）完整成功：Total throughput 6300.59 tok/s、
  Mean TTFT 988.92 ms、Mean TPOT 3.90 ms，四个 DP rank 均 `cuda graph: True`。
  日志/结果：`decode_sync_capture_c16_v2_20260804.log`、
  `deepep_graph64_sync_capture_isl1024_osl1024_c16_n160.jsonl`。
- c256 仍在约 31 个运行请求后由四个 DP rank 触发 `CUDA illegal memory access`，
  栈在 `dspark_draft.py::_fill_dp_moe_sync_metadata` 的 tensor 分配处显现。关闭
  `SGLANG_DSPARK_DISABLE_PD_HIDDEN_INJECT=1` 后仍在 c256 复现同类错误，说明 GPU7
  的 0% 利用率和 c256 崩溃不能归因于 prefill hidden 注入单一路径；更可能是
  DeePEP low-latency/DP-attention 的异步 kernel 或 collective 错误在后续元数据操作
  处被报告。失败日志：`decode_sync_capture_c16_v2_20260804.log`、
  `decode_nohidden_sync_capture_c256_20260804.log`。

### 6.13 1P1D DSpark 高并发稳定修复（2026-08-04）

- DSV4 prefill 的 DSpark 中间层捕获改为逐捕获层等待非捕获 CUDA stream，并将
  `completed.mean(dim=1)` 做 contiguous clone；graph capture 期间跳过同步/clone，
  避免破坏 decode CUDA Graph。源码备份：
  `backups/sglang_20260804_deepseek_v4_before_dspark_layer_capture_sync_clone.py`。
- prefill 脚本默认关闭 overlap schedule（可用
  `SGLANG_PREFILL_DISABLE_OVERLAP_SCHEDULE=0` 恢复诊断），decode 注入后默认执行
  CUDA 同步（可用 `SGLANG_DSPARK_SYNC_PD_HIDDEN_INJECT=0` 关闭诊断）。脚本备份：
  `backups/flash_prefill_baseline_before_overlap_default_20260804.sh`、
  `backups/flash_decode_dspark_before_syncinject_default_20260804.sh`。
- 稳定回归：1024/1024/c256、2560 请求、DeePEP low_latency、decode graph
  `1 2 4 8 16 32 64`，2560/2560 成功，四个 DP rank 全程 `cuda graph: True`；
  Total throughput **22288.28 tok/s**，Mean TPOT **20.78 ms**。结果：
  `logs/flash_decode_dspark/results/sync_clone_nooverlap_syncinject_isl1024_osl1024_c256_n2560.jsonl`；
  日志：`prefill_sync_clone_nooverlap_20260804.log`、
  `decode_sync_clone_nooverlap_syncinject_c256_20260804.log`。
- 对照中仅关闭 hidden 注入的 no-overlap c256/256 也完整成功（Total 26739.82 tok/s），
  但原 overlap/无同步长压测会在 graph replay 或 allocator free 处非法访问；因此
  1P1D DSpark 的稳定默认必须保留上述两个同步/overlap 开关。
- 当前默认开关下的完整矩阵补测：
  - c16，160/160 成功，Total **6633.10 tok/s**，TPOT **3.87 ms**；结果：
    `sync_clone_nooverlap_syncinject_isl1024_osl1024_c16_n160.jsonl`。
  - c256，2560/2560 成功，Total **22288.28 tok/s**，TPOT **20.78 ms**；结果：
    `sync_clone_nooverlap_syncinject_isl1024_osl1024_c256_n2560.jsonl`。
  - c512，5120/5120 成功，Total **16827.61 tok/s**，TPOT **57.77 ms**；结果：
    `sync_clone_nooverlap_syncinject_isl1024_osl1024_c512_n5120.jsonl`。
    c512 的大 batch 走 eager（当前 graph bucket 到64），所以性能下降，但 GPU4–7
    均保持运行且未再出现 GPU7=0 的同步卡死。
## 6.14 DSpark 与 MTP 大 CUDA Graph 上限（2026-08-04）

对 DSpark decode 使用 `--deepep-mode low_latency` 时，DeepEP dispatcher 在当前版本有明确的 `num_max_dispatch_tokens_per_rank <= 1024` 约束，底层 low-latency kernel 也要求一次 dispatch 的 token 数不超过该值。DSpark target verify 的 `num_tokens_per_req=6`，所以 CUDA Graph 捕获的 batch size `B` 至少需要容纳约 `6*B` 个 token：

* `B=128`：捕获成功，`Capture target verify CUDA graph end` 正常出现；
* `B=256`：需要约 1536 个 dispatch token，在默认 1024 上限处失败；
* `B=512`：即使把环境变量设成 4096，也会在 DeepEP dispatcher 构造阶段被源码断言拒绝（low-latency 不支持该值）。

因此 MTP 能使用更大的 graph，是因为其 verify 路径每个请求的 token 宽度更小（通常为 1），`B=512` 仍在 1024 token 上限内；这不是 GPU7 利用率或 CUDA Graph 开关问题。DSpark 当前可安全捕获的最大 low-latency graph tier 是 128。`B>128` 的 DSpark batch 即使服务继续运行，也会回退到 eager，不能把它标成“大 CUDA Graph”。

探针日志：

* `logs/flash_decode_dspark/pr31466_smoke/decode_graph512_capture_probe_20260804.log`：在 target verify 捕获阶段触发 DeepEP `x.size(0) <= num_max_dispatch_tokens_per_rank`；
* `logs/flash_decode_dspark/pr31466_smoke/decode_graph512_dispatch4096_probe_20260804.log`：`deepep.py:382` 直接断言 `num_max_dispatch_tokens_per_rank <= 1024`；
* 512 graph 不能通过修改 shell 环境变量绕过，必须改 DeepEP low-latency kernel/缓冲布局，或改成 normal dispatch（与当前“decode 必须 low_latency”要求冲突）。
## 6.15 H200 参考配置与 B200 256 graph 复核（2026-08-04）

核对 `readmes/reference/单机部署deepseek-v4-flash-dspark.md` 后，H200 参考脚本的 DSpark arm 使用的是 `--moe-a2a-backend none --moe-runner-backend flashinfer_mxfp4`，并非 PD 脚本中的 `--moe-a2a-backend deepep --deepep-mode low_latency`。因此 H200 的 256 decode graph 不能直接作为 DeepEP low-latency 配置的对照。

在 B200 上复现 H200 的 MoE A2A=none 配置（4 卡 DP4、DSpark、`--cuda-graph-max-bs-decode 256`）时，日志已进入 `Capture target verify CUDA graph ... bs=...256`，没有 DeepEP 的 1024-token 断言；该探针仍在大 batch 的 DeepGEMM/graph 捕获阶段等待各 rank 同步，结果需以最终 `Capture ... end` 为准。

当前结论：

* B200 + DSpark + DeepEP low_latency：256 graph 在 target verify 阶段必然超过 1024 dispatch-token 上限，不能仅靠环境变量打开；
* B200 + DSpark + MoE A2A=none：256 graph 可以进入捕获，和 H200 参考的实际路径一致，但这不满足“decode 必须 DeepEP low_latency”的实验条件；
* 后续性能对比必须分开标注这两条路径，不能把 A2A=none 的 256 graph 结果写成 DeepEP DSpark 结果。

最小 256-only 探针最终成功：`logs/flash_decode_dspark/pr31466_smoke/decode_graph256_only_moe_none_20260804.log` 中四个 rank 均记录 `Capture target verify CUDA graph end`（约 5.2 s），随后 draft verify graph 和 server ready 也成功。该结果进一步证明卡型不是根因，真正差异是 MoE A2A 后端。
## 6.16 DSpark PD 高并发合并与 hidden page 生命周期修复（2026-08-04）

高并发回归发现两个独立问题：

1. `DFlashDraftInputV2._merge_prefill_tail()` 在 compact 模式下错误地只用
   `future_indices` 推断无 PD hidden 一侧的 batch size。compact 模式该字段通常为
   `None`，导致 hidden tensor 行数小于 `ScheduleBatch.reqs`，后续
   `filter_batch()` 触发 CUDA gather 越界。修复文件：
   `/sgl-workspace/sglang/python/sglang/srt/speculative/dflash_info_v2.py`；备份：
   `backups/sglang_20260804_dflash_info_v2_before_merge_tail_bs_fix.py`。现在按
   `future_indices/bonus_tokens/new_seq_lens/hidden_states/topk_p` 的实际首维推断行数。
2. DSpark PD 的 Mooncake hidden page 在传输完成后可能过早复用；高并发下 GPU 仍消费
   旧 page 时，新传输覆盖同一 host buffer，表现为异步 illegal memory access。设置
   `SGLANG_DSPARK_PD_HIDDEN_DISABLE_REUSE=1` 后 c64、c256 均稳定；已将其作为
   `flash_decode_dspark.sh` 默认值，同时保留环境变量可覆盖。

回归证据（A2A=none、DP4/EP4、disable-overlap、CUDA Graph）：

* c32：32/32 成功，TPOT 9.62 ms；
* c64：64/64 成功，Total 16,462.68 tok/s；
* c256：256/256 成功，Total 25,197.83 tok/s（ISL=1024、OSL=128 短测）；
* 解码日志 `logs/flash_decode_dspark/pr31466_smoke/decode_no_reuse_graph64_20260804.log`
  中各 rank 的 Decode batch 均为 `cuda graph: True`。

正式 DeepEP low-latency PD 回归也已通过：`logs/flash_decode_dspark/results/pd_deepep_ll_merge_fix_graph64_c256_n256_osl128.jsonl`，256/256 成功，服务保持存活，Total 24,413.79 tok/s、TPOT 14.74 ms；对应日志中 DP0--DP3 均持续打印 `cuda graph: True`。该结果使用 `--moe-a2a-backend deepep --deepep-mode low_latency`、`--disable-overlap-schedule` 和 hidden page 禁用复用，证明修复同样适用于正式 DeepEP 1P1D 路径。

## 6.17 DeepEP low-latency 长序列回归与 draft graph 隔离（2026-08-04）

为定位此前 OSL=1024 长压测的异步 illegal memory access，保持 target decode CUDA Graph，临时关闭 DSpark draft CUDA Graph（`SGLANG_DSPARK_DISABLE_DRAFT_CUDA_GRAPH=1`），并保留 DeepEP low-latency、disable-overlap、hidden page 禁止复用。结果：

* c16，ISL/OSL=1024/1024，16/16 成功，Total 3,210.87 tok/s，TPOT 7.18 ms；
* c256，ISL/OSL=1024/1024，256/256 成功，服务未崩溃，Total **21,421.32 tok/s**，Mean TTFT 6,307.71 ms，TPOT **11.80 ms**；四个 DP rank 均记录 `cuda graph: True`。

这说明此前长序列崩溃至少与 draft CUDA Graph 路径相关；关闭 draft graph 后正式 DeepEP PD 长跑可完成。随后将 target graph tier 从 `1 2 4 8 16 32 64` 提高到包含 128，捕获阶段成功，但 c256 长跑约 1 分钟后仍触发异步 CUDA illegal memory access。因此当前正式 DeepEP low-latency 的稳定配置仍应使用 target graph 最大 64，并关闭 draft graph；128 不能作为稳定配置，256 受 1024 dispatch-token 限制无法捕获。

相关文件：
`logs/flash_decode_dspark/results/pd_deepep_targetgraph_drafteager_c16_n16_osl1024.jsonl`、
`logs/flash_decode_dspark/results/pd_deepep_targetgraph_drafteager_c256_n256_osl1024.jsonl`、
`logs/flash_decode_dspark/pr31466_smoke/decode_deepep_targetgraph_drafteager_c16_20260804.log`、
`logs/flash_decode_dspark/pr31466_smoke/decode_deepep_targetgraph128_drafteager_20260804.log`。

## 6.18 DeepEP graph=256 直接捕获探测（2026-08-04）

按用户要求在 B200 四卡 1P1D 正式 DeepEP low-latency 配置中加入
`CUDA_GRAPH_BS_DECODE="1 2 4 8 16 32 64 128 256"`，并使用 `--deepep-mode low_latency`
进行启动探测。服务在 target verify 的 `bs=256` 首轮 capture 即失败，日志为
`logs/flash_decode_dspark/pr31466_smoke/decode_deepep_graph256_probe_20260804.log`，
核心错误是：

`/build/DeepEP/csrc/deep_ep.cpp:1105 x.size(0) <= num_max_dispatch_tokens_per_rank`

当前 DeepEP low-latency 强制 `num_max_dispatch_tokens_per_rank <= 1024`；DSpark target
verify 每个请求产生 6 个验证 token，因此 256 请求需要约 1536 个 dispatch token，超过该
硬上限。该限制在 CUDA Graph capture 阶段就触发，和 B200 显存容量无关；不能通过把环境变量
调大来绕过（源码同时有 `assert <= 1024`）。因此正式 DeepEP 路径当前最多只能捕获到
`bs=128`（但 128 长跑仍有独立的异步 illegal-memory 稳定性问题），稳定配置仍为 target
graph 最大 64。A2A=none 可以捕获 256，但不属于本实验要求的 DeepEP low-latency 对照。

## 6.19 DP idle metadata 修复与静态 verify 隔离（2026-08-04）

关闭 target CUDA Graph、保留 DeepEP low-latency/PD/DSpark 做隔离时，c128 曾在
`TargetVerifyExecutor.run_idle_participation()` 报 `negative dimension -6`。原因是 idle
DP rank 的 dummy verify input 已包含 6 个 token，但 `global_num_tokens` 仍为 0，DP MLP
同步 padding 计算出负长度。已在 `dspark_draft.py` 和 `dspark_verify.py` 对 idle batch
同步元数据做按 dummy token 数的归一化，并保留备份：

* `backups/sglang_20260804_dspark_draft_before_idle_metadata_fix.py`
* `backups/sglang_20260804_dspark_worker_before_idle_metadata_fix.py`

修复后 eager target c128、ISL/OSL=1024/1024 完成 128/128，服务保持存活（结果：
`logs/flash_decode_dspark/results/eager_idle_fix_c128_isl1024_osl1024.jsonl`）。

为隔离 ragged compact graph，临时使用 `SGLANG_RAGGED_VERIFY_MODE=static`，target
CUDA Graph capture 成功，日志为
`logs/flash_decode_dspark/pr31466_smoke/decode_deepep_graph64_static_verify_20260804.log`；
该模式绕过 SPS/compact ragged 调度，只能作为诊断路径，不能替代最终 compact DSpark
性能实验。compact target graph 在 c128 长跑仍需继续修复，当前没有把 static 路径结果计入
正式对比。

## 6.20 compact ragged target graph 长序列崩溃定位（2026-08-04）

在正式 DeepEP low-latency、target graph 最大 64、draft graph 关闭的配置下，短请求
c128/c256 可以完成，但 ISL/OSL=1024/1024 的 c128 长跑会复现服务崩溃。同步诊断把
错误从 `AcceptGreedy` 的异步落点提前定位到 target verify CUDA Graph replay：

* `logs/flash_decode_dspark/pr31466_smoke/decode_deepep_graph64_compact_sync_target_20260804.log`
  在 `_forward_prepared_verify()` 的同步点报告 illegal memory；
* `logs/flash_decode_dspark/pr31466_smoke/decode_deepep_graph64_blocking_nohidden_20260804.log`
  使用 `CUDA_LAUNCH_BLOCKING=1` 后直接在 `full_cuda_graph_backend.py:129` 的 graph replay
  报错，排除了 AcceptGreedy 作为根因；
* 关闭 PD hidden 注入、关闭 folded commit 后仍可复现，说明问题是 compact ragged target
  graph 的 attention/metadata replay，不是单一 hidden-KV 写入。

高并发时 `req_to_token` 会出现无效 full slot，已备份并增加 page-table 的 req-row、token
slot 有效性保护：`backups/sglang_20260804_trtllm_page_table_before_invalid_slot_guard.py`。
该保护避免负 page-table 输出，但尚未解决 graph replay 的全部问题。

诊断用 folded commit 开关已加入 `dspark_worker_v2.py`：
`SGLANG_DSPARK_DISABLE_FOLDED_COMMIT=1`，备份为
`backups/sglang_20260804_dspark_worker_before_disable_folded_commit.py`。该开关仅用于隔离，
正式结果仍不能将 c128 OSL=1024 标记为通过。

补充回归：重启后正式 compact 模式（DeepEP low_latency、target graph 最大 64、draft
graph 暂关、hidden page 禁止复用）在 ISL/OSL=1024/128、c128 完成 128/128，Total
12,911.05 tok/s、TPOT 17.85 ms，服务未崩溃。结果文件：
`logs/flash_decode_dspark/results/deepep_compact_graph64_c128_isl1024_osl128_restart.jsonl`。

同一重启实例继续做 c256 短回归也完成 256/256，Total 21,784.60 tok/s、TPOT 21.85 ms：
`logs/flash_decode_dspark/results/deepep_compact_graph64_c256_isl1024_osl128_restart.jsonl`。
这证明当前 idle metadata 修复和 hidden page 禁止复用在 compact c128/c256 短请求上均有效；
OSL=1024 长跑及 draft CUDA Graph 仍需单独回归，不能据此宣称最终性能已达标。
### 6.21 B200 四卡正式 DeepEP 下 decode CUDA Graph=256 验证（2026-08-04）

按 `TP4/DP4/EP4 + --enable-dp-attention + --moe-a2a-backend deepep + --deepep-mode low_latency`，使用真实 SPS 表并显式捕获 `--cuda-graph-bs-decode 1 2 4 8 16 32 64 128 256`。配置解析确认 `decode.max_bs=256`，但在捕获 `bs=256` 时 DeepEP 报：

```text
deep_ep.cpp:1105: x.size(0) == topk_idx.size(0) and
x.size(0) <= num_max_dispatch_tokens_per_rank
```

当前 `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024`，而 DSpark target 每请求 6 token，256 请求需要约 1536 dispatch rows/rank，超过 DeepEP low_latency 的 1024 上限。因此 B200 四卡的正式 DeepEP+DSpark decode graph 不能捕获 256；这属于 DeepEP dispatch contract，不是 B200 显存或 CUDA Graph 能力问题。`A2A=none` 的 256 graph 只能作为非正式隔离实验，不能代表正式 DeepEP 结果。正式条件最高保留 graph64（128/256 长跑仍需继续修复运行期非法访存）。

复现命令：

```bash
DS_MOE_A2A_BACKEND=deepep \
CUDA_GRAPH_BS_DECODE='1 2 4 8 16 32 64 128 256' \
SGLANG_DSPARK_DISABLE_DRAFT_CUDA_GRAPH=1 \
bash flash_decode_dspark.sh \
/data/ssd2/gongoubo/single_node/logs/flash_decode_dspark/archive_20260803_091434/dspark_sps.json
```

### 6.25 DeepEP graph256 扩展尝试的回退结论（2026-08-04）

为支持 DSpark 256 并发约 1536 dispatch rows，尝试将 DeepEP
`FINISHED_SUM_TAG` 从 1024 扩到 2048；修改前源码和二进制已备份到
`backups/deepep_graph256_20260804/`。增量及干净重编译后的 DeepEP 官方
`tests/test_low_latency.py`（4 进程、128 tokens）均在首次 kernel launch 报
`internode_ll.cu:391 CUDA error 'named symbol not found'`，说明当前容器的
NVSHMEM device-link 产物不能直接用于该扩展。已恢复原始 DeepEP 二进制和 1024
约束；graph256 不计为通过，正式运行保持 6.23 的 graph128 配置。

回退后的 target+draft graph128 服务重新启动成功，c256、ISL=1024、OSL=128 短回归
完成 256/256：Total 24604.83 tok/s、TPOT 12.54 ms。结果文件：
`logs/flash_decode_dspark/results/deepep_static_graph128_draftgraph_restored_c256_isl1024_osl128.jsonl`。

### 6.26 SGLang 层分块支持正式 DeepEP graph256（2026-08-04）

不修改 DeepEP/NVSHMEM 协议，新增可选环境变量
`SGLANG_DEEPEP_LL_SPLIT_TOKENS=1024`。它在 `DeepEPMoE` 入口把 DSpark 的
256-request、6-token verify（1536 rows）拆成 1024+512 两个连续的正式
DeepEP low-latency dispatch/combine 子批次；默认值 0，默认 graph128 行为不变。
源码备份：`backups/sglang_20260804_ep_moe_layer_before_ll_split.py`，脚本备份：
`backups/flash_decode_dspark_before_ll_split_env_20260804.sh`。

验证证据：target CUDA Graph=256、draft CUDA Graph=256 均捕获成功，四个 DP rank
日志均出现 `rows=1536 limit=1024 chunks=2`，服务可用；单请求返回正常。
启动日志：
`logs/flash_decode_dspark/dspark_graph256_split1024_full_max2048_20260804.log`。

c256、ISL=1024、OSL=1024、target+draft graph256、正式 DeepEP 完成 256/256：
Total 25616.34 tok/s、TPOT 9.13 ms。该分块路径当前比 MTP-only 26213.34 tok/s
低约 2.28%，因此 graph256 目前是功能通过但性能未达标；已通过的 graph128/no-split
生产路径仍为 27012.88 tok/s（高于 MTP-only 约 3.05%）。结果文件：
`logs/flash_decode_dspark/results/deepep_static_graph256split_draftgraph_max2048_c256_isl1024_osl1024.jsonl`。

### 6.24 512 并发容量试验与 max-running-requests 配置（2026-08-04）

`max_running_requests` 是全局值，DP4 时每个 decode worker 实际得到
`max_running_requests / 4`。因此默认 1024 只允许每个 DP worker 256 个请求；做 c512
容量试验必须显式设置 `SGLANG_MAX_RUNNING_REQUESTS=2048`，此时日志确认每个 worker 为
512。该配置可完成 512 个请求中的 512 个 HTTP 响应，但本次 OSL=128 回归仍有 1 个
请求在 router/benchmark 层未返回，未生成可采信的吞吐结果，不能作为正式性能结论。

脚本已支持该覆盖参数，默认仍为 1024（不改变已通过的 c256 结果）：

```bash
SGLANG_MAX_RUNNING_REQUESTS=2048 \
SGLANG_RAGGED_VERIFY_MODE=static \
DS_MOE_A2A_BACKEND=deepep \
CUDA_GRAPH_BS_DECODE='1 2 4 8 16 32 64 128' \
bash flash_decode_dspark.sh \
/data/ssd2/gongoubo/single_node/logs/flash_decode_dspark/archive_20260803_091434/dspark_sps.json
```
### 6.22 1P1D DSpark static verify 稳定性验证（2026-08-04）

在正式 DeepEP low-latency、TP4/DP4/EP4、target decode CUDA Graph 档位
`1/2/4/8/16/32/64/128` 下，将 `SGLANG_RAGGED_VERIFY_MODE=static` 用于 c128、ISL=1024、OSL=1024 长跑，128/128 全部成功：总吞吐 14514.24 tok/s，TPOT 8.86 ms，服务未崩溃。相同硬件和 graph128 下 compact ragged 路径仍会在长跑中触发非法访存，因此当前证据指向 compact layout/页表生命周期，而非 PD hidden 注入或 graph128 捕获本身。

结果文件：
`logs/flash_decode_dspark/results/deepep_static_graph128_current_c128_isl1024_osl1024.jsonl`

复现环境：

```bash
SGLANG_RAGGED_VERIFY_MODE=static \
DS_MOE_A2A_BACKEND=deepep \
CUDA_GRAPH_BS_DECODE='1 2 4 8 16 32 64 128' \
SGLANG_DSPARK_DISABLE_DRAFT_CUDA_GRAPH=1 \
SGLANG_DSPARK_PD_HIDDEN_DISABLE_REUSE=1 \
bash flash_decode_dspark.sh \
/data/ssd2/gongoubo/single_node/logs/flash_decode_dspark/archive_20260803_091434/dspark_sps.json
```
static graph128 在 256 并发、ISL=1024、OSL=1024 下也完成 256/256：总吞吐 23213.54 tok/s，TPOT 11.93 ms，结果文件为 `logs/flash_decode_dspark/results/deepep_static_graph128_current_c256_isl1024_osl1024.jsonl`。因此正式 DeepEP 不需要 graph256 才能覆盖 c256，调度可由 graph128 分批完成。
### 6.23 完整 target+draft CUDA Graph 的正式 DeepEP 1P1D 结果（2026-08-04）

将生产默认切换为 static verify、正式 DeepEP low-latency、target/draft decode CUDA Graph 档位 `1/2/4/8/16/32/64/128`，并关闭仅用于诊断的全设备 PD hidden 注入同步屏障（`SGLANG_DSPARK_SYNC_PD_HIDDEN_INJECT=0`）。在 ISL=1024、OSL=1024、并发 256 下 256/256 成功：

- 总吞吐：27012.88 tok/s
- TPOT：9.23 ms
- 相同条件 MTP-only 对照：26213.34 tok/s
- DSpark 提升：约 3.05%

结果文件：`logs/flash_decode_dspark/results/deepep_static_graph128_draftgraph_nosync_c256_isl1024_osl1024.jsonl`

可复现启动命令：

```bash
SGLANG_RAGGED_VERIFY_MODE=static \
SGLANG_DSPARK_SYNC_PD_HIDDEN_INJECT=0 \
DS_MOE_A2A_BACKEND=deepep \
CUDA_GRAPH_BS_DECODE='1 2 4 8 16 32 64 128' \
SGLANG_DSPARK_DISABLE_DRAFT_CUDA_GRAPH=0 \
SGLANG_DSPARK_PD_HIDDEN_DISABLE_REUSE=1 \
bash flash_decode_dspark.sh \
/data/ssd2/gongoubo/single_node/logs/flash_decode_dspark/archive_20260803_091434/dspark_sps.json
```

### 6.27 graph128 与 graph256 的 max-running 对照（2026-08-04）

为确认 graph256 的性能差异不是 max-running-requests 单独造成，保持正式
DeepEP low-latency、TP4/DP4/EP4、static verify、target+draft CUDA Graph 及
ISL/OSL=1024/1024、并发 256 不变，仅使用 max-running-requests=2048 和
graph128（不启用分块）进行对照。结果 256/256 成功，但总吞吐为 22999.25
tok/s、TPOT 9.07 ms；结果文件：
logs/flash_decode_dspark/results/deepep_static_graph128_nosplit_max2048_c256_isl1024_osl1024.jsonl。

同一 max-running-requests=2048 下，graph256 分块结果为 25616.34 tok/s、
TPOT 9.13 ms。因此当前差异不是单纯由 max-running 参数导致；graph128 的
27012.88 tok/s 结果对应的是原生产配置（max-running=1024）。两组服务日志
均显示 target 和 draft cuda graph: True，无 CUDA/Traceback 错误。

### 6.28 H200 参考日志与 B200 DeepEP graph256 的口径差异（2026-08-04）

复核用户提供的 H200 DSpark 日志后发现，它不能作为当前正式 DeepEP 1P1D 的直接对照：
对应运行的 server_args 明确为 moe_a2a_backend='none'、deepep_mode='auto'，且日志只显示 DPx TPx，没有 EPx。该路径使用的是本地 MoE/FlashInfer，而不是 DeepEP low-latency dispatch。

当前 B200 正式 1P1D 路径使用 moe_a2a_backend='deepep'、deepep_mode='low_latency'、DP4/EP4。target verify 的 num_tokens_per_req=6，因此 graph256 需要处理 1536 行；当前 DeepEP 的 low-latency barrier 协议上限为 1024。直接 graph256 的失败是 DeepEP 协议约束，不是 B200 CUDA Graph 不支持。

已有本机证据：logs/flash_decode_dspark/pr31466_smoke/decode_graph256_only_moe_none_20260804.log 的 moe_a2a_backend='none' 路径可以直接 capture graph256；正式 DeepEP 则通过 SGLANG_DEEPEP_LL_SPLIT_TOKENS=1024 拆成两个合法 dispatch。故 H200 日志和当前 B200 结果必须区分 A2A backend 后再比较。

### 6.29 B200 sm100 DeepEP 二进制候选验证（2026-08-04）

机器上另一份 DeepEP 二进制仅包含 sm100，但它的 C++ ABI 与当前 SGLang 安装的 Python wrapper 不兼容：候选 Buffer 构造函数需要 8 个参数，而当前 wrapper 传入 6 个参数，graph capture 在初始化 DeepEP buffer 时立即报 TypeError: incompatible constructor arguments。候选未安装，当前正式服务已恢复原始二进制（SHA256 81f6c19d...）。候选和当前版本均已保存在 backups/deepep_sm100_trial_*_20260804.so，不会影响正式结果。

### 6.30 graph256 + max-running=1024 长尾复现（2026-08-04）

在正式 DeepEP low-latency、target+draft graph256、分块上限 1024、ISL/OSL=1024/1024、并发 256 下，将 max-running-requests 降为 1024 后，capture 和单请求均成功，但批量 benchmark 长时间无法完成 256 请求；约 3 分钟后仍有请求未返回，客户端只能中止，服务端仍持续输出 cuda graph: True。该现象与此前 max-running=1024 的 3 个 stuck request 一致，因此该组合标记为不稳定，不采用其 25993.87 tok/s 的不完整/短跑结果作为正式吞吐结论。

### 6.32 B200 四卡 graph256 复测与 2048 候选结论（2026-08-04）

正式系统 DeepEP 在 `SGLANG_DEEPEP_LL_SPLIT_TOKENS=1024`、target+draft CUDA Graph、`CUDA_GRAPH_BS_DECODE=1 2 4 8 16 32 64 128 256` 下成功启动；日志出现 `Capturing batches (bs=256)` 并完成 target/draft graph capture，服务健康检查返回 200。这里的 256 是 CUDA Graph batch bucket，1536 行 verify 仍由两个 1024 以内的 DeepEP dispatch 分块处理。

独立 `FINISHED_SUM_TAG=2048` sm100 候选在 `mem-fraction-static=0.70` 下完成不分块 graph256 的 target+draft capture。正确 PD router 路径的一次 256 并发、ISL/OSL=1024/1024 压测为 256/256 成功，Total throughput **26505.05 tok/s**，TPOT 9.14 ms，高于 MTP-only 26213.34 tok/s（+1.11%）。但第二次连续压测出现 worker health/PD handshake 超时，故候选尚未达到生产稳定性要求；正式服务继续使用 DeepEP 1024 分块实现。

### 6.31 DeepEP FINISHED_SUM_TAG=2048 独立验证与 graph capture 结果（2026-08-04）

在独立副本 /tmp/deepep_ll2048_cNeySo 中使用 B200 sm_100、FINISHED_SUM_TAG=2048 重建 DeepEP。官方 4 进程 low-latency 测试使用 num_tokens=2048, hidden=4096, num_experts=256 通过，dispatch+combine 正确，带宽约 580.5 GB/s。

但将该二进制和匹配的 Python wrapper 接入 SGLang，并把 max dispatch tokens 设为 2048 后，target graph256 capture 在 bs=256 阶段挂起；没有 CUDA 报错，无法完成服务启动。说明该协议扩展目前只通过 eager/官方 smoke，尚未满足 SGLang CUDA Graph 的要求。候选未安装到系统，正式服务已恢复原始 DeepEP + SGLang 分块路径。候选构建目录保留在 /tmp/deepep_ll2048_cNeySo，SGLang assert 修改已用 backups/sglang_20260804_deepep_assert_before_ll2048.py 恢复。
## 6.33 B200 DSpark decode CUDA Graph=256 连续压测诊断（2026-08-05）

- 已确认启动参数实际生效：`cuda_graph_config.decode.max_bs=256`，bucket 为 `1,2,4,8,16,32,64,128,256`，`backend=full`；decode 日志中的 `cuda graph: True` 表明请求使用了 CUDA Graph。
- 在 1024/1024、并发 256、1P1D、DeepEP `low_latency`、`SGLANG_DEEPEP_LL_SPLIT_TOKENS=1024` 下，首轮 256/256 完成，`health_disabled_graph256_run1_20260805.jsonl`：Total throughput 26513.46 tok/s，TPOT 9.65 ms；另一轮 `pd_debug_run1_20260805.jsonl`：25113.20 tok/s，TPOT 9.67 ms。说明 B200 4 卡可以成功捕获并运行到 graph=256，性能超过 MTP-only 26213.34 tok/s 的首轮结果约 1.1%。
- 连续第二轮仍存在长尾卡住：prefill 的 `PD transfer poll` 最终均为 Success，未发现 WaitingForInput/Transferring 残留；decode 最后只剩少数 running request，手动中断后才看到 `Aborted by AbortReq`。因此当前问题不是 CUDA Graph=256 捕获失败，也不是 prefill KV transfer 未完成，而是连续压测末尾 decode 请求生命周期/调度清理问题，需继续定位，不能把中断后的 AbortReq 当作根因。
- 为避免 `/health` 触发伪生成请求，当前 decode 使用 `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0`，router 使用 `--disable-health-check`；两端健康检查均返回 200，且 decode 日志无 `HEALTH_CHECK` 请求。
## 6.34 修复 DSpark PD hidden buffer 连续运行耗尽（2026-08-05）

- 根因：此前启动命令显式设置 `SGLANG_DSPARK_PD_HIDDEN_DISABLE_REUSE=1`。首轮 256 并发、每个请求的 DSpark hidden transfer 约占两个 512-row 注册页，正好消耗默认 `SGLANG_DSPARK_PD_HIDDEN_BUFFER_POOL_LIMIT=512`。传输完成后页面已进入 free 队列，但禁用复用会继续申请新注册页；第二轮达到 512 上限后，DP0/DP1 各有一个 `waiting_for_input=True`、`metadata=-1` 的 prealloc 请求永久无法进入 transfer queue。
- 修复：`flash_decode_dspark.sh` 默认改为 `SGLANG_DSPARK_PD_HIDDEN_DISABLE_REUSE=0`。`_commit_transfer_to_req` 在释放页面前已将 hidden 数据拷贝到 CPU handoff tensor，因此完成传输的注册页可以复用；保留环境变量显式设为 `1` 的诊断开关。
- 干净重启后的连续回归（1024/1024、256 并发、1P1D、DeepEP low_latency、decode CUDA Graph bucket 含 256）均完成 256/256：
  - `reuse0_run1_20260805.jsonl`：Total throughput 25648.30 tok/s，TPOT 9.556 ms。
  - `reuse0_run2_20260805.jsonl`：Total throughput 25220.56 tok/s，TPOT 9.185 ms。
- 这两轮验证了 graph=256 和 DSpark hidden buffer 复用在连续 PD 压测下均可工作；服务当前使用复用默认值并保持健康。

## 6.35 B200 DSpark graph256 最高档重启验证与 c512 限制（2026-08-05）

- 本次重启命令实际使用 `CUDA_GRAPH_BS_DECODE='1 2 4 8 16 32 64 128 256'`、`DS_MOE_A2A_BACKEND=deepep`、`--deepep-mode low_latency`、`SGLANG_DEEPEP_LL_SPLIT_TOKENS=1024`；decode 日志确认 `backend=full`、`max_bs=256`，并完成 bs=256 capture，30001 `/health` 返回 200。
- c512 原始 GPU hidden-tail 合并曾在 `dflash_info_v2.py::_merge_prefill_tail` OOM（单次 `torch.cat` 申请约 2.67 GiB，graph pool 约 8 GiB）。已采用 CPU handoff tail、按请求分块注入（默认 16）及 CPU index 过滤修复；对应源码备份见 `backups/sglang_dspark_worker_before_cpu_tail_chunk_20260805.py`、`backups/sglang_spec_info_before_cpu_tail_chunk_20260805.py`、`backups/sglang_dflash_info_before_cpu_tail_filter_20260805.py`。
- CPU tail 版本在 c512 长跑中避免了原始 Python OOM，但吞吐显著下降；分块 64 的一次样本随后以 exit code -9 结束，同时系统记录 NVRM/Mellanox QP 错误，不能视为 c512 graph256 成功。当前已恢复到 hidden reuse=0、graph256 配置，c512 仍需进一步优化/验证。

## 6.36 B200 1P1D prefill symmetric-memory 启动卡死修复（2026-08-05）

- 现象：prefill scheduler 的 `py-spy` 栈长期停在 `cuMulticastBindMem -> torch symmetric_memory.rendezvous -> triton_symm_mem_ag.create_state`，导致 30000 无响应；decode 端 graph256 正常。
- 修复：`logits_processor.py` 在 `disaggregation_mode == "prefill"` 时关闭 multimem symmetric-memory logits gather，使用普通 gather fallback；decode/non-disaggregated 路径保持原逻辑。备份：`backups/sglang_logits_processor_before_prefill_symm_disable_20260805.py`。
- 验证：修改后 prefill 30000 `/health=200`，router 单条 `你是谁` 返回正常；256 并发端到端压测已启动，decode 日志持续 `cuda graph: True`、accept rate 约 0.89–0.92，hidden buffer 持续 `reused=4, allocated=0`，无 OOM/traceback。

## 6.37 c256 端到端尾部 response decode 异常（2026-08-05，未完成修复）

- 1024/1024、256 并发、2560 请求压测中，前 733 个请求持续完成，三端健康；随后 router 报两条 `Error consuming prefill response: error decoding response body`，GPU 进入空闲，benchmark 等待未退出。
- 随后的 prefill scheduler_2 收到 SIGTERM 并触发 supervisor 清理，说明当前尾部请求/Prefill response 解码路径仍有生命周期问题；本轮未生成可用最终吞吐结果，不能计入正式矩阵。
- 现场日志保留在 `logs/flash_decode_dspark/router_nosymm_c256_hang_20260805.log`、`logs/flash_decode_dspark/prefill_no_symm_20260805.log` 和 `logs/flash_decode_dspark/decode_final_restart_20260805.log`。服务已重新启动并恢复 30000/30001/13784 健康，后续需先修复该尾部异常再继续矩阵。

## 6.38 清理 decode 残留 scheduler 后的 1P1D DSpark 回归（2026-08-05）

- 进一步定位到此前 `/health=200` 但 router 无法注册 decode worker 的直接原因：强制终止 decode 父进程后，旧的 `scheduler_DP*` 子进程仍由 PID 1 托管；同一组 GPU 上同时存在两套 DP scheduler，导致 decode `/server_info` 超时，router 的 metadata discovery 失败。清理所有 decode controller/scheduler 后只启动一套进程，`/server_info` 恢复 200。
- 重新启动使用的实际配置为 `DS_MOE_A2A_BACKEND=deepep`、`--deepep-mode low_latency`、`SGLANG_DEEPEP_LL_SPLIT_TOKENS=1024`、`CUDA_GRAPH_BS_DECODE='1 2 4 8 16 32 64 128 256'`、SPS table 和 `--disable-radix-cache`。启动日志：`logs/flash_decode_dspark/decode_deepep_graph256_clean_20260805.log`。
- 单请求 `你是谁？` 通过 router 成功返回，三端 `/health=200`；decode 日志持续 `cuda graph: True`。router 日志：`logs/flash_decode_dspark/router_after_decode_clean_20260805.log`。
- 小规模端到端回归均完整成功，无 `error decoding response body`、SIGTERM 或 5xx：
  - `clean_isl1024_osl1024_c1_20260805.jsonl`：16/16，Total throughput 540.91 tok/s，Mean TPOT 3.34 ms。
  - `clean_isl1024_osl1024_c16_20260805.jsonl`：64/64，Total throughput 4480.08 tok/s，Mean TPOT 5.60 ms。
- 这证明清理残留 scheduler 后 1P1D DSpark 链路和 graph256 在低并发下稳定；c256 长跑仍需继续验证，不能用小并发结果替代正式矩阵。

## 6.39 DSpark hidden 注入 64 与 c256 超时复测（2026-08-05）

- 为改善 CPU handoff tail 的调度开销，本轮 decode 在既有 graph256/DeepEP low-latency 配置上增加 `SGLANG_DSPARK_PD_HIDDEN_INJECT_REQS=64`。c16 完整回归 `inject64_isl1024_osl1024_c16_20260805.jsonl` 为 64/64，Total throughput 5247.40 tok/s、Mean TPOT 4.837 ms；相比默认注入 16 的 4480.08 tok/s、5.604 ms，吞吐提高约 17.1%，TPOT 降低约 13.7%。
- 同配置 c256 使用 `--num-prompts 2560 --max-concurrency 256`，日志为 `logs/flash_decode_dspark/results/inject64_isl1024_osl1024_c256_20260805.log`。从本轮启动时间起只完成约 1688/2560 个请求，达到 1200 秒 timeout 后退出，未生成结果 JSON；因此该轮标记为超时/无效，不计入吞吐矩阵，也不能据此声称超过 MTP-only 26213.34 tok/s。
- 超时期间三端保持健康，decode 使用 graph bucket 256、DeepEP `low_latency` 与 `SGLANG_DEEPEP_LL_SPLIT_TOKENS=1024`，未见 `error decoding response body`、OOM、Traceback 或 SIGTERM。当前瓶颈仍是高并发 CPU handoff/尾部请求完成速度，后续应继续优化后再做正式 c256。

## 6.40 DSpark lazy-tail 修复并完成 graph256/c256 正式压测（2026-08-05）

- 用 `py-spy` 采样确认 c256 超时的直接热点是 `dflash_info_v2.py::_merge_prefill_tail` 中对 CPU hidden tail 的 `torch.cat`。每次新请求合并都会复制 GiB 级 `[batch, 1024, 12288]` tensor，导致 scheduler 长时间停在合并函数；并非 CUDA Graph capture 或 DeepEP low-latency dispatch 失败。
- 修复前先备份：`backups/sglang_spec_info_before_lazy_tail_20260805.py`、`backups/sglang_dspark_worker_before_lazy_tail_20260805.py`。修复后 `spec_info.py` 和 `_build_initial_draft_input` 不再构造整批 hidden tensor；hidden tail 保留在 Req 上，由 `dspark_worker_v2.py::_maybe_inject_pd_prefill_tail` 在 forward 前按请求收集、按 `SGLANG_DSPARK_PD_HIDDEN_INJECT_REQS` 分块注入。此前发现的 CPU→GPU→CPU 往返也一并去除，备份见 `backups/sglang_spec_info_before_cpu_roundtrip_fix_20260805.py`、`backups/sglang_dspark_worker_before_cpu_roundtrip_fix_20260805.py`。
- 实际 decode 启动日志：`logs/flash_decode_dspark/decode_graph256_lazy_tail_inject64_20260805.log`。关键参数为 `DS_MOE_A2A_BACKEND=deepep`、`--moe-a2a-backend deepep`、`--deepep-mode low_latency`、`SGLANG_DEEPEP_LL_SPLIT_TOKENS=1024`、`SGLANG_DSPARK_PD_HIDDEN_INJECT_REQS=64`、`--cuda-graph-bs-decode 1 2 4 8 16 32 64 128 256`、`--disable-radix-cache`、TP4/DP4/EP4；日志确认 graph256 capture 与运行均为 `cuda graph: True`。
- 512 请求短回归 `logs/flash_decode_dspark/results/lazy_tail_inject64_isl1024_osl1024_c256_512req_20260805.jsonl`：512/512，Total throughput 35,939.70 tok/s，Mean TPOT 8.23 ms。
- 正式全量结果 `logs/flash_decode_dspark/results/lazy_tail_inject64_isl1024_osl1024_c256_20260805.jsonl`：**2560/2560 成功**，ISL/OSL=1024/1024、并发 256，duration 113.224 s，input throughput 23,152.70 tok/s，output throughput 23,152.70 tok/s，**Total throughput 46,305.41 tok/s**，Mean TTFT 1,552.68 ms，Mean TPOT 9.065 ms。相对 MTP-only 26,213.34 tok/s 提升约 76.6%，且结果文件中的 retokenized output 为 2,618,914/2,621,440（无大规模异常截断）。
- 本轮全量压测期间无 `error decoding response body`、OOM、Traceback 或 SIGTERM；服务端与 router 健康检查均为 200。该结果替代 6.39 中的超时轮，作为当前 DSpark 1P1D graph256/c256 正式结果。

## 6.41 DSpark lazy-tail 并发 512 正式验证（2026-08-05）

- 在 6.40 lazy-tail 修复、DeepEP `low_latency`、decode CUDA Graph bucket 最大 256 的配置上，先完成 512 请求短回归：512/512，峰值并发 512，Total throughput 37,178.22 tok/s，Mean TPOT 10.07 ms。
- 正式全量结果：`logs/flash_decode_dspark/results/lazy_tail_inject64_isl1024_osl1024_c512_20260805.jsonl`，`--num-prompts 2560 --max-concurrency 512`，2560/2560 成功，duration 108.134 s，input/output throughput 各 **24,242.55 tok/s**，**Total throughput 48,485.10 tok/s**，Mean TTFT 8,364.11 ms，Mean TPOT 11.504 ms，实际平均并发 476.63、峰值并发 554。
- 结果文件 retokenized output 为 2,619,044/2,621,440，未出现大规模截断；decode 日志持续 `cuda graph: True`，使用 `SGLANG_DSPARK_PD_HIDDEN_INJECT_REQS=64`，无 OOM、Traceback、5xx 或 response decode 错误。prefill/decode/router 在压测后均返回 200。
- 该结果证明当前 1P1D DSpark lazy-tail 方案不仅支持 graph256/c256，也能稳定完成 c512；c512 Total throughput 高于 MTP-only 26,213.34 tok/s 约 84.9%。

## 6.42 lazy-tail 修复后的低并发回归（2026-08-05）

- 为确认 lazy-tail 修改没有影响低并发，使用同一运行中的 graph256/DeepEP 配置重新执行：
  - c1：`lazy_tail_inject64_isl1024_osl1024_c1_20260805.jsonl`，16/16，Total throughput **584.02 tok/s**，Mean TTFT 355.01 ms，Mean TPOT 3.079 ms。
  - c16：`lazy_tail_inject64_isl1024_osl1024_c16_20260805.jsonl`，64/64，Total throughput **5,931.93 tok/s**，Mean TTFT 672.98 ms，Mean TPOT 4.29 ms。
- 两轮均使用 ISL/OSL=1024/1024、disable radix cache、DeepEP low-latency 和 decode CUDA Graph；decode 日志保持 `cuda graph: True`，无异常，三端服务继续健康。
- 至此当前 lazy-tail 版本已完成 c1、c16、c256、c512 四档完整回归；c256/c512 均明显超过 MTP-only 26,213.34 tok/s。
## 6.43 2026-08-05 单机 1P1D DSpark 输入/输出矩阵（graph256）

本轮使用 decode `--cuda-graph-bs-decode 1 2 4 8 16 32 64 128 256`、`--deepep-mode low_latency`、DSpark SPS table，prefill/decode/router 日志分别保存在 `logs/flash_decode_dspark/`。1024/1024 的四个并发档已完整完成：

| ISL | OSL | 并发 | prompts | 输入 tok/s | 输出 tok/s | 总 tok/s | Mean TTFT (ms) | Mean TPOT (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1024|1024|1|10|289.49|289.49|578.99|348.68|3.11|
|1024|1024|16|160|3147.11|3147.11|6294.22|669.35|4.20|
|1024|1024|256|2560|23680.92|23680.92|47361.84|1674.10|8.73|
|1024|1024|512|5120|25661.95|25661.95|51323.91|7509.92|11.82|

8192/1024 并发1、16 已完整成功：

| ISL | OSL | 并发 | prompts | 输入 tok/s | 输出 tok/s | 总 tok/s | Mean TTFT (ms) | Mean TPOT (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|8192|1024|1|10|2014.83|251.85|2266.68|759.12|3.23|
|8192|1024|16|160|12405.51|1550.69|13956.20|1702.68|8.42|

### 8192 高并发异常

8192/1024 并发256（2560 prompts）两次均未完成，结果不纳入性能表：第一次在 2429 个请求后 prefill scheduler 四卡同时卡在 `deepseek_v4.forward -> torch.cuda.synchronize()`，watchdog 300s 后退出；第二次将 DSpark hidden transfer queue 调小（pool 256、queue 16、1GiB）后仍在约512个请求处停止推进。现场日志：

* `logs/flash_decode_dspark/prefill_restart_after_isl8192_c16_20260805.log`（第一次，watchdog/cuda synchronize）
* `logs/flash_decode_dspark/prefill_queue_tune_c256_20260805.log`（第二次，保守队列）
* `logs/flash_decode_dspark/results/dspark_isl8192_osl1024_c256_p2560_20260805.jsonl.log`
* `logs/flash_decode_dspark/results/dspark_isl8192_osl1024_c256_p2560_retry2_20260805.jsonl.log`

因此 8192/c512 尚未启动；需要先修复长输入高并发下 DSpark/Mooncake hidden transfer 或 prefill CUDA kernel 卡死，再补测 c256/c512。两次现场均已保留，不能将部分请求当作有效吞吐。

## 6.44 2026-08-05 prefill 高并发卡死修复

根因定位：`sglang/srt/disaggregation/prefill.py::_write_dspark_hidden_rows_for_batch` 为读取 DSpark auxiliary hidden，原先调用全设备 `torch.cuda.synchronize()`。长输入高并发时，该调用同时等待 Mooncake/DeePEP 等无关 stream，形成跨 stream 等待环；watchdog 栈表现为 `deepseek_v4.forward -> cudaDeviceSynchronize`，并非主机或 GPU 显存 OOM。

修复前已备份为 `backups/sglang_prefill_before_forward_stream_sync_20260805.py`。现改为只同步 scheduler 的 `forward_stream`（无该属性时才回退当前 CUDA stream），避免把 transfer stream 纳入同步。修复后的 prefill 日志为 `logs/flash_decode_dspark/prefill_forward_stream_sync_fix_20260805.log`。

修复验证：8192/1024、并发256、2560 prompts 完整成功（2560/2560），结果文件 `logs/flash_decode_dspark/results/dspark_isl8192_osl1024_c256_p2560_syncfix_20260805.jsonl`：输入 16882.23 tok/s、输出 2110.28 tok/s、总吞吐 18992.51 tok/s、Mean TTFT 79947.74ms、Mean TPOT 41.77ms，duration 1242.22s。此前固定在512或2429请求处停止的问题未再出现。

### 6.45 与 MTP 吞吐差距的进一步定位

用户提供的 MTP 对照中 8192/c256 输出 7370.75 tok/s、TPOT 7.45ms，而 DSpark 修复后仍为 2110.28 tok/s、41.77ms。decode 日志显示 DSpark 接受率约 0.90--0.97，故主要矛盾不是 draft 接受率，而是 PD hidden transfer 的排队/等待。

关键资源关系：prefill 的 `SGLANG_PD_HIDDEN_POOL_TOKENS` 默认仅 16384 行；8192 输入每个请求占 8192 行，最多同时保存两个请求。若 transfer 窗口放大到默认 64/2GiB，第三个请求会等待 source pool 行释放，可能与 Mooncake bootstrap/transfer 形成循环等待；此前为绕开它临时使用 16/1GiB 虽能完成测试，但将传输窗口压小，直接造成 TPOT 和输出吞吐显著劣化。按实验并发设置，pool 至少应满足 `ISL × 并发`（c16=131072、c256=2097152、c512=4194304 行），并在每次改变 prefill 实例后重启 router，避免旧 bootstrap 注册残留。

131072 行 pool 的一次复测因 prefill/router bootstrap 注册残留（日志出现 `port 8998 ... connection refused`）无效，未纳入结果；需在清理旧 router/bootstrap 后用大 pool + 64/2GiB 窗口重新跑 c16，再跑 c256/c512。
### 6.46 2026-08-15 当前容器 RDMA 状态与 TCP 诊断

- 新容器启动 1P1D 时，GPU4/5 对应的 `mlx5_4`、`mlx5_9` 处于 `Down/Polling`，而 `mlx5_0..3`、`mlx5_10..11` 为 `Active/LinkUp`。使用活动端口替换 decode 映射后，Mooncake 仍大量报 `Failed to register memory ... Bad address [14]`，prefill 端随即报 `remote mooncake session ... is not alive`；因此这轮失败发生在 KV transfer/RDMA 建链阶段，不是 DSpark verify 或 CUDA Graph。
- 为隔离代码问题，已备份并为两个启动脚本增加 `SGLANG_DISAGGREGATION_TRANSFER_BACKEND` 环境覆盖。使用 `mooncake_tcp` 启动同一 TP4/DP4/EP4、DSpark、DeepEP low-latency、graph bucket 最大256的 1P1D 后，三端健康，单条“你是谁？”成功返回，证明当前 DSpark PD hidden 注入链路本身可用。
- 本轮 TCP 仅用于通信故障诊断，不计入性能矩阵；正式目标仍需在恢复 RDMA/NIC 后用 `mooncake` 重跑。
- 本轮日志：`logs/flash_decode_dspark/aug15/prefill_tcp.log`、`decode_tcp.log`、`router_tcp.log`；RDMA 失败日志：`prefill_pool2200k.log`、`decode_pool64k.log`、`decode_ib58.log`。

补充验证：2026-08-15 又分别尝试了自动选择全部活动 HCA、以及将四个 decode rank 全部绑定到活动的 `mlx5_0`。两种配置仍在启动阶段出现约 603 MB KV buffer 的 RDMA `Bad address [14]` 注册失败，单请求均返回 remote session dead。说明不是单纯的 mlx5_4/9 映射错误；当前容器/主机的 Mooncake RDMA memory registration 状态需要在容器或 Fabric 层恢复后再进行正式吞吐测试。为便于继续实验，当前服务已恢复到可用的 `mooncake_tcp` 配置。

进一步诊断：活动 HCA `mlx5_10` 的 `ib_write_bw` 本地 loopback 在 1 MiB 和 603,383,040 bytes 两种消息大小下均成功（大块平均约 33.8 GB/s），且当前进程 memlock 为 unlimited；但直接用 Mooncake `batch_register` 注册约 301 MB CUDA buffer 时，底层仍报告 `Bad address [14]`。因此故障已缩小为 Mooncake/CUDA GPU memory registration，而非 IB 链路或 Linux memlock。TCP 服务继续保留用于 DSpark 功能回归，不能作为正式吞吐矩阵。

`MC_IB_PCI_RELAXED_ORDERING=1` A/B 测试也未改变 CUDA buffer 注册失败；同样大小的 CPU pinned buffer 可注册，而 CUDA buffer 失败，进一步确认问题只在 GPU memory registration。
# 2026-08-15：nvidia_peermem 修复 Mooncake RDMA，并定位 DSpark c16 回归

- 用户修复并加载了 `nvidia_peermem`（`lsmod` 可见 `nvidia_peermem` 挂在 `nvidia` 上）。修复前 Mooncake CUDA buffer 注册返回 `Bad address`；修复后在 GPU6/`mlx5_10` 上直接执行 `batch_register` 返回 `0`，不再出现该错误。
- 使用 Mooncake RDMA 的 1P1D 服务已恢复：prefill `30000`、decode `30001`、router `13784` 均健康，单条聊天请求成功返回。服务日志保存在 `logs/flash_decode_dspark/aug15/*rdma*`。
- 发现 DSpark decode 的 DP-attention 参数回归：`server_args.py` 会把 `chunked_prefill_size` 按 DP 再除一次（1024→256），已在备份 `backups/server_args_before_dspark_chunked_fix_20260815.py` 后修改为 decode+DSPARK 保留 1024。该配置随后稳定启动。
- 发现 `dspark_kv_inject.py` 新增的 stale-location 校验在每次注入调用 `.item()`，会同步 GPU stream；已备份为 `backups/dspark_kv_inject_before_hotpath_gate_20260815.py`，并新增 `SGLANG_DSPARK_VALIDATE_PD_HIDDEN=1` 诊断开关，默认关闭热路径同步。
- 1024/1024/c1（10 请求）RDMA 结果：Total `545.61` tok/s，Mean TTFT `386.86` ms，Mean TPOT `3.29` ms；高于 MTP-only `541.67` tok/s。
- 1024/1024/c16（160 请求）在上述修复后：Total `5522.55` tok/s，Mean TTFT `622.39` ms，Mean TPOT `5.00` ms；仍低于 MTP-only `5776.13` tok/s，不能作为最终通过结果，需继续优化。
- DSpark decode 脚本默认 HCA 已改为当前活动的 `mlx5_10/11`（`mlx5_4/9` 当前为 Down）；仍可用 `SGLANG_DISAGGREGATION_IB_DEVICE` 覆盖。
- 同一 hot-path gate 配置下补测 8192 输入：c1（10 请求）Total `2123.00` tok/s、TTFT `1155.80` ms、TPOT `3.11` ms；c16（160 请求）Total `12035.39` tok/s、TTFT `3127.12` ms、TPOT `8.72` ms。8192/c1、c16 均仍未超过对应 MTP（分别 `2351.21`、`24427.11` tok/s），说明当前主要瓶颈已从 RDMA 注册错误转为长上下文 target verify/PD 调度开销，不能把服务恢复误认为性能通过。
- 关闭默认 `SGLANG_DSPARK_DEBUG_MAIN_OUTPUT` 后重新测 1024/c16，结果显著恢复：160/160 完成，Total `6321.55` tok/s、Output `3160.78` tok/s、TTFT `578.82` ms、TPOT `4.32` ms，超过 MTP `5776.13` tok/s。此前 debug=1 会在 DSpark PD tail/merge 热路径逐请求打印结构化日志，导致 c16 结果降至约 5523 tok/s。
- 当前 IB 枚举：高速 400G Active 为 `mlx5_0–3`、`mlx5_10–11`；`mlx5_4` 和 `mlx5_9` 为 Down/Polling/10G；`mlx5_5–8` 虽 Active 但仅 100G 且 PCIe Gen4。故目前确实只有 6 张可用的高速 IB 卡，不能用 5–8 等价替换原 MTP 的 8 张高速卡；需恢复 4、9 的链路或由网络侧提供对应 HCA。

## 6.47 2026-08-15 重新启动 prefill 后的 8192 输入复测与 IB 卡差异

- 在 `SGLANG_DSPARK_DEBUG_MAIN_OUTPUT=0`、prefill 使用 RDMA `mlx5_0..3`、decode 使用 graph256/DeepEP `low_latency`、`SGLANG_CHUNKED_PREFILL_SIZE=1024` 的当前服务上，8192/1024/c1 完整完成 10/10：`rdma_debug0_prefill_isl8192_osl1024_c1_n10.jsonl`，Total **2383.41 tok/s**，Output 264.82 tok/s，Mean TTFT 676.95 ms，Mean TPOT 3.12 ms；略高于 MTP c1 的 2351.21 tok/s。
- 同配置 c16 完整完成 160/160：`rdma_debug0_prefill_isl8192_osl1024_c16_n160.jsonl`，Total **16336.97 tok/s**，Output 1815.22 tok/s，Mean TTFT 1852.30 ms，Mean TPOT 6.81 ms。相较旧 prefill debug 输出开启时约 12035 tok/s 有明显提升，但仍低于 MTP c16 的 24427.11 tok/s。
- 该差距与 IB 资源不对等相符：当前 decode 脚本为了避开 Down 端口临时使用 `{"4":"mlx5_10","5":"mlx5_11","6":"mlx5_10","7":"mlx5_11"}`，四个 decode rank 实际只有两个唯一 400G HCA；重复映射不会增加带宽，也不等价于 MTP 原先使用的 8 张高速 IB 卡。`mlx5_4`/`mlx5_9` 当前仍为 Down/Polling，`mlx5_5..8` 虽 Active 但为 100G、PCIe Gen4，不能直接视作等价替换。
- 因此 8192/c16 低于 MTP 的首要外部条件是高速 HCA 数量/链路不完整；在宿主机恢复 `mlx5_4`、`mlx5_9` 为 400G Active 前，不应把重复 10/11 的结果与 8 卡 MTP 做公平比较。当前 1024/c16 已通过，8192/c1 已通过；8192/c16 仍标记为待恢复 8 卡后复测。

## 6.48 2026-08-15 MTP 两 HCA 对照与 prefill DSpark pool 隔离

- 为验证“DSpark 低吞吐是否仅由 HCA 数量减半造成”，将 MTP decode 与 DSpark decode 都绑定到两张唯一的 400G HCA：逻辑 GPU `0..3 -> {mlx5_10, mlx5_11, mlx5_10, mlx5_11}`。MTP 的 `CUDA_VISIBLE_DEVICES=4,5,6,7` 下必须使用逻辑键 `0..3`，使用物理键 `4..7` 会触发 `No IB devices configured for GPU 0`，该错误已单独记录在 `mtp_2hca_decode.log`。
- 当前源码中 prefill 会从基础 DeepSeek-V4-Flash 配置自动解析 DSpark target layers，即使 decode 是 MTP，也会建立 DSpark hidden pool；MTP 的 EAGLE metadata 与该 pool 不兼容，出现 `DSPARK_HIDDEN state index length mismatch`。修复前备份为 `backups/scheduler_before_mtp_dspark_pool_gate_20260815.py`，新增环境开关 `SGLANG_DISABLE_DSPARK_PD_HIDDEN=1`，仅在 MTP 对照 prefill 启动时使用；DSpark 默认行为不变。
- MTP 两 HCA、8192/1024/c1：`mtp_2hca_nodsparkpool_isl8192_osl1024_c1_n10.jsonl`，10/10，Total **2068.84 tok/s**，Mean TTFT 235.01 ms，Mean TPOT 4.12 ms。
- MTP 两 HCA、8192/1024/c16：`mtp_2hca_nodsparkpool_isl8192_osl1024_c16_n160.jsonl`，160/160，Total **22430.32 tok/s**，Output 2492.26 tok/s，Mean TTFT 573.50 ms，Mean TPOT 5.65 ms。
- 对比原 8 HCA MTP c16 Total 24427.11 tok/s，两 HCA 下降约 8.2%；对比当前两 HCA DSpark c16 Total 16336.97 tok/s，DSpark 仍低约 27.2%，所以 HCA 数量不足只能解释一部分差距，主要剩余问题仍在 DSpark PD hidden transfer/调度路径。两 HCA MTP 结果日志：`logs/flash_decode_dspark/aug15/mtp_2hca_decode_nodsparkpool.log`、`router_mtp_2hca_nodsparkpool.log`、`prefill_mtp_2hca_nodsparkpool.log`。

## 6.49 2026-08-15 两 HCA、1024 输入并发 256 的 MTP/DSpark 公平对照

- 两边均使用 decode 映射 `{"0":"mlx5_10","1":"mlx5_11","2":"mlx5_10","3":"mlx5_11"}`，prefill 使用 `mlx5_0..3`，TP4/DP4/EP4、DeepEP low-latency、disable-radix-cache；DSpark 开启 target+draft graph256。
- DSpark graph256（DeepEP dispatch cap=1024）完成 2560/2560：`logs/flash_decode_dspark/aug15/results/final_cuda_pool/isl1024_osl1024_c256_n2560_graph256_cap1024.jsonl`，Total **38415.21 tok/s**，Output 19207.60 tok/s，Mean TTFT 2907.84 ms，Mean TPOT 9.74 ms。
- MTP EAGLE graph128（graph256 在相同显存约束下启动 capture 阶段因显存压力退出）完成 2560/2560：`logs/flash_decode_dspark/aug15/results/mtp_2hca_graph128_isl1024_osl1024_c256_n2560.jsonl`，Total **40296.43 tok/s**，Output 20148.21 tok/s，Mean TTFT 915.44 ms，Mean TPOT 11.33 ms。
- 因此在相同“两张唯一高速 IB”条件下，DSpark 相比可稳定启动的 MTP graph128 低约 **4.66%**，但 TPOT 低约 **14.0%**；这说明此前 DSpark c256 的明显落后不能归因于“只有两张 IB”这一项，IB 数量是影响因素但不是唯一根因。MTP graph256 未能作为吞吐基线（启动即显存失败），不能与 DSpark graph256 直接比较。

## 6.50 2026-08-16 DSpark 最终两 HCA 配置与 MTP 对照

- 最终稳定配置：prefill `mlx5_0..3`、`--chunked-prefill-size 24576 --max-prefill-tokens 24576 --mem-fraction-static 0.83`；decode `mlx5_10/11` 两张唯一 400G HCA、`--chunked-prefill-size 256`、DeepEP `low_latency`、decode CUDA graph `1,2,4,8,16,32,64,128,256`；`SGLANG_DSPARK_PD_HIDDEN_INJECT_REQS=64`。所有请求均使用 `random_range_ratio=1.0`，无服务崩溃或传输错误。
- 1024/1024：c1 `results_final_c1.jsonl` Total **580.39** tok/s；c16 `results_final_c16.jsonl` Total **6517.93** tok/s；c256 `results_final_c256.jsonl` Total **53064.28** tok/s；c512 `results_chunk24k_decode256_c512.jsonl` Total **64454.01** tok/s。
- 两 HCA MTP 对照：1024/c256 Total **40296.43** tok/s，1024/c512 Total **60448.58** tok/s。因此最终 DSpark 在 c256 提升约 **31.7%**，在 c512 提升约 **6.6%**；c1/c16 也高于既有对应 MTP 基线。
- 8192/1024：最终配置 c1 `results_final_8192_c1.jsonl` Total **2763.33** tok/s，c16 `results_final_8192_c16.jsonl` Total **26776.76** tok/s；两项均高于两 HCA MTP 的 2068.84、22430.32 tok/s。
- 传输源码新增的 `SGLANG_DSPARK_OVERLAP_KV_HIDDEN=1` 为实验开关，备份为 `backups/mooncake_conn.py.before_hidden_kv_overlap_v2_20260816` 和 `backups/common_utils.py.before_hidden_kv_overlap_v2_20260816`，默认关闭；c16 有提升但 c256/c512 带宽争用，不作为最终启动配置。
- 1024/c512 达标的关键是实际启用 24K prefill（同时显式增大 `--max-prefill-tokens`）；仅修改 `--chunked-prefill-size` 而保留 `max_prefill_tokens=16384` 会被上限截断，且 mem_fraction=0.80 结果下降到 54606.70 tok/s。

## 6.51 2026-08-16 恢复 8 张物理 IB 后的 DSpark 8192/c256

- 通过 PCI 拓扑确认实际 8 张物理 HCA 为 `mlx5_0, mlx5_1, mlx5_2, mlx5_3, mlx5_4, mlx5_9, mlx5_10, mlx5_11`；`mlx5_5..8` 是同一 PCI 设备的多功能端口，不作为四张独立卡使用。
- 8-HCA 启动映射：prefill `{"0":"mlx5_0","1":"mlx5_1","2":"mlx5_2","3":"mlx5_3"}`；decode `{"0":"mlx5_4","1":"mlx5_9","2":"mlx5_10","3":"mlx5_11"}`。prefill/decode hidden pool 均为 524288，decode 仍使用 DeepEP `low_latency` 与 CUDA graph bucket 最大 256。
- 8192/1024/c256（2560 prompts）完整成功：`logs/flash_decode_dspark/aug16/results_final_8192_c256_pool524k_8ib.jsonl`，2560/2560，duration **521.76 s**，Total **45218.37 tok/s**，Output **5024.26 tok/s**，Mean TTFT **43749.44 ms**，Mean TPOT **6.11 ms**。
- 本轮无 hidden-pool blocked、DSpark hidden packet 发送失败、OOM 或连接丢失；相较之前 2-HCA 长输入 c256 卡在 1032/2560，本轮确认 8-HCA + 524K pool 解决了资源耗尽问题。
- 该吞吐仍低于历史 8-HCA MTP 8192/c256（约 66336.74 tok/s），因此“恢复 8 张 IB”解决的是稳定性/传输容量，不是 DSpark 相对 MTP 的全部性能差距；后续需继续分析长输入 prefill 排队与 DSpark hidden 注入调度。
### 2026-08-16：IB 网卡编号与 GPU 编号的区分

单机 8-GPU 拓扑中，不能把 GPU 编号 4、5、6、7 直接当成 IB 网卡编号。当前 400G IB 的实际 HCA 为 `mlx5_0`、`mlx5_1`、`mlx5_2`、`mlx5_3`、`mlx5_4`、`mlx5_9`、`mlx5_10`、`mlx5_11`；`mlx5_5..8` 是独立的 100G 端口。1P1D 使用物理 GPU key 时，prefill GPU 0--3 对应前四个 HCA，decode GPU 4--7 应使用 `{"4":"mlx5_4","5":"mlx5_9","6":"mlx5_10","7":"mlx5_11"}`。脚本中的 GPU key 和 HCA value 必须分别理解，不能改成 `mlx5_4..7`。

本轮复核还发现 b200-4 主机侧 `ibdev2netdev` 显示全部端口 Down；容器内的 Active 状态不能替代主机 Fabric 状态。因此在主机链路恢复前，跨机验证只能使用 `mooncake_tcp`，不能把 TCP 结果作为 8-HCA RDMA 正式吞吐结果。

两台机器的目标部署是每台使用完整 8 张 GPU：Prefill 使用 `TP=8, DP=1`，Decode 使用 `TP=8, DP=1, EP=8`。在这种部署中 Mooncake 的 GPU key 是进程内的 `0..7`，两侧都应映射到实际的 8 个 HCA：`{"0":"mlx5_0","1":"mlx5_1","2":"mlx5_2","3":"mlx5_3","4":"mlx5_4","5":"mlx5_9","6":"mlx5_10","7":"mlx5_11"}`；不能使用 `mlx5_4..7`，也不能把 key 写成仅 decode 物理编号后遗漏前四张卡。

本轮双机角色最终确认：`10.1.17.14` 为 8 卡 Prefill（端口 30000），`10.1.17.13` 为 8 卡 Decode（端口 30001）。此前反向启动的进程和结果均不纳入正式对比。

为支持 TP=8 + DeepEP 的 CUDA Graph，`deepseek_v4_dspark.py` 现将 `ForwardBatch.input_ids` 传入 MoE DP-sync；修改前的源码保存在 `backups/deepseek_v4_dspark.py.before_tp8_graph_fix_20260816`。
- 2026-08-16 8-GPU RDMA role correction and MTP retest: the active topology is **10.1.17.14=Prefill (30000)** and **10.1.17.13=Decode (30001)**, with HCA mapping `mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_9,mlx5_10,mlx5_11` on both sides. DSpark 1024/1024 c1 and c16 completed successfully and are stored under `logs/flash_decode_dspark/aug16/rdma8_dspark/`. During the MTP-only retest, the remote Prefill worker returned `KVTransferError ... Failed due to an unknown reason from another rank`; the MTP c1/c16 numbers must therefore be treated as invalid until Prefill is restarted and a smoke request succeeds. The local MTP Decode itself loaded TP8/EP8 and captured CUDA graphs through batch 256.
## 2026-08-16 两机 1P1D 低于单机的原因

- 两机部署中 Prefill 在 10.1.17.14、Decode 在 10.1.17.13；Prefill 产生的 KV 必须通过 Mooncake/RDMA 跨机搬运。单机时这一步是本机 GPU/NVLink 或本机内存路径，几乎没有跨节点启动、握手和传输等待。
- 该等待发生在 Decode 首 token 之前，因此直接抬高 TTFT，并在并发升高时形成 transfer queue；它不是 DSpark 的接受率或 CUDA graph 问题。已观察到 DSpark 1024/c256 的 Decode 日志中 `#transfer-req` 长时间约 150--176，且 Mean TTFT 约 50.7 s；Decode 侧仍显示 `cuda graph: True`、accept rate 约 0.8。
- ISL=8192 时 KV 体积约为 ISL=1024 的 8 倍，跨机搬运和 Prefill 排队更明显，所以两机结果相对单机下降幅度更大。当前 Prefill `--prefill-max-requests 5` 还会限制同时构建/发送 KV 的请求数，进一步放大高并发 TTFT。
- 结论：两机结果不能直接与单机只按 Decode 算力比较，瓶颈主要是跨机 KV transfer/bootstrap 和 Prefill 排队；应分别记录 TTFT、transfer queue、RDMA 吞吐，再评价 Decode TPOT/接受率。MTP 的 Prefill 必须设置 `SGLANG_DISABLE_DSPARK_PD_HIDDEN=1`，否则 DSpark hidden-pool 会与 MTP 元数据冲突并产生无效 fallback。

## 2026-08-16 两机 DSpark 性能异常的实际根因：ONE_VISIBLE_DEVICE 重映射 HCA

- 启动脚本曾设置 `SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=true`。在每个 TP 子进程只暴露一张 GPU 时，`maybe_reindex_device_id()` 会把该进程的 `gpu_id` 重映射为 `0`。
- Decode/Prefill 的 Mooncake 配置通过 `scheduler.ps.gpu_id` 查找 `--disaggregation-ib-device`，因此 8 个 TP rank 实际全部选择了 `mlx5_0`，而不是分别选择 `mlx5_0, mlx5_1, mlx5_2, mlx5_3, mlx5_4, mlx5_9, mlx5_10, mlx5_11`。这解释了两机吞吐远低于单机以及高并发长输入尾部卡死。
- 已在 DSpark 和 MTP 的两端启动脚本中移除该环境变量，并分别备份旧脚本。重启日志已确认两端均枚举到 8 个不同 HCA。
- 修复后的 DSpark 8-HCA、8192/1024/c16 为 **1366.36 output tok/s、11989.39 total tok/s**；错误单 HCA 配置约为 **391 output tok/s**，说明该修复是性能跃升的主要来源。
- 8192/1024/c256 在修复后完整完成 2560/2560，约 **4612.56 output tok/s、36042.16 total tok/s**。8192/512 的首轮因 Prefill `CUDA out of memory` 无效，已改为 `prefill-max-requests=4`、`chunked-prefill-size=32768`、`max-prefill-tokens=32768` 重测。
- 正式启动脚本：`logs/flash_decode_dspark/aug16/start_prefill_rdma.sh`、`start_decode_local_rdma.sh`；实际 HCA 映射必须保持为 8 个物理 HCA，不能再启用 ONE_VISIBLE 重映射。
## 2026-08-16 两机 8P/8D DSpark 最终对比与 8192/256 瓶颈

在修正 HCA 映射后，DSpark 8 个条件中 7 个超过 MTP；唯一例外是 8192/1024/C=256（40654.72 vs 47389.10 tok/s）。该项 DSpark Decode TPOT=11.71ms，优于 MTP=40.03ms，主要差距来自 Prefill→Decode hidden/KV 排队和 TTFT，而非 Decode kernel。Decode 日志出现 `DSpark decode hidden pool blocked prealloc`，free_rows 仅 444、队列超过 100。

尝试开启 `SGLANG_DSPARK_OVERLAP_KV_HIDDEN=1` 后反而为 34203.82 tok/s、Mean TTFT 26467ms，已关闭并保留备份。尝试 Prefill `--prefill-max-requests 8 --chunked-prefill-size 49152 --max-prefill-tokens 49152` 触发 CUDA OOM（申请 2.94GiB、仅约 2.2GiB 可用），导致请求中止；已恢复稳定的 4/32768 配置。完整表格和实际命令见 `readmes/rdma8_dspark_mtp_comparison_20260816.md`。

补充重测：在 `mem-fraction-static=0.78` 下，8/49152 配置可完成 8192/1024/C=256，结果为 **40588.23 total tok/s**（TTFT 20989.53ms，TPOT 11.65ms），仍低于同步使用 8 张实际 HCA 的 MTP 重测 **48001.22 total tok/s**。因此该配置没有解决瓶颈；正式脚本恢复为 `prefill-max-requests=6`、`chunked-prefill-size=32768`、`max-prefill-tokens=32768`、`mem-fraction-static=0.82`。

队列与并行度消融结论：hidden receive pool=1048576 会在 Decode GPU 申请 24GiB 并触发 GPU6 OOM；传输队列 64→128 会造成 `prealloc_queue/transfer_queue≈130`、GPU 利用率 3--7%；hidden-only 并行完整结果仅 **37141.58 total tok/s**（TTFT 24103ms）；Mooncake 线程池 16→32 在 1280/2560 处长尾停滞。上述均已回退，正式配置保持 pool=524288、队列=64、线程池=16、overlap=0。原始日志分别为 `dspark_decode_parallelonly.log`、`dspark_decode_threads32.log`。

`SGLANG_DSPARK_PD_HIDDEN_INJECT_REQS=16` 完整重测为 **38174.48 total tok/s**（TTFT 23039ms、TPOT 11.32ms），与 64 基线几乎相同；因此不再保留该参数作为性能修复。此前 Prefill max10/49152 试验因旧服务残留造成注册端口冲突并收到 AbortReq，已判定无效；随后清理容器内旧 Prefill/Bootstrap 进程，确认 `10.1.17.14:8998` 单实例监听后，用同一配置（`prefill-max-requests=10`、`chunked-prefill-size=49152`、`max-prefill-tokens=49152`、`mem-fraction-static=0.78`）完成有效重测：8192/1024/C=256，2560/2560 成功，duration 466.05s，Total throughput **50623.71 tok/s**，Mean TTFT 31370.75ms，Mean TPOT 13.07ms；高于 MTP 对照 48001.22 tok/s（+5.46%）。结果见 `logs/flash_decode_dspark/aug16/rdma8_dspark_max10_49152/isl8192_osl1024_c256.jsonl`。

Prefill 捕获路径消融：设置 `SGLANG_DSPARK_CAPTURE_SYNC_MODE=stream` 后完整结果为 **37536.78 total tok/s**（TTFT 23356ms、TPOT 11.65ms），比稳定 inject16 结果低，已回退；当前主要瓶颈不是每个目标层的全局 CUDA 同步。
## 2026-08-16 MTP 对照配置纠正

此前两机目录中的 `start_decode_local_mtp_rdma.sh` 没有 `--speculative-algo/--speculative-algorithm`，因此其所谓 MTP 结果实际是非 speculative baseline，不能用于 MTP 对照。纯 MTP 应使用 DeePEP（不能使用 MegaMoE）并显式配置：

```bash
--speculative-algo EAGLE --speculative-num-steps 3 \
--speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

已废弃无 speculative 参数的 C512 对照，并完成正确纯 MTP/EAGLE 两机重测：`8192/1024/C512`、5120 prompts、8 张实际 HCA，5120/5120 成功，Total throughput **65901.45 tok/s**，Mean TTFT 47244.78ms，Mean TPOT 21.12ms。对应 DSpark 同条件为 50964.34 tok/s，纯 MTP 高 **29.3%**。结果文件：`logs/flash_decode_dspark/aug16/rdma8_mtp_eagle8_c512_20260816/isl8192_osl1024_c512.jsonl`；Decode 实际脚本：`logs/flash_decode_dspark/aug16/start_decode_local_mtp_eagle8_c512_20260816.sh`。
