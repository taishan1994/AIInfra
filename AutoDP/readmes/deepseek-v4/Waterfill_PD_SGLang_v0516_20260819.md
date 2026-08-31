# 原生 SGLang 0.5.16 PD 分离 Waterfill 实验报告

日期：2026-08-19

## 1. 实验目标与源码基线

目标是在未经历史 TBO、DSpark、MegaMoE 等修改的 SGLang 0.5.16 上，将 Waterfill 集成到 PD 分离 decode，并超过
`PD_COMPLETED_REPORT_20260803.md` 中的 baseline。

源码基线为 `/data/ssd2/sglang_v0.5.16`，commit `fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1`，对应 v0.5.16 tag。
DeepGEMM baseline 使用该干净 tag；之后为本轮 FlashInfer MxFP4 + DeepEP low_latency 适配，在同一源码树中加入了显式隔离的 MxFP4 兼容路径。baseline 数值不应与后续适配代码混用。

## 2. 最终可用配置

Prefill 使用原有 PD baseline 服务（GPU 0–3，端口 30000）。Decode 使用 GPU 4–7，端口 30001，Router 端口 13784。

Decode 关键参数：

```text
--tp-size 4 --dp-size 4 --ep-size 4
--moe-a2a-backend deepep
--moe-runner-backend deep_gemm
--deepep-mode low_latency
--enable-waterfill
--cuda-graph-backend-decode full
--cuda-graph-bs-decode 1 2 4 8 16 32 64 128
--disable-custom-all-reduce
SGLANG_OPT_USE_ONLINE_COMPRESS=1
```

Waterfill 在 decode 服务启动日志中实际输出：

```text
Waterfill is enabled with moe_a2a_backend='deepep'.
Prepared 43 Waterfill TopK modules.
```

运行时 DeepEP low_latency dispatch 与 CUDA Graph 均生效。Graph128 是必要的：C512 时每个 DP rank 的 batch 为 128，Graph64 会退回 eager。

## 3. 与 baseline 对比

baseline 数据来自 [PD_COMPLETED_REPORT_20260803.md](PD_COMPLETED_REPORT_20260803.md)。本轮数据均来自 benchmark JSONL，指标顺序为 Out tok/s、Total tok/s、Mean TTFT、Mean TPOT。

| ISL | OSL | Concurrency | 本轮 Out tok/s | baseline Out tok/s | Out 变化 | 本轮 Total tok/s | baseline Total tok/s | Total 变化 | 本轮 TTFT ms | baseline TTFT ms | 本轮 TPOT ms | baseline TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 105.83 | 107.55 | -1.60% | 211.66 | 215.09 | -1.60% | 286.89 | 332.51 | 9.17 | 8.98 |
| 1024 | 1024 | 16 | 1354.95 | 1348.95 | +0.45% | 2709.90 | 2697.90 | +0.45% | 398.52 | 631.91 | 11.40 | 11.25 |
| 1024 | 1024 | 256 | 13786.86 | 13106.67 | +5.19% | 27573.71 | 26213.34 | +5.19% | 1352.34 | 2284.50 | 16.95 | 16.82 |
| 1024 | 1024 | 512 | 17390.36 | 13243.63 | +31.31% | 34780.72 | 26487.25 | +31.31% | 6836.11 | 16554.06 | 21.36 | 21.32 |
| 8192 | 1024 | 1 | 110.38 | 106.52 | +3.63% | 993.45 | 958.65 | +3.63% | 235.89 | 348.65 | 8.84 | 9.05 |
| 8192 | 1024 | 16 | 1378.12 | 1328.45 | +3.74% | 12403.04 | 11956.04 | +3.74% | 644.60 | 607.99 | 10.86 | 11.23 |
| 8192 | 1024 | 256 | 7349.02 | 6861.28 | +7.11% | 66141.15 | 61751.53 | +7.11% | 19551.08 | 21913.55 | 14.24 | 14.34 |
| 8192 | 1024 | 512 | 7401.80 | 7110.57 | +4.10% | 66616.22 | 63995.17 | +4.10% | 53028.33 | 55535.20 | 14.25 | 14.42 |

结论：除 1024/C1 外，其余 7 组 Total tok/s 均超过 baseline；8192 输入的四组全部超过 baseline。高并发目标已达到，且所有对比均包含 Total tok/s、TTFT、TPOT。

## 4. 已验证的 Waterfill A/B

在 1024/C256、其余配置完全相同（DeepGEMM、DeepEP low_latency、Graph64、online c128、PD 路由）的情况下：

| 配置 | Out tok/s | Total tok/s | TTFT ms | TPOT ms |
|---|---:|---:|---:|---:|
| Waterfill 开启 | 13786.86 | 27573.71 | 1352.34 | 16.95 |
| Waterfill 关闭 | 14499.21 | 28998.41 | 1452.47 | 15.91 |

因此可以确认 Waterfill 真实执行，但在本机、随机均匀 workload 和当前 low_latency CUDA Graph 约束下，Waterfill 本身增加了约 4.3% decode 开销；本轮超过旧 baseline 的主要来源是原生 DeepGEMM masked runner、online c128，以及覆盖实际并发 batch 的 Graph64/Graph128。不能将全部收益归因于 Waterfill。

## 5. 失败与修复记录

1. 原生 MxFP4 runner 直接接 DeepEP low_latency dispatch output 时缺少 `topk_output`；临时将 expert-major masked buffer 展平的兼容补丁导致语义错误和严重 padding 计算膨胀，短请求达到 11–13 秒。该补丁已删除，未纳入最终结果。
2. 改用原生 `--moe-runner-backend deep_gemm` 后，使用正确的 masked DeepGEMM 路径，16 token 短请求约 0.41 秒，恢复正常。
3. 未设置 `SGLANG_OPT_USE_ONLINE_COMPRESS=1` 时，DSV4 c128 CUDA Graph capture 出现非法内存访问；开启原生 online c128 后 Graph capture 成功。
4. Graph16 在 C256 以上退回 eager，每卡约 420 tok/s；扩到 Graph64 后 C256 恢复到每卡约 3.7k tok/s。
5. Graph64 在 C512 每卡 batch 128 时仍退回 eager，每卡约 800 tok/s；扩到 Graph128 后 capture 成功并恢复 `cuda graph: True`。
6. 将 `SGLANG_DISABLE_STATIC_WATERFILL=1` 切换到 dynamic Waterfill 后，Graph64 capture 在 bs=64 阶段 scheduler 退出，未作为有效配置使用。最终保留原生 static Waterfill。

## 6. 当前结论

最终配置满足：原生 SGLang 0.5.16、DeepEP low_latency、decode CUDA Graph、PD 分离，并在 7/8 组超过历史 baseline，8192 输入四组全部超过 baseline。

但严格 A/B 表明 Waterfill 当前不是吞吐提升来源，而是一个已生效、但在该随机 workload 上有额外代价的 dispatch 策略。后续若要证明 Waterfill 自身增益，应使用真实共享专家热点/偏斜路由 workload，或继续优化 static Waterfill 的 count 与 materialize 路径；不能用本轮 Graph/DeepGEMM 的收益替代 Waterfill 的独立收益。

## 7. FlashInfer MxFP4 + DeepEP low_latency 适配实验

原生 v0.5.16 的 `flashinfer_mxfp4` 读取 `dispatch_output.topk_output`，与 DeepEP low_latency 的 `DeepEPLLDispatchOutput` 不兼容。适配位于 `python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py`，并在 decode 脚本中固定使用 `--deepep-dispatcher-output-dtype bf16`。

适配内容包括：识别 expert-major LL buffer；保留原始 `topk_ids/topk_weights` 给 DeepEP combine；按静态 `topk_ids.shape` 压缩每 expert 的计算 M；用 GPU `topk(masked_m)` 选择 active experts；结果回填到 DeepEP 完整输出布局；保留 Waterfill 与 CUDA Graph。

适配实验结果（PD Router，ISL=1024，OSL=128）：

| 配置 | 并发 | Out tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms |
|---|---:|---:|---:|---:|---:|
| 初始 E×2048 padding | 1 | 7.84 | 70.60 | 385.90 | 125.43 |
| 固定计算 M=512 | 1 | 12.75 | 114.73 | 338.87 | 76.37 |
| active expert + M 压缩 | 1 | 41.41 | 372.72 | 314.21 | 21.84 |
| PDL + active expert + M 压缩 | 16 | 541.97 | 4877.75 | 592.81 | 24.69 |

`flashinfer_mxfp4_moe_precision=bf16` A/B 的 C16 TPOT 为 25.46 ms，慢于 default；补充 `enable_pdl` 后为 24.69 ms，基本无改善。`masked_m.max().item()` 会在 Graph capture 中触发 host sync，最终已改为静态 shape 推导。

当前适配已经做到服务启动、Graph128 捕获和请求正确返回，但尚未超过 DeepGEMM baseline（C16 baseline TPOT 约 11.25 ms）。主要瓶颈是当前 TRT-LLM MxFP4 routed kernel 不接受 `masked_m`，小 batch 仍有固定 routed-MoE 调度开销。上游当前也仍将 `flashinfer_mxfp4` 作为 standard dispatch 路径，DeepEP low_latency 官方 masked 路径是 DeepGEMM/CuteDSL。因此还需要 FlashInfer masked MXFP8×MXFP4 grouped kernel，或等价的自定义 masked mixed-MoE kernel，才能完成超过 baseline 的目标。

## 8. 2026-08-19 追加：完整 OSL=1024 A/B 与 masked fallback

为确认前面的 OSL=128 结果不是测量长度造成的误判，补测了完整 OSL=1024，并新增了一个显式实验开关：

```text
SGLANG_MXFP4_LL_BACKEND=deep_gemm
DEEPEP_DISPATCHER_OUTPUT_DTYPE=fp8
```

该开关仍使用 `--moe-runner-backend flashinfer_mxfp4` 和 MXFP4 权重格式，只把 DeepEP low_latency 的 masked GEMM 临时转交给已有 DeepGEMM masked 实现，用于验证权重/scale 转换的性能上限；它不是纯 FlashInfer kernel 结果。脚本已支持通过 `DEEPEP_DISPATCHER_OUTPUT_DTYPE` 切换 dtype，默认值仍为 `bf16`，供纯 FlashInfer 路径使用。

| 路径 | ISL | OSL | C | Out tok/s | Total tok/s | TTFT ms | TPOT ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| MxFP4 + DeepGEMM masked fallback | 1024 | 1024 | 1 | 105.55 | 211.10 | 280.19 | 9.21 |
| MxFP4 + DeepGEMM masked fallback | 1024 | 1024 | 16 | 1352.65 | 2705.29 | 394.21 | 11.42 |
| MxFP4 + DeepGEMM masked fallback | 1024 | 1024 | 256 | 13494.94 | 26989.89 | 1222.01 | 17.52 |
| 纯 FlashInfer MxFP4 routed | 1024 | 1024 | 16 | 631.30 | 1262.60 | 581.28 | 24.63 |
| 原 DeepGEMM baseline | 1024 | 1024 | 1 | 105.83 | 211.66 | 286.89 | 9.17 |
| 原 DeepGEMM baseline | 1024 | 1024 | 16 | 1354.95 | 2709.90 | 398.52 | 11.40 |
| 原 DeepGEMM baseline | 1024 | 1024 | 256 | 13786.86 | 27573.71 | 1352.34 | 16.95 |

结论：fallback 在 C1/C16 基本复现 baseline，证明 MXFP4 原始权重、E8M0 scale 和 DeepEP LL 数据本身没有造成数量级损失；但 C256 仍比 baseline 低约 2.1%。纯 FlashInfer C16 的 Out tok/s 只有 631.30，TPOT 24.63 ms，明显低于 baseline，故当前 expert-major flatten + routed kernel 的方案不能作为最终适配。

本轮还验证了两个启动问题：BF16 dispatch 没有 activation scale，不能直接喂 DeepGEMM masked kernel；把 activation group size 误设为 32 会触发 DSV4 masked quant kernel 的 `static_assert(kGroupSize == 128)`。最终 fallback 使用 FP8 dispatch、activation group size 128，权重 recipe 仍保持 `(1,32)`。

当前源码工作树包含上述显式 fallback 和纯 FlashInfer 适配实验代码，默认路径不受 fallback 环境变量影响。要让“纯 flashinfer_mxfp4 + DeepEP low_latency”超过 baseline，下一步必须实现 FlashInfer 侧支持 `masked_m/expected_m` 的 MXFP8 activation × MXFP4 weight grouped kernel，或者将 TRT-LLM routed kernel 改造成真正的 expert-major masked 执行；仅继续调整 padding、active expert 选择或 PDL 参数无法消除当前 1.9 倍左右的 TPOT 差距。

## 9. 2026-08-19 追加：FlashInfer grouped MXFP8×MXFP4 实验

尝试使用 FlashInfer 0.6.14 的 `group_gemm_mxfp8_mxfp4_nt_groupwise`，将 DeepEP low_latency 已完成的 expert-major buffer 按 active expert 分组，分别执行 up/down GEMM。为满足 Graph-safe 约束，计算 M 使用静态 `round_up(batch_tokens, 4)`，不用 `masked_m.item()`；权重和 A scale 按 API 要求转换为 column-major storage。

独立真实 checkpoint 测试表明：G=65、down GEMM 一次性调用时，FlashInfer SM100 grouped kernel 会出现 NaN/Internal Error；拆成每次最多 64 个 group 可以规避该特定问题，`tile_n=256` 也比 `tile_n=128` 稳定。但整模型 decode CUDA Graph capture 仍在 grouped GEMM 内部报 `CutlassMXFP4GroupwiseScaledGroupGEMMSM100 ... cutlass gemm.run failed: Error Internal`，尚未形成可运行的服务，因此没有吞吐结果，不能宣称超过 baseline。

本轮结论：FlashInfer grouped API 可以在独立真实权重上运行，但当前版本在 DeepEP LL 的多层、动态 active expert 和 CUDA Graph 组合下仍不稳定；当前可复现的有效路径仍是 DeepGEMM masked fallback。相关修改均已备份：

```text
backups/mxfp4_flashinfer_trtllm_moe_grouped_torch_activation_20260819.py
backups/mxfp4_flashinfer_mxfp8_grouped_colmajor_20260819.py
backups/mxfp4_flashinfer_mxfp8_grouped_active_colmajor_20260819.py
backups/mxfp4_flashinfer_mxfp8_grouped_tile256_20260819.py
backups/mxfp4_flashinfer_mxfp8_grouped_chunk64_20260819.py
```

## 10. 2026-08-19 追加：有效超过 baseline 的 MxFP4 配置

保留以下外部配置：`--moe-runner-backend flashinfer_mxfp4`、DeepEP `low_latency`、Waterfill，以及 decode Graph `1 2 4 8 16 32 64 128`。针对 DeepEP low-latency 的 expert-major masked buffer，当前有效路径使用显式隔离开关 `SGLANG_MXFP4_LL_BACKEND=deep_gemm`：保留 MxFP4 权重加载和 `flashinfer_mxfp4` 服务配置，但将 masked GEMM 交给已有 DeepGEMM masked runner。

额外配置为 `SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1`、`SGLANG_OPT_USE_JIT_EP_ACTIVATION=1`、`SGLANG_OPT_SWIGLU_CLAMP_FUSION=1`，DeepEP normal dispatch/combine SMS=64。PD Router 真实测试（ISL=1024、OSL=1024、C=256、2560 requests）：

完整的 1024/1024 PD Router 结果如下；每组均为 `10 × concurrency` 请求，包含 Total tok/s、Mean TTFT 和 Mean TPOT：

| ISL | OSL | C | Out tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms | 相对 baseline |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1024 | 1024 | 1 | 105.39 | 210.79 | 285.28 | 9.22 | Out -0.42%，TPOT +0.55% |
| 1024 | 1024 | 16 | 1448.16 | 2896.33 | 459.24 | 10.55 | Out +6.88%，TPOT -7.43% |
| 1024 | 1024 | 256 | 14445.98 | 28891.96 | 1932.32 | 15.35 | Out +4.78%，TPOT -9.44% |
| 1024 | 1024 | 512 | 15211.23 | 30422.45 | 11397.88 | 21.25 | Out -12.53%，TPOT -0.52% |

其中 C256 的对照为：MxFP4 + masked fallback + fixmem + DeepEP SMS64 的 Out/Total 为 `14445.98/28891.96`，原 DeepGEMM baseline 为 `13786.86/27573.71`，TTFT 为 `1932.32/1352.34 ms`，TPOT 为 `15.35/16.95 ms`。C16 和 C256 均超过 baseline；C1 基本持平，C512 仍未超过 baseline，且高并发 TTFT 明显变差。因此当前结果应表述为“在 C16/C256 中并发超过 baseline”，不能表述为所有并发均超过。

结果文件：

```text
logs/mxfp4_fallback_fixmem_sms64_20260819_c1.jsonl
logs/mxfp4_fallback_fixmem_sms64_20260819_c16.jsonl
logs/mxfp4_fallback_fixmem_sms64_20260819_c256.jsonl
logs/mxfp4_fallback_fixmem_sms64_20260819_c512.jsonl
```

同一配置补测长输入 8192/1024：

| ISL | OSL | C | Out tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms | 相对历史 baseline |
|---:|---:|---:|---:|---:|---:|---:|---|
| 8192 | 1024 | 1 | 110.74 | 996.68 | 232.06 | 8.81 | Out +0.33%，TPOT -0.34% |
| 8192 | 1024 | 16 | 1481.09 | 13329.80 | 581.93 | 10.10 | Out +7.48%，TPOT -6.99% |

长输入 C1/C16 也已超过或基本超过历史 baseline；8192/C256、C512 本轮未在该最终 fallback 配置下重复，避免把其他阶段的历史结果混入当前配置结论。

结果文件：

```text
logs/mxfp4_fallback_fixmem_sms64_20260819_isl8192_c1.jsonl
logs/mxfp4_fallback_fixmem_sms64_20260819_isl8192_c16.jsonl
```

## 11. 2026-08-19 追加：源码路径审计与 grouped 修复

重新审计启动链路时发现，之前的 `flash_decode_waterfill.sh` 没有设置 `PYTHONPATH`，服务实际导入的是 `/sgl-workspace/sglang`，而修改和备份的原生 v0.5.16 工作树是 `/data/ssd2/sglang_v0.5.16`；两份源码并不相同。因此第 10 节中使用旧启动链路得到的 fallback 吞吐结果，不能作为当前工作树修改已生效的证据，后续必须重新启动并重测。

启动脚本现已固定：

```text
PYTHONPATH=/data/ssd2/sglang_v0.5.16/python:${PYTHONPATH:-}
```

在真正的 v0.5.16 grouped 路径上，Graph bs=1 已同步定位到 FlashInfer `group_gemm_mxfp8_mxfp4_nt_groupwise` 的 down-GEMM，而不是 attention。根因修复方向是：父类 `Fp8MoEMethod` 会在 `flashinfer_mxfp4` 服务后端下自动执行 TRT-LLM 专用权重/scale shuffle；纯 grouped 模式现通过 `SGLANG_MXFP4_LL_BACKEND=flashinfer_grouped` 跳过该 shuffle，再由 grouped 适配函数自行转换为列主序 B、列主序 activation scale 和行主序 weight scale。该条件修改位于 `python/sglang/srt/layers/quantization/fp8.py`，普通 TRT-LLM/fallback 路径不受影响。

本次新增备份：

```text
backups/mxfp4_flashinfer_before_pure_grouped_retry_20260819.py
backups/mxfp4_flashinfer_before_down_tile_sweep_20260819.py
backups/mxfp4_flashinfer_before_grouped_metadata_debug_20260819.py
backups/mxfp4_flashinfer_before_grouped_layer_trace_20260819.py
backups/fp8_before_grouped_alignment_bypass_20260819.py
backups/flash_decode_waterfill_before_graph_isolation_20260819.sh
backups/flash_decode_waterfill_before_pythonpath_fix_20260819.sh
backups/mxfp4_flashinfer_grouped_alignment_bypass_current_20260819.py
backups/fp8_grouped_alignment_bypass_current_20260819.py
backups/flash_decode_waterfill_pythonpath_graph_override_current_20260819.sh
backups/mxfp4_flashinfer_dynamic_groups_current_20260819.py
backups/fp8_dynamic_groups_current_20260819.py
```

当前尚未完成修复后的真实服务启动和吞吐验证：连续 grouped 失败后 GPU 4–7 保留约 94.5 GiB 显存，但 NVML 看不到对应进程；`nvidia-smi --gpu-reset -i 4,5,6,7` 报设备仍被其他客户端占用。纯 FlashInfer grouped 超过 baseline 的目标保持未完成，待 GPU 状态清理后继续验证。

## 12. 2026-08-19 追加：真实 checkpoint 单层验证

在完整服务无法启动期间，使用真实 checkpoint 的 layer0 权重进行脱离服务验证。读取了 65 个 expert 的真实 `w1/w2/w3` 和 `float8_e8m0fnu` scale，按 grouped 路径转换为 expert-major、B 列主序、activation scale 列主序、weight scale 行主序；FlashInfer 0.6.14 的 up/down GEMM 均返回 finite 输出。

| grouped expert 数 | M/expert | up/down 结果 |
|---:|---:|---|
| 6 | 128 | 成功，down 输出 `[768, 4096]`，finite |
| 16 | 128 | 成功，down 输出 `[2048, 4096]`，finite |
| 64 | 128 | 成功，down 输出 `[8192, 4096]`，finite |

这证明父类 TRT-LLM shuffle bypass 后，真实权重布局和小 group 数均可被 FlashInfer grouped kernel 接受。为避免 C1/C16 无谓计算，当前实现将默认的最小 group 数从 64 改为 1，并保留 `SGLANG_FLASHINFER_GROUPED_MIN_GROUPS=64` 作为旧行为回退开关。该结果仍不是端到端吞吐证据；完整 43 层 CUDA Graph 和 PD Router benchmark 必须在显存恢复后完成。

另外，`segment_m >= 256` 时现在自动使用 FlashInfer 文档推荐的 `mma_sm=2`，小 batch 仍使用 `mma_sm=1`；可通过 `SGLANG_FLASHINFER_GROUPED_MMA_SM=1/2` 强制覆盖。真实 layer0、G=16、M=256 的 up/down grouped GEMM 使用 `mma_sm=2` 成功且输出 finite。

新增纯 grouped 启动入口：`flash_decode_waterfill_pure_grouped.sh`。它固定 `SGLANG_MXFP4_LL_BACKEND=flashinfer_grouped`、默认 `SGLANG_FLASHINFER_GROUPED_MIN_GROUPS=1`、DeepEP SMS=64，并继承基础脚本的原生 v0.5.16 `PYTHONPATH`、Waterfill 和 CUDA Graph 设置。

需要明确区分：此前“超过 baseline”的结果来自 `DeepGEMM fallback`，不能作为本次纯 FlashInfer grouped 目标的结果；它只能说明外部 `flashinfer_mxfp4` 配置、MxFP4 权重、DeepEP `low_latency` 和 Waterfill 的组合在 fallback 路径上可行。当前纯 FlashInfer grouped 已完成真实 checkpoint 单层和小 group 的 finite 验证，但尚未完成 43 层 CUDA Graph、PD Router 和吞吐复现，因此不能宣称已经超过 baseline。

本次有效代码备份：

```text
backups/mxfp4_flashinfer_final_fallback_fixmem_sms64_20260819.py
backups/flash_decode_waterfill_final_config_20260819.sh
backups/mxfp4_flashinfer_dynamic_mma_sm_current_20260819.py
backups/flash_decode_waterfill_pure_grouped_current_20260819.sh
backups/mxfp4_flashinfer_before_expected_m_segment_20260819.py
```

## 13. 2026-08-19 追加：按 DeepEP expected_m 修正 grouped segment

审计 DeepEP low-latency 接口后确认，dispatch output 除 `masked_m` 外还返回 `expected_m`。`hidden_states` 是跨 rank 聚合后的固定专家 buffer，本地 `batch_tokens` 不是每个专家的实际/期望 token 数。此前 grouped 路径直接使用 `batch_tokens` 生成 `m_indptr`，在 EP 场景下会造成 segment 估计偏大，或者在跨 rank 热点专家场景下存在截断风险。

当前默认改为使用 `dispatch_output.expected_m`，再按 4 对齐并至少保留一个 128-row tile；`SGLANG_FLASHINFER_GROUPED_SEGMENT_M` 可显式覆盖旧实验的 segment。该修改只影响 `flashinfer_grouped` 路径，普通 TRT-LLM/fallback 不变。由于 GPU 仍被不可见的残留上下文占用，本轮只能完成源码审计和静态校验，真实 Graph/吞吐结果待设备恢复后补测。

## 14. 2026-08-19 追加：按 DeepEP 通信组修正 active expert 数量

进一步审计发现，`masked_m` 统计的是 DeepEP 通信组内所有 rank 汇聚到本地专家的接收数量。原实现使用 `batch_tokens * top_k` 估计 active expert 数量，只覆盖本 rank 路由；在小 decode batch 下可能只选择部分非零专家，导致合法专家输出未参与 combine。

当前改为使用 `batch_tokens * dispatch_group_size * top_k`，其中 `dispatch_group_size` 来自 DeepEP 使用的 TP device group，并限制在 `num_local_experts` 以内。该修复与上一节的 `expected_m` segment 修复配套：前者保证不漏专家，后者控制每个专家的 GEMM 行数。修改前后备份如下：

```text
backups/mxfp4_flashinfer_before_group_size_active_count_20260819.py
backups/mxfp4_flashinfer_group_size_active_count_current_20260819.py
```

本轮仍因 GPU 残留显存无法完成真实 CUDA Graph 和吞吐验证。

## 15. 2026-08-19 追加：FlashInfer grouped chunk 上限收紧到 32

使用真实 checkpoint 的 layer0 权重、M=128 复测发现：G=28 的 up/down 均 finite；G=32 的 up/down 均 finite；但 G=65 按 64+1 分块时，首个 G=64 chunk 的 up/down 返回 non-finite。将同一测试改为 32+32+1 分块后，三个 chunk 的 up/down 全部 finite。

因此纯 grouped 路径的 `max_groups_per_call` 从 64 收紧为 32。该限制只增加 grouped kernel 调用次数，不改变专家选择和结果布局；它规避了 FlashInfer 0.6.14 在真实 DSV4 权重、M=128、64-group launch 下的非有限输出问题。该测试仍是单层真实权重验证，不等价于完整服务 CUDA Graph 吞吐结果。

本轮备份：

```text
backups/mxfp4_flashinfer_before_chunk32_20260819.py
```

## 16. 2026-08-19 追加：真实权重 grouped CUDA Graph kernel-only 验证

在 GPU0 的低显存环境中，使用真实 layer0 checkpoint 权重和 scale，固定 G=32、M=128、`tile_m=128`、`tile_n=256`、`tile_k=128`、`mma_sm=1`，对 up/down 两个 grouped GEMM 做 CUDA Graph warmup、capture，并 replay 5 次。结果为：

```text
cuda graph replay: ok
up/down shape: [4096, 4096]
```

每次 replay 的 up/down 输出均 finite。这证明 G=32 分块不仅 eager 可运行，也能进入 CUDA Graph；但该测试只覆盖 grouped kernel，不包含 DeepEP 通信、43 层模型和 PD Router 吞吐。

## 17. 2026-08-19 追加：包含 active expert/mask/quantize 的 Graph 验证

进一步构造与服务 decode 形态一致的低显存测试：本地专家数 65、通信组 4、`top_k=7`、C1，因此 active expert 上界为 28；`masked_m` 只给前 28 个专家有效计数，其余专家为零，segment 固定为 128。测试完整执行了 `topk(masked_m)`、专家索引、zero padding、MXFP8 quantize、SwiGLU、G=28 的 up/down grouped GEMM，并 capture/replay CUDA Graph 3 次。结果：

```text
core grouped graph with topk/mask/quantize: ok
output shape: [3584, 4096]
```

这验证了当前 active expert 修复与 32-group chunk 在 Graph 内可以组合运行；仍不代表 DeepEP IBGDA 通信和完整服务已经通过。

## 18. 2026-08-19 追加：补齐 DeepSeek-V4 clamped SwiGLU 语义

检查 checkpoint 的 `config.json` 确认 `swiglu_limit=10.0`。原 grouped 路径只执行普通 `SiLU(gate) * up`，与 SGLang 原生 DeepSeek-V4 路径不一致。当前已在 down activation quantize 之前补齐：

```text
gate = gate.clamp(max=swiglu_limit)
up   = up.clamp(min=-swiglu_limit, max=swiglu_limit)
down_input = silu(gate) * up
```

这样纯 FlashInfer grouped 与 baseline 使用相同的 SwiGLU 数值语义。修改前备份：

```text
backups/mxfp4_flashinfer_before_swiglu_clamp_20260819.py
```

## 19. 2026-08-19 追加：补齐 C256/C512 的 CUDA Graph decode batch

历史 baseline 对比包含 C256 和 C512，但两个启动入口默认只注册到 decode Graph batch 128。为保证高并发对比仍使用 CUDA Graph，已将基础入口和纯 grouped 入口的默认列表统一扩展为：

```text
1 2 4 8 16 32 64 128 256 512
```

用户仍可通过 `CUDA_GRAPH_BS_DECODE` 显式覆盖。相关脚本备份：

```text
backups/flash_decode_waterfill_before_graph_bs256_512_20260819.sh
backups/flash_decode_waterfill_pure_grouped_before_graph_bs256_512_20260819.sh
```

## 20. 2026-08-19 追加：M=64 可运行但未采用为默认值

使用真实 layer0 MXFP4 权重验证 G=32、M=64：up/down eager 均 finite；包含 clamp、MXFP8 quantize 和 CUDA Graph capture/replay 的核心路径也连续 replay 成功且 finite。M=64 保留为可选实验参数，但没有直接替换默认值。

```text
backups/mxfp4_flashinfer_before_segment_min64_20260819.py
```

## 21. 2026-08-19 追加：M=64/M=128 kernel 定时对比

使用相同真实权重、G=32、tile 配置和 20 次重复测量，up+down 平均耗时如下：

| segment M | avg up+down | finite |
|---:|---:|---|
| 64 | 0.229 ms | 是 |
| 128 | 0.209 ms | 是 |

由于 M=64 在 SM100 上受到 tile/scheduling 固定开销影响，实际略慢于 M=128。因此当前源码默认恢复 `segment_m >= 128`；如需专项实验仍可设置 `SGLANG_FLASHINFER_GROUPED_SEGMENT_M=64`。本次恢复前备份：

```text
backups/mxfp4_flashinfer_before_revert_segment128_20260819.py
```

## 22. 2026-08-19 追加：SM100 grouped tile_n 调优到 192

使用真实 layer0 权重、G=32、M=128、20 次重复测量：

| tile_n | avg up+down | finite |
|---:|---:|---|
| 192 | 0.165 ms | 是 |
| 256 | 0.214 ms | 是 |

随后使用 G=65 的真实权重按 32+32+1 分块复测，tile_n=192 的所有 up/down chunk 均 finite。当前 grouped 路径将 up/down 默认 tile_n 调整为 192；down 仍可通过 `SGLANG_FLASHINFER_GROUPED_DOWN_TILE_N` 覆盖。修改前备份：

```text
backups/mxfp4_flashinfer_before_tile_n192_20260819.py
```

## 23. 2026-08-19 追加：tile_n=192 CUDA Graph 复测

真实 layer0 权重、G=32、M=128、up/down 均使用 `tile_n=192`，包含 clamp 和 MXFP8 activation quantize 的核心路径完成 CUDA Graph capture，并 replay 5 次；结果为：

```text
tile_n192 graph: ok
output shape: [4096, 4096]
```

每次 replay 输出均 finite，确认 tile_n=192 的性能优化没有破坏 Graph 稳定性。

## 24. 2026-08-19 追加：去除 grouped chunk 的 torch.cat 拷贝

FlashInfer grouped API 支持传入预分配的 `out` buffer。当前实现为 gate/up 和 down 分别预分配连续输出，把每个最多 32-expert chunk 直接写入对应切片，删除原来的 `gateup_parts/output_parts + torch.cat`。真实 layer0 权重、G=65、M=128、32+32+1 分块、tile_n=192 验证结果为三个 chunk 的 up/down 全部 finite。

修改前备份：

```text
backups/mxfp4_flashinfer_before_preallocated_chunk_outputs_20260819.py
```

本次 G=32、M=128、tile_n=192 预分配 buffer 定时为 `0.164 ms`（20 次平均，输出 finite）；与此前约 `0.165 ms` 的 eager 结果基本持平，说明该修改主要减少中间内存和拷贝压力，单层 kernel 总耗时提升有限。

另外，预分配 `out=` 路径使用真实 G=32、M=128、tile_n=192 完成 CUDA Graph capture，并 replay 5 次；结果为 `preallocated graph: ok`，输出 `[4096, 4096]` 且每次 finite。

## 25. 2026-08-19 追加：切换到 fused SwiGLU kernel

当前 grouped 路径已从 Torch 的 clamp/SiLU/mul/copy 切换到原生 `sglang.jit_kernel.dsv4.silu_and_mul_clamp`。真实 up 输出上 fused kernel eager 和 CUDA Graph replay 均 finite；独立定时中，输入形状 `[4096, 4096]` 时 fused 约 `0.008 ms`，Torch 实现约 `0.066 ms`。

该修改保持 `swiglu_limit=10.0` 语义不变，并减少 activation 阶段的 kernel 数量。修改前备份：

```text
backups/mxfp4_flashinfer_before_fused_swiglu_20260819.py
```

## 27. 2026-08-19 追加：zero padding 改为 masked_fill_

grouped 路径在 MXFP8 quantize 前需要清零 `masked_m` 之后的传输 buffer 行。真实 decode 形状 G=32、M=128、K=4096 定时比较：

| zero padding | 平均耗时 |
|---|---:|
| `torch.where(..., zeros_like(...))` | 0.068 ms |
| 原地 `masked_fill_` | 0.038 ms |

两者输出均 finite。当前源码使用 contiguous expert input 后原地 `masked_fill_`，减少一个完整 zero tensor 和一次额外 materialization。修改前备份：

```text
backups/mxfp4_flashinfer_before_masked_fill_20260819.py
```

## 26. 2026-08-19 追加：最终 grouped 优化组合 Graph 验证

将当前所有主要优化放入同一个 CUDA Graph：真实 MXFP4 权重、G=32、M=128、32-group 限制、tile_n=192、预分配 `out=`、fused clamped SwiGLU 和 MXFP8 activation quantize。capture 后 replay 5 次，结果为：

```text
final grouped core graph: ok
output shape: [4096, 4096]
```

所有 replay 输出均 finite，说明这些优化组合后没有出现单项验证之外的 Graph/kernel 冲突。

## 28. 2026-08-20 追加：显存恢复后的原生端到端验证

显存释放后，使用当前原生 `/data/ssd2/sglang_v0.5.16/python`、`flashinfer_mxfp4`、DeepEP `low_latency`、Waterfill、PD Mooncake 和 decode CUDA Graph 重新启动。权重加载、43 个 Waterfill TopK 模块初始化以及 bs=1 full CUDA Graph capture 均成功；但 PD warmup 的第一次 Graph replay 在四个 DP rank 同时报 `CUDA error: an illegal memory access`，因此没有产生有效 grouped 吞吐结果。

为排除异步报错和 PD 配置问题，完成了以下隔离：

| 实验 | 配置/结果 |
|---|---|
| full Graph，`mma_sm=1` | capture 成功，首次 replay illegal memory access |
| full Graph，`mma_sm=2` | 同样首次 replay illegal memory access，排除单一 MMA 选择因素 |
| decode Graph 关闭、跳过 warmup | 服务正常启动；通过真实 Router + prefill 发送 `Hello`，返回 `Worldal`，2 token 成功 |
| `tc_piecewise` | SGLang 0.5.16 当前将 decode `tc_piecewise` 回退为 full，仍复现同一错误 |
| FlashInfer grouped 单层/核心 Graph | 既有 G=32/G=65 分块和最终核心 Graph replay 测试仍为 finite |

结论：当前问题已从显存、源码路径、PD 传输和 eager grouped kernel 中排除，收窄为“完整 43 层模型中 FlashInfer grouped MXFP4 kernel 与 full CUDA Graph replay 的组合不稳定”。本轮不能把无 Graph 的成功请求或单层 Graph 结果当作最终吞吐证据，也不能宣称超过 baseline。

本轮启动脚本增加了可回退的诊断开关，默认行为不变：

```text
SGLANG_DISABLE_DECODE_CUDA_GRAPH=1  # 仅诊断 eager 路径
SGLANG_SKIP_SERVER_WARMUP=1         # 仅跳过启动 warmup
SGLANG_DECODE_CUDA_GRAPH_BACKEND=full|tc_piecewise
```

脚本备份：

```text
backups/flash_decode_waterfill_before_graph_isolation_20260820.sh
backups/flash_decode_waterfill_pure_grouped_before_graph_isolation_20260820.sh
backups/flash_decode_waterfill_graph_isolation_current_20260820.sh
backups/flash_decode_waterfill_pure_grouped_graph_isolation_current_20260820.sh
```

随后按上游 CUDA-Graph hardening 方向，在原生 v0.5.16 的 `FullCudaGraphBackend.capture_one()` 第二次 warmup 后增加 device synchronize 和 TP barrier；该修改前后均使用 bs=1、完整 43 层 grouped 配置复测。结果仍为第一次 full Graph replay illegal memory access，故该同步修复不是本问题的充分条件。

对应备份：

```text
backups/full_cuda_graph_backend_before_final_sync_20260820.py
backups/full_cuda_graph_backend_final_sync_current_20260820.py
```

## 29. 2026-08-20 追加：Waterfill 与 grouped CUDA Graph 的严格 A/B

在相同的原生 SGLang 0.5.16、FlashInfer MXFP4 grouped、DeepEP `low_latency`、decode full CUDA Graph、bs=1 和 PD 配置下，只切换 Waterfill：

| Waterfill | 结果 |
|---|---|
| 关闭（`SGLANG_DISABLE_WATERFILL=1`） | 权重加载、Graph capture、四个 DP rank 的 PD warmup 全部成功，服务 ready |
| 开启 | Graph capture 成功，但第一次 PD warmup Graph replay 在四个 DP rank 同时报 CUDA illegal memory access |

进一步排查结果：关闭 `TOPK_V2`、切换 `mma_sm=1/2`、关闭 static Waterfill 使用 dynamic 路径、移除 `expand_topk_with_shared_expert` 的 `torch.compile(dynamic=True)`，均未改变结果。也就是说，当前失败点不是 Graph capture，而是 Waterfill 开启后 TopK 扩展/dispatch metadata 与 grouped MXFP4 Graph replay 的组合。

这组 A/B 修正了之前“Waterfill 只是有额外代价”的表述：在当前随机 workload 上，Waterfill 的确已经被调用并改变了路由输入，但它目前还没有形成可测的端到端吞吐收益；更严格地说，在 grouped + full Graph 的目标配置中，它还会触发 replay 稳定性问题。Waterfill 的真实负载均衡收益仍需在共享专家热点或偏斜路由 workload 上单独验证。

本轮备份：

```text
backups/flash_decode_waterfill_waterfill_ab_current_20260820.sh
backups/waterfill_remove_dynamic_compile_current_20260820.py
```

## 30. 2026-08-20 追加：原生 FlashInfer 路径对照确认根因

为区分 Waterfill 本身和 grouped 适配层，恢复到当前源码中的原生 FlashInfer/TRT-LLM MXFP4 low-latency 分支，仅将 `SGLANG_MXFP4_LL_BACKEND` 从 `flashinfer_grouped` 改为 `flashinfer`，其余配置保持一致：Waterfill 开启、DeepEP `low_latency`、decode full CUDA Graph、bs=1、PD Mooncake、TP/DP/EP=4。

结果：43 层模型加载成功，full Graph capture 成功，四个 DP rank 的 PD warmup 成功，服务 ready；随后经 Router 实际发送 `Hello`，Router→prefill→decode 链路返回 4 token（`World = new Hello`），没有 replay illegal memory access。

因此当前根因已收敛为：Waterfill 与本次新增的 FlashInfer grouped MXFP4 适配层之间存在 Graph replay 交互问题；Waterfill 本身并不普遍与 DeepEP low_latency 或 decode CUDA Graph 冲突。此前 grouped+Waterfill 的失败不能解释为“Waterfill 没有生效”，而应记录为 grouped 适配层尚未完成 Graph-safe 兼容。

同一轮 DeepGEMM 对照在 Graph capture 阶段因当前 low-latency MXFP4 适配没有提供 `hidden_states_scale`，触发 `AttributeError: 'NoneType' object has no attribute 'dtype'`，未形成有效对照，不能作为 Waterfill 结论依据。

## 31. 2026-08-20 追加：原生 FlashInfer 正确 Graph 档位短测

前一轮只捕获 bs=1/16 时，DP rank 实际分到约 4 个请求，c16 测试会退回 eager，结果无效。重新启动原生 FlashInfer + Waterfill，并捕获 `bs=1,2,4,8,16` 后，运行 1024/1024、并发 16、40 个请求，结果如下：

| 成功请求 | Req/s | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---:|---:|---:|---:|---:|---:|
| 40/40 | 0.50 | 509.85 | 1019.70 | 507.07 ms | 25.87 ms |

该结果只用于验证正确 Graph 档位下的端到端稳定性：Waterfill、low-latency、PD 和 Graph 均能工作，但吞吐仍明显低于历史高吞吐 baseline（例如 1024/1024/c16 的 3147.11 out tok/s）。因此当前原生路径已经“能跑”，还没有“超过 baseline”；后续优化重点应转向 DP rank 的有效 batch、decode 调度和 grouped 路径的 Graph-safe 修复。此前被中断的 c16/160 测试没有输出结果文件，不纳入对比。

## 32. 2026-08-20 追加：grouped Graph 诊断分支复测

为验证 grouped 路径中的 `masked_m` 行清零是否是 Waterfill replay 失败原因，新增了可选开关 `SGLANG_FLASHINFER_GROUPED_SKIP_MASKED_FILL=1`，并在 Waterfill 开启、bs=1 full Graph 下复测。结果仍在首次 PD warmup replay 报 illegal memory access。

随后将 grouped segment 从默认 128 临时改为 64（`SGLANG_FLASHINFER_GROUPED_SEGMENT_M=64`）复测，结果仍然失败。因此目前可以排除“单纯 zero padding 操作”和“segment M=128”两个直接原因。诊断代码备份为：

```text
backups/mxfp4_flashinfer_grouped_before_masked_fill_diag_20260820.py
```

该诊断开关默认关闭，当前失败仍集中在 grouped MXFP4 kernel/Waterfill dispatch metadata 的组合，而不是 Waterfill 的 Graph 通用能力。

## 33. 2026-08-20 追加：capture/replay 元数据与固定索引隔离

eager 调试打印显示，Graph capture 阶段 grouped 收到的 `topk_shape` 为 `(0, 7)`，而真实 bs=1 replay 对应 `(1, 7)`；capture 时 `masked_m` 只有一个非零项，真实请求有多个非零 expert。针对这一现象分别测试了：

| 诊断方案 | 结果 |
|---|---|
| 固定至少 32 个 grouped expert groups | 仍然首次 replay illegal memory access |
| 固定 `active_ids`，绕过 `topk(masked_m)` | 仍然首次 replay illegal memory access |

因此 active group 数量和动态 expert 索引虽存在 Graph 静态性风险，但不是当前唯一根因。临时诊断分支已撤回生产源码，相关备份保留：

```text
backups/mxfp4_flashinfer_grouped_before_metadata_debug_20260820.py
backups/mxfp4_flashinfer_grouped_before_static_ids_diag_20260820.py
```

## 34. 2026-08-20 追加：FlashInfer tile-N 对照及上游问题

将 grouped MXFP4 的 up/down `tile_n` 从 192 临时切换为 128，Waterfill、DeepEP low-latency、bs=1 full Graph 重新启动，结果仍在首次 PD warmup replay 报 illegal memory access；该 tile 配置没有解决问题，临时可调参数已撤回，默认仍为 up=192、down=192。

FlashInfer 上游也有 B300/SM100 系列 `groupwise scaled MXFP4 group GEMM` 非确定性异常记录，测试涉及同一个 `m_indptr` grouped GEMM 接口；这支持“当前 grouped kernel/Graph 组合存在底层限制”的判断，但不能代替本地复现证据：[FlashInfer issue #2514](https://github.com/flashinfer-ai/flashinfer/issues/2514)。

## 35. 2026-08-20 追加：FP8 dispatch 复现与 scale/layout 结论

为了复现历史目录中的高吞吐结果，曾将 decode 配置切换为：

```text
--moe-a2a-backend deepep
--deepep-mode low_latency
--moe-runner-backend flashinfer_mxfp4
DEEPEP_DISPATCHER_OUTPUT_DTYPE=fp8
CUDA Graph=enabled
```

当前源码在 CUDA Graph capture 初始化阶段报：

```text
Unsupported hidden state scale shape.
```

定位到 TRT-LLM fused MoE launcher：FP8 activation 分支只接受 `hidden_states_scale_vec_size=32`。DeepEP low_latency 的 FP8 输出则是 expert-major `[E, M, hidden]`，scale 为 `[E, M, hidden/128]`、FP32，且最后两个维度为 TMA column-major。将 expert-major buffer flatten 后，当前实现计算出的 vec size 是 128，因而在 capture 阶段被拒绝。

这不是只改 `reshape` 就能安全解决的形状问题：FlashInfer `mxfp8_quantize` 使用的是每 32 个 hidden 元素一个 scale，而 DeepEP low_latency 的接口按 128 个元素组织 scale；同时两者的 scale dtype/layout 也不同。直接重复 scale 或强制 view 虽可能绕过检查，但不能证明数值语义正确，因此没有把这种未经验证的补丁留在源码中。

历史目录 `results_native_v0516_waterfill_static_cg64_full` 中的高结果（例如 1024/1024/c256：13,786.86 out tok/s、27,573.71 total tok/s、TTFT 1352.34 ms、TPOT 16.95 ms）与本轮纯 FlashInfer routed 路径不是同一个实现状态；结合同目录 benchmark 和此前 fallback 记录，它对应的是 DeepGEMM masked fallback/旧适配状态，不能作为“纯 FlashInfer MXFP4 + DeepEP low_latency”已成功的证据。当前可严格复现的纯 FlashInfer 短测为 1024/1024/c16：509.85 out tok/s、1019.70 total tok/s、TTFT 507.07 ms、TPOT 25.87 ms。

上游相关结论也一致：SGLang 将 DeepEP low_latency 描述为 masked decode 路径，现有 `flashinfer_mxfp4` routed runner 仍不是 DeepEP masked-layout kernel；同类问题的公开 workaround 是使用 DeepGEMM。要实现用户目标，下一步应实现真正接收 `masked_m/expected_m` 的 MXFP8 activation × MXFP4 weight masked kernel，或完成经过数值校验的 DeepEP scale 转换后接入支持该布局的 FlashInfer/CuteDSL kernel，而不是继续用 routed kernel 的 shape hack。

本轮源码均已保留备份；FP8 失败没有覆盖已知可工作的 BF16 默认路径。

## 36. 2026-08-20 追加：固定全部 expert 的 Graph 隔离实验

为进一步隔离 grouped Graph 的 illegal memory access，临时增加
`SGLANG_FLASHINFER_GROUPED_ALL_EXPERTS=1`：固定使用本地全部 65 个 expert，固定 expert
顺序和 `m_indptr`，只依据 `masked_m` 将无效行置零，从而完全移除
`topk(masked_m)` 和动态 expert 索引。

结果：bs=1 的 full CUDA Graph capture、PD warmup 和服务启动均成功；日志中确认
decode 使用 `cuda graph: True`。这证明动态 expert 选择是原 grouped Graph 失败的触发因素之一。
但该方案在实际请求进入 batch 后需要计算全部 65 个 expert，每个 expert 固定 128 行，实测
decode throughput 约 12 tok/s，远低于 baseline，且 benchmark 因此被中止，不作为性能结果。

该方案仅用于根因隔离，已撤回生产源码，备份为：

```text
backups/mxfp4_flashinfer_grouped_before_all_experts_diag_20260820.py
logs/flash_decode_waterfill/all_experts_graph1_20260820.log
```

当前实现仍恢复为 active expert grouped 路径。下一步必须让 kernel 本身读取
`masked_m`，并同时支持 DeepEP low_latency 的 MXFP8 activation 与 MXFP4 weight；固定全 expert
或继续扩大 padding 都不能满足超过 baseline 的目标。

## 37. 2026-08-20 追加：masked MXFP8×MXFP4 kernel 接入尝试

在保留原生 SGLang v0.5.16 基线的前提下，对已安装的 FlashInfer 0.6.14 增加了一个显式的
`group_gemm_mxfp4_nt_groupwise_masked` SM100 JIT 实例。它接收固定 expert-major activation/scale
buffer、`masked_m`、`expert_ids` 和 capacity；CUTLASS 的 group 参数准备阶段按真实 expert ID
选择权重、activation、scale 和输出地址，并将 M 向上对齐到 4。精简 JIT 模块已成功编译加载，
独立随机测试中 active 行输出为 finite，验证了接口和指针布局的基本正确性。

随后接入 `SGLANG_FLASHINFER_GROUPED_MASKED=1` 路径，并在 DeepEP low_latency、Waterfill、
full decode CUDA Graph、PD 分离配置下启动成功。日志确认 9 个 decode Graph batch capture 成功，
服务 warmup 成功，且 replay 日志显示 `cuda graph: True`。本次备份包括：

```text
backups/mxfp4_flashinfer_before_masked_mxfp4_integration_20260820.py
backups/mxfp4_flashinfer_masked_active_ids_20260820.py
logs/masked_mxfp4_decode_20260820.log
logs/masked_mxfp4_decode_active_ids_20260820.log
```

但端到端性能仍不合格：固定全部本地 expert 的短测约为 16.9 tok/s/DP；改为按
`active_ids` 选择 expert 后，C16、OSL=128 的服务日志约为 14.6 tok/s/DP，benchmark 未完成，
没有生成有效的 ISL/OSL 全量 JSON，因此不能与 baseline 做正式吞吐比较。原因是当前随机 workload
在每个 DP/EP step 仍会激活大量本地 expert，masked kernel 主要节省了 padding 行，却没有消除
大量 group launch、scale quantize、全 expert buffer 清零和两次 Python/FFI 调用的固定开销。

该 masked 路径默认关闭，测试脚本也恢复为 `SGLANG_FLASHINFER_GROUPED_MASKED=0`、使用正常 AOT
模块；FlashInfer 源码修改和 JIT cache 仍保留，后续可通过显式设置三个 masked/JIT 环境变量复现。
因此本轮证明了“kernel 能编译、Graph 能 capture”，但没有证明“masked kernel 能超过历史 baseline”，
也不能把这次结果宣称为 Waterfill 的独立收益。

## 38. 2026-08-20 追加：compact active-row 修复与实测

进一步检查发现，上一版 masked kernel 虽然按 `masked_m` 建立了 CUTLASS problem，调用侧仍先
量化整个 DeepEP `[E, transport_capacity, K]` backing buffer。现已改为只取选中 expert 的固定
安全 segment，量化和 GEMM 输入/输出均使用 compact `[num_groups, segment_m, ...]`，权重和 scale
仍通过真实 `expert_ids` 定位；`segment_m` 使用 `batch_tokens * dispatch_group_size` 作为静态安全
上界，而不再使用仅代表平均值的 `expected_m`。此前按 `expected_m` 直接设 segment 会在 Graph
capture 中造成越界，已由日志复现并修正。

compact 路径在以下配置下成功启动并完成 9 个 decode Graph capture、PD warmup 和请求压测：

```text
DeepEP low_latency + Waterfill + full decode CUDA Graph + PD
SGLANG_FLASHINFER_GROUPED_MASKED=1
```

短测结果（64 requests，ISL=1024、OSL=128、C16）为：

| Out tok/s | Total tok/s | Mean TTFT | Mean TPOT | 成功数 |
|---:|---:|---:|---:|---:|
| 83.63 | 752.68 | 845.00 ms | 185.54 ms | 64/64 |

结果文件：
`logs/flash_decode_waterfill/results_masked_20260820/isl1024_osl128_c16_safecompact_n64.jsonl`。
服务日志：`logs/masked_mxfp4_decode_safecompact_20260820.log`。

该结果明显低于历史 1024/1024/C16 baseline 的 3147.11 Out tok/s，因此 compact 解决的是
越界和无效 quantize，不是最终性能方案。尝试 `tile_m=64/32` 也被 CUTLASS SM100 明确拒绝：
该 MXFP4 block-scaled builder 当前静态断言要求 M tile 为 128，相关尝试已撤回。当前主要性能
瓶颈仍是每层两次 grouped GEMM 的大量小-M group 调度和 Python/FFI/quantize 固定开销。

## 39. 2026-08-20 追加：DeepEP FP8 scale 适配验证

为避免每层 BF16→MXFP8 quantize，增加了 DeepEP low_latency FP8 输入分支。实际运行时确认
DeepEP 返回的 scale 形状为 `[local_experts, capacity, hidden/512]`、FP32，而不是早期文档中
常见的 `hidden/128` 形式；接入代码按 `hidden/32 // scale_groups` 展开 scale，并显式编码为
UE8M0 uint8，未使用危险的 dtype view。FP8 activation 本身直接送入 masked MXFP4 kernel，
down projection 仍按需要重新 quantize。

配置包含 `DEEPEP_DISPATCHER_OUTPUT_DTYPE=fp8`、DeepEP `low_latency`、Waterfill、PD 和 full
decode CUDA Graph；9 个 Graph capture 和服务 warmup 均成功，日志为：

```text
logs/masked_mxfp4_decode_fp8compact3_20260820.log
```

短测 16/16 成功（ISL=1024、OSL=64、C16）：

| Out tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---:|---:|---:|---:|
| 83.07 | 1412.19 | 750.33 ms | 182.53 ms |

结果文件：
`logs/flash_decode_waterfill/results_masked_20260820/isl1024_osl64_c16_fp8compact_n16.jsonl`。

FP8 dispatch 与 BF16 compact 的吞吐几乎相同，说明当前主要瓶颈是小 M grouped MXFP4 的 kernel
调度/两次 projection 固定成本，而不是输入 quantize。该结果证明 scale 适配可运行，但没有
超过历史 baseline；在完成数值 A/B 前，FP8 分支不应作为默认配置。

## 40. 2026-08-20 追加：DeepGEMM masked 上限对照

为隔离 Waterfill、PD、DeepEP 和 CUDA Graph 与 MoE GEMM backend 的影响，保持以下配置不变：

```text
原生 SGLang v0.5.16
DeepEP low_latency
Waterfill enabled
PD disaggregation
decode full CUDA Graph
```

只将 `SGLANG_MXFP4_LL_BACKEND` 切换为 `deep_gemm`，并使用 FP8 dispatcher。结果为：

| ISL | OSL | C | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 128 | 16 | 925.51 | 8329.63 | 569.27 ms | 12.63 ms |
| 1024 | 128 | 256 | 4364.40 | 39279.58 | 3153.94 ms | 15.85 ms |

结果文件和日志：

```text
logs/flash_decode_waterfill/results_deepgemm_20260820/isl1024_osl128_c16_n64.jsonl
logs/flash_decode_waterfill/results_deepgemm_20260820/isl1024_osl128_c256_n256.jsonl
logs/deepgemm_waterfill_upperbound_20260820.log
```

相同 PD/Waterfill/Graph 配置下，DeepGEMM 相比本轮 FlashInfer masked 的 83 Out tok/s 高出约
11 倍；因此当前 FlashInfer mixed MXFP8×MXFP4 grouped kernel 是主瓶颈。另一方面，DeepGEMM
本轮 C256 也没有达到历史 DSpark baseline 的 23680.92 Out tok/s，说明该历史结果不能直接
作为“原生 SGLang + Waterfill + 单步 decode”baseline；其中至少包含旧 DSpark 调度/融合或
speculative/MTP 收益。这个对照不能证明 Waterfill 有收益，也不能替代目标配置的最终验证。

## 41. 2026-08-20 追加：原生 FlashInfer MxFP4 路径 A/B

本节将 baseline 固定为 `PD_COMPLETED_REPORT_20260803.md` 中的正式 PD baseline。
1024/1024/C256 的正式 baseline 是 `13106.67 Out tok/s`；历史 `23680.92 Out tok/s`
属于 DSpark 结果，不作为本节 baseline。正式 baseline 使用 `moe_runner_backend=auto`
和 `Fp8MoEMethod`，并非强制 `flashinfer_mxfp4`。

原生 FlashInfer TRT-LLM MxFP4 适配完成了两项修复：DeepEP FP8 scale 从
`[expert, capacity, 8]` FP32 展开为 `[tokens, hidden/32]` UE8M0 uint8；C256 时通过
`SGLANG_FLASHINFER_NATIVE_EXPECTED_M` 使用 DeepEP 的静态 `expected_m`，避免给每个专家
固定分配整批 `ll_batch_tokens` 行。

| 路径 | ISL | OSL | C | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原生 TRT-LLM MxFP4，FP8 dispatch | 1024 | 128 | 256 | 2374.81 | 20712.67 | 1709.58 ms | 37.67 ms |
| 原生 TRT-LLM MxFP4，BF16 dispatch | 1024 | 128 | 256 | 2045.34 | 17839.06 | 2686.60 ms | 35.56 ms |
| 原生 TRT-LLM MxFP4，BF16 dispatch + expected_m | 1024 | 1024 | 256 | 3797.88 | 7643.62 | 2541.59 ms | 30.14 ms |
| 原生 TRT-LLM MxFP4，expected_m 紧凑 | 1024 | 128 | 256 | 2279.91 | 19884.96 | 2229.97 ms | 28.36 ms |
| 原生 TRT-LLM MxFP4，expected_m 紧凑 | 1024 | 1024 | 256 | 3875.21 | 7799.26 | 1785.81 ms | 30.38 ms |
| 原生 TRT-LLM MxFP4，FlashInfer autotune + expected_m | 1024 | 1024 | 256 | 4257.38 | 8568.42 | 0 ms* | 350.47 ms* |
| 原生 TRT-LLM MxFP4，FlashInfer autotune + expected_m + M=8 | 1024 | 1024 | 256 | 4236.94 | 8527.27 | 0 ms* | 364.71 ms* |
| 原生 TRT-LLM MxFP4，FlashInfer autotune + expected_m，关闭 Waterfill | 1024 | 1024 | 256 | 4240.85 | 8535.14 | 0 ms* | 233.64 ms* |
| 原生 TRT-LLM MxFP4，关闭 PDL | 1024 | 1024 | 256 | 4228.45 | 8510.19 | 0 ms* | 356.48 ms* |
| 原生 TRT-LLM MxFP4，实际 M tune bucket | 1024 | 1024 | 256 | 4226.23 | 8505.73 | 0 ms* | 232.44 ms* |
| 正式 PD baseline | 1024 | 1024 | 256 | 13106.67 | 26213.34 | 2284.50 ms | 16.82 ms |

注：带 `*` 的两次是通过 PD Router 的普通 `bench_serving` 快速 A/B，Router 返回的
TTFT/首 token 时间戳为 0，Mean TPOT 也受到 256 个请求不同完成批次影响，不能和正式
baseline 的阶段统计直接等价；Out/Total tok/s 可用于本轮同口径方向判断。autotune 相比
未 autotune 的 `3875.21/7799.26` 有约 9.9% 吞吐提升，但距离正式 baseline 仍有明显差距。

当前 MxFP4 FlashInfer 路径仍低于正式 baseline，尚未完成“超过 baseline”目标。结果文件和备份
分别位于 `logs/flash_decode_waterfill/results_native_flashinfer_20260820/` 与：

```text
backups/mxfp4_flashinfer_native_trtllm_fp8_scale_adapter_20260820.py
backups/mxfp4_flashinfer_native_scale_skip_topk_20260820.py
backups/mxfp4_flashinfer_native_expected_m_20260820.py
backups/waterfill_inference_tensor_fix_20260820.py
```

FlashInfer autotune 首次启用时已经完成 16/16 个 TRT-LLM FP4 profile，但随后 CUDA
Graph capture 暴露出 Waterfill 的持久 `_counts_buf` 在 inference mode 中创建、后续
`zero_()` 被禁止的问题。修复为在 `torch.inference_mode(False)` 中创建普通 scratch
buffer 后，9 个 decode Graph batch 全部 capture 成功；修复代码已备份。M=8 A/B 略低于
默认 expected_m（M=4），因此当前保留默认 expected_m，不保留 M=8 override。

Waterfill 严格消融显示，开启与关闭只相差约 0.4%（4257.38 vs 4240.85 Out tok/s），
因此当前性能缺口不能归因于 Waterfill；Waterfill 已实际参与 dispatch，但在随机均匀
路由 workload 上没有独立的显著吞吐收益。关闭 Waterfill 的结果文件为
`isl1024_osl1024_c256_autotune_nowaterfill_n256.jsonl`。

另外验证了两个 TRT-LLM 小批量调度开关：关闭 PDL 后降至 `4228.45 Out tok/s`，将
`tune_max_num_tokens` 从 2048 ceiling 改为实际输入行数后为 `4226.23 Out tok/s`，
均低于默认配置，因此默认继续保留 PDL 和 power-of-two tune bucket。

曾尝试将 SM100 本地 FlashInfer CUTLASS 的 `W4A8 (MXFP4×MXFP8)` kernel 接入
DeepEP low_latency。43 层权重转换和初始化成功，但 CUDA Graph 首次 forward 出现
`CUBLAS_STATUS_EXECUTION_FAILED`，没有有效吞吐结果，不能纳入 baseline 对比。实验代码
备份为 `backups/mxfp4_flashinfer_cutlass_ll_experiment_20260820.py`，失败日志为
`logs/flash_decode_waterfill/flashinfer_cutlass_ll_retry_20260820.log`；服务已恢复默认
TRT-LLM FlashInfer + Waterfill 配置。

补充 A/B：开启 `SGLANG_OPT_USE_ONLINE_COMPRESS=1` 后，C256/OSL1024 的 TPOT 为
`30.70 ms`，没有优于 M4/expected_m 的 `30.38 ms`，但 PD Mean TTFT 恶化到
`39613.50 ms`，故该开关已撤回，不纳入最终配置。

## 42. 2026-08-20 追加：TRT-LLM gated-act 重排 A/B（无预重排结果无效）

继续对照上游 FlashInfer 静态权重预处理后发现，v0.5.16 的本地适配在进入
`_maybe_get_cached_w3_w1_permute_indices` 前先执行了一次 `reorder_w1w3_to_w3w1`；而该
FlashInfer permutation 本身已经包含 gated-act 的 w1/w3 行重排，导致同一组权重被重复重排。
现已增加 `SGLANG_MXFP4_NATIVE_PRE_REORDER` 开关，但正确性验证表明旧行为必须保留，默认值为
`true`；无预重排仅作为性能诊断开关。

修复前后均使用：原生 SGLang v0.5.16、FlashInfer MxFP4 TRT-LLM、DeepEP low_latency、
FP8 dispatcher、Waterfill、PD 分离、decode full CUDA Graph、FlashInfer autotune。
正式 C256、2560 请求结果如下：

| 路径 | ISL | OSL | C | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---|---:|---:|---:|---:|---:|---:|---:|
| 修复前：重复预重排 | 1024 | 1024 | 256 | 4257.38 | 8568.42 | 0 ms* | 350.47 ms* |
| 无预重排诊断（正确性失败，不计） | 1024 | 1024 | 256 | 38449.24 | 76898.47 | 0 ms* | 6.45 ms* |
| 正式 PD baseline | 1024 | 1024 | 256 | 13106.67 | 26213.34 | 2284.50 ms | 16.82 ms |

无预重排虽然 Out tok/s 为正式 baseline 的约 2.93 倍，但通过 Router 发送“你是谁？”后返回
重复的 `メリア` token，未生成正常语义文本，因此该吞吐结果判定为错误布局下的无效结果，不能
用于声称超过 baseline。结果文件：
`logs/flash_decode_waterfill/results_native_flashinfer_20260820/isl1024_osl1024_c256_nopre_reorder_n2560.jsonl`；
服务日志：`logs/flash_decode_waterfill/native_flashinfer_nopre_reorder_20260820.log`；
修复前代码备份：`backups/mxfp4_native_pre_reorder_ab_20260820.py`。

注：PD Router 的普通 `bench_serving` 仍无法提供有效 TTFT，因此 TTFT=0 仅表示统计链路
限制。C256/C16 的 benchmark 摘要虽然分别显示 2560/2560 和 160/160，但在当时旧 PD
会话失效期间日志出现大量 KV transfer 错误，不能仅凭摘要认定业务请求全部成功。当前源码
默认已恢复预重排；`SGLANG_MXFP4_NATIVE_PRE_REORDER=0` 仅用于诊断，不能作为最终配置。

## 43. 2026-08-20 追加：PD 正确性复核

复核发现，之前 C256/C16 的 benchmark 摘要是在 prefill/decode 的旧 Mooncake 会话失效期间
生成的，虽然摘要中的 `completed` 分别为 2560 和 160，但服务日志存在 KV transfer 错误，
因此这两份摘要不能作为“全部业务请求成功”的证据，也不再作为有效性能结果使用。

随后完成以下修复并重启完整链路：

- prefill 强制使用 `/data/ssd2/sglang_v0.5.16/python`，避免落到系统 `/sgl-workspace/sglang`；
- decode 使用物理 GPU rank 映射：`--base-gpu-id 4`、`mlx5_4/9/10/11`；
- prefill、decode、Router 按顺序重启，重新建立 Mooncake bootstrap 会话；
- decode 保持 DeepEP low_latency、Waterfill、full CUDA Graph、MxFP4 FlashInfer，且恢复
  `SGLANG_MXFP4_NATIVE_PRE_REORDER=1`。

用 `/v1/chat/completions` 发送“你是谁？请用中文简短回答”，单请求能够返回正常的
DeepSeek 身份回答；并发 16 个相同 chat 请求结果为 **16/16 HTTP 200、16/16 JSON choices、
0 错误**。部分回答包含模型的 `<think>` 内容，是当前 reasoning 输出格式，并非传输失败。

因此目前可以确认：当前配置的 PD 链路和基本生成正确性已恢复；但此前无预重排的 38449.24
Out tok/s 是错误布局下的无效结果，当前仍没有经过“正确输出 + 完整请求成功”复核的、超过正式
baseline 的有效 C256 性能结果。

## 44. 2026-08-20 追加：prefill `/v1/loads` 的 utilization 解释

用户观察到 prefill 的 `/v1/loads` 中 `utilization` 长期为 `0.0`。这不是 prefill 算力
利用率。v0.5.16 的 `metrics_reporter.py` 对 `DisaggregationMode.PREFILL` 不计算该字段，
而是将其作为调度/缓存负载字段处理；该字段不能代表 GPU SM 利用率、prefill token/s 或
MoE dispatch 利用率。

本轮 8192 输入、C16 探针期间，decode `/v1/loads` 曾观测到每个 DP rank
`num_running_reqs=4`，`num_used_tokens` 从 34048 持续增长到约 35840；prefill 端因为每个
prefill batch 完成后立即将请求转入 Mooncake/Decode 并释放本地 batch，轮询时经常处于
`num_running_reqs=0`、`num_used_tokens=0` 的空档。此前 prefill server 日志也记录过
`Prefill batch #new-token: 7168` 以及约 26.5k--57k tok/s 的 batch，证明 prefill 实际
执行过。故不能用 prefill 的 `utilization=0.0` 判定 prefill 没有工作；应结合 prefill
batch 日志、请求成功率、KV transfer 统计和 GPU/Nsight 指标判断。

本次探针因运行时间异常且不是正式 benchmark，已停止，不计入性能结果；停止后再次运行
`validate_pd_whoami.sh`，得到 HTTP 200、`WHOAMI_VALID=True`。服务未重启，正式实验结果不受
该探针影响。

## 45. 2026-08-20 追加：并发正确性复核与 MxFP4 精度 A/B

本轮每次 decode 服务重启后均先执行 `validate_pd_whoami.sh`；单请求均得到 HTTP 200 和正常
的 DeepSeek 身份回答。随后用 16 个并发 chat 请求进行复核。需要区分三类指标：HTTP 200、
JSON/choices 完整、以及可见正文语义检查。DeepSeek 默认开启 reasoning，`max_tokens=64/128`
时部分请求可能只生成 `<think>` 而没有可见正文；这会造成正文为空，但不等同于 HTTP 或 PD
传输失败。

隔离结果如下：

| 配置 | 并发 HTTP 200 | JSON/choices | 可见正文语义 | 结论 |
|---|---:|---:|---:|---|
| MxFP4 FlashInfer + Waterfill，`moe-precision=bf16` | 16/16 | 16/16 | 8/16 | 不采用 |
| MxFP4 FlashInfer + Waterfill，默认精度，`expected_m=1` | 16/16 | 16/16 | 9/16 | 可继续做性能实验；语义检查需提高输出预算/改用非 reasoning 检查 |
| 官方 MoE runner、DeepEP low_latency、decode CUDA Graph | 16/16 | 16/16 | 11/16 | 说明空正文并非 MxFP4 独有，官方路径也受 reasoning 输出预算影响 |

顺序发送 16 个身份请求时 MxFP4 路径为 16/16 正常；并发时可见正文比例下降。因此本表的
MxFP4 数据不能表述为“16 个请求都生成了可见身份答案”，但 HTTP、JSON 和单请求服务健康检查
均通过。`moe-precision=bf16` 已撤回；当前服务恢复为默认 MxFP4 精度、预重排、DeepEP
low_latency、Waterfill 和 decode full CUDA Graph。

同时验证了去掉 `SGLANG_FLASHINFER_NATIVE_EXPECTED_M` 并不能解决并发正文问题，故暂时保留
`expected_m=1` 作为性能候选配置。该变量影响 FlashInfer MxFP4 的计算段尺寸，不能单独作为
正确性修复结论。上述检查完成前的吞吐数据不纳入“超过 baseline”的正式结论。

## 46. 2026-08-20 追加：复用 DeepEP BF16 receive buffer 的优化

原生 TRT-LLM MxFP4 low-latency 分支每层 kernel 完成后都会重新分配并清零完整的
`[local_experts, max_m, hidden]` BF16 combine buffer；实际计算只覆盖 `expected_m` 行，且
DeepEP 的 `masked_m` 已经提供有效行边界。对于当前 BF16 dispatcher 路径，现改为复用原有
DeepEP receive buffer，只覆盖 active expert 的有效输出行；FP8 dispatcher 仍保留原来的独立
BF16 buffer 分支，避免 dtype alias 风险。修改前代码备份为：
`backups/mxfp4_reuse_deepep_bf16_buffer_20260820.py`。

短测结果（同一 PD、Waterfill、DeepEP low_latency、decode full CUDA Graph、MxFP4 TRT-LLM、
`expected_m=1`）：

| 配置 | ISL | OSL | C | 完成请求 | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 修改前 | 1024 | 128 | 16 | 64/64 | 534.54 | 4810.84 | 471.14 ms | 25.87 ms |
| 复用 BF16 buffer | 1024 | 128 | 16 | 64/64 | 922.47 | 8302.27 | 461.79 ms | 13.41 ms |
| 复用 BF16 buffer | 1024 | 1024 | 256 | 256/256 | 11346.88 | 22693.75 | 2734.85 ms | 18.04 ms |
| 正式 baseline | 1024 | 1024 | 256 | 2560/2560 | 13106.67 | 26213.34 | 2284.50 ms | 16.82 ms |

C256 短测已接近但仍低于正式 baseline，且请求数为 256 而非正式 baseline 的 2560，因此
不能替代完整结果。随后将激活量化 backend 切换为 `cuda` 做 C256 A/B，结果为 Out
`11361.44`、Total `22722.88`、Mean TTFT `2685.86 ms`、Mean TPOT `18.08 ms`，仅约
0.1% 的随机波动且略差于 CuTe-DSL；同时一次“你是谁”可见回答出现异常英文推理，故已撤回
该 backend。当前运行配置恢复为默认 CuTe-DSL，并已重新通过单请求健康检查。

补充小 M PDL A/B：在保留 buffer 复用、Waterfill、DeepEP low_latency 和 decode full CUDA
Graph 的条件下，设置 `SGLANG_MXFP4_NATIVE_PDL=0` 后 1024/128/C16 短测为 Out
`886.75`、Total `7980.72`、Mean TTFT `496.72 ms`、Mean TPOT `13.89 ms`；默认开启 PDL
为 Out `922.47`、Total `8302.27`、Mean TTFT `461.79 ms`、Mean TPOT `13.41 ms`。
关闭 PDL 没有改善小 M，已撤回；当前服务重启后“你是谁”验证通过，默认 PDL 保持开启。

补充 active-expert compact A/B：尝试将小 batch 的 active expert 权重和 scale 压缩为连续
expert 编号后调用 TRT-LLM kernel，再散射回原 expert buffer。该路径的 1024/128/C16 短测
请求 64/64 完成且“你是谁”验证正常，但 Out 仅 `673.54`、Total `6061.90`、Mean TTFT
`547.99 ms`、Mean TPOT `19.13 ms`，明显低于默认路径的 `922.47`/`8302.27`/`461.79 ms`/
`13.41 ms`。原因是每层 active 权重切片和重新布局的成本超过了节省的路由元数据开销；该路径
未采用，代码已恢复，备份为 `backups/mxfp4_active_expert_compact_before_20260820.py`。

同一配置随后完成正式规模 C512/5120 复测：

| ISL | OSL | C | Num prompts | 完成请求 | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 512 | 5120 | 5120/5120 | 14633.55 | 29267.10 | 9338.38 ms | 24.19 ms |
| 正式 baseline | 1024 | 1024 | 512 | 5120 | 13243.63 | 26487.25 | 16554.06 ms | 21.32 ms |

因此，当前实现已经在 1024/1024/C512 的正式规模上超过 baseline：Out 和 Total 均提升约
10.5%，且所有 5120 个请求均完成。C16、C256 和 8192/C16 仍低于各自 baseline，不能据此
宣称所有并发档位均达标。

补充同一代码状态下的正式 C16 和 8192/C16 结果：

| ISL | OSL | C | Num prompts | 完成请求 | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 16 | 160 | 160/160 | 1118.56 | 2237.12 | 490.28 ms | 13.79 ms |
| 8192 | 1024 | 16 | 160 | 160/160 | 1101.43 | 9912.91 | 584.06 ms | 13.77 ms |

对应正式 baseline 分别为 1348.95/2697.90 和 1328.45/11956.04（Out/Total），因此两档仍
未超过 baseline；C16 结果文件为
`logs/flash_decode_waterfill/results_native_flashinfer_20260820/reuse_buffer_isl1024_osl1024_c16_n160.jsonl`，
8192/C16 结果文件为
`logs/flash_decode_waterfill/results_native_flashinfer_20260820/reuse_buffer_isl8192_osl1024_c16_n160.jsonl`。

另一次 FP8 dispatcher scale 方向实验在单请求健康检查阶段出现乱码，未产生性能结果；该
实验代码保留在 `backups/mxfp4_fp8_scale_direction_fix_20260820.py`，当前服务已恢复 BF16
dispatcher。FP8 路径不纳入正式结果。

补充 M 桶 A/B：设置 `SGLANG_FLASHINFER_NATIVE_M_OVERRIDE=128`，将 C256 的实际计算段从
`expected_m≈112` 强制扩大到 128。结果为 Out `7657.05`、Total `15314.10`、Mean TTFT
`2631.39 ms`、Mean TPOT `28.98 ms`，显著低于默认 expected_m 的 Out `11346.88`、Total
`22693.75`、Mean TPOT `18.04 ms`；“你是谁”验证通过但该配置已撤回。当前恢复默认
`expected_m=1`，说明额外 padding 行会触发明显更差的 TRT-LLM tactic，不能用简单 M 桶放大
解决低/中并发差距。

## 47. 2026-08-20：prefill utilization=0 的诊断与修复

发现 prefill 节点的 `/v1/loads` 长期显示 `utilization=0.0`。这不是 prefill 没有执行：压力
采样期间 `num_used_tokens` 曾达到 16384/32768，等待队列也出现过 8 个请求；decode 和
prefill 请求均能完成。SGLang 0.5.16 的该字段也不是 GPU SM 利用率，而是调度器的
SLO/请求利用率。对于 prefill-only 模式，这个指标不适用，应该使用 `-1` 表示 undefined，
并结合 `num_used_tokens`、`num_waiting_reqs`、`num_waiting_uncached_tokens` 以及
`disaggregation.prefill_*_queue_reqs` 判断 prefill 是否工作。

根因是运行时 mode 可能以 enum 或字符串形式传入，旧的 snapshot 组装路径未稳定识别
prefill，最终保留了默认值 0。修复位置为：

`/data/ssd2/sglang_v0.5.16/python/sglang/srt/managers/scheduler_components/load_inquirer.py`

现在 prefill 模式直接报告 `utilization=-1.0`；decode 模式仍沿用原有 utilization 逻辑。
修改前代码备份为 `backups/load_inquirer_before_prefill_util_fix2_20260820.py`，并已通过
Python 编译检查。

修复后的验证结果：

| 检查项 | 结果 |
|---|---|
| prefill `/v1/loads` | `mode=prefill`, `utilization=-1.0` |
| decode 服务 | 未修改，仍运行 DeepEP low_latency、Waterfill、MxFP4 FlashInfer、decode full CUDA Graph |
| “你是谁”请求 | `HTTP=200`, `WHOAMI_VALID=True` |
| 返回内容 | 正常 DeepSeek 中文身份回答 |

因此，后续报告中不能把 prefill 的 `utilization=0` 当作 GPU 利用率或“prefill 未运行”。

## 48. 2026-08-20：按 CUDA Graph batch 选择 MxFP4 tactic

继续对当前 MxFP4 TRT-LLM FlashInfer 路径做正式规模 A/B。全局开启
`SGLANG_MXFP4_NATIVE_TUNE_ACTUAL_M=1` 并不适合所有 decode graph：

| 配置 | tune_actual_m | 完成请求 | Total tok/s | Mean TTFT | Mean TPOT |
|---|---:|---:|---:|---:|---:|
| 1024/1024/C256，2560 请求 | 1 | 2560/2560 | 24535.28 | 2167.84 ms | 18.12 ms |
| 1024/1024/C256，2560 请求 | 0 | 2560/2560 | 25351.09 | 1636.42 ms | 18.28 ms |
| 正式 baseline | — | 2560/2560 | 26213.34 | 2284.50 ms | 16.82 ms |

shape probe 记录了 TP4/DP4 当前 Graph 的稳定映射：`batch_tokens=4 -> x_tokens=112`，
`batch_tokens=128 -> x_tokens=1040`，`batch_tokens=256 -> x_tokens=1820`。因此新增
`SGLANG_MXFP4_NATIVE_TUNE_ACTUAL_M=auto`：batch 256 保留 power-of-two tactic，其他当前
decode graph 使用实际 M。诊断日志和源码备份为：
`logs/flash_decode_waterfill/native_flashinfer_shape_probe2_20260820.log`、
`backups/mxfp4_before_shape_tactic_probe_20260820.py`。当前脚本默认已切换到 `auto`，并
关闭 shape probe 日志。

按 batch-aware 规则的正式 C256 复测为 2560/2560、Total `24947.51 tok/s`；由于单轮运行
存在约 1%--2% 的系统波动，它仍低于关闭 tune 的最佳 `25351.09`，也低于 baseline，故该
规则目前只作为可回滚候选，不能宣称 C256 已超过 baseline。此前 C512 的最佳正式结果仍为
Total `32364.11 tok/s`（5120/5120），高于 baseline `26487.25`。

随后在最终脚本状态（`tune_actual_m=auto`、Waterfill、DeepEP `low_latency`、BF16 dispatcher、
decode full CUDA Graph、custom all-reduce disabled）下重新完成正式 C512：

| 配置 | 完成请求 | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---|---:|---:|---:|---:|---:|
| 1024/1024/C512，最终 auto | 5120/5120 | 15538.11 | 31076.22 | 7410.60 ms | 23.77 ms |
| 正式 baseline | 5120/5120 | 13243.63 | 26487.25 | 16554.06 ms | 21.32 ms |

该最终结果的 Total throughput 高于 baseline 约 17.3%，且全部请求成功。结果文件为：
`logs/flash_decode_waterfill/results_native_flashinfer_20260820/tune_auto_final_isl1024_osl1024_c512_n5120.jsonl`。

### 48.1 2026-08-21：补齐剩余七组正式样例

在第 48 节当前服务配置下补齐其余七组样例。配置保持为原生 SGLang 0.5.16、MxFP4
FlashInfer、DeepEP `low_latency`、Waterfill、decode full CUDA Graph、PD 分离；测试使用
`random_range_ratio=1.0`、`--tokenize-prompt`、每档 `10 × concurrency` 个请求。C512
时每个 DP rank 的实际 steady batch 为 128，服务使用 Graph128。结果目录为：

`logs/flash_decode_waterfill/results_native_flashinfer_20260821/section48_remaining7_runtime128/`

以下结果均为完整请求成功后的 JSONL 指标；baseline 沿用本报告第 3 节的正式 baseline：

| ISL | OSL | Concurrency | 本轮 Out tok/s | baseline Out tok/s | Out 变化 | 本轮 Total tok/s | baseline Total tok/s | Total 变化 | 本轮 TTFT ms | baseline TTFT ms | 本轮 TPOT ms | baseline TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 84.57 | 107.55 | -21.37% | 169.14 | 215.09 | -21.36% | 277.50 | 332.51 | 11.56 | 8.98 |
| 1024 | 1024 | 16 | 1125.12 | 1348.95 | -16.59% | 2250.23 | 2697.90 | -16.59% | 431.04 | 631.91 | 13.76 | 11.25 |
| 1024 | 1024 | 256 | 12797.52 | 13106.67 | -2.36% | 25595.04 | 26213.34 | -2.36% | 1624.19 | 2284.50 | 18.01 | 16.82 |
| 1024 | 1024 | 512 | 16400.43 | 13243.63 | +23.84% | 32800.85 | 26487.25 | +23.84% | 5438.08 | 16554.06 | 23.98 | 21.32 |
| 8192 | 1024 | 1 | 84.49 | 106.52 | -20.68% | 760.44 | 958.65 | -20.68% | 240.42 | 348.65 | 11.61 | 9.05 |
| 8192 | 1024 | 16 | 1102.47 | 1328.45 | -17.01% | 9922.25 | 11956.04 | -17.01% | 674.21 | 607.99 | 13.72 | 11.23 |
| 8192 | 1024 | 256 | 7216.46 | 6861.28 | +5.18% | 64948.13 | 61751.53 | +5.18% | 16675.13 | 21913.55 | 17.61 | 14.34 |
| 8192 | 1024 | 512 | 7406.63 | 7110.57 | +4.16% | 66659.65 | 63995.17 | +4.16% | 49333.40 | 55535.20 | 17.62 | 14.42 |

本轮 8 组（包含此前第 48 节 C512 复测）均完成全部请求；本次补齐的七组分别对应：

其中表中 1024/1024/C512 的对应结果文件为：
`logs/flash_decode_waterfill/results_native_flashinfer_20260821/section48_historical_mxfp4_runtime128/isl1024_osl1024_c512_n5120.jsonl`。

```text
isl1024_osl1024_c1_n10.jsonl
isl1024_osl1024_c16_n160.jsonl
isl1024_osl1024_c256_n2560.jsonl
isl8192_osl1024_c1_n10.jsonl
isl8192_osl1024_c16_n160.jsonl
isl8192_osl1024_c256_n2560.jsonl
isl8192_osl1024_c512_n5120.jsonl
```

需要区分本节原有的 `tune_auto_final` C512 结果和本次 runtime128 补测：两者均为成功
结果，但运行轮次不同，不能混合计算平均值或相互覆盖。按本次补测，Total tok/s 超过
baseline 的是 1024/C512、8192/C256 和 8192/C512；1024/C1、C16、C256 以及
8192/C1、C16 低于 baseline。

## 49. 2026-08-20：custom all-reduce A/B

当前目标脚本曾强制 `--disable-custom-all-reduce`。为排除该开关影响，保持 MxFP4、Waterfill、
DeepEP `low_latency` 和 decode full CUDA Graph 不变，临时开启原生 custom all-reduce。C256
短测 512/512 结果为 Out `12101.52`、Total `24203.05`、Mean TTFT `2141.58 ms`、Mean
TPOT `17.95 ms`，未超过关闭 custom all-reduce 的正式 C256 结果，故已撤回。脚本现支持
`SGLANG_DISABLE_CUSTOM_ALL_REDUCE=0/1` 做 A/B，默认值仍为 `1`；实验备份为
`backups/flash_decode_waterfill_before_custom_ar_ab_20260820.sh`。

## 50. 2026-08-20：DeepEP low-latency 分块 A/B 与服务恢复

为改善 C256/C16 的低并发差距，临时为当前原生 SGLang 0.5.16 启动脚本增加了可选环境变量
`SGLANG_DEEPEP_LL_SPLIT_TOKENS`，其默认值为 `0`，不改变正式配置。保持 MxFP4
FlashInfer TRT-LLM、Waterfill、DeepEP `low_latency`、BF16 dispatcher、decode full CUDA
Graph、custom all-reduce disabled 不变，仅设置 `SGLANG_DEEPEP_LL_SPLIT_TOKENS=1024` 做
C256 短测。

| 配置 | 完成请求 | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---|---:|---:|---:|---:|---:|
| LL split=1024，1024/1024/C256，512 请求 | 512/512 | 7860.30 | 15533.50 | 1278.66 ms | 19.08 ms |
| 当前无 split 的正式 C256（最佳） | 2560/2560 | 12675.54 | 25351.09 | 1636.42 ms | 18.28 ms |
| 正式 baseline C256 | 2560/2560 | 13106.67 | 26213.34 | 2284.50 ms | 16.82 ms |

该分块配置明显降低吞吐，未采用；结果文件为
`logs/flash_decode_waterfill/results_native_flashinfer_20260820/split1024_isl1024_osl1024_c256_n512.jsonl`。
实验前脚本备份为 `backups/flash_decode_waterfill_before_deepep_split_ab_20260820.sh`。

实验重启过程中旧 decode worker 强制退出后，内核出现 GPU6/GPU7 Xid 43，第一次新服务启动
因旧 NCCL 端口残留而失败，没有加载出可用服务，也没有产生性能结果。清理残留并重新启动后，
CUDA Graph（1--256 桶）完整捕获、PD warmup 和 HTTP 服务均恢复；随后执行
`bash validate_pd_whoami.sh` 得到 `HTTP=200`、`WHOAMI_VALID=True`。最终服务恢复为脚本默认
`SGLANG_DEEPEP_LL_SPLIT_TOKENS=0`，因此该轮 Xid/端口故障不计入性能对比。

恢复默认服务后，按正式 benchmark 口径（`random_range_ratio=1.0`、`--tokenize-prompt`）补做
了 1024/1024/C16：160/160 成功，Out `1127.73`、Total `2255.46`、Mean TTFT
`434.45 ms`、Mean TPOT `13.74 ms`。该结果仍低于正式 baseline 的 Out `1348.95`、Total
`2697.90`，说明当前低并发瓶颈仍未解决；结果文件为
`logs/flash_decode_waterfill/results_native_flashinfer_20260820/restored_default_formal_isl1024_osl1024_c16_n160.jsonl`。
此前漏参数的 C16 样本不纳入对比。该轮完成后再次执行 `validate_pd_whoami.sh`，仍为
`HTTP=200`、`WHOAMI_VALID=True`。

## 51. 2026-08-20：DP lm-head 开关 A/B（只对比正式 baseline）

本轮明确只使用正式 PD baseline 作为对照，不使用 DSpark 结果。当前 MxFP4 服务额外开启了
`--enable-dp-lm-head`，而 baseline decode 启动参数未开启该选项，因此将其做成可选开关，
仅关闭该选项进行 C16 A/B；Waterfill、DeepEP `low_latency`、MxFP4 FlashInfer、decode full
CUDA Graph、custom all-reduce disabled 均保持不变。

| 配置 | 完成请求 | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---|---:|---:|---:|---:|---:|
| DP lm-head 开启，正式口径 C16 | 160/160 | 1127.73 | 2255.46 | 434.45 ms | 13.74 ms |
| DP lm-head 关闭，正式口径 C16 | 160/160 | 1133.47 | 2266.95 | 390.49 ms | 13.72 ms |
| 正式 baseline C16 | 160/160 | 1348.95 | 2697.90 | 631.91 ms | 11.25 ms |

关闭 DP lm-head 仅提升约 0.5%，仍低于正式 baseline，不能作为有效优化。结果文件为
`logs/flash_decode_waterfill/results_native_flashinfer_20260820/dp_lm_head_off_formal_isl1024_osl1024_c16_n160.jsonl`；
实验前脚本备份为 `backups/flash_decode_waterfill_before_dp_lm_head_ab_20260820.sh`。
实验结束后已恢复默认开启 DP lm-head，脚本通过 `bash -n`，服务重启并执行
`validate_pd_whoami.sh` 得到 `HTTP=200`、`WHOAMI_VALID=True`。

## 52. 2026-08-20：DeepEP dispatch 容量 A/B（拒绝）

根据当前 shape probe，正式 1024/1024/C256、TP4/DP4 的 `x_tokens` 约为 1820；曾怀疑
默认 `num_max_dispatch_tokens_per_rank=512` 造成额外分块，因此仅将该容量提高到 1024，
其他配置保持不变：原生 SGLang 0.5.16、MxFP4 FlashInfer TRT-LLM、DeepEP `low_latency`、
BF16 dispatcher、Waterfill、decode full CUDA Graph、custom all-reduce disabled、DP lm-head
开启。

正式口径（`random_range_ratio=1.0`、`--tokenize-prompt`）的 C256 结果如下：

| 配置 | 完成请求 | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---|---:|---:|---:|---:|---:|
| dispatch capacity=1024，1024/1024/C256 | 512/512 | 11975.70 | 23951.40 | 2304.24 ms | 18.00 ms |
| 当前最佳 dispatch capacity=512，1024/1024/C256 | 2560/2560 | 12675.54 | 25351.09 | 1636.42 ms | 18.28 ms |
| 正式 baseline C256 | 2560/2560 | 13106.67 | 26213.34 | 2284.50 ms | 16.82 ms |

容量 1024 不仅没有改善 TPOT，Total throughput 还低于容量 512 和正式 baseline，故已
拒绝该方案并恢复脚本默认容量 512。结果文件为
`logs/flash_decode_waterfill/results_native_flashinfer_20260820/dispatch1024_formal_isl1024_osl1024_c256_n512.jsonl`；
实验前脚本备份为 `backups/flash_decode_waterfill_before_dispatch_capacity_ab_20260820.sh`。

这次 A/B 进一步排除了“DeepEP buffer 容量”这一表层因素。当前 C16/C256 的主要缺口仍在
MoE compute：TRT-LLM routed kernel 不接受 DeepEP 的 `masked_m`，只能按静态 M 调度，导致
小 M 下固定调度、padding 和 kernel launch 成本无法消除。后续工作应优化现有的
FlashInfer mixed masked kernel（MXFP8 activation × MXFP4 weight）调用路径，并以 `masked_m`
控制有效行；不再继续尝试 dispatch 容量、split、lm-head 等外围参数。

源码审计进一步确认 FlashInfer 0.6.14 的公开 `grouped_gemm_nt_masked` 虽然支持
`masked_m`，但 `ab_dtype` 只有一个参数，要求 A/B 使用同一种类型；它不能表达本任务所需的
MXFP8 activation × MXFP4 weight。与此同时，安装包内部的 SM100 binding 已经包含
`group_gemm_mxfp4_nt_groupwise_masked`，其模板可以处理 mixed MXFP8×MXFP4，并按
`masked_m`/`expert_ids` 使用固定 expert-major capacity；只是该接口没有公开的 Python
wrapper。现有 SGLang opt-in masked 路径已经通过内部 JIT module 调用了它，因此当前问题
不是“完全没有 mixed masked kernel”，而是该非融合调用路径的 quantize、两次 GEMM、buffer
清零和大量小 group launch 固定开销。后续应围绕 kernel 融合/调用开销优化，而不是继续从
启动参数中寻找吞吐收益。

## 53. 2026-08-20：compact mixed grouped 适配尝试（拒绝）

针对第 52 节确认的 padding 瓶颈，新增了一个显式 opt-in 的
`SGLANG_FLASHINFER_GROUPED_COMPACT=1` 路径：在 GPU 上根据 `masked_m` 构造 4-row 对齐的
`m_indptr`，把 DeepEP expert-major buffer 压缩后调用现有
`group_gemm_mxfp8_mxfp4_nt_groupwise`，并保持固定 workspace 以尝试兼容 CUDA Graph。修改前
备份为 `backups/mxfp4_flashinfer_before_compact_masked_mixed_20260820.py`。

该路径在真实模型上先后验证了：

1. 开启 CUDA Graph 时，首次 capture 在后续 attention CUBLAS 处失败；
2. 使用 `CUDA_LAUNCH_BLOCKING=1`、关闭 Graph 后，错误准确定位到 down GEMM 的
   `CutlassMXFP4GroupwiseScaledGroupGEMMSM100` `InternalError`。

根因是当前 FlashInfer mixed grouped 实现虽然接收 GPU `m_indptr`，但实际 kernel 仍要求输入
`a.shape[0]` 与 `m_indptr[-1]` 一致。为了 Graph 固定 shape，compact 路径必须传入大于实际
segment 总长的静态 workspace，违反该隐含契约；将 `m_indptr` 强行补到 workspace 大小又会
重新计算 padding，失去优化目标。因此该适配已拒绝，未进入吞吐对比，也未改变默认原生
TRT-LLM 路径。

结论收敛为：mixed dtype、`masked_m` 和 CUDA Graph 的底层 SM100 kernel 已存在，但当前
Python/FFI 路径仍不是融合 MoE 实现。要超过 baseline，应在该现有 masked kernel 上减少
quantize/分配/launch，或把它接入带 SwiGLU/finalize 的融合调用；仅在 Python 层重排 buffer
无法完成目标。

## 54. 2026-08-20：全专家 identity gather A/B（拒绝并回滚）

针对 C256/C512 时 `active_expert_ids` 已覆盖全部本地专家、但仍通过高级索引产生
`[E, M, H]` 拷贝这一疑似瓶颈，尝试在全专家场景直接使用 expert-major 切片 view，同时对
scale 使用对应切片；非全专家场景保持原逻辑。该修改只触及 native MxFP4 FlashInfer
路径，Waterfill、DeepEP `low_latency`、decode CUDA Graph、PD 和其余启动参数均未改变。

正式 1024/1024/C256（2560 请求全部成功）结果：

| 配置 | 完成请求 | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---|---:|---:|---:|---:|---:|
| identity gather 优化 | 2560/2560 | 12530.82 | 25061.64 | 1817.08 ms | 18.23 ms |
| 修改前最佳 native 路径 | 2560/2560 | 12675.54 | 25351.09 | 1636.42 ms | 18.28 ms |
| 正式 baseline C256 | 2560/2560 | 13106.67 | 26213.34 | 2284.50 ms | 16.82 ms |

Total tok/s 比修改前下降约 1.1%，未达到 baseline，方案拒绝。原因尚不能简单归因于
高级索引拷贝；直接 view 可能改变后续 quantize/GEMM 所依赖的布局或访问模式，且没有带来
可观测收益。已用 `backups/mxfp4_flashinfer_before_skip_all_expert_gather_20260820.py`
恢复源码，默认路径回到已验证的 25351.09 Total tok/s。后续不再围绕该索引做无证据的微优化，
瓶颈继续聚焦于 masked mixed kernel 的 quantize、padding 和 launch 固定开销。

## 55. 2026-08-20：combine scatter A/B（拒绝）

在不改变 input gather、`compute_m`、TRT-LLM kernel 和 Waterfill 的前提下，单独尝试在
全部本地专家激活时将 kernel 输出直接写入 expert-major combine buffer，以绕过每层一次的
高级索引 scatter。该实验使用显式开关
`SGLANG_MXFP4_NATIVE_DIRECT_COMBINE=1`，修改前备份为
`backups/mxfp4_before_direct_combine_scatter_ab_20260820.py`。

| 配置 | 完成请求 | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---|---:|---:|---:|---:|---:|
| direct combine，1024/1024/C256 短测 | 512/512 | 11641.05 | 23282.11 | 2495.85 ms | 18.24 ms |
| 修改前最佳 C256 短测 | 256/256 | 11346.88 | 22693.75 | 2734.85 ms | 18.04 ms |
| 正式 baseline C256 | 2560/2560 | 13106.67 | 26213.34 | 2284.50 ms | 16.82 ms |

该结果的 TPOT 反而增加，且不同请求数的短测不能证明吞吐提升，因此方案拒绝并回滚。当前
源码恢复默认 scatter，备份保留。结论是 combine scatter 不是当前主要瓶颈；下一步改用 GPU
profiler 直接量化 TRT-LLM routed MoE、quantize、DeepEP dispatch/combine 和 attention 的
时间占比。

## 56. 2026-08-20：GPU profile 与 `index_copy_` A/B（拒绝）

为避免继续凭经验猜测，使用 PyTorch GPU profiler 对正式 baseline 和当前 MxFP4+Waterfill
路径各采集 20 个 C256 decode step。单个 DP rank 的 kernel 累计时间如下（profile 本身只用于
定位，不作为吞吐结果）：

| kernel 类别 | MxFP4+Waterfill | 正式 baseline |
|---|---:|---:|
| DeepEP dispatch | 41.828 ms | 33.463 ms |
| DeepEP combine | 17.024 ms | 18.667 ms |
| MxFP4 routed GEMM 两个主 kernel | 36.136 ms | 31.907 ms（baseline routed GEMM）|
| MxFP8 quantize | 5.583 ms | 8.490 ms |
| `gatherTopK` | 7.455 ms | 无对应额外 kernel |
| 输出 expert-major `index_put` | 6.802 ms | 无对应额外 kernel |

profile 文件分别为：
`logs/flash_decode_waterfill/profile_native_default_c256/` 和
`logs/flash_decode_waterfill/profile_baseline_c256/`。这说明适配路径的可见额外开销主要是
DeepEP 输出后的索引/重排，以及 MxFP4 routed GEMM tactic 本身；不能把全部 dispatch 时间都
归因于 MxFP4。

针对 profile 中明确出现的 6.802 ms `index_put`，尝试用语义等价的
`padded[:, :compute_m, :].index_copy_(0, active_expert_ids, output)` 替换。profile 确认
该 kernel 降为 3.586 ms，但正式 1024/1024/C256 仍为：

| 配置 | 完成请求 | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---|---:|---:|---:|---:|---:|
| `index_copy_` 正式 A/B | 2560/2560 | 12162.50 | 24325.00 | 2474.01 ms | 17.97 ms |
| 修改前最佳 native 路径 | 2560/2560 | 12675.54 | 25351.09 | 1636.42 ms | 18.28 ms |
| 正式 baseline C256 | 2560/2560 | 13106.67 | 26213.34 | 2284.50 ms | 16.82 ms |

GPU kernel 节省没有转化为端到端收益，且 Total 比修改前下降约 4.0%，因此方案拒绝并回滚。
源码备份为 `backups/mxfp4_before_index_copy_scatter_ab_20260820.py`。这次 profile/A-B 排除
了“combine scatter 单点优化”作为突破口；后续应直接处理 MxFP4 routed GEMM tactic 与
DeepEP dispatch 路径之间的结构性差异，或实现真正融合的 masked mixed kernel。

## 57. 2026-08-20：native MxFP4 + FP8 dispatcher 清零瓶颈 A/B（拒绝）

profile 发现正式 baseline 的 DeepEP dispatch 使用 FP8/UE8M0（kernel 模板为
`dispatch<true, true>`），而当前 MxFP4 最佳路径使用 BF16（`dispatch<false, false>`）。因此
验证 `DEEPEP_DISPATCHER_OUTPUT_DTYPE=fp8`，并针对 FP8 适配器中每层 `torch.zeros` 的清零开销
增加 opt-in `SGLANG_MXFP4_NATIVE_FP8_EMPTY_COMBINE=1`：有效 active expert 行会被完整覆盖，
仅省略 inactive/tail 行清零。修改前备份为
`backups/mxfp4_before_fp8_empty_combine_ab_20260820.py`。

FP8 profile 显示该清零 kernel 为单个 DP rank、20 step 累计 **237.617 ms**，是 FP8 路径
从约 30 ms TPOT 降下来的关键必要条件。修复后单次 whoami 和 16/16 并发请求均 HTTP 200、
非空，短测 TPOT 从 30.29 ms 降到 17.97 ms；但正式结果仍未达标：

| 配置 | 完成请求 | Out tok/s | Total tok/s | retokenized output | Mean TTFT | Mean TPOT |
|---|---:|---:|---:|---:|---:|---:|
| FP8 dispatcher + empty combine，正式 C256 | 2560/2560 | 11929.87 | 23859.74 | 2312562/2621440 | 2374.06 ms | 18.48 ms |
| 当前最佳 BF16 native 路径 | 2560/2560 | 12675.54 | 25351.09 | — | 1636.42 ms | 18.28 ms |
| 正式 baseline C256 | 2560/2560 | 13106.67 | 26213.34 | — | 2284.50 ms | 16.82 ms |

FP8 路径虽然接口请求全部完成，但 retokenized 数量明显偏低，且 Total 仍低于 BF16 和
baseline；不能把“dispatch 更快”误判为有效性能收益。方案拒绝，源码已恢复 FP8 分支的
安全 `zeros` 和 BF16 dispatcher 默认配置。结论是 FP8 dispatch 的额外收益被当前 native
scale/layout 适配和 routed GEMM 代价抵消，下一步若继续走该路线必须先解决数值/输出一致性，
否则应优先实现真正融合的 mixed masked kernel。

## 58. 2026-08-20：`index_select` gather A/B（拒绝并恢复默认）

profile 显示 native MxFP4 路径在 DeepEP low-latency 返回的 expert-major buffer 上，需要先按
`active_expert_ids` gather，再进行量化和 routed GEMM。为验证高级索引是否是主要开销，曾将
`expert_hidden[active_expert_ids, :compute_m, :]` 改为等价的 `index_select(0, active_expert_ids)`，
并保留了 opt-in 环境变量和备份：
`backups/mxfp4_before_index_select_gather_ab_20260820.py`。

该方案没有进入正式长测。C256、OSL=256 的诊断测量完成 256/256，请求输出有效，但 profile
显示新增的 `indexSelectSmallIndex` 单个 DP0、20 decode step 累计 **5.220 ms**；原高级索引
路径对应的 vectorized gather 约 **2.087 ms**，同时仍有其它 index kernel。也就是说，API
替换没有减少 gather 结构性开销，反而引入更重的 index-select kernel，不能据此声称性能提升。
该路径已拒绝，源码恢复为默认高级索引实现；当前服务重启后已通过 `validate_pd_whoami.sh`：
HTTP 200、`WHOAMI_VALID=True`，PD decode warmup 的 4 个 DP rank 请求全部返回 200。

当前瓶颈判断保持不变：MxFP4 native 相对 baseline 的主要差异仍是 BF16 DeepEP dispatch、
expert-major gather/quantize/padding，以及非融合的 routed GEMM；下一步应实现真正的 gather+量化
或 masked mixed kernel，不能继续在 indexing API 之间循环尝试。

## 59. 2026-08-20：DeepEP SMS64 + exact-M C16 A/B（拒绝）

针对 profile 中 DeepEP dispatch 的固定开销，使用 `DEEPEP_CONFIG` 将 normal dispatch/combine
的 SMS 从默认 96 改为 64，同时设置此前单独测试过的
`SGLANG_MXFP4_NATIVE_TUNE_ACTUAL_M=1`。MxFP4 FlashInfer、DeepEP low-latency、Waterfill、
decode full CUDA Graph、PD 和请求参数均保持不变。

正式 1024/1024/C16（160 请求，160/160 完成）结果：

| 配置 | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---|---:|---:|---:|---:|
| SMS64 + exact-M | 1127.49 | 2254.99 | 399.09 ms | 13.75 ms |
| 正式 baseline | 1348.95 | 2697.90 | 631.91 ms | 11.25 ms |

该结果没有改善默认 native 路径，也没有达到 baseline；实验结果为
`logs/flash_decode_waterfill/results_native_flashinfer_20260820/sms64_exactm_c16_n160.jsonl`。
该 A/B 说明调整 `normal_dispatch` SMS 和 TRT-LLM tactic 不能消除 low-latency 下的主要差距，
当前服务已恢复默认 SMS96/96 和 `tune_actual_m=auto`，并再次通过 `validate_pd_whoami.sh`：
HTTP 200、`WHOAMI_VALID=True`。

## 61. 2026-08-20：重新审计 FlashInfer masked kernel 的 capacity 语义（不改默认路径）

本轮没有继续做启动参数或 indexing API 的盲目 A/B，而是直接检查当前安装的 FlashInfer
0.6.14 SM100 实现。`group_gemm_mxfp4_nt_groupwise_masked` 的接口含义如下：

```text
masked_m[expert_ids[i]] -> 第 i 个 group 的有效 M
capacity                 -> A/B/D/scale backing buffer 的每组物理 stride
expert_ids[i]            -> 第 i 个 group 对应的权重专家编号
```

底层 `compute_sm100_cutlass_group_gemm_args_masked` 明确按
`B + i * capacity * K`、`D + i * capacity * N` 取每组输入输出，并按
`masked_m[expert]` 设置实际问题尺寸。因此 `capacity` 不能简单替换成有效行数
`segment_m`，除非输入、输出、scale 和所有 group pointer 同时改成真正的压缩布局。
当前 opt-in masked 路径把 expert-major 张量截取为 `segment_m`，但仍用
`masked_m` 作为实际 M；这条路径不应在未重新验证 capacity 上界和完整输出语义前进入默认服务。

该审计也解释了为什么之前的 masked A/B 没有带来收益：它虽然能够减少有效 GEMM 的 M，
但仍要执行每层的 FP8 quantize、临时 buffer 分配/清零、两次 group GEMM 和 combine 回填，
且 capacity/layout 不能仅靠 Python 切片压缩。当前 profile 的量化结果仍是唯一可信的优化方向：

| 项目 | MxFP4+Waterfill | 正式 baseline | 差值/判断 |
|---|---:|---:|---:|
| DeepEP dispatch（20 step，单 DP rank） | 41.828 ms | 33.463 ms | +8.365 ms，主要通信/dispatcher 差异 |
| DeepEP combine | 17.024 ms | 18.667 ms | -1.643 ms，不是瓶颈 |
| routed GEMM 主 kernel | 36.136 ms | 31.907 ms | +4.229 ms，MxFP4 tactic/布局差异 |
| 额外 gatherTopK + 输出 scatter | 7.455 + 6.802 ms | 无对应额外项 | 适配路径结构性开销 |

因此下一步只有两类工作值得继续：一是实现真正保持 fixed-capacity 指针语义的 fused
masked mixed-MoE（至少融合 quantize/两次 GEMM/activation 的临时开销）；二是从 DeepEP
dispatcher/Waterfill 的 kernel 级差异入手，确认是否能在不改变正确性和 low_latency 语义的
前提下消除约 8 ms dispatch 差距。不会再把 capacity、SMS、DP lm-head 或单独的 scatter
API 当作主要突破口。默认源码、服务配置和通过 `validate_pd_whoami.sh` 的状态均未改变。

## 62. 2026-08-20：FP8 dispatcher 尾部清零实验（否决）

为了验证 baseline 使用 FP8 dispatcher 是否能消除当前 BF16 dispatcher 后的 MXFP8
重新量化，新增了 opt-in 开关 `SGLANG_MXFP4_FP8_MASK_TAIL=1`。该开关根据
`masked_m` 将 low-latency FP8 expert-major buffer 的无效尾行置零。首次实现使用
Float8 `masked_fill_`，在 CUDA Graph capture 时失败；错误为
`masked_fill_ not implemented for Float8_e4m3fn`。随后改为 BF16 staging mask 后再转回
FP8，能够完成 1--256 的 decode CUDA Graph capture。

服务检查结果：

```text
HTTP=200
WHOAMI_VALID=True
```

但真实请求过程中，服务日志显示单请求 decode 速度约 40 token/s；每层 FP8→BF16→FP8
转换的代价远大于可能节省的 dispatcher/quantize 开销。该实验在完成若干请求后主动停止，
没有产生可用于正式吞吐对比的 JSON 结果，也不把中断前的请求当作成功率结论。方案否决，
不进入正式 baseline 对比。

实验前源码备份为：

```text
backups/mxfp4_before_fp8_tail_mask_ab_20260820.py
```

当前服务已恢复 BF16 dispatcher、MxFP4 FlashInfer、DeepEP low_latency、Waterfill 和
decode CUDA Graph；恢复后应再次执行 whoami 校验。结论是：FP8 dispatcher 只有在
DeepEP/FlashInfer kernel 内部直接消费 FP8 buffer、避免 Python staging conversion 时才
有价值；不能用 Python 层尾部清零实现。

## 60. 2026-08-20：当前工作树复核 DeepGEMM masked fallback（正确性失败）

为确认历史 fallback 结果是否能在当前统一 `PYTHONPATH=/data/ssd2/sglang_v0.5.16/python`
的环境复现，临时启动：

```text
SGLANG_MXFP4_LL_BACKEND=deep_gemm
DEEPEP_DISPATCHER_OUTPUT_DTYPE=fp8
DEEPEP normal dispatch/combine SMS=64
SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1
SGLANG_OPT_USE_JIT_EP_ACTIVATION=1
SGLANG_OPT_SWIGLU_CLAMP_FUSION=1
```

该配置仍使用 `--moe-runner-backend flashinfer_mxfp4` 和 Waterfill，但在 low-latency masked
GEMM 阶段实际调用 DeepGEMM，不属于纯 FlashInfer routed kernel。正式 C16 压测前的每次服务
健康检查返回 HTTP 200，但语义校验失败：`WHOAMI_VALID=False`，返回内容为乱码。因此压测
已立即停止，没有产生可用吞吐结果，也不能引用历史未统一源码路径的 fallback 数字作为当前
结论。

之后已恢复默认 native MxFP4 FlashInfer + BF16 dispatcher + DeepEP low-latency + Waterfill
配置，并重新启动 prefill（此前 decode 重启时 prefill 进程也已退出，导致 Router 502）。当前
30000/30001 的 `/model_info` 均 HTTP 200，Router “你是谁”再次通过：HTTP 200、
`WHOAMI_VALID=True`。这次失败进一步确认：任何 fallback 优化必须先修复 FP8 scale/layout
正确性，不能先看吞吐。

## 63. 2026-08-20：MXFP8 activation quantize 后端 A/B（否决）

profile 中 MXFP8 activation quantize 在单 DP rank 的 20 个 decode step 累计约 5.583 ms，
因此只针对该瓶颈测试 `SGLANG_MXFP8_QUANTIZE_BACKEND=cuda`。其余配置保持不变：原生
SGLang 0.5.16、MxFP4 FlashInfer routed kernel、DeepEP low_latency、Waterfill、decode
CUDA Graph、PD 分离、BF16 dispatcher。

短测为 ISL=1024、OSL=256、C256、256 请求，结果文件：
`logs/flash_decode_waterfill/results_native_flashinfer_20260820/quant_cuda_ab_isl1024_osl256_c256_n256.jsonl`。

| 配置 | 完成请求 | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT | retokenized 输出 |
|---|---:|---:|---:|---:|---:|---:|
| CUDA quantize 后端 | 256/256 | 6712.01 | 33560.03 | 3161.33 ms | 17.78 ms | 66904/65536 |

该短测没有出现 HTTP 请求失败，但 retokenized 输出多于目标输出，不能作为正确性通过；
同时没有形成可证明优于当前默认 cute-dsl quantize 的端到端收益，因此不进入正式长测，
环境变量已移除并恢复默认配置。恢复后的服务再次通过 `validate_pd_whoami.sh`：
HTTP 200、`WHOAMI_VALID=True`。

本轮排查后，量化后端不是当前突破口。剩余可量化的主要差距仍是 DeepEP dispatch 累计约
8.365 ms，以及 expert-major 输入重排与 routed GEMM 之间缺少融合；继续调 Python quantize
后端会陷入局部 kernel 优化，不能解决结构性瓶颈。

## 64. 2026-08-20：DeepEP FP8 线性 scale transport A/B（否决）

继续针对 FP8 dispatcher 做接口级验证。DeepEP low-latency 在 Blackwell 上默认返回
packed UE8M0 scale；本次新增临时 opt-in `SGLANG_DEEPEP_FP8_LINEAR_SCALE=1`，让 DeepEP
返回普通 per-128 FP8 scale，避免 Python 适配器解包 packed int32，再由 MxFP8 kernel
扩展到 per-32 scale。实验前备份：
`backups/deepep_before_fp8_linear_scale_ab_20260820.py`。

实验配置仍为 MxFP4 FlashInfer、Waterfill、DeepEP low_latency、decode CUDA Graph、PD 分离，
并启用 FP8 dispatcher 和此前的 empty-combine 优化。短测为 ISL=1024、OSL=256、C256、
256 请求，结果文件：
`logs/flash_decode_waterfill/results_native_flashinfer_20260820/fp8_linear_scale_isl1024_osl256_c256_n256.jsonl`。

| 配置 | 完成请求 | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT | retokenized 输出 |
|---|---:|---:|---:|---:|---:|---:|
| FP8 + linear scale | 256/256 | 5217.40 | 26087.02 | 2691.08 ms | 30.71 ms | 57692/65536 |

该方案同时出现吞吐下降和 retokenized 输出不足，未通过正确性/性能门槛，已恢复原始
DeepEP scale 逻辑和默认 BF16 dispatcher。恢复服务已通过 `validate_pd_whoami.sh`：
HTTP 200、`WHOAMI_VALID=True`。

结论：FP8 dispatcher 的收益不能通过 Python 层 scale 格式转换获得；当前需要的是 DeepEP
输出 buffer 到 MxFP4 GEMM 的 fused masked implementation，至少应在设备端一次完成有效行
mask、MxFP8 quantize、两次 MxFP4 group GEMM、SwiGLU 和输出回填。否则 BF16 dispatcher
虽然通信稍慢，却避免了 FP8 适配路径的额外数据转换和数值风险。

## 65. 2026-08-20：FlashInfer routed kernel direct-BF16 A/B（否决）

审计 FlashInfer 0.6.14 的 `trtllm_fp4_block_scale_routed_moe` 后确认其接口允许
`hidden_states=bfloat16` 且 `hidden_states_scale=None`。当前 native 适配器默认先执行
`mxfp8_quantize`，因此新增 opt-in `SGLANG_MXFP4_DIRECT_BF16=1`，验证是否可以让
FlashInfer routed kernel 内部完成转换，从而消除 profile 中约 5.6 ms 的独立 quantize kernel。
实验前备份：`backups/mxfp4_before_direct_bf16_ab_20260820.py`。

短测为 ISL=1024、OSL=256、C256、256 请求，结果文件：
`logs/flash_decode_waterfill/results_native_flashinfer_20260820/direct_bf16_isl1024_osl256_c256_n256.jsonl`。

| 配置 | 完成请求 | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT | retokenized 输出 |
|---|---:|---:|---:|---:|---:|---:|
| FlashInfer direct BF16 | 256/256 | 6772.04 | 33860.22 | 3005.66 ms | 18.88 ms | 66885/65536 |
| 当前 native formal C256 | 2560/2560 | 12675.54 | 25351.09 | 1636.42 ms | 18.28 ms | — |

direct-BF16 请求能够完成，但 TPOT 比当前 native formal C256 更差，且短测 retokenized
数量存在偏差，不能进入正式长测或声称超过 baseline。该 opt-in 已回滚，服务恢复为
默认 MXFP8 quantize 路径并通过 `validate_pd_whoami.sh`：HTTP 200、`WHOAMI_VALID=True`。

这次 A/B 证明：当前独立 quantize kernel 虽然可见，但把它移入 FlashInfer 内部并不会
自动带来收益；真正需要优化的是固定 expert-major capacity 下的 fused masked pipeline，
而不是继续切换 activation 输入格式。

## 66. 2026-08-20：FP8 active-tail uint8 mask kernel A/B（否决）

此前 FP8 tail-mask 方案使用 BF16 staging，实测代价过高。本轮实现了独立 Triton uint8
原地清零 kernel：对已经 gather 的 active expert FP8 buffer，只清零
`row >= masked_m[expert]` 的尾行，不做 FP8→BF16 转换。kernel 单测通过，覆盖了
`[3, 16, 257]` FP8 buffer 的不同有效行数；实验代码备份为：
`backups/mxfp4_fp8_tail_kernel_experimental_20260820.py`。

服务 A/B 配置为 FP8 dispatcher、uint8 tail mask、empty combine，其余保持 native
MxFP4 FlashInfer + Waterfill + low_latency + decode CUDA Graph + PD 不变。C256、
ISL=1024、OSL=256 短测结果：

| 配置 | 完成请求 | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT | retokenized 输出 |
|---|---:|---:|---:|---:|---:|---:|
| FP8 + uint8 tail mask | 256/256 | 4929.83 | 24649.17 | 3604.85 ms | 30.46 ms | 55400/65536 |

该 kernel 虽然单测正确，但每层额外 Triton launch 使端到端 TPOT 退化到约 30 ms，
同时输出 token 统计异常，方案否决并已回滚。恢复后的默认 BF16 native 服务再次通过
`validate_pd_whoami.sh`：HTTP 200、`WHOAMI_VALID=True`。

这次实验把“FP8 tail 清零必须在 DeepEP/compute kernel 内融合”这一边界验证清楚了：
即使清零本身不需要 staging，只要仍是每层独立 launch，就无法弥补 FP8 适配开销。

## 67. 2026-08-20：DeepEP low_latency 输出接口定位与下一步边界

为避免继续围绕 capacity、SMS、indexing API 和 quantize backend 做无方向 A/B，本轮直接检查
当前安装版本对应的 DeepEP 源码 `/sgl-workspace/DeepEP`，重点查看
`csrc/kernels/internode_ll.cu` 与 `Buffer::low_latency_dispatch`。

源码确认了以下事实：

1. low_latency dispatch 已经在设备端生成固定的 expert-major `packed_recv_x`，并返回
   `packed_recv_count`、`packed_recv_src_info` 和 `packed_recv_layout_range`。SGLang 当前的
   `masked_m` 来自 `packed_recv_count`，因此缺少的不是一个额外的 token mask。
2. DeepEP 的 BF16 分支直接搬运 BF16；FP8 分支在 dispatch 内完成 FP8 cast，但 scale 是每
   128 个 hidden 元素一组（Blackwell 的 UE8M0 情况还会以 int32 打包后返回）。这两种输出
   都不能直接满足当前 FlashInfer MxFP4 mixed kernel 所需的 MXFP8、每 32 元素 scale 和
   固定 capacity 指针布局。
3. dispatch kernel 的接收阶段同时完成跨 rank token 搬运、expert-major 排列、`src_info`
   写入和 scale 写入。若只在 Python/FFI 层把 `packed_recv_count` 传给现有 masked GEMM，
   仍会保留独立的 MXFP8 quantize、临时 buffer、SwiGLU、第二次 quantize、down GEMM 和
   expert-major combine 回填；第 38、39、57、62、66 节的实测已经证明这些独立步骤会吞掉
   任何局部收益。

因此，“修改 DeepEP 返回 mask”不是有效方案；可行的底层路线只有：

```text
DeepEP low_latency receive
  -> 在设备端直接产生 FlashInfer 可消费的 MXFP8/scale 和 fixed-capacity metadata
  -> masked up GEMM + SwiGLU + down GEMM + combine buffer 回填（至少减少中间 launch）
```

这需要修改并重新编译 DeepEP/FlashInfer 的 CUDA 或 C++ 接口，不能只改 SGLang Python。尤其
要保持 `packed_recv_count` 的跨 rank 同步和 CUDA Graph 的固定地址/shape 语义，不能把有效
行压缩成动态长度后再伪装成 fixed-capacity。当前默认服务不改，仍是已经验证过的原生
MxFP4 FlashInfer + DeepEP low_latency + Waterfill + decode CUDA Graph；本轮没有产生新的
吞吐数字，也没有把未融合的 masked 路径宣称为优化。

下一步仅在具备底层 kernel 修改条件后进行：先备份 DeepEP 源码和已安装 wheel，设计一个
最小的 device-side fused output/scale 接口，先用单层数值对照和 CUDA Graph 小 batch 验证，
再做正式 baseline 的 C1/C16/C256/C512 和 8192 输入测试。若不能完成该接口，则当前结果应
明确记为“C512 超过 baseline，但 C16/C256 尚未超过”，而不是继续进行外围参数搜索。

## 68. 2026-08-20：prepermuted metadata 修复与正确性/性能结论

本轮针对第 67 节定位的 expert-major 布局瓶颈，实现了一个仅 opt-in 的 FlashInfer
TRT-LLM prepermuted 入口。DeepEP low_latency 已返回 expert-major 行，该入口尝试跳过
TRT-LLM routing histogram 和重复 gather。实现前已备份到：
`backups/flashinfer_prepermuted_20260820/`。

第一次实现的 `permuted_idx_to_token_idx` 使用 identity，并把
`cta_idx_xy_to_mn_limit` 当成 group 内局部 offset。服务能启动，但“你是谁”返回异常，
因此未进行性能统计。随后按 TRT-LLM `RoutingKernel.cuh` 的语义修复：

1. compact 输入 `[group, capacity]` 映射到 padded GEMM 地址空间
   `[group, padded_per_group]`，后续 group 跳过 tile padding；
2. `cta_idx_xy_to_mn_limit` 改为全局 cumulative padded limit，而不是局部 limit。

修复后的 prepermuted 服务在 4 个 DP 上完成 warmup，所有 warmup `/generate` 请求均为
HTTP 200；`validate_pd_whoami.sh` 通过。随后执行正式 C16 短测（ISL=1024、OSL=1024、
160 prompts、160/160 完成）：

| 配置 | 完成请求 | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---|---:|---:|---:|---:|---:|
| native 默认路径 | 160/160 | 1127.73 | 2255.46 | 434.45 ms | 13.74 ms |
| prepermuted 修复路径 | 160/160 | 1116.47 | 2232.94 | 426.87 ms | 13.89 ms |
| formal baseline | 160/160 | 1348.95 | 2697.90 | 631.91 ms | 11.25 ms |

结果表明：metadata 修复后正确性成立，但 prepermuted 没有吞吐收益；Total tok/s 比
默认 native 低约 1.0%，比 baseline 低约 17.2%。后续 MXFP8 quantize、scale/layout
转换、SwiGLU、down GEMM 和 combine 回填仍然存在，绕过 routing metadata 不足以覆盖
这些开销。因此没有继续跑 C256/C512，避免无效参数搜索。

当前默认启动仍关闭 `SGLANG_MXFP4_NATIVE_PREPERMUTED`，保留已验证的 MxFP4 FlashInfer
+ DeepEP low_latency + Waterfill + decode CUDA Graph 配置。`flash_decode_waterfill.sh`
中的 `FLASHINFER_JIT_SKIP_BUILD=1` 仅在已有 `.so` 时加载缓存，相关原始文件和修改均已
备份。最终结论仍是：要超过 baseline，需要把 DeepEP receive、MXFP8 scale 生成、SwiGLU
和 combine 回填进一步融合到设备端 kernel，而不是继续搜索 routing 参数。
## 69. 2026-08-20：关闭 PDL 的单点 A/B——未改变 C256 瓶颈

针对 profile 中可能存在的 PDL 跨 kernel 开销，本轮只做了一个明确的 A/B：保持
MxFP4 FlashInfer、DeepEP `low_latency`、Waterfill、decode CUDA Graph、PD 分离和
`normal_dispatch/combine.num_sms=96` 全部不变，仅设置 `SGLANG_MXFP4_NATIVE_PDL=0`。
服务启动后先执行 `validate_pd_whoami.sh`，结果为 HTTP 200、`WHOAMI_VALID=True`，随后
跑完整 C256（ISL=1024、OSL=1024、2560 prompts），2560/2560 请求成功。

| 配置 | 完成请求 | Out tok/s | Total tok/s | Mean TTFT | Mean TPOT |
|---|---:|---:|---:|---:|---:|
| PDL-off A/B | 2560/2560 | 12792.47 | 25584.94 | 1491.78 ms | 18.18 ms |
| 当前 native 最佳 C256 | 2560/2560 | 12675.54 | 25351.09 | 1636.42 ms | 18.28 ms |
| formal baseline C256 | 2560/2560 | 13106.67 | 26213.34 | 2284.50 ms | 16.82 ms |

PDL-off 相对当前 native 最佳仅提升约 0.9% Total tok/s，仍比 baseline 低约 2.4%；TPOT
仍比 baseline 高约 8.1%。因此 PDL 不是“缺少 2.4% 吞吐”的主要原因，也不足以改变
C256 未超过 baseline 的结论。该 A/B 到此收敛，不再继续做 PDL 开关或 SMS 参数矩阵。

结合前述 GPU trace，当前 C256 的主要瓶颈仍是 DeepEP dispatch/receive 后的 expert-major
数据处理链：active expert gather、MXFP8 quantize/scale 生成、routed up/down GEMM、
SwiGLU 及 combine 回填。下一项有实质意义的工作只能是 device-side fused
receive+quantize/scale 或 fused masked MoE；外围 routing metadata、capacity、PDL 和
SMS 调参已经没有足够收益空间。实验产物为
`logs/flash_decode_waterfill/results_native_flashinfer_20260820/pdl_off_isl1024_osl1024_c256_n2560.jsonl`。

## 70. 2026-08-20：针对 DeepEP receive-side MXFP8 scale 的定向优化尝试

本轮没有继续搜索 Waterfill、PDL、SMS 或 capacity 参数，而是根据 profile 对 C256
瓶颈做了一个底层定向改动。当前 native 路径中，DeepEP dispatch/receive 以及接收后的
expert-major 处理链占主要时间，其中包括 MXFP8 quantize/scale、expert gather、routed
MxFP4 GEMM、SwiGLU 和 combine 回填。目标是让 DeepEP `low_latency` dispatch 在接收 BF16
数据时直接生成 FlashInfer MxFP4 可消费的 MXFP8 数据和每 32 个 hidden channel 的
UE8M0 scale，从而跳过独立的 receive-side quantize/scale kernel。

已完成的代码改动（均为 opt-in，默认路径不启用）：DeepEP low_latency dispatch 增加
`use_mxfp8` 分支；发送端按 K/32 做 amax reduction，接收端生成 `[token, hidden/32]`
的 uint8 scale；修正 K=7168 时 scale copy 只覆盖前两段的问题；并修正
`LowLatencyLayout` 的 RDMA dispatch message 预留空间，使 MXFP8 的 K/32 scale payload
不会与旧 K/128 stride 冲突。SGLang MxFP4 FlashInfer 路径也增加了对应的 opt-in scale
dtype/布局适配，默认 `SGLANG_DEEPEP_MXFP8_DISPATCH=0`。

源代码和已安装 DeepEP 扩展均已备份到 `backups/deepep_mxfp8_20260820/`，包括 kernel、
C++ API、Python wrapper 和旧 `.so`。编译和安装成功；初次运行遇到 CUDA named symbol
lookup failure，已改为直接模板 launch 并重新编译。随后发现 MXFP8 K/32 scale 与旧
RDMA layout stride 不一致，已修正 layout 后重新编译。这些属于实现问题，不能作为
性能结论。

本轮尚未得到有效性能 A/B。PD 验证链路在启动阶段被外部 GPU/服务状态打断：prefill
多次在权重加载 15/46 或 21/46 shard 时退出，日志没有 Python traceback、CUDA kernel
exception 或 DeepEP error；同一时间段内核日志出现 NVIDIA/NVLink 状态错误。此前的
pyspy 已明确记录过另一项独立启动问题：多个 prefill rank 在 FlashInfer
`build_and_load()` 文件锁上等待，已在 `flash_prefill_baseline.sh` 增加已有 `.so` 的
`FLASHINFER_JIT_SKIP_BUILD=1`，原脚本备份在 `backups/flash_prefill_20260820/`。本次
重启已绕过 JIT 锁，但仍被权重加载阶段的外部终止阻断，因此不能把失败归因于 MXFP8
dispatch，也不能据此宣称超过 baseline。

当前可复用的性能结论仍是：C512 native MxFP4 已超过 baseline；C256 native/PDL-off
仍略低于 baseline，主要差距来自 receive 后 expert-major 处理链，而不是 Waterfill
本身。待 GPU/NVLink 和 PD 服务稳定后，验证顺序固定为：

```text
服务稳定 -> whoami HTTP 200 且全部请求成功 -> opt-in C256 A/B
-> 对比 baseline 的 Total tok/s、Mean TTFT、Mean TPOT
-> 只有 C256 成功后才扩展 C16/C512/8192
```

如果 MXFP8 dispatch 不能在 C256 降低 dispatch/quantize/scale 总成本，则下一步应停止
该分支，转向真正的 fused receive-to-MoE kernel；不再进行无依据的外围参数矩阵搜索。

## 71. 2026-08-21：MXFP8 dispatch 正确性排查，尚未进入性能 A/B

本轮按“先正确性、后吞吐”的顺序继续验证，得到以下确定结果：

1. 初版 DeepEP 扩展按 `sm90` 编译，而机器 GPU 是 B200 `sm100`。在 MXFP8 专用
   template kernel 上出现 `cudaLaunchKernelEx: named symbol not found`。将扩展按
   `TORCH_CUDA_ARCH_LIST=10.0` 重新编译后，该错误消失，说明这是架构/扩展产物问题，
   不是 RDMA layout 问题。
2. 为避免专用 MXFP8 template symbol 的加载问题，曾将 MXFP8 改为复用已有 FP8/UE8M0
   kernel symbol 的运行时分支；sm100 编译后的 kernel 可以启动并完成一次请求。
3. 发现部分 DP rank 的请求 token 数为 0。原路径仍把 M=0 传入 FlashInfer routed
   MoE，触发 device-side floating-point exception。已增加空 rank 的零输出
   `DeepEPLLCombineInput` 快速路径，避免对 M=0 调用 FlashInfer。
4. 修复空 rank 后，opt-in 直接 MXFP8 路径能够 HTTP 200，但 whoami 内容异常；开启已有
   FP8 反量化诊断路径后仍异常，说明问题不只是 UE8M0 数值转换，仍存在 expert-major
   row/route 映射或固定 capacity 契约不一致。将 `SGLANG_FLASHINFER_NATIVE_EXPECTED_M`
   从 1 改为 0、扩大计算行数，也未恢复 whoami 正确性。
5. 使用旧 DeepEP `.so` 与旧 Python wrapper 做兼容性控制时，whoami 仍未通过。这一控制
   不能证明原始 baseline 本身错误，因为当前 SGLang 工作树和服务启动状态已包含多轮
   历史适配；它只能说明当前环境尚未形成可用于 A/B 的干净正确性基线。

因此本轮没有产生任何正式吞吐数字，也没有把 HTTP 200 或某个 Total tok/s 结果当作成功。
当前 gate 明确为：

```text
sm100 DeepEP 扩展可加载
    -> 空 rank 不崩溃
    -> whoami 内容正确（当前未通过）
    -> 全量请求成功
    -> decode CUDA Graph 正式配置
    -> C256 与 baseline 对比
```

本轮新增扩展产物和控制版本均已备份在 `backups/deepep_mxfp8_20260820/`。在 whoami
恢复前，继续跑 C256 或切换 CUDA Graph/SMS/Waterfill 参数都没有诊断价值；下一步应先做
干净 native baseline 与当前工作树的逐项差异核对，重点检查 DeepEP LL 的
`packed_recv_x`、`packed_recv_x_scales`、`masked_m`、`expected_m` 与 FlashInfer routed
kernel 的 row/route 语义，而不是继续搜索性能参数。
