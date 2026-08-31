# PD 分离最终推荐部署方案（按 workload 选择 Prefill + Decode 纯 DSpark）

## 结论

当前在本机、当前源码和 DeepSeek-V4-Flash workload 下，Decode 侧的最佳稳定组合
已经固定为 DSpark；Prefill 侧则按输入长度和并发选择 runner/chunk：

- Prefill：GPU 0--3，TP4/DP1/EP1，Mooncake；低/中并发短输入使用 MegaMoE，
  高并发或长输入使用直接 `flashinfer_mxfp4`。
- Decode：DSpark，GPU 4--7，TP4/DP4/EP4，DeepEP `low_latency`。
- Decode CUDA Graph：graph128，static verify。
- Decode `moe-runner-backend=auto`；不叠加 Waterfill、LPLB、DeepGEMM target、TBO、FP4 indexer 或 HiSparse。
- Prefill transfer worker：16；PD hidden pool/queue 使用现有已验证配置。

Prefill profile 建议如下：

```text
ISL=1024, C=1/16/256：MegaMoE，max/chunk=8192（或 16384，按已有部署脚本）
ISL=1024, C=512：flashinfer_mxfp4，max/chunk=16384
ISL=8192, C=1：MegaMoE，max/chunk=8192
ISL=8192, C=16/256：flashinfer_mxfp4，max/chunk=32768
ISL=8192, C=512：flashinfer_mxfp4，max/chunk=49152
```

这里的 49152 是直接 `flashinfer_mxfp4` Prefill 的已验证 profile；此前 OOM 的是
`MegaMoE + chunk32768` 或其他显式 runner 组合，不能把这些失败结果等同于直接
flashinfer_mxfp4 路径。单一常驻服务无法根据每个请求的 ISL/C 自动切换 profile，
生产部署应按业务 workload 选择并重启对应 Prefill profile。

## 启动

使用当前源码：

```text
/data/ssd2/sglang_v0.5.16_pr32281_fix7_repro_20260823
```

Prefill：

```bash
SGLANG_MAX_PREFILL_TOKENS=8192 \
SGLANG_CHUNKED_PREFILL_SIZE=8192 \
SGLANG_SERVICE_LOG_DIR=/data/ssd2/gongoubo/single_node/repro_pr32281_20260823/step24_prefill_megamoe_decode_dspark/logs/services/prefill_final \
bash /data/ssd2/gongoubo/single_node/repro_pr32281_20260823/step24_prefill_megamoe_decode_dspark/flash_prefill_megamoe.sh
```

1024 输入 workload 可将两个 Prefill 参数同时改为 `16384`。脚本会自动把
`SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` 同步到该值，使 MegaMoE
不会因为 cap 小于 Prefill chunk 而意外回退到普通 MoE runner。

Decode：

```bash
SGLANG_SERVICE_LOG_DIR=/data/ssd2/gongoubo/single_node/repro_pr32281_20260823/step24_prefill_megamoe_decode_dspark/logs/services/decode_final \
bash /data/ssd2/gongoubo/single_node/repro_pr32281_20260823/step24_prefill_megamoe_decode_dspark/flash_decode_dspark.sh
```

Decode 固定启用 DeepEP `low_latency`、CUDA Graph 1/2/4/8/16/32/64/128、DSpark
draft CUDA Graph 和 static verify。Router 使用同目录的 `flash_router_baseline.sh`。

每次重启后必须依次检查 health、Router `whoami`，再开始压测：

```bash
bash /data/ssd2/gongoubo/single_node/validate_pd_whoami.sh
```

## 有效性能结果

| ISL | OSL | C | Requests | Out tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 1 | 10/10 | 299.41 | 598.82 | 185.71 | 3.16 |
| 1024 | 1024 | 16 | 160/160 | 3445.61 | 6891.22 | 353.38 | 4.12 |
| 1024 | 1024 | 256 | 2560/2560 | 31842.97 | 63685.94 | 714.44 | 6.95 |
| 1024 | 1024 | 512 | 5120/5120 | 36572.46 | 73144.91 | 1101.05 | 12.36 |
| 8192 | 1024 | 1 | 10/10 | 302.15 | 2719.31 | 205.19 | 3.11 |
| 8192 | 1024 | 16 | 160/160 | 3411.87 | 30706.81 | 548.56 | 3.98 |
| 8192 | 1024 | 256 | 2560/2560 | 5452.67 | 49074.07 | 41318.77 | 4.27 |
| 8192 | 1024 | 512 | 5120/5120 | 5486.57 | 49379.13 | 86588.80 | 4.34 |

证据目录：

```text
/data/ssd2/gongoubo/single_node/repro_pr32281_20260823/step24_prefill_megamoe_decode_dspark
```

该目录保存 Prefill/Decode/Router 独立日志、whoami、8 组 JSONL、配置记录和
失败 A/B 记录；脚本备份位于：

```text
/data/ssd2/gongoubo/single_node/backups/prefill_megamoe_decode_dspark_20260823
```

## 不纳入最终方案的路径

- MegaMoE + chunk32768：高并发 attention/PD hidden OOM。
- Prefill 显式 DeepGEMM：高并发临时 workspace OOM。
- MegaMoE 回退到显式 flashinfer_mxfp4：缺少 `output1_scale_scalar`；这不影响直接
  `flashinfer_mxfp4` Prefill profile。
- Decode 端 Waterfill/LPLB/DeepGEMM/TBO/FP4 indexer/HiSparse：已有消融结果显示
  并非在当前随机 workload 上都产生独立收益，且部分组合有功能或稳定性边界；
  在没有新的 workload-specific 证据前不纳入最终部署。

## 2026-08-23：纯 DSpark 独立验收

按当前最佳 PD 配置重新启动后，Decode 仅启用 DSpark：DeepEP `low_latency`、Decode
CUDA Graph batch `1/2/4/8/16/32/64/128`、static verify；没有启用 TBO、Waterfill、
LPLB、FP4 indexer 或 DeepGEMM MoE runner。

- Prefill：MegaMoE，`max-prefill-tokens=8192`、`chunked-prefill-size=8192`。
- Decode：`--speculative-algorithm DSPARK`、`--deepep-mode low_latency`、
  `moe-runner-backend=auto`、`enable-two-batch-overlap=false`。
- `你是谁`：`HTTP=200`，`WHOAMI_VALID=True`。
- 1024/1024、并发 1：10/10 请求成功，retokenized output tokens=10236，Out tok/s=295.63，
  Total tok/s=591.27，Mean TTFT=195.03 ms，Mean TPOT=3.19 ms。

日志与结果：

- [Decode 启动日志](/data/ssd2/gongoubo/single_node/repro_pr32281_20260823/step24_prefill_megamoe_decode_dspark/logs/services/decode_pure_dspark_fg/decode_20260823_150630_pid3456055.log)
- [Prefill 启动日志](/data/ssd2/gongoubo/single_node/repro_pr32281_20260823/step24_prefill_megamoe_decode_dspark/logs/services/prefill_tbo_fg/prefill_20260823_145913_pid3447860.log)
- [whoami 验证](/data/ssd2/gongoubo/single_node/repro_pr32281_20260823/step24_prefill_megamoe_decode_dspark/logs/validation/whoami_pure_dspark_best_20260823.log)
- [10 请求结果](/data/ssd2/gongoubo/single_node/repro_pr32281_20260823/step24_prefill_megamoe_decode_dspark/logs/results/dspark_only_best/isl1024_osl1024_c1_n10.jsonl)

TBO 已完成兼容性修复并通过 Graph、whoami 和 10/10 请求功能验收，但同 workload
Total tok/s=365.81，低于纯 DSpark 的 591.27；因此最终部署保持 TBO 关闭。TBO
修复源码和专项日志保存在同一实验目录及 `backups/prefill_megamoe_decode_dspark_20260823/source_patches/`。

## HiSparse 兼容性结论

已单独验证 HiSparse（关闭 DSpark）可以在当前 PD 配置下启动并完成 CUDA Graph、
host C4 pool、whoami 和 10/10 请求；1024/1024/C1 结果为 Out=72.31、Total=144.63
tok/s、TTFT=201.97 ms、TPOT=13.64 ms。相比纯 DSpark 的 Out=295.63、Total=591.27
tok/s、TPOT=3.19 ms，HiSparse 当前没有性能收益。

HiSparse 与 speculative decoding/DSpark 不再允许静默运行：历史组合虽然能启动，
但只得到 8131/10240 retokenized tokens，存在输出损坏风险。当前源码已在启动时
fail-fast 拒绝该组合；最终部署继续使用纯 DSpark，不启用 HiSparse。HiSparse 独立
实验及日志见 `repro_pr32281_20260823/step26_hisparse_no_spec/`。

## 最终恢复复核（2026-08-23）

最终推荐配置已重新启动并完成运行态检查。Prefill 和 Decode 均 ready；Decode 日志
确认使用 `DeepseekV4ForCausalLMDSpark`，target/draft CUDA Graph capture 完成，
Router `whoami` 返回 `HTTP=200`、`WHOAMI_VALID=True`。8192/1024 的 C1/C16/C256/C512
四组均达到目标请求数，结果与上方有效性能表一致；因此早期 49152 分块的 OOM 不能
再被视为当前部署失败。

本次复核日志：

```text
repro_pr32281_20260823/step24_prefill_megamoe_decode_dspark/logs/services/prefill_final_restore2/
repro_pr32281_20260823/step24_prefill_megamoe_decode_dspark/logs/services/decode_final_restore2/
repro_pr32281_20260823/step24_prefill_megamoe_decode_dspark/logs/validation/whoami_final_restore2_20260823.log
```

最终启动脚本、HiSparse fail-fast 修复和 TBO 兼容性修复的源码副本已保存在：
`backups/prefill_megamoe_decode_dspark_20260823/source_patches/`。

## 2026-08-23：面向 workload 的 Prefill 最优 profile 复验

保持 Decode pure DSpark 不变，新增对 `flashinfer_mxfp4` Prefill 的 A/B。高并发
8192 输入的最佳已验证组合为：`flashinfer_mxfp4 + chunk49152 + Decode
DeepEP low_latency + DSpark + Decode CUDA Graph`。8192/C512 达到
Out=8221.08、Total=73989.76 tok/s、TTFT=56203.96 ms、TPOT=4.32 ms，
5120/5120 请求成功，retokenized=5241532。

不过全矩阵最优是 workload-dependent：1024/C1、C16、C256 仍由
MegaMoE/chunk8192 领先，1024/C512 由 flashinfer_mxfp4/chunk16384 领先；
8192/C1 由 MegaMoE/chunk8192 领先，C16/C256 使用 flashinfer_mxfp4/chunk32768，
C512 使用 chunk49152。部署时应按业务输入长度和并发选择 profile，而不是把
chunk49152 固定用于所有请求。

以历史原始 baseline（见 `readmes/LPLB_PD_SGLang_v0516_baseline_20260821.md`）为
唯一对照，选择各 workload 的最佳 profile 后，8/8 组均超过 baseline：

| ISL | C | 最佳 Prefill profile | Total tok/s | baseline Total tok/s | Total 变化 | TTFT ms | TPOT ms | 请求完整性 |
|---:|---:|---|---:|---:|---:|---:|---:|---|
| 1024 | 1 | MegaMoE/chunk8192 | 598.82 | 215.09 | +178.40% | 185.71 | 3.16 | 10/10 |
| 1024 | 16 | MegaMoE/chunk8192 | 6891.22 | 2697.90 | +155.43% | 353.38 | 4.12 | 160/160 |
| 1024 | 256 | MegaMoE/chunk8192 | 63685.94 | 26213.34 | +142.95% | 714.44 | 6.95 | 2560/2560 |
| 1024 | 512 | flashinfer_mxfp4/chunk16384 | 84766.97 | 26487.25 | +220.03% | 1498.50 | 9.82 | 5120/5120 |
| 8192 | 1 | MegaMoE/chunk8192 | 2719.31 | 958.65 | +183.66% | 205.19 | 3.11 | 10/10 |
| 8192 | 16 | flashinfer_mxfp4/chunk32768 | 30895.17 | 11956.04 | +158.41% | 570.50 | 3.94 | 160/160 |
| 8192 | 256 | flashinfer_mxfp4/chunk32768 | 72049.21 | 61751.53 | +16.68% | 26918.51 | 4.17 | 2560/2560 |
| 8192 | 512 | flashinfer_mxfp4/chunk49152 | 73989.76 | 63995.17 | +15.62% | 56203.96 | 4.32 | 5120/5120 |

这张表是“按 workload 选择 profile”的最佳方案，不表示同一个 Prefill 进程同时
动态切换全部 chunk；切换输入长度/并发档位时需要按对应脚本重启 Prefill，Decode
可以保持不变。

新增实验目录：

```text
repro_pr32281_20260823/step29_prefill_flashinfer_mxfp4_chunk32768/
repro_pr32281_20260823/step30_prefill_flashinfer_mxfp4_short/
repro_pr32281_20260823/step31_prefill_flashinfer_mxfp4_chunk32768_c1/
repro_pr32281_20260823/step32_prefill_flashinfer_mxfp4_chunk49152/
```
