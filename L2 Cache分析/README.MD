# 固定 Decode Batch 的 NSYS/NCU 采集流程

本文整理一套可复用的单机固定 decode batch profiling 流程。后续切换模型时，
只需要替换模型路径、served name、tokenizer、parser 和模型专属启动参数。

## 1. 实验目标

实验需要保持以下配置不变：

```text
单机、8 张 GPU、非 PD 部署
TP=8，DP=1，EP=8
decode_batch_size=BATCH_SIZE
max_running_requests=BATCH_SIZE
```

当前 B200 实验使用 `BATCH_SIZE=256`。客户端也必须同时提交相同数量的请求，
否则即使设置了 `--decode-batch-size`，服务实际运行时仍可能使用更小的 decode
batch。

典型启动参数如下：

```bash
--tp-size 8 --dp-size 1 --ep-size 8 \
--decode-batch-size "$BATCH_SIZE" \
--max-running-requests "$BATCH_SIZE" \
--disable-custom-all-reduce \
--skip-server-warmup
```

模型相关配置集中放在变量中：

```bash
MODEL_PATH=/path/to/model
SERVED_MODEL_NAME=your-model-name
TOKENIZER_PATH=/path/to/model
TOOL_CALL_PARSER=your_parser       # 模型不需要时删除
INPUT_LEN=4096                     # 或 16384
BATCH_SIZE=256
MEASURED_OUTPUT_LEN=8              # 较短输出可让 profiling 更快结束
DECODE_PROFILE_STEPS=1             # 需要多个 decode step 时增大
```

## 2. 为什么必须 warmup

不要直接采集第一批请求。服务启动后应先完成一批完整 warmup，再开始测量。
warmup 用于初始化 CUDA graph、Triton/FlashInfer autotune、显存池、通信 buffer
和运行时状态。未 warmup 就采集容易出现启动 kernel、metric 缺失、时间不稳定或
decode batch 不正确。

推荐顺序：

```text
关闭服务自带 warmup 后启动服务
等待 Application startup complete
提交 warmup：num-prompts=BATCH_SIZE，output-len=8
确认日志中 completed=BATCH_SIZE
调用 native decode-only profiler 接口
提交 measured batch
再次确认 completed=BATCH_SIZE
```

warmup 和 measured request 必须使用相同模型、输入长度、batch size 和 tokenizer。
只有 measured profiling 的 output length 可以缩短。

## 3. 只采集 decode 的边界

warmup 完成后再 arm profiler，并且请求中必须明确指定 decode stage。推荐使用
SGLang `/start_profile` 接口，payload 如下：

```json
{
  "activities": ["CUDA_PROFILER"],
  "num_steps": 1,
  "profile_by_stage": true,
  "profile_stages": ["decode"],
  "with_stack": false,
  "record_shapes": false
}
```

measured log 应包含类似证据：

```text
native /start_profile armed for decode
completed=256 input_len=4096 output_len=8
```

如果使用 trigger-file 版本，则在启动服务前设置：

```bash
SGLANG_NCU_DECODE_ONLY=1
SGLANG_NCU_PROFILE_TRIGGER=/tmp/sglang_decode_nsys_start
SGLANG_NCU_DECODE_STEPS=1
```

启动前删除 trigger，先完成 warmup，warmup 结束后再创建 trigger。不能仅凭“服务
正在运行”推断采集的是 decode；必须检查 profile API 返回内容和日志证据。

## 3.1 必须修改的 SGLang 源码

仅修改启动参数不足以保证 decode batch 固定，也不足以保证采集范围只包含 decode。
实际使用的 SGLang checkout 需要两个 opt-in 能力。不同版本的文件名可能不同，可以
先定位 scheduler 和 profiler：

```bash
rg -n "decode_batch_size|start_profile|profile_by_stage|CUDA_PROFILER" \
  /path/to/sglang/python/sglang
```

### A. 在 scheduler 中增加 decode batch gate

增加或启用类似 `decode_batch_size` 的 scheduler 配置。在 decode-forward 边界统计
下一次 decode forward 的 request 数量，等待 ready/admitted 数量达到目标后再执行
decode，并输出可审计日志：

```text
DECODE_BATCH_GATE released ready=256 admitted=256 target=256
DECODE_BATCH size=256
```

gate 只能作用于 decode，必须是 opt-in，不能让普通服务死锁。CLI 参数和 scheduler
状态必须使用同一个目标值；流程要支持其他模型或 batch 时不要把 `256` 直接写死。
如果允许不足 batch，必须明确记录，而不能静默当成满 batch。

### B. 增加 decode-stage profiler hook

扩展 `/start_profile`，使其支持 `profile_by_stage=true`、
`profile_stages=["decode"]` 和 `num_steps=N`。只有进入指定 decode stage 后才 arm
CUDA profiling，统计 decode forward，并在完成 `N` 个 step 后停止。prefill 和 warmup
decode 不能调用 profiler-start hook。使用 trigger file 的版本则在 decode 路径检查
trigger，并用 `SGLANG_NCU_DECODE_ONLY=1` 之类的 opt-in 变量保护。

同时记录 arm 和 stop 日志；没有 `profile_by_stage` 时保留旧行为，保证 API 向后兼容。
这些源码级日志是证明 NSYS/NCU 没有采集 prefill 的依据。

### C. 修改源码后的验证

切换到其他 SGLang 版本后，验证普通非 profiling 服务仍可用、未显式请求时 gate 关闭、
warmup 不产生 profiling range、`/start_profile` 接受 decode stage、日志包含 batch gate
和 decode-only 证据，并且停止 profiling 不会重启服务。每次 profiling 都记录 SGLang
版本或 commit ID。

## 4. NSYS：先找重要 kernel

NSYS 是发现阶段，用于找出 kernel，不负责一次性采集所有 NCU metric。采集时长要
足够覆盖稳定 decode activity，同时避免无目的地扩大范围。

示例启动方式：

```bash
NSYS_OUTPUT=logs/${MODEL_TAG}_${INPUT_LEN}_decode_nsys \
DECODE_LOG=logs/${MODEL_TAG}_${INPUT_LEN}_nsys.log \
NSYS_DELAY=180 NSYS_DURATION=90 \
MODEL_PATH="$MODEL_PATH" \
  bash experimental/dsv4_ncu_batch256/profile_nsys_pro.sh
```

该 wrapper 的关键要求：

- report 和日志统一放到 `logs/`，不要写在仓库当前目录；
- 不要加入自动重启循环；
- 使用固定 batch 配置启动服务；
- 只 arm decode profiling；
- 排查单卡问题时可以只观察指定 device。

从 NSYS report 导出 CUDA kernel summary。不同 Nsight Systems 版本的 report 名称
可能不同，必要时先执行 `nsys stats --help` 查看可用 report。导出的 CSV 至少要
包含 kernel name、duration；多卡 trace 还应包含 device。

kernel 选择应按设备累计耗时超过 90%，不要直接按任意 top-N：

```bash
python3 experimental/dsv4_ncu_batch256/select_nsys_operators.py \
  logs/${MODEL_TAG}_${INPUT_LEN}_cuda_gpu_kern_sum.csv \
  --device 0 \
  --threshold 0.90 \
  --json logs/${MODEL_TAG}_${INPUT_LEN}_selected_operators.json \
  --tsv logs/${MODEL_TAG}_${INPUT_LEN}_selected_operators.tsv
```

JSON 会保存总耗时、每个 kernel 占比和累计占比。之后 NCU 必须以该 JSON 作为
kernel 列表来源，不要手工凭记忆重写 kernel 名称。

## 5. NCU：只采集 NSYS 选出的 kernel

NCU 使用 selected JSON，并使用精确的 demangled kernel name：

```bash
SELECTED_JSON=logs/${MODEL_TAG}_${INPUT_LEN}_selected_operators.json \
NCU_DIR=logs/${MODEL_TAG}_${INPUT_LEN}_ncu \
NCU_LOG=logs/${MODEL_TAG}_${INPUT_LEN}_ncu.log \
INPUT_LEN="$INPUT_LEN" \
NCU_LAUNCH_COUNT=3 \
DECODE_PROFILE_STEPS="$DECODE_PROFILE_STEPS" \
MEASURED_OUTPUT_LEN="$MEASURED_OUTPUT_LEN" \
NCU_EXACT_DEMANGLED=1 \
NCU_METRIC_GROUPS="$METRIC_GROUP" \
  bash experimental/dsv4_ncu_batch256/profile_selected_operators_pro_orchestrated.sh
```

需要 3 个样本时设置 `--launch-count 3`。最终报告用 `/` 分隔三个值，不要静默
求平均。

普通计算 kernel 通常可以一次采集一个 metric group。NCCL 或互联敏感 kernel 建议
先完成非通信 kernel，再按 metric 或 metric group 单独采集。NCCL 若需要，使用已经
验证过的 communicator 配置：

```text
--communicator shmem
--communicator-shmem-num-peers 8
--lockstep-kernel-launch
```

kernel filter 必须区分模板实例。类似
`sm100_fp8_fp4_gemm_1d1d_impl` 的短函数名不足以唯一标识 kernel；多个模板实例
同时存在时，必须用完整 demangled signature（包括模板参数）验证 report。

## 6. metric 与单位

始终从 NCU raw CSV 的 units row 读取单位，不要给所有 kernel 的同一个 metric 指定
统一单位。

- `sector` 和 `request` 是计数，不转换为字节；
- `lts__t_request_hit_rate.ratio=0.96` 表示 `96%`，不是 `0.96%`；
- `lts__t_sector_hit_rate.ratio` 同样需要从 ratio 转换为百分比；
- `gpu__dram_throughput...pct...` 和 `sm__throughput...pct...` 本身就是百分比；
- 字节 metric 可能由不同 kernel 以 `MB`、`KB` 或 `B` 输出，表格中保留各自单位；
- 整数不显示小数，带小数的值保留两位。

## 7. 结果验收清单

```text
[ ] 服务启动完成，且使用了正确的 warmup 配置
[ ] warmup 恰好完成 BATCH_SIZE 个请求
[ ] profiler 明确 arm 在 decode，而不是 prefill
[ ] measured request 恰好完成 BATCH_SIZE 个请求
[ ] NSYS kernel 列表已导出，并按累计设备耗时 >=90% 选择
[ ] NCU report 中是完整 demangled 目标 kernel
[ ] 没有把同名不同模板实例混到同一个结果中
[ ] launch 数量和 report 行数符合实验设计
[ ] 单位来自每个 report 的 NCU units row
[ ] 所有日志和 report 都在 logs/<run-id>/ 下
```

常用检查：

```bash
python3 experimental/dsv4_ncu_batch256/verify_decode_batch.py \
  logs/${MODEL_TAG}_${INPUT_LEN}_measured.log \
  --expected "$BATCH_SIZE" --min-count 1

ncu --import path/to/report.ncu-rep --csv --page raw > logs/report_raw.csv
```

检查 raw CSV 中的 `Kernel Name`、`Block Size`、`Grid Size` 和完整 metric 值。
短函数名相同不代表 kernel 相同。

## 8. 失败处理

单次服务 wrapper 不应包含自动重启循环。如果任务被外部 kill，不要自动重启；保留
日志并记录失败的 operator/metric。

NCU 失败时按以下顺序处理：

1. 先检查是否已经生成 `.ncu-rep`。如果 `status=143` 但
   `report_present=1`，仍可在完整性验证通过后使用；
2. 检查服务日志和 kernel filter 是否匹配过宽；
3. 只重试缺失的 operator/metric，不要重复已完成任务；
4. 精确 kernel 采集优先使用 `--replay-mode kernel`；
5. 如果出现 device memory backup warning，再考虑
   `--replay-mode application`，但要明确它可能扩大采集范围；
6. 出现 interconnect/Xid149 时，隔离 metric 或 metric group，并使用验证过的
   communicator 配置；不要合并不同模板实例的行。

如果某个 metric 在不同 launch 间差异很大，先比较完整 kernel name、grid/block、
duration 和相关计数器。缓存 hit rate 会受 cache 状态和数据相关稀疏计算影响，差异
不一定意味着错误；通过完整 kernel 校验后应保留样本并在报告中说明原因。

## 9. 切换到其他模型

复制一份新的 run 目录，只替换：

```text
MODEL_PATH
SERVED_MODEL_NAME
TOKENIZER_PATH
TOOL_CALL_PARSER 和模型专属启动参数
INPUT_LEN
BATCH_SIZE
DECODE_PROFILE_STEPS
```

仍然按照以下顺序执行：

```text
检查 metric 是否存在
启动一个固定 batch 的服务
warmup
arm decode-only profiling
运行 NSYS
按累计设备耗时 >=90% 选择 kernel
对精确 kernel 运行 NCU
校验 report
生成 metric × kernel 表格
```

每次实验使用独立目录，例如：

```text
logs/<model-tag>_<input-len>_nsys/
logs/<model-tag>_<input-len>_ncu/
logs/<model-tag>_<input-len>_reports/
```

## 10. 仓库中的实际代码

下面代码块来自当前仓库实际使用的脚本，方便后续切换模型时直接修改和复用，避免
依赖旧终端里的手工命令。如果代码片段与新版本 SGLang 行为不一致，以仓库脚本和
实际安装的 SGLang 版本为准。

### 10.1 NSYS wrapper：`profile_nsys_pro.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_PATH:=/data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Pro}"
: "${SERVER_HOST:=0.0.0.0}"
: "${SERVER_PORT:=30001}"
: "${NSYS_OUTPUT:=logs/dsv4_pro_decode_nsys}"
: "${DECODE_LOG:=logs/dsv4_pro_nsys_decode.log}"
: "${PROFILE_TRIGGER:=/tmp/sglang_decode_nsys_start}"
: "${NSYS_DELAY:=180}"
: "${NSYS_DURATION:=90}"

rm -f "$PROFILE_TRIGGER"
exec env SGLANG_NCU_DECODE_ONLY=1 \
  SGLANG_NCU_PROFILE_TRIGGER="$PROFILE_TRIGGER" \
  SGLANG_NCU_DECODE_STEPS=128 SGLANG_DISABLE_TRITON_SYMM_MEM_AG=1 \
  nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --cuda-memory-usage=false --delay="$NSYS_DELAY" \
  --duration="$NSYS_DURATION" --stop-on-exit=false \
  --force-overwrite=true -o "$NSYS_OUTPUT" \
  python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --served-model-name deepseek-ai/DeepSeek-V4-Pro \
    --trust-remote-code --tool-call-parser deepseekv4 \
    --decode-batch-size 256 --max-running-requests 256 \
    --tp-size 8 --dp-size 1 --ep-size 8 \
    --moe-runner-backend deep_gemm --moe-a2a-backend deepep \
    --deepep-mode normal --disable-flashinfer-autotune \
    --disable-custom-all-reduce --skip-server-warmup \
    --cuda-graph-config '{"decode":{"backend":"full","max_bs":256,"bs":[256]}}' \
    --mem-fraction-static 0.9 --swa-full-tokens-ratio 0.1 \
    --host "$SERVER_HOST" --port "$SERVER_PORT" 2>&1 | tee "$DECODE_LOG"
```

切换模型时替换模型路径、served name、parser 和模型专属参数，固定 batch 和
decode-only 控制不要删除。

### 10.2 warmup 与 decode profile：`run_nsys_once_pro.sh`

实际 wrapper 的关键执行顺序如下：

```bash
python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --served-model-name deepseek-ai/DeepSeek-V4-Pro \
  --trust-remote-code --tool-call-parser deepseekv4 \
  --decode-batch-size 256 --max-running-requests 256 \
  --tp-size 8 --dp-size 1 --ep-size 8 \
  --moe-runner-backend deep_gemm --moe-a2a-backend deepep \
  --deepep-mode normal --disable-flashinfer-autotune \
  --disable-custom-all-reduce --skip-server-warmup \
  --cuda-graph-config '{"decode":{"backend":"full","max_bs":256,"bs":[256]}}' \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --swa-full-tokens-ratio 0.1 --watchdog-timeout 7200 \
  --host 0.0.0.0 --port "$PORT" >"$SERVER_LOG" 2>&1 &

python3 experimental/dsv4_ncu_batch256/run_batch.py \
  --url "http://127.0.0.1:$PORT" \
  --model deepseek-ai/DeepSeek-V4-Pro --tokenizer "$MODEL_PATH" \
  --num-prompts 256 --input-len "$INPUT_LEN" --output-len 8 \
  --timeout "$REQUEST_TIMEOUT" 2>&1 | tee "$WARMUP_LOG"

python3 - "$PORT" <<'PY'
import json, os, sys, urllib.request
port = int(sys.argv[1])
payload = json.dumps({
    "activities": ["CUDA_PROFILER"],
    "num_steps": int(os.environ.get("DECODE_PROFILE_STEPS", "4")),
    "profile_by_stage": True,
    "profile_stages": ["decode"],
    "with_stack": False,
    "record_shapes": False,
}).encode()
request = urllib.request.Request(
    f"http://127.0.0.1:{port}/start_profile", data=payload,
    headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(request, timeout=30) as response:
    if response.status != 200:
        raise RuntimeError(response.read().decode())
print("native /start_profile armed for decode")
PY

python3 experimental/dsv4_ncu_batch256/run_batch.py \
  --url "http://127.0.0.1:$PORT" \
  --model deepseek-ai/DeepSeek-V4-Pro --tokenizer "$MODEL_PATH" \
  --num-prompts 256 --input-len "$INPUT_LEN" \
  --output-len "$MEASURED_OUTPUT_LEN" --timeout "$REQUEST_TIMEOUT" \
  2>&1 | tee "$MEASURED_LOG"
test "$(rg -c '^completed=256 ' "$MEASURED_LOG")" -eq 1
```

### 10.3 NSYS 累计耗时选择：`select_nsys_operators.py`

该脚本按耗时降序排列 kernel，在累计占比达到目标后停止：

```python
kernels.sort(key=lambda item: item["total_time_ns"], reverse=True)
total_ns = sum(item["total_time_ns"] for item in kernels)
selected = []
cumulative_ns = 0.0
for item in kernels:
    cumulative_ns += item["total_time_ns"]
    selected.append({
        **item,
        "fraction": item["total_time_ns"] / total_ns,
        "cumulative_fraction": cumulative_ns / total_ns,
    })
    if cumulative_ns / total_ns >= args.threshold:
        break
result = {
    "threshold": args.threshold,
    "summary_total_time_ns": total_ns,
    "all_kernel_count": len(kernels),
    "selected_kernel_count": len(selected),
    "selected": selected,
}
args.json.write_text(json.dumps(result, indent=2) + "\n")
```

使用 `--device` 时，实际脚本会忽略没有 `GrdX` 的行，因此 memcpy/memset 不会进入
kernel shortlist。

### 10.4 精确 kernel 的 NCU 命令构造

NCU orchestrator 从 selected JSON 读取完整名称。关键代码如下：

```bash
if [[ "$name" == ncclDevKernel_* ]]; then
  ncu_target_args=(--target-processes all --devices 0)
  ncu_kernel_args=(--kernel-name-base demangled --kernel-name "regex:^$name.*")
  ncu_filter_args=()
  if (( NCU_USE_COMMUNICATOR )); then
    ncu_communicator_args=(--communicator shmem \
      --communicator-shmem-num-peers 8 --lockstep-kernel-launch)
  fi
else
  ncu_kernel_args=(--kernel-name-base demangled --kernel-name "$full_name")
fi

env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  INPUT_LEN="$INPUT_LEN" DECODE_PROFILE_STEPS="$DECODE_PROFILE_STEPS" \
  "$NCU_BIN" --replay-mode "$metric_replay_mode" --kill "$NCU_KILL" \
  --launch-count "$metric_launch_count" "${ncu_kernel_args[@]}" \
  --metrics "$current_metrics" --force-overwrite -o "$output" \
  ./experimental/dsv4_ncu_batch256/run_nsys_once_pro.sh
```

完整实现位于 `profile_selected_operators_pro_orchestrated.sh`。通信 kernel 重试时
使用 `NCU_FORCE_REPLAY_MODE=kernel`、`NCU_KILL=0`，并只隔离缺失的 metric 或 metric
group。

### 10.5 decode batch 验证：`verify_decode_batch.py`

```python
text = args.log.read_text()
sizes = [int(value) for value in re.findall(r"DECODE_BATCH size=(\d+)", text)]
gates = re.findall(
    r"DECODE_BATCH_GATE released ready=(\d+)(?: admitted=(\d+))? target=(\d+)",
    text,
)
if len(sizes) < args.min_count:
    raise SystemExit(f"FAIL: found {len(sizes)} decode batches")
if any(size != args.expected for size in sizes):
    raise SystemExit(f"FAIL: observed decode batch sizes={sizes}")
if not any(
    int(ready) == args.expected
    and (admitted == "" or int(admitted) == args.expected)
    and int(target) == args.expected
    for ready, admitted, target in gates
):
    raise SystemExit(f"FAIL: no gate release for target={args.expected}")
print(f"PASS: {len(sizes)} decode batches, every batch_size={args.expected}")
```

报表脚本 `make_current_report.py` 和 `make_input4096_report.py` 会读取 raw NCU CSV，
保留每份 report 的单位，将 ratio 转为百分比，并按完整 kernel signature 过滤混入的
模板实例，再生成 metric × kernel 表格。
