# Runtime / NCCL API Profiling Workflow

本文档总结如何用 CUPTI 获取 Runtime / Driver / NCCL API，以及如何获取 Runtime API 参数。适合后续更换模型后快速复用。

## 目标输出

核心输出格式：

| API | Captured | Sample params |
|---|---:|---|
| `cudaMemcpyAsync` | 1127612 | `{"dst":"0x7c7373000000","src":"0x7ffd8cd72670","count":8,"kind":4,"kindName":"cudaMemcpyDefault","stream":"0x1fb9af00"}` |
| `cudaLaunchKernelExC` | 5621584 | `{"func":"0x7c73bb2608b0","config":"0x7ffd8cd73aa0","grid":[1,1,1],"block":[256,1,1],"args":"0x7ffd8cd738c0","dynamicSmemBytes":0,"stream":"0x26525550","attrs":"(nil)","numAttrs":0}` |

实际生成文件：

- `runtime_api_param_coverage.md`: API / Captured / Sample params / status。
- `runtime_api_param_coverage.csv`: 机器可读版 API 参数覆盖表。
- `runtime_api_to_kernel_signatures.md`: API 对应 kernel signature。
- `runtime_kernel_api_params_full.csv`: 按 `pid + correlation_id` join 后的全量明细。
- `runtime_launch_args_samples.md`: Runtime launch kernel 参数 raw dump 样例。

## 总体方法

使用两类 CUPTI 数据：

1. CUPTI Callback API

捕获 Runtime / Driver API 的调用名、`correlation_id`、线程、返回值和参数 JSON。

2. CUPTI Activity API

捕获 GPU kernel activity，包括 kernel name、device、stream、grid/block、shared memory、duration、`correlation_id`。

二者通过：

```text
pid + correlation_id
```

关联，得到：

```text
Runtime/Driver API 参数 -> 实际 GPU kernel name/signature
```

NCCL kernel 通常不是通过 `nccl*` host API 名直接落到 kernel activity，而是表现为 Driver launch：

```text
cuLaunchKernel
cuLaunchKernelEx
```

再通过 CUPTI Activity 得到实际 NCCL kernel，例如：

```text
ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>)
```

## 快速执行流程

### 1. 编译 CUPTI profiler

```bash
g++ -std=c++17 -fPIC -shared tools/cupti_api_param_profiler.cpp \
  -o tools/libcupti_api_param_profiler.so \
  -I/usr/local/cuda-13.0/targets/x86_64-linux/include \
  -L/usr/local/cuda-13.0/targets/x86_64-linux/lib \
  -lcupti -ldl -pthread
```

### 2. 启动采集

常规参数采集：

```bash
RUN_ID=flash_pd_ep_decode_cupti_api_kernel_params_$(date +%Y%m%d_%H%M%S) \
MODEL_PATH=/path/to/model \
TOKENIZER=/path/to/model \
INPUT_LEN=4096 \
OUTPUT_LEN=1024 \
BATCH_SIZE=256 \
NUM_PROMPTS=256 \
MAX_CONCURRENCY=256 \
./profile_flash_pd_cupti_api_kernel_params.sh
```

如果要额外 dump Runtime launch kernel 的 raw argument slot：

```bash
RUN_ID=flash_pd_ep_decode_cupti_runtime_args_$(date +%Y%m%d_%H%M%S) \
MODEL_PATH=/path/to/model \
TOKENIZER=/path/to/model \
INPUT_LEN=4096 \
OUTPUT_LEN=1024 \
BATCH_SIZE=256 \
NUM_PROMPTS=256 \
MAX_CONCURRENCY=256 \
CUPTI_RUNTIME_ARG_SLOTS=8 \
CUPTI_RUNTIME_ARG_BYTES=16 \
CUPTI_EXTRA_ARG_DUMP_BYTES=0 \
./profile_flash_pd_cupti_api_kernel_params.sh
```

说明：

- `CUPTI_RUNTIME_ARG_SLOTS=8`: 每次 Runtime launch 读取前 8 个 `args` slot。
- `CUPTI_RUNTIME_ARG_BYTES=16`: 每个 slot 读取 16 bytes。
- `CUPTI_EXTRA_ARG_DUMP_BYTES=0`: 关闭 Driver `extra` buffer dump，避免 NCCL/Driver extra 数据放大 CSV。

### 3. 后处理

```bash
python3 tools/join_cupti_api_kernel_params.py profiles/<RUN_ID>
python3 tools/make_runtime_api_kernel_signature_report.py profiles/<RUN_ID>
python3 tools/make_runtime_api_param_coverage_report.py profiles/<RUN_ID>
python3 tools/make_runtime_launch_args_sample_report.py profiles/<RUN_ID> 80
```

## 实际采集脚本代码

文件：[profile_flash_pd_cupti_api_kernel_params.sh](/data/ssd2/gongoubo/pd/profile_flash_pd_cupti_api_kernel_params.sh)

关键逻辑是只给 decode 注入 `LD_PRELOAD`，prefill/router 正常启动：

```bash
CUPTI_API_PARAM_PROFILER_LIB="${CUPTI_API_PARAM_PROFILER_LIB:-$ROOT_DIR/tools/libcupti_api_param_profiler.so}"
CUPTI_API_PARAM_LOG_PATTERN="${CUPTI_API_PARAM_LOG_PATTERN:-$OUT_DIR/runtime_api_params_%p.csv}"
CUPTI_KERNEL_ACTIVITY_LOG_PATTERN="${CUPTI_KERNEL_ACTIVITY_LOG_PATTERN:-$OUT_DIR/kernel_activity_%p.csv}"
CUPTI_API_PARAM_LD_LIBRARY_PATH="${CUPTI_API_PARAM_LD_LIBRARY_PATH:-/usr/local/cuda-13.0/targets/x86_64-linux/lib}"

log "Starting decode with CUPTI API/kernel profiler"
setsid env \
  CUPTI_API_PARAM_LOG="$CUPTI_API_PARAM_LOG_PATTERN" \
  CUPTI_KERNEL_ACTIVITY_LOG="$CUPTI_KERNEL_ACTIVITY_LOG_PATTERN" \
  LD_LIBRARY_PATH="$CUPTI_API_PARAM_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}" \
  LD_PRELOAD="$CUPTI_API_PARAM_PROFILER_LIB${LD_PRELOAD:+:$LD_PRELOAD}" \
  bash "$ROOT_DIR/flash_d.sh" >"$OUT_DIR/decode_cupti.log" 2>&1 &
wait_health decode "$DECODE_PORT"
```

benchmark 部分：

```bash
python3 "$ROOT_DIR/InferenceX/utils/bench_serving/benchmark_serving.py" \
  --backend vllm \
  --base-url "http://$HOST:$ROUTER_PORT" \
  --endpoint /v1/completions \
  --model "$MODEL_NAME" \
  --tokenizer "$TOKENIZER" \
  --dataset-name random \
  --random-input-len "$INPUT_LEN" \
  --random-output-len "$OUTPUT_LEN" \
  --random-range-ratio 1.0 \
  --num-prompts "$NUM_PROMPTS" \
  --request-rate "$REQUEST_RATE" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --ignore-eos \
  --trust-remote-code \
  --disable-tqdm \
  ${BENCHMARK_EXTRA_ARGS:-} 2>&1 | tee "$OUT_DIR/benchmark.log"
```

## CUPTI Profiler 关键代码

文件：[tools/cupti_api_param_profiler.cpp](/data/ssd2/gongoubo/pd/tools/cupti_api_param_profiler.cpp)

### 1. 目标 API 列表

```cpp
static bool is_requested_api(const char *name) {
  if (!name) return false;
  static const char *names[] = {
      "cudaMemcpyAsync", "cudaLaunchKernelExC", "cuLaunchKernelEx",
      "cudaLaunchKernel", "cudaEventQuery", "cuMemSetAccess",
      "cudaStreamSynchronize", "cuMemCreate", "cuLibraryLoadData",
      "cudaEventRecordWithFlags", "cuModuleLoadData", "cudaStreamWaitEvent",
      "cudaEventCreateWithFlags", "cudaMalloc", "cudaEventRecord",
      "cuTensorMapEncodeTiled", "cudaStreamIsCapturing", "cudaEventDestroy",
      "cuMemMap", "cuMemImportFromShareableHandle", "cuKernelGetName",
      "cudaThreadExchangeStreamCaptureMode", "cudaGetFuncBySymbol",
      "cudaGetDriverEntryPointByVersion", "cuKernelGetAttribute",
      "cudaStreamGetCaptureInfo", "cudaStreamCreateWithPriority",
      "cudaIpcOpenMemHandle", "cudaFree", "cuMemAddressReserve",
      "cuKernelGetFunction", "cuLibraryLoadFromFile", "cuKernelSetAttribute",
      "cudaHostAlloc", "cudaMemsetAsync", "cuLaunchKernel",
      "cudaEventSynchronize", "cudaMemPoolCreate", "cudaMemGetInfo",
      "cuGetProcAddress_v2", "cudaStreamCreateWithFlags", "cuMemRelease",
      "cuLibraryGetKernel", "cudaKernelSetAttributeForDevice",
      "cuMemGetAllocationGranularity", "cudaLibraryLoadData", "cuModuleLoad",
      "cudaMemcpy", "cudaMemset", "cudaDeviceSynchronize",
      "cudaOccupancyMaxActiveClusters", "cuLibraryGetKernelCount", "cuInit",
      "cudaGetDeviceProperties", "cudaOccupancyAvailableDynamicSMemPerBlock",
      "cuLibraryEnumerateKernels", "cudaMemPoolSetAttribute",
      "cuModuleGetLoadingMode", "cudaLibraryGetKernel"};
  for (const char *x : names) {
    if (std::strcmp(name, x) == 0) return true;
  }
  return false;
}
```

### 2. Callback API 行格式

```cpp
static void log_row(const CUpti_CallbackData *cb, const char *domain,
                    CUpti_CallbackId cbid, const char *params_json) {
  if (!g_file || !cb || cb->callbackSite != CUPTI_API_ENTER) return;
  std::lock_guard<std::mutex> lock(g_mu);
  fprintf(g_file,
          "%llu,%d,%s,%u,%s,%s,%u,%u,%llu,%llu,",
          g_seq.fetch_add(1), getpid(), domain, cbid, phase_name(cb->callbackSite),
          cb->functionName ? cb->functionName : "", cb->correlationId,
          cb->contextUid, (unsigned long long)syscall(SYS_gettid),
          (unsigned long long)cb->functionReturnValue);
  write_csv_escaped(params_json ? params_json : "{}");
  fputc('\n', g_file);
}
```

生成 CSV：

```text
runtime_api_params_<pid>.csv
```

字段包括：

```text
seq,pid,domain,cbid,phase,function_name,correlation_id,context_uid,thread_id,function_return_value,params_json
```

### 3. Runtime launch 参数捕获

`cudaLaunchKernel`：

```cpp
case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000: {
  const auto *x = static_cast<const cudaLaunchKernel_v7000_params *>(cb->functionParams);
  snprintf(p, sizeof(p),
           "{\"func\":\"%p\",\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],"
           "\"args\":\"%p\",\"sharedMem\":%zu,\"stream\":\"%p\"",
           x->func, x->gridDim.x, x->gridDim.y, x->gridDim.z,
           x->blockDim.x, x->blockDim.y, x->blockDim.z, x->args,
           x->sharedMem, x->stream);
  std::string params = std::string(p) + runtime_args_json_fields(x->args) + "}";
  log_row(cb, "runtime", cbid, params.c_str());
  break;
}
```

`cudaLaunchKernelExC`：

```cpp
case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060: {
  const auto *x = static_cast<const cudaLaunchKernelExC_v11060_params *>(cb->functionParams);
  if (x->config) {
    snprintf(p, sizeof(p),
             "{\"func\":\"%p\",\"config\":\"%p\",\"grid\":[%u,%u,%u],"
             "\"block\":[%u,%u,%u],\"args\":\"%p\",\"dynamicSmemBytes\":%zu,"
             "\"stream\":\"%p\",\"attrs\":\"%p\",\"numAttrs\":%u",
             x->func, x->config, x->config->gridDim.x, x->config->gridDim.y,
             x->config->gridDim.z, x->config->blockDim.x, x->config->blockDim.y,
             x->config->blockDim.z, x->args, x->config->dynamicSmemBytes,
             x->config->stream, x->config->attrs, x->config->numAttrs);
  } else {
    snprintf(p, sizeof(p),
             "{\"func\":\"%p\",\"config\":\"%p\",\"args\":\"%p\"",
             x->func, x->config, x->args);
  }
  std::string params = std::string(p) + runtime_args_json_fields(x->args) + "}";
  log_row(cb, "runtime", cbid, params.c_str());
  break;
}
```

### 4. Runtime kernel raw args dump

Runtime callback 只能给 `void **args`，不提供参数个数和每个参数类型。当前实现用 `process_vm_readv` 安全读取本进程内存，避免无效指针导致进程崩溃。

```cpp
static ssize_t safe_read_process_memory(const void *src, void *dst, size_t size) {
  if (!src || !dst || size == 0) return 0;
  struct iovec local_iov;
  struct iovec remote_iov;
  local_iov.iov_base = dst;
  local_iov.iov_len = size;
  remote_iov.iov_base = const_cast<void *>(src);
  remote_iov.iov_len = size;
  return process_vm_readv(getpid(), &local_iov, 1, &remote_iov, 1, 0);
}
```

raw args 输出：

```cpp
static std::string runtime_args_json_fields(void **args) {
  std::string out = ",\"runtimeArgsParseStatus\":\"";
  if (!args) {
    out += "no_args\",\"runtimeArgSlotsDumped\":0,\"runtimeArgs\":[]";
    return out;
  }
  out += "raw_runtime_args_dump_no_abi\","
         "\"runtimeArgDumpCaveat\":\"cuda runtime callback does not expose argument count or sizes; slots beyond the real kernel ABI are best-effort memory reads and must be ignored during ABI-aware decoding\","
         "\"runtimeArgSlotLimit\":";
  char buf[256];
  std::snprintf(buf, sizeof(buf), "%zu,\"runtimeArgBytesPerSlot\":%zu,\"runtimeArgs\":[",
                max_runtime_arg_slots(), max_runtime_arg_bytes());
  out += buf;
  for (size_t i = 0; i < max_runtime_arg_slots(); ++i) {
    void *arg_ptr = nullptr;
    errno = 0;
    ssize_t slot_read = safe_read_process_memory(args + i, &arg_ptr, sizeof(arg_ptr));
    if (i) out += ",";
    if (slot_read != static_cast<ssize_t>(sizeof(arg_ptr))) {
      std::snprintf(buf, sizeof(buf),
                    "{\"slot\":%zu,\"argPtrReadStatus\":\"%s\",\"argPtr\":\"(unreadable)\","
                    "\"dumpBytes\":0,\"hex\":\"\"}",
                    i, read_status(slot_read, sizeof(arg_ptr)));
      out += buf;
      break;
    }
    size_t bytes_read = 0;
    std::string arg_hex = safe_hex_dump(arg_ptr, max_runtime_arg_bytes(), &bytes_read);
    std::snprintf(buf, sizeof(buf),
                  "{\"slot\":%zu,\"argPtrReadStatus\":\"ok\",\"argPtr\":\"%p\","
                  "\"dumpBytes\":%zu,\"hex\":\"",
                  i, arg_ptr, bytes_read);
    out += buf;
    out += arg_hex;
    out += "\"}";
  }
  out += "]";
  return out;
}
```

输出示例：

```json
{
  "runtimeArgsParseStatus": "raw_runtime_args_dump_no_abi",
  "runtimeArgSlotLimit": 8,
  "runtimeArgBytesPerSlot": 16,
  "runtimeArgs": [
    {"slot":0,"argPtrReadStatus":"ok","argPtr":"0x7ffd...","dumpBytes":16,"hex":"..."}
  ]
}
```

注意：超过真实 ABI 参数数量的 slot 是 best-effort memory reads，不应当当作真实参数解释。

### 5. Runtime 普通 API 参数捕获

示例：`cudaMemcpyAsync`

```cpp
case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020: {
  const auto *x = static_cast<const cudaMemcpyAsync_v3020_params *>(cb->functionParams);
  snprintf(p, sizeof(p),
           "{\"dst\":\"%p\",\"src\":\"%p\",\"count\":%zu,\"kind\":%d,"
           "\"kindName\":\"%s\",\"stream\":\"%p\"}",
           x->dst, x->src, x->count, (int)x->kind,
           memcpy_kind_name(x->kind), x->stream);
  log_row(cb, "runtime", cbid, p);
  break;
}
```

示例输出：

```json
{"dst":"0x7c7373000000","src":"0x7ffd8cd72670","count":8,"kind":4,"kindName":"cudaMemcpyDefault","stream":"0x1fb9af00"}
```

### 6. Driver launch / NCCL kernel 捕获

NCCL kernel 通常从 Driver launch 进入：

```cpp
case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx:
case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz: {
  const auto *x = static_cast<const cuLaunchKernelEx_params *>(cb->functionParams);
  std::string extra_fields = launch_extra_json_fields(x->extra);
  if (x->config) {
    snprintf(p, sizeof(p),
             "{\"func\":\"%p\",\"config\":\"%p\",\"grid\":[%u,%u,%u],"
             "\"block\":[%u,%u,%u],\"sharedMemBytes\":%u,\"stream\":\"%p\","
             "\"attrs\":\"%p\",\"numAttrs\":%u,\"kernelParams\":\"%p\","
             "\"extra\":\"%p\"%s}",
             x->f, x->config, x->config->gridDimX, x->config->gridDimY,
             x->config->gridDimZ, x->config->blockDimX, x->config->blockDimY,
             x->config->blockDimZ, x->config->sharedMemBytes,
             x->config->hStream, x->config->attrs, x->config->numAttrs,
             x->kernelParams, x->extra, extra_fields.c_str());
  } else {
    snprintf(p, sizeof(p),
             "{\"func\":\"%p\",\"config\":\"%p\",\"kernelParams\":\"%p\","
             "\"extra\":\"%p\"%s}",
             x->f, x->config, x->kernelParams, x->extra, extra_fields.c_str());
  }
  log_row(cb, "driver", cbid, p);
  break;
}
```

Driver `extra` buffer 解析：

```cpp
static void parse_launch_extra(void **extra, const void **arg_buffer,
                               size_t *arg_buffer_size, bool *truncated,
                               const char **status) {
  *arg_buffer = nullptr;
  *arg_buffer_size = 0;
  *truncated = false;
  *status = extra ? "extra_present_no_arg_buffer" : "no_extra";
  if (!extra) return;

  for (int i = 0; i < 64; i += 2) {
    void *key = extra[i];
    if (key == nullptr || key == CU_LAUNCH_PARAM_END) {
      *status = *arg_buffer ? "decoded_extra_arg_buffer" : "extra_present_no_arg_buffer";
      return;
    }
    void *value = extra[i + 1];
    if (key == CU_LAUNCH_PARAM_BUFFER_POINTER) {
      *arg_buffer = value;
    } else if (key == CU_LAUNCH_PARAM_BUFFER_SIZE && value) {
      *arg_buffer_size = *static_cast<const size_t *>(value);
    }
  }
  *truncated = true;
  *status = *arg_buffer ? "decoded_extra_arg_buffer_extra_list_truncated"
                        : "extra_list_truncated_no_arg_buffer";
}
```

### 7. CUPTI Activity kernel 捕获

```cpp
static void log_kernel_record(const CUpti_ActivityKernel10 *k) {
  if (!g_kernel_file || !k) return;
  std::lock_guard<std::mutex> lock(g_kernel_mu);
  fprintf(g_kernel_file,
          "%llu,%d,%u,%llu,%llu,%llu,%u,%u,%u,%d,%d,%d,%d,%d,%d,%d,%d,%u,%u,%lld,%llu,%u,%u,%u,%u,%u,%u,%u,%llu,",
          g_kernel_seq.fetch_add(1), getpid(), (unsigned)k->kind,
          (unsigned long long)k->start, (unsigned long long)k->end,
          (unsigned long long)(k->end >= k->start ? k->end - k->start : 0),
          k->deviceId, k->contextId, k->streamId, k->gridX, k->gridY, k->gridZ,
          k->blockX, k->blockY, k->blockZ, k->staticSharedMemory,
          k->dynamicSharedMemory, k->registersPerThread, k->correlationId,
          (long long)k->gridId, (unsigned long long)k->graphNodeId, k->graphId,
          k->clusterX, k->clusterY, k->clusterZ, k->channelID,
          (unsigned)k->channelType, (unsigned)k->launchType,
          (unsigned long long)k->localMemoryTotal_v2);
  write_csv_escaped_file(g_kernel_file, k->name);
  fputc('\n', g_kernel_file);
}
```

Activity buffer callback：

```cpp
static void CUPTIAPI buffer_completed(CUcontext, uint32_t, uint8_t *buffer,
                                      size_t, size_t validSize) {
  if (!buffer) return;
  CUpti_Activity *record = nullptr;
  while (cuptiActivityGetNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS) {
    if (record->kind == CUPTI_ACTIVITY_KIND_KERNEL ||
        record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
      log_kernel_record(reinterpret_cast<const CUpti_ActivityKernel10 *>(record));
    }
  }
  free(buffer);
}
```

生成 CSV：

```text
kernel_activity_<pid>.csv
```

## Join 逻辑

文件：[tools/join_cupti_api_kernel_params.py](/data/ssd2/gongoubo/pd/tools/join_cupti_api_kernel_params.py)

只把 launch API 和 kernel activity join：

```python
LAUNCH_APIS = {
    "cudaLaunchKernel",
    "cudaLaunchKernelExC",
    "cuLaunchKernel",
    "cuLaunchKernel_ptsz",
    "cuLaunchKernelEx",
    "cuLaunchKernelEx_ptsz",
}
```

读取 API CSV，以 `correlation_id` 建索引：

```python
def load_api_map(path):
    by_corr = {}
    launch_rows = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row.get("function_name", "")
            if fn not in LAUNCH_APIS:
                continue
            corr = row.get("correlation_id", "")
            if not corr:
                continue
            entry = by_corr.setdefault(corr, {})
            if row.get("domain") == "runtime":
                entry["runtime_function_name"] = fn
                entry["runtime_params_json"] = row.get("params_json", "")
                entry["runtime_thread_id"] = row.get("thread_id", "")
            elif row.get("domain") == "driver":
                entry["driver_function_name"] = fn
                entry["driver_params_json"] = row.get("params_json", "")
                entry["driver_thread_id"] = row.get("thread_id", "")
            launch_rows += 1
    return by_corr, launch_rows
```

join kernel activity：

```python
api = api_map.get(corr, {})
runtime_fn = api.get("runtime_function_name", "")
driver_fn = api.get("driver_function_name", "")
api_key = runtime_fn or driver_fn or "UNMATCHED"
kernel_name = k.get("kernel_name", "")

writer.writerow({
    "pid": pid,
    "device_id": k.get("device_id", ""),
    "stream_id": k.get("stream_id", ""),
    "correlation_id": corr,
    "kernel_duration_ns": k.get("duration_ns", ""),
    "kernel_grid": f'[{k.get("grid_x","")},{k.get("grid_y","")},{k.get("grid_z","")}]',
    "kernel_block": f'[{k.get("block_x","")},{k.get("block_y","")},{k.get("block_z","")}]',
    "kernel_name": kernel_name,
    "runtime_function_name": runtime_fn,
    "runtime_params_json": api.get("runtime_params_json", ""),
    "driver_function_name": driver_fn,
    "driver_params_json": api.get("driver_params_json", ""),
})
```

输出：

```text
runtime_kernel_api_params_full.csv
runtime_kernel_api_params_summary.csv
runtime_kernel_api_params_report.md
```

## API / Captured / Sample params 报告生成

文件：[tools/make_runtime_api_param_coverage_report.py](/data/ssd2/gongoubo/pd/tools/make_runtime_api_param_coverage_report.py)

核心逻辑：

```python
counts = Counter()
decoded = Counter()
undecoded = Counter()
sample_decoded = {}
sample_undecoded = {}

for name in os.listdir(out_dir):
    if not name.startswith("runtime_api_params_") or not name.endswith(".csv"):
        continue
    path = os.path.join(out_dir, name)
    if os.path.getsize(path) <= 111:
        continue
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            api = row.get("function_name", "")
            if api not in REQUESTED_APIS:
                continue
            counts[api] += 1
            params = row.get("params_json", "")
            parsed = parse_params(params)
            if parsed.get("capture_status") == "api_seen_params_not_decoded":
                undecoded[api] += 1
                sample_undecoded.setdefault(api, params)
            else:
                decoded[api] += 1
                sample_decoded.setdefault(api, params)
```

写 CSV 字段：

```python
fieldnames = [
    "api_name",
    "captured_calls",
    "decoded_param_calls",
    "undecoded_param_calls",
    "unique_kernel_signatures",
    "status",
    "reason",
    "domains",
    "cbids",
    "sample_params_json",
    "sample_undecoded_json",
]
```

后续如果只想看三列，可以从 CSV 中取：

```text
api_name,captured_calls,sample_params_json
```

## Runtime launch args 样例报告

文件：[tools/make_runtime_launch_args_sample_report.py](/data/ssd2/gongoubo/pd/tools/make_runtime_launch_args_sample_report.py)

用途：按耗时 top Runtime launch kernel，提取实际参数样例。

```python
RUNTIME_LAUNCH_APIS = {"cudaLaunchKernel", "cudaLaunchKernelExC"}
```

从 summary 中选择 Runtime launch kernel：

```python
with open(summary_path, newline="") as f:
    for row in csv.DictReader(f):
        if row.get("api_name") not in RUNTIME_LAUNCH_APIS:
            continue
        row["total_duration_ms"] = float(row["total_duration_ms"])
        row["count"] = int(row["count"])
        targets.append(row)
targets.sort(key=lambda r: r["total_duration_ms"], reverse=True)
targets = targets[:top_n]
```

从 full CSV 中找实际 sample params：

```python
wanted = {(r["api_name"], r["kernel_name"]) for r in targets}
samples = {}
with open(full_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row.get("runtime_function_name"), row.get("kernel_name"))
        if key in wanted and key not in samples:
            samples[key] = row
            if len(samples) == len(wanted):
                break
```

输出：

```text
runtime_launch_args_samples.md
runtime_launch_args_samples.csv
```

## 结果解读

### `cudaDeviceSynchronize = 0`

如果 `cudaDeviceSynchronize` 为 0，含义是 workload 没有实际调用这个 API，不是 profiler 漏采。高性能服务通常避免全 device 同步，更多使用：

```text
cudaEventQuery
cudaEventRecord
cudaEventSynchronize
cudaStreamWaitEvent
cudaStreamSynchronize
```

### `func` 地址不能直接当作 kernel 名

`func` 是进程内 kernel function handle，例如：

```json
{"func":"0x2297ead0"}
```

它不是稳定符号名，不能跨进程复用。要得到实际 kernel 名，需要通过 CUPTI Activity 的 `kernel_name`，用 `pid + correlation_id` join。

### Runtime raw args 不是字段级 ABI

Runtime launch callback 只有：

```text
void **args
```

没有参数个数、类型、大小。因此 `runtimeArgs` 是原始 bytes，需要结合 demangled kernel signature 离线解释。

### NCCL kernel 参数

NCCL kernel 常见路径：

```text
cuLaunchKernelEx -> CUPTI Activity kernel_name -> ncclDevKernel_...
```

Driver `extra` 可以 dump 原始 launch buffer，但 NCCL 内部结构字段名需要对应 NCCL 版本 ABI 才能完整解释。

## 换模型后的最小复用命令

```bash
g++ -std=c++17 -fPIC -shared tools/cupti_api_param_profiler.cpp \
  -o tools/libcupti_api_param_profiler.so \
  -I/usr/local/cuda-13.0/targets/x86_64-linux/include \
  -L/usr/local/cuda-13.0/targets/x86_64-linux/lib \
  -lcupti -ldl -pthread

RUN_ID=<new_model_run_id> \
MODEL_PATH=<new_model_path> \
TOKENIZER=<new_model_path> \
INPUT_LEN=<input_len> \
OUTPUT_LEN=<output_len> \
BATCH_SIZE=<batch> \
NUM_PROMPTS=<batch> \
MAX_CONCURRENCY=<batch> \
CUPTI_RUNTIME_ARG_SLOTS=8 \
CUPTI_RUNTIME_ARG_BYTES=16 \
CUPTI_EXTRA_ARG_DUMP_BYTES=0 \
./profile_flash_pd_cupti_api_kernel_params.sh

python3 tools/join_cupti_api_kernel_params.py profiles/<new_model_run_id>
python3 tools/make_runtime_api_kernel_signature_report.py profiles/<new_model_run_id>
python3 tools/make_runtime_api_param_coverage_report.py profiles/<new_model_run_id>
python3 tools/make_runtime_launch_args_sample_report.py profiles/<new_model_run_id> 80
```

优先看：

```text
profiles/<new_model_run_id>/runtime_api_param_coverage.md
profiles/<new_model_run_id>/runtime_api_to_kernel_signatures.md
profiles/<new_model_run_id>/runtime_launch_args_samples.md
```
