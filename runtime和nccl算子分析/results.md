# TP=4, EP=1

## 部署脚本

部署环境：1台B200

部署方法：prefill和decode都是TP4, EP1

```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server \
  --model-path /data/ssd2/checkpoints/deepseek-ai/DeepSeek-V4-Flash \
  --served-model-name deepseek-ai/DeepSeek-V4-Flash \
  --trust-remote-code \
  --tool-call-parser deepseekv4 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --tp-size 4 \
  --dp-size 1 \
  --ep-size 1 \
  --moe-runner-backend flashinfer_mxfp4 \
  --disable-flashinfer-autotune \
  --mem-fraction-static 0.9 \
  --disable-radix-cache \
  --host 0.0.0.0 \
  --port 30000
  
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m sglang.launch_server \
  --model-path /data/ssd2/checkpoints/deepseek-ai/DeepSeek-V4-Flash \
  --served-model-name deepseek-ai/DeepSeek-V4-Flash \
  --trust-remote-code \
  --tool-call-parser deepseekv4 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --tp-size 4 \
  --dp-size 1 \
  --ep-size 1 \
  --moe-runner-backend flashinfer_mxfp4 \
  --disable-flashinfer-autotune \
  --mem-fraction-static 0.9 \
  --swa-full-tokens-ratio 0.1 \
  --host 0.0.0.0 \
  --port 30001 
  
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://10.1.17.13:30000 \
  --decode http://10.1.17.13:30001 \
  --host 0.0.0.0 --port 13784 \
  --disable-circuit-breaker \
  --health-check-interval-secs 999999
```

## Configuration

| Item                | Value                            |
| ------------------- | -------------------------------- |
| Model               | DeepSeek-V4-Flash                |
| Mode                | Local single-node PD separation  |
| Prefill GPUs        | 0,1,2,3                          |
| Decode GPUs         | 4,5,6,7                          |
| Batch / concurrency | 256                              |
| Input length        | 4096                             |
| Output length       | 1024                             |
| nsys trace          | cuda,nvtx,osrt,cublas,cudnn,nccl |
| CUDA graph trace    | node                             |

## Runtime API Summary

| #    | API                                       | Time % | Total Time | Calls     | Avg        | Med        | Min        | Max        | StdDev     |
| ---- | ----------------------------------------- | ------ | ---------- | --------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| 1    | cudaMemcpyAsync                           | 38.50% | 67.166 s   | 403,244   | 166.565 us | 47.595 us  | 1.573 us   | 87.477 ms  | 631.570 us |
| 2    | cudaEventSynchronize                      | 17.60% | 30.641 s   | 10,360    | 2.958 ms   | 2.642 ms   | 3.923 us   | 10.760 ms  | 2.732 ms   |
| 3    | cudaGraphLaunch                           | 8.90%  | 15.606 s   | 10,360    | 1.506 ms   | 1.514 ms   | 762.998 us | 6.640 ms   | 303.680 us |
| 4    | cudaStreamSynchronize                     | 4.30%  | 7.464 s    | 351,185   | 21.254 us  | 16.521 us  | 2.500 us   | 13.801 ms  | 60.322 us  |
| 5    | cudaLaunchKernel                          | 4.00%  | 6.991 s    | 869,084   | 8.044 us   | 5.269 us   | 1.146 us   | 39.995 ms  | 123.160 us |
| 6    | cudaGraphInstantiateWithFlags             | 2.20%  | 3.817 s    | 144       | 26.506 ms  | 26.581 ms  | 21.940 ms  | 31.812 ms  | 1.986 ms   |
| 7    | cuLaunchKernelEx                          | 2.20%  | 3.789 s    | 423,624   | 8.943 us   | 7.860 us   | 1.395 us   | 7.818 ms   | 22.690 us  |
| 8    | cudaLaunchKernelExC                       | 2.20%  | 3.769 s    | 464,656   | 8.111 us   | 7.880 us   | 1.255 us   | 28.968 ms  | 84.158 us  |
| 9    | cuKernelSetAttribute                      | 2.10%  | 3.639 s    | 568       | 6.407 ms   | 41.624 us  | 498 ns     | 599.442 ms | 43.690 ms  |
| 10   | cuMemSetAccess                            | 1.90%  | 3.307 s    | 1,572     | 2.104 ms   | 1.245 ms   | 116.896 us | 133.365 ms | 5.359 ms   |
| 11   | cudaDeviceSynchronize                     | 1.80%  | 3.227 s    | 432       | 7.471 ms   | 212.763 us | 31.011 us  | 65.769 ms  | 14.198 ms  |
| 12   | cudaEventRecordWithFlags                  | 1.70%  | 2.986 s    | 416,976   | 7.162 us   | 7.595 us   | 262 ns     | 11.619 ms  | 38.038 us  |
| 13   | cudaMalloc                                | 1.60%  | 2.772 s    | 3,454     | 802.661 us | 598.840 us | 5.494 us   | 50.814 ms  | 1.483 ms   |
| 14   | cudaEventCreateWithFlags                  | 1.60%  | 2.763 s    | 401,936   | 6.875 us   | 5.761 us   | 259 ns     | 1.346 ms   | 7.711 us   |
| 15   | cuLibraryLoadData                         | 1.20%  | 2.180 s    | 612       | 3.563 ms   | 87.939 us  | 39.290 us  | 57.126 ms  | 7.902 ms   |
| 16   | cuMemCreate                               | 1.20%  | 2.026 s    | 1,988     | 1.019 ms   | 414.384 us | 10.936 us  | 131.386 ms | 3.411 ms   |
| 17   | cudaStreamWaitEvent                       | 1.10%  | 1.994 s    | 387,896   | 5.141 us   | 5.493 us   | 313 ns     | 161.828 us | 3.342 us   |
| 18   | cuModuleLoadData                          | 1.10%  | 1.916 s    | 280       | 6.843 ms   | 145.248 us | 50.600 us  | 646.511 ms | 61.345 ms  |
| 19   | cudaFree                                  | 1.10%  | 1.852 s    | 2,260     | 819.408 us | 430.862 us | 22.000 us  | 5.116 ms   | 917.465 us |
| 20   | cuMemExportToShareableHandle              | 0.90%  | 1.574 s    | 256       | 6.148 ms   | 5.970 ms   | 668.331 us | 13.370 ms  | 2.222 ms   |
| 21   | cudaStreamIsCapturing                     | 0.90%  | 1.552 s    | 1,123,198 | 1.382 us   | 961 ns     | 168 ns     | 153.421 us | 1.519 us   |
| 22   | cudaEventDestroy                          | 0.40%  | 753.384 ms | 401,816   | 1.875 us   | 1.968 us   | 221 ns     | 597.031 us | 2.520 us   |
| 23   | cuTensorMapEncodeTiled                    | 0.30%  | 447.537 ms | 1,104,360 | 405 ns     | 171 ns     | 102 ns     | 108.828 us | 736 ns     |
| 24   | cuMemMap                                  | 0.20%  | 435.665 ms | 1,572     | 277.141 us | 131.567 us | 1.131 us   | 84.115 ms  | 2.135 ms   |
| 25   | cuMemImportFromShareableHandle            | 0.20%  | 268.570 ms | 256       | 1.049 ms   | 829.283 us | 128.971 us | 3.773 ms   | 693.152 us |
| 26   | cuKernelGetName                           | 0.10%  | 240.818 ms | 906,668   | 266 ns     | 144 ns     | 77 ns      | 396.297 us | 669 ns     |
| 27   | cudaEventQuery                            | 0.10%  | 209.449 ms | 30,744    | 6.813 us   | 3.554 us   | 523 ns     | 130.723 us | 6.058 us   |
| 28   | cudaGraphDestroy                          | 0.10%  | 172.991 ms | 144       | 1.201 ms   | 1.193 ms   | 932.105 us | 1.647 ms   | 142.375 us |
| 29   | cudaStreamGetCaptureInfo                  | 0.10%  | 171.873 ms | 433,444   | 396 ns     | 283 ns     | 177 ns     | 85.972 us  | 578 ns     |
| 30   | cuKernelGetFunction                       | 0.10%  | 130.436 ms | 172       | 758.348 us | 281.000 us | 56.818 us  | 27.861 ms  | 3.300 ms   |
| 31   | cudaStreamCreateWithPriority              | 0.10%  | 123.092 ms | 512       | 240.414 us | 4.096 us   | 2.204 us   | 4.878 ms   | 584.239 us |
| 32   | cudaMemGetInfo                            | 0.10%  | 122.716 ms | 97        | 1.265 ms   | 67.130 us  | 7.177 us   | 102.213 ms | 10.389 ms  |
| 33   | cudaMemsetAsync                           | 0.00%  | 72.145 ms  | 12,536    | 5.755 us   | 3.906 us   | 1.776 us   | 103.172 us | 5.087 us   |
| 34   | cudaStreamEndCapture                      | 0.00%  | 55.590 ms  | 144       | 386.042 us | 359.721 us | 208.310 us | 647.030 us | 100.988 us |
| 35   | cudaGetDriverEntryPointByVersion          | 0.00%  | 55.110 ms  | 109,896   | 502 ns     | 197 ns     | 167 ns     | 30.287 us  | 795 ns     |
| 36   | cuMemAddressReserve                       | 0.00%  | 45.991 ms  | 1,572     | 29.256 us  | 26.457 us  | 583 ns     | 208.721 us | 27.967 us  |
| 37   | cuLibraryLoadFromFile                     | 0.00%  | 40.373 ms  | 172       | 234.727 us | 226.658 us | 61.309 us  | 421.276 us | 72.615 us  |
| 38   | cuKernelGetAttribute                      | 0.00%  | 38.323 ms  | 109,800   | 349 ns     | 219 ns     | 70 ns      | 67.007 us  | 489 ns     |
| 39   | cudaIpcOpenMemHandle                      | 0.00%  | 28.703 ms  | 12        | 2.392 ms   | 2.652 ms   | 919.191 us | 3.668 ms   | 996.143 us |
| 40   | cudaStreamCreateWithFlags                 | 0.00%  | 24.225 ms  | 164       | 147.713 us | 46.083 us  | 3.942 us   | 882.755 us | 199.682 us |
| 41   | cudaHostAlloc                             | 0.00%  | 18.092 ms  | 319       | 56.715 us  | 31.153 us  | 3.109 us   | 2.038 ms   | 198.734 us |
| 42   | cuCtxGetId                                | 0.00%  | 12.292 ms  | 37,152    | 331 ns     | 280 ns     | 71 ns      | 32.982 us  | 513 ns     |
| 43   | cuMemRelease                              | 0.00%  | 7.156 ms   | 4         | 1.789 ms   | 1.566 ms   | 1.362 ms   | 2.662 ms   | 604.692 us |
| 44   | cuLaunchKernel                            | 0.00%  | 4.442 ms   | 444       | 10.004 us  | 9.848 us   | 2.290 us   | 120.898 us | 9.613 us   |
| 45   | cudaFreeHost                              | 0.00%  | 3.950 ms   | 148       | 26.691 us  | 24.239 us  | 6.641 us   | 97.241 us  | 9.901 us   |
| 46   | cudaEventRecord                           | 0.00%  | 3.027 ms   | 1,596     | 1.897 us   | 940 ns     | 339 ns     | 23.943 us  | 2.314 us   |
| 47   | cudaStreamDestroy                         | 0.00%  | 2.879 ms   | 144       | 19.992 us  | 19.967 us  | 10.525 us  | 39.857 us  | 3.585 us   |
| 48   | cuModuleLoad                              | 0.00%  | 2.801 ms   | 16        | 175.078 us | 172.204 us | 117.943 us | 237.525 us | 42.356 us  |
| 49   | cuGetProcAddress_v2                       | 0.00%  | 2.619 ms   | 10,086    | 260 ns     | 143 ns     | 69 ns      | 22.201 us  | 544 ns     |
| 50   | cudaKernelSetAttributeForDevice           | 0.00%  | 2.055 ms   | 128       | 16.058 us  | 503 ns     | 202 ns     | 162.665 us | 41.413 us  |
| 51   | cudaStreamBeginCapture                    | 0.00%  | 1.977 ms   | 144       | 13.729 us  | 8.438 us   | 3.504 us   | 37.212 us  | 9.559 us   |
| 52   | cudaGetFuncBySymbol                       | 0.00%  | 1.743 ms   | 436       | 3.998 us   | 3.854 us   | 1.889 us   | 26.753 us  | 1.440 us   |
| 53   | cuLibraryGetKernel                        | 0.00%  | 1.716 ms   | 884       | 1.941 us   | 1.016 us   | 133 ns     | 19.108 us  | 2.789 us   |
| 54   | cuMemRetainAllocationHandle               | 0.00%  | 1.525 ms   | 116       | 13.146 us  | 12.473 us  | 8.407 us   | 31.994 us  | 2.941 us   |
| 55   | cudaGraphGetNodes                         | 0.00%  | 1.494 ms   | 144       | 10.376 us  | 10.088 us  | 8.866 us   | 23.025 us  | 1.360 us   |
| 56   | cudaLibraryLoadData                       | 0.00%  | 1.429 ms   | 16        | 89.286 us  | 87.094 us  | 70.480 us  | 107.827 us | 11.250 us  |
| 57   | cudaThreadExchangeStreamCaptureMode       | 0.00%  | 910.254 us | 2,264     | 402 ns     | 179 ns     | 150 ns     | 29.750 us  | 1.090 us   |
| 58   | cudaMemPoolCreate                         | 0.00%  | 808.362 us | 4         | 202.090 us | 204.026 us | 175.032 us | 225.278 us | 21.698 us  |
| 59   | cudaGraphAddEventRecordNode               | 0.00%  | 753.667 us | 288       | 2.617 us   | 2.277 us   | 647 ns     | 21.971 us  | 2.500 us   |
| 60   | cuMemGetAllocationGranularity             | 0.00%  | 650.355 us | 1,576     | 413 ns     | 336 ns     | 85 ns      | 18.136 us  | 832 ns     |
| 61   | cudaLaunchHostFunc                        | 0.00%  | 512.070 us | 144       | 3.556 us   | 3.209 us   | 2.024 us   | 26.055 us  | 2.111 us   |
| 62   | cudaStreamUpdateCaptureDependencies       | 0.00%  | 511.038 us | 720       | 710 ns     | 380 ns     | 228 ns     | 12.516 us  | 828 ns     |
| 63   | cudaGraphAddDependencies                  | 0.00%  | 392.956 us | 288       | 1.364 us   | 1.511 us   | 311 ns     | 4.360 us   | 771 ns     |
| 64   | cudaGraphRetainUserObject                 | 0.00%  | 287.890 us | 144       | 1.999 us   | 2.002 us   | 1.134 us   | 2.831 us   | 309 ns     |
| 65   | cudaMemset                                | 0.00%  | 271.966 us | 12        | 22.664 us  | 7.731 us   | 4.206 us   | 65.404 us  | 24.936 us  |
| 66   | cudaUserObjectCreate                      | 0.00%  | 226.888 us | 144       | 1.576 us   | 1.558 us   | 1.046 us   | 3.225 us   | 243 ns     |
| 67   | cudaMemcpy                                | 0.00%  | 151.065 us | 4         | 37.766 us  | 37.809 us  | 33.941 us  | 41.505 us  | 3.837 us   |
| 68   | cuInit                                    | 0.00%  | 148.112 us | 30        | 4.937 us   | 3.429 us   | 1.445 us   | 42.797 us  | 7.381 us   |
| 69   | cuLibraryGetKernelCount                   | 0.00%  | 110.508 us | 172       | 642 ns     | 552 ns     | 208 ns     | 2.009 us   | 296 ns     |
| 70   | cudaGetDeviceProperties                   | 0.00%  | 85.782 us  | 88        | 975 ns     | 376 ns     | 90 ns      | 7.120 us   | 1.361 us   |
| 71   | cuModuleGetLoadingMode                    | 0.00%  | 72.986 us  | 26        | 2.807 us   | 250 ns     | 63 ns      | 29.918 us  | 5.917 us   |
| 72   | cudaOccupancyMaxActiveClusters            | 0.00%  | 62.227 us  | 120       | 519 ns     | 429 ns     | 387 ns     | 3.640 us   | 453 ns     |
| 73   | cuLibraryEnumerateKernels                 | 0.00%  | 53.039 us  | 172       | 308 ns     | 250 ns     | 214 ns     | 1.473 us   | 157 ns     |
| 74   | cudaOccupancyAvailableDynamicSMemPerBlock | 0.00%  | 48.216 us  | 4         | 12.054 us  | 10.117 us  | 9.944 us   | 18.038 us  | 3.991 us   |
| 75   | cudaLibraryGetKernel                      | 0.00%  | 40.396 us  | 64        | 631 ns     | 406 ns     | 329 ns     | 2.080 us   | 436 ns     |
| 76   | cudaMemPoolSetAttribute                   | 0.00%  | 12.730 us  | 4         | 3.183 us   | 2.758 us   | 2.030 us   | 5.184 us   | 1.492 us   |

## NCCL Host/NVTX Summary

| #    | Range             | Style    | Time % | Total Time | Instances | Avg        | Med        | Min       | Max        | StdDev     |
| ---- | ----------------- | -------- | ------ | ---------- | --------- | ---------- | ---------- | --------- | ---------- | ---------- |
| 1    | NCCL:AllGather    | StartEnd | 66.00% | 32.445 s   | 339,200   | 95.651 us  | 64.128 us  | 1.071 us  | 28.308 ms  | 436.374 us |
| 2    | NCCL:API Group    | StartEnd | 6.90%  | 3.370 s    | 436       | 7.730 ms   | 145.887 us | 66.789 us | 832.414 ms | 78.397 ms  |
| 3    | NCCL:GroupLaunch  | StartEnd | 6.80%  | 3.365 s    | 436       | 7.717 ms   | 133.042 us | 58.495 us | 832.387 ms | 78.395 ms  |
| 4    | NCCL:GroupRuntime | StartEnd | 0.80%  | 413.117 ms | 10,652    | 38.783 us  | 39.323 us  | 4.172 us  | 785.049 us | 18.638 us  |
| 5    | NCCL:AllReduce    | StartEnd | 0.00%  | 5.772 ms   | 8         | 721.519 us | 153.712 us | 2.324 us  | 3.622 ms   | 1.269 ms   |
| 6    | NCCL:CommInit     | StartEnd | 0.00%  | 184.234 us | 4         | 46.059 us  | 45.361 us  | 42.093 us | 51.418 us  | 3.975 us   |

| NVTX Range        | 可对应的 NCCL API/内部函数                                   | 对应 GPU Kernel                                  |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------ |
| NCCL:AllGather    | pncclAllGather，即公共 API ncclAllGather                     | ncclDevKernel_AllGather_RING_LL(...)             |
| NCCL:AllReduce    | pncclAllReduce，即公共 API ncclAllReduce                     | ncclDevKernel_AllReduce_Sum_f32_RING_LL(...)     |
| NCCL:CommInit     | pncclCommInitRank / ncclCommInitRankDev / ncclCommInitRankFunc | 一般属于初始化流程，不一定对应 collective kernel |
| NCCL:API Group    | NCCL Group API 范围，与 group 调用或封装相关                 | 内部可能覆盖 AllGather/AllReduce kernel launch   |
| NCCL:GroupLaunch  | 与 groupLaunch(...) / ncclGroupEndInternal(...) 相关         | 内部可能覆盖 NCCL collective kernel launch       |
| NCCL:GroupRuntime | 与 ncclLaunchKernel(...) 相关                                | 主要对应实际 NCCL kernel launch/runtime          |

## NCCL GPU Kernel Summary

| #    | Kernel                                                       | Time % | Total Time | Instances | Avg        | Med        | Min        | Max       | StdDev     |
| ---- | ------------------------------------------------------------ | ------ | ---------- | --------- | ---------- | ---------- | ---------- | --------- | ---------- |
| 1    | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>) | 0.40%  | 1.121 s    | 10,648    | 105.260 us | 73.760 us  | 14.944 us  | 28.316 ms | 434.531 us |
| 2    | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>) | 0.00%  | 5.779 ms   | 4         | 1.445 ms   | 922.607 us | 307.199 us | 3.627 ms  | 1.544 ms   |

# TP=4, EP=4

## 配置

| 项目                   | 值                                                   |
| ---------------------- | ---------------------------------------------------- |
| 模型                   | deepseek-ai/DeepSeek-V4-Flash                        |
| 部署                   | 本机 PD disaggregation                               |
| Prefill                | GPU 0-3, TP=4, DP=1, EP=1, flashinfer_mxfp4          |
| Decode                 | GPU 4-7, TP=4, DP=1, EP=4, deep_gemm + deepep normal |
| Batch / Input / Output | 256 / 4096 / 1024                                    |
| nsys trace             | cuda,nvtx,nccl                                       |
| cuda graph trace       | graph                                                |
| nsys report            | nsys_decode.nsys-rep                                 |
| CSV                    | nsys_kernels.csv, nsys_cuda_api.csv, nsys_nvtx.csv   |

## NCCL Kernel Summary

| Time (%) | Total Time (ns) | Instances | Avg (ns)  | Med (ns)  | Min (ns) | Max (ns) | StdDev (ns) | Name                                                         |
| -------- | --------------- | --------- | --------- | --------- | -------- | -------- | ----------- | ------------------------------------------------------------ |
| 7.3      | 8490220137      | 51104     | 166136.1  | 141680    | 6432     | 17497350 | 188335.5    | ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>) |
| 0        | 8962578         | 4         | 2240644.5 | 2242220.5 | 155296   | 4322841  | 1701665.7   | ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>) |

## DeepEP A2A Kernel Summary

当前 decode 使用 `--moe-a2a-backend deepep --deepep-mode normal`，EP all-to-all 没有表现为 NCCL `AllToAll` API/kernel，而是表现为 DeepEP 的 intranode dispatch/combine kernel。

| Time (%) | Total Time (ns) | Instances | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) | Name                                                         |
| -------- | --------------- | --------- | -------- | -------- | -------- | -------- | ----------- | ------------------------------------------------------------ |
| 25.6     | 29688328682     | 49931     | 594587.1 | 521728   | 18560    | 9125580  | 524943.7    | void deep_ep::intranode::notify_dispatch<(int)4>(const int *, int *, const int *, int *, int, int, int, const bool *, int *, int *, int, int, void , int , int) |
| 6.1      | 7076229921      | 49929     | 141725.8 | 94560    | 9504     | 46123023 | 425389.7    | void deep_ep::intranode::cached_notify_combine<(int)4>(void , int *, int, int, int, int , int) |
| 1.6      | 1912090615      | 49929     | 38296.2  | 40544    | 7232     | 288319   | 8191.6      | void deep_ep::intranode::combine<__nv_bfloat16, (int)4, (int)768, (int)4096>(T1 *, float *, const T1 *, const float *, const T1 *, const T1 *, const int *, const int *, const int *, int *, int, int, int, int, void **, int, int, int) |
| 1.5      | 1739940606      | 49931     | 34846.9  | 34304    | 11008    | 1149569  | 32473.4     | void deep_ep::intranode::dispatch<(int)4, (int)768, (int)8192>(int4 *, float *, int *, long *, float *, int *, int *, const int4 *, const float *, const long *, const float *, const bool *, const int *, int, int, int, int, int, int, int, int, void **, int, int, int) |
| 0.2      | 193863966       | 49931     | 3882.6   | 3872     | 2880     | 12320    | 442.1       | void deep_ep::layout::get_dispatch_layout<(int)256, (int)4, (int)8>(const long *, int *, int *, int *, bool *, int, int, int, int) |

## NCCL Host/NVTX Summary Range

这些名称来自 nsys 的 NVTX summary `Range` 原始字段。

| Time (%) | Total Time (ns) | Instances | Avg (ns) | Med (ns) | Min (ns) | Max (ns)  | StdDev (ns) | Style    | Range             |
| -------- | --------------- | --------- | -------- | -------- | -------- | --------- | ----------- | -------- | ----------------- |
| 95.3     | 956421579586    | 6274731   | 152424.3 | 111584   | 720      | 17488614  | 226887.5    | StartEnd | NCCL:AllGather    |
| 2.3      | 23433797074     | 194667    | 120378.9 | 86534    | 32262    | 807253883 | 5145296     | StartEnd | NCCL:API Group    |
| 2        | 19754010156     | 194667    | 101475.9 | 66889    | 27631    | 807228235 | 5145188.1   | StartEnd | NCCL:GroupLaunch  |
| 0.4      | 3526544897      | 194667    | 18115.8  | 15661    | 1995     | 1093698   | 8729.3      | StartEnd | NCCL:GroupRuntime |
| 0        | 8812328         | 8         | 1101541  | 5069.5   | 2596     | 4318009   | 1645386.6   | StartEnd | NCCL:AllReduce    |
| 0        | 262621          | 8         | 32827.6  | 31262.5  | 17357    | 56041     | 15641.8     | StartEnd | NCCL:CommInit     |

## CUDA Runtime/API Summary

`Total Time/Avg/Med/Min/Max/StdDev` 单位均为 ns。

| Time (%) | Total Time (ns) | Num Calls | Avg (ns)  | Med (ns) | Min (ns) | Max (ns)  | StdDev (ns) | Name                                      |
| -------- | --------------- | --------- | --------- | -------- | -------- | --------- | ----------- | ----------------------------------------- |
| 89.6     | 531995586684    | 395803    | 1344091.9 | 7761     | 2461     | 137119561 | 3899392.2   | cudaMemcpyAsync                           |
| 2.1      | 12373055565     | 1491745   | 8294.4    | 7674     | 3266     | 7795130   | 17354.4     | cudaLaunchKernelExC                       |
| 1.5      | 9109793129      | 1141132   | 7983.1    | 7504     | 2543     | 7750139   | 13286.8     | cuLaunchKernelEx                          |
| 1.5      | 8685347214      | 862078    | 10074.9   | 7506     | 2801     | 147907221 | 242291      | cudaLaunchKernel                          |
| 1.1      | 6411857657      | 1584976   | 4045.4    | 2931     | 434      | 11954912  | 17916.9     | cudaEventQuery                            |
| 0.6      | 3533977575      | 1964      | 1799377.6 | 1136143  | 113023   | 126431104 | 4219515.2   | cuMemSetAccess                            |
| 0.4      | 2407997805      | 131438    | 18320.4   | 16439    | 2330     | 19824669  | 69469       | cudaStreamSynchronize                     |
| 0.4      | 2192113464      | 2124      | 1032068.5 | 425309   | 11738    | 127478187 | 4123175.1   | cuMemCreate                               |
| 0.3      | 1895745285      | 600       | 3159575.5 | 83450.5  | 32220    | 54002271  | 6643522.4   | cuLibraryLoadData                         |
| 0.3      | 1779015015      | 319664    | 5565.3    | 4791     | 406      | 11415733  | 40314.2     | cudaEventRecordWithFlags                  |
| 0.3      | 1731343167      | 224       | 7729210.6 | 158644.5 | 66042    | 557360037 | 63052437.7  | cuModuleLoadData                          |
| 0.3      | 1672189603      | 566038    | 2954.2    | 1419     | 374      | 1011865   | 3042.8      | cudaStreamWaitEvent                       |
| 0.3      | 1645249476      | 516073    | 3188      | 2360     | 298      | 1070118   | 6569.7      | cudaEventCreateWithFlags                  |
| 0.3      | 1583548845      | 1741      | 909562.8  | 678501   | 5497     | 65092564  | 2395326.7   | cudaMalloc                                |
| 0.3      | 1523185463      | 1300193   | 1171.5    | 817      | 340      | 5970415   | 10839.6     | cudaEventRecord                           |
| 0.2      | 1087547823      | 2871633   | 378.7     | 171      | 101      | 213791    | 687.7       | cuTensorMapEncodeTiled                    |
| 0.1      | 832886595       | 727751    | 1144.5    | 828      | 178      | 1425424   | 7474.4      | cudaStreamIsCapturing                     |
| 0.1      | 678426488       | 515808    | 1315.3    | 1092     | 216      | 1134708   | 3017        | cudaEventDestroy                          |
| 0.1      | 490707038       | 1964      | 249850.8  | 112950   | 1046     | 41799752  | 1359135.6   | cuMemMap                                  |
| 0.1      | 479849834       | 512       | 937206.7  | 673506.5 | 134023   | 4567746   | 671954.9    | cuMemImportFromShareableHandle            |
| 0.1      | 317512547       | 913170    | 347.7     | 280      | 69       | 1013910   | 1252.8      | cuKernelGetName                           |
| 0        | 280026093       | 304262    | 920.3     | 810      | 148      | 899045    | 3379.1      | cudaThreadExchangeStreamCaptureMode       |
| 0        | 155539382       | 51108     | 3043.3    | 2970     | 1185     | 103014    | 1420.7      | cudaGetFuncBySymbol                       |
| 0        | 144678656       | 295098    | 490.3     | 189      | 156      | 107085    | 896.7       | cudaGetDriverEntryPointByVersion          |
| 0        | 110389973       | 294956    | 374.3     | 215      | 71       | 43814     | 566.8       | cuKernelGetAttribute                      |
| 0        | 106652627       | 51108     | 2086.8    | 2105     | 575      | 96659     | 1158        | cudaStreamGetCaptureInfo                  |
| 0        | 97465976        | 512       | 190363.2  | 3447.5   | 1874     | 3633333   | 432446.8    | cudaStreamCreateWithPriority              |
| 0        | 71537572        | 24        | 2980732.2 | 2876046  | 1013252  | 4763095   | 882134.3    | cudaIpcOpenMemHandle                      |
| 0        | 68578150        | 292       | 234856.7  | 193828.5 | 5185     | 1404397   | 175800.9    | cudaFree                                  |
| 0        | 47490416        | 1964      | 24180.5   | 17943.5  | 784      | 171102    | 25370.8     | cuMemAddressReserve                       |
| 0        | 45337628        | 181       | 250484.1  | 259518   | 31779    | 597651    | 82444.4     | cuKernelGetFunction                       |
| 0        | 40471738        | 181       | 223600.8  | 213612   | 35875    | 482105    | 86105.3     | cuLibraryLoadFromFile                     |
| 0        | 31957024        | 552       | 57893.2   | 33073.5  | 464      | 315326    | 59087.1     | cuKernelSetAttribute                      |
| 0        | 18965936        | 132       | 143681.3  | 37269.5  | 3310     | 2322904   | 423564.9    | cudaHostAlloc                             |
| 0        | 12031224        | 792       | 15190.9   | 6325.5   | 2993     | 58995     | 13932.2     | cudaMemsetAsync                           |
| 0        | 11243464        | 1164      | 9659.3    | 9053.5   | 5457     | 78294     | 4958.8      | cuLaunchKernel                            |
| 0        | 9271550         | 1156      | 8020.4    | 7911     | 4380     | 41739     | 2432.4      | cudaEventSynchronize                      |
| 0        | 8556402         | 8         | 1069550.3 | 227391   | 133774   | 7035140   | 2411783     | cudaMemPoolCreate                         |
| 0        | 7250144         | 50        | 145002.9  | 45909    | 6417     | 3084789   | 444342.4    | cudaMemGetInfo                            |
| 0        | 2578614         | 9595      | 268.7     | 142      | 70       | 26992     | 481.6       | cuGetProcAddress_v2                       |
| 0        | 1652322         | 20        | 82616.1   | 25175.5  | 4713     | 360554    | 119656.6    | cudaStreamCreateWithFlags                 |
| 0        | 1645893         | 4         | 411473.3  | 347621   | 118071   | 832580    | 348954      | cuMemRelease                              |
| 0        | 1571930         | 848       | 1853.7    | 918      | 131      | 18430     | 2784        | cuLibraryGetKernel                        |
| 0        | 1090495         | 64        | 17039     | 587.5    | 199      | 164725    | 44074.1     | cudaKernelSetAttributeForDevice           |
| 0        | 712741          | 1968      | 362.2     | 327.5    | 80       | 16195     | 466.7       | cuMemGetAllocationGranularity             |
| 0        | 651292          | 8         | 81411.5   | 74790.5  | 68147    | 101219    | 13651.4     | cudaLibraryLoadData                       |
| 0        | 600073          | 4         | 150018.3  | 150937   | 131926   | 166273    | 14269.9     | cuModuleLoad                              |
| 0        | 398002          | 12        | 33166.8   | 40101    | 8265     | 57681     | 18694.9     | cudaMemcpy                                |
| 0        | 296303          | 12        | 24691.9   | 8186.5   | 4216     | 68603     | 27133.2     | cudaMemset                                |
| 0        | 240437          | 4         | 60109.3   | 60615.5  | 55073    | 64133     | 4000        | cudaDeviceSynchronize                     |
| 0        | 110695          | 120       | 922.5     | 424      | 384      | 8124      | 997.5       | cudaOccupancyMaxActiveClusters            |
| 0        | 103619          | 181       | 572.5     | 467      | 191      | 1778      | 281.9       | cuLibraryGetKernelCount                   |
| 0        | 98734           | 29        | 3404.6    | 3267     | 902      | 6559      | 1421.2      | cuInit                                    |
| 0        | 73853           | 56        | 1318.8    | 532.5    | 91       | 5138      | 1422.5      | cudaGetDeviceProperties                   |
| 0        | 59681           | 4         | 14920.3   | 13482    | 9705     | 23012     | 5706.7      | cudaOccupancyAvailableDynamicSMemPerBlock |
| 0        | 56043           | 181       | 309.6     | 242      | 206      | 1188      | 151.2       | cuLibraryEnumerateKernels                 |
| 0        | 28581           | 8         | 3572.6    | 3453.5   | 1660     | 5795      | 1553.7      | cudaMemPoolSetAttribute                   |
| 0        | 28022           | 21        | 1334.4    | 262      | 83       | 6159      | 2044.5      | cuModuleGetLoadingMode                    |
| 0        | 21956           | 32        | 686.1     | 399.5    | 335      | 2175      | 562.8       | cudaLibraryGetKernel                      |

## Runtime api参数说明

1. CUPTI Callback API

1. 用来捕获 Runtime/Driver API 调用及参数，比如：
   - cudaLaunchKernel
   - cudaLaunchKernelExC
   - cuLaunchKernel
   - cuLaunchKernelEx
   - cudaMemcpyAsync
   - cudaEventRecord
   - cudaStreamWaitEvent
   - cuMemMap
   - cuLibraryGetKernel
   - 等等

1. CUPTI Activity API

1. 用来捕获实际 GPU kernel activity，比如 kernel name、device、stream、grid/block、duration、correlation_id。然后用：pid + correlation_id 把 Callback API 记录和 kernel activity 记录 join 起来，得到：Runtime/Driver API 参数 -> 实际 GPU kernel name/signature

| API                                       | Captured | Sample params                                                | Sample params 说明                                           |
| ----------------------------------------- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| cudaMemcpyAsync                           | 1127612  | {"dst":"0x7c7373000000","src":"0x7ffd8cd72670","count":8,"kind":4,"kindName":"cudaMemcpyDefault","stream":"0x1fb9af00"} | 异步内存拷贝操作。dst 表示目标 GPU 地址，src 表示源地址，count 为拷贝字节数，kind/kindName 表示拷贝方向，stream 指定执行该异步操作的 CUDA stream。 |
| cudaLaunchKernelExC                       | 5621584  | {"func":"0x7c73bb2608b0","config":"0x7ffd8cd73aa0","grid":[1,1,1],"block":[256,1,1],"args":"0x7ffd8cd738c0","dynamicSmemBytes":0,"stream":"0x26525550","attrs":"(nil)","numAttrs":0} | 扩展版 CUDA kernel 启动接口。func 是 kernel 函数句柄，grid/block 描述执行配置，args 保存 kernel 参数，dynamicSmemBytes 表示动态共享内存大小，stream 指定执行流。 |
| cuLaunchKernelEx                          | 9802812  | {"func":"0x2297ead0","config":"0x7ffd8cd72b20","grid":[1,1,1],"block":[128,1,1],"sharedMemBytes":82240,"stream":"0x2ba833f0","attrs":"0x7ffd8cd72b60","numAttrs":4,"kernelParams":"(nil)","extra":"0x7ffd8cd72af0"} | Driver API 扩展 kernel 启动接口。func 为 kernel，grid/block 为网格和线程块配置，sharedMemBytes 是共享内存大小，kernelParams/extra 保存参数传递信息。 |
| cudaLaunchKernel                          | 3047962  | {"func":"0x268c7cb0","grid":[1,1,1],"block":[128,1,1],"args":"0x7ffd8cd72a30","sharedMem":0,"stream":"(nil)"} | Runtime API kernel 启动接口。func 为 kernel 地址，grid/block 定义并行规模，args 保存参数列表，sharedMem 表示动态共享内存，stream 表示执行流。 |
| cudaEventQuery                            | 4416652  | {"event":"0x4148dfc0"}                                       | 查询 CUDA event 状态。event 是事件对象句柄，用于判断此前提交到 GPU 的任务是否完成。 |
| cuMemSetAccess                            | 2476     | {"ptr":"0x7c7372e00000","size":2097152,"desc":"0x7ffd8cd72574","count":1} | 设置 CUDA 虚拟内存访问权限。ptr 为虚拟地址，size 为区域大小，desc 描述访问属性，count 表示属性数量。 |
| cudaStreamSynchronize                     | 144666   | {"stream":"0x1fb9af00"}                                      | 等待指定 CUDA stream 中所有任务完成。stream 为需要同步的执行流句柄。 |
| cuMemCreate                               | 2636     | {"handle_out":"0x7ffd8cd72d90","size":2097152,"prop":"0x7ffd8cd72da0","flags":0} | 创建 CUDA 物理内存分配对象。handle_out 返回内存句柄，size 为申请大小，prop 描述内存属性，flags 为分配选项。 |
| cuLibraryLoadData                         | 704      | {"library_out":"0x7ffd8cd720e0","code":"0x7c775ba0a270","jitOptions":"0x7ffd8cd720f4","jitOptionsValues":"0x7ffd8cd72110","numJitOptions":0,"libraryOptions":"0x7ffd8cd720ec","libraryOptionValues":"0x7ffd8cd72100","numLibraryOptions":1} | 从内存加载 CUDA Library/CUBIN。library_out 返回库对象，code 指向代码数据，jitOptions 控制 JIT 编译选项。 |
| cudaEventRecordWithFlags                  | 1039765  | {"event":"0x4067f150","stream":"0x26525550","flags":0}       | 在指定 stream 上记录 event。event 是事件对象，stream 是记录所在执行流，flags 控制记录行为。 |
| cuModuleLoadData                          | 248      | {"module_out":"0x7ffd8cd73f98","image":"0x3f6e1d10"}         | 从内存加载 CUDA module。module_out 返回模块句柄，image 指向模块二进制数据。 |
| cudaStreamWaitEvent                       | 1969180  | {"stream":"0x1fb9af00","event":"0x26cba970","flags":0}       | 让 stream 等待指定 event 完成。stream 是等待方，event 是依赖事件，flags 控制等待模式。 |
| cudaEventCreateWithFlags                  | 1781293  | {"event_out":"0x26cba7d8","flags":2}                         | 创建 CUDA event。event_out 返回事件句柄，flags 设置事件属性。 |
| cudaMalloc                                | 1742     | {"devPtr_out":"0x7ffd8cd71ee8","size":2097152}               | 分配 GPU 显存。devPtr_out 返回设备指针，size 为申请字节数。  |
| cudaEventRecord                           | 4895092  | {"event":"0x23acfd70","stream":"0x22721ab0"}                 | 向 stream 插入 event 记录操作。event 为事件对象，stream 为执行流。 |
| cuTensorMapEncodeTiled                    | 10635412 | {"tensorMap":"0x7ffd8cd72200","tensorDataType":9,"tensorRank":2,"globalAddress":"0x7e42a00000","globalDim":"0x7ffd8cd71dd0","globalStrides":"0x7ffd8cd71da0","boxDim":"0x7ffd8cd71dc8","elementStrides":"0x7ffd8cd71da8","interleave":0,"swiz... | 构造 Tensor Map 描述符，常用于 Hopper/Blackwell TMA。tensorDataType、tensorRank 描述数据类型和维度，globalAddress/globalDim/strides 描述张量布局。 |
| cudaStreamIsCapturing                     | 2177393  | {"stream":"(nil)","captureStatus_out":"0x7ffd8cd71de0"}      | 查询 stream 是否处于 CUDA Graph capture 状态。captureStatus_out 返回捕获状态。 |
| cudaEventDestroy                          | 1781059  | {"event":"0x4067f150"}                                       | 销毁 CUDA event 对象。event 为待释放事件句柄。               |
| cuMemMap                                  | 2476     | {"ptr":"0x7c7372e00000","size":2097152,"offset":0,"handle":649804304,"flags":0} | 将物理内存映射到 CUDA 虚拟地址空间。ptr 为地址，size 为映射大小，handle 为物理内存句柄。 |
| cuMemImportFromShareableHandle            | 512      | {"handle_out":"0x7ffd8cd72928","osHandle":"0xc0","shHandleType":1} | 从外部共享句柄导入 CUDA 内存。handle_out 返回 CUDA 内存句柄，osHandle 是操作系统共享句柄。 |
| cuKernelGetName                           | 3240154  | {"name_out":"0x7ffd8cd721d8","kernel":"0x268c7cb0"}          | 获取 kernel 名称。name_out 返回字符串，kernel 为 kernel 对象。 |
| cudaThreadExchangeStreamCaptureMode       | 388752   | {"mode_inout":"0x7ffd8cd72dfc"}                              | 修改线程级 stream capture 模式。mode_inout 保存输入输出模式。 |
| cudaGetFuncBySymbol                       | 192212   | {"functionPtr_out":"0x7ffd8cd72ae0","symbolPtr":"0x7c7776771ce0"} | 通过 CUDA symbol 获取 kernel/function 指针。functionPtr_out 返回函数地址。 |
| cudaGetDriverEntryPointByVersion          | 1109640  | {"symbol":"cuGetErrorString","funcPtr_out":"0x7c7781d0b478","cudaVersion":6000,"flags":0,"driverStatus_out":"0x7ffd8cd72dec"} | 参数字段描述：Sample params 展示该 CUDA API 调用时的关键参数，包括句柄、地址、尺寸、执行配置或控制选项，用于分析运行时行为。 |
| cuKernelGetAttribute                      | 1119560  | {"pi_out":"0x7ffd8cd72c48","attrib":0,"kernel":"0x22620000","dev":0} | 查询 kernel 属性。pi_out 返回属性值，attrib 指定属性类型，kernel 是目标 kernel。 |
| cudaStreamGetCaptureInfo                  | 192212   | {"stream":"0x2ba833f0","captureStatus_out":"0x7ffd8cd72dec","id_out":"0x7ffd8cd72e30","graph_out":"0x7ffd8cd72e28","dependencies_out":"(nil)","edgeData_out":"(nil)","numDependencies_out":"(nil)"} | 获取 stream capture 信息。包括捕获状态、graph ID、graph 对象以及依赖信息。 |
| cudaStreamCreateWithPriority              | 512      | {"pStream_out":"0x7c77f3dda560","flags":1,"priority":0}      | 创建带优先级 CUDA stream。pStream_out 返回 stream，flags 设置属性，priority 设置优先级。 |
| cudaIpcOpenMemHandle                      | 24       | {"devPtr_out":"0x7ffd8cd728d0","handle_ptr":"0x7ffd8cd72758","flags":1} | 打开跨进程共享 GPU 内存句柄。devPtr_out 返回设备地址，handle_ptr 是共享句柄。 |
| cudaFree                                  | 292      | {"devPtr":"(nil)"}                                           | 释放 GPU 内存。devPtr 是需要释放的设备地址。                 |
| cuMemAddressReserve                       | 2476     | {"ptr_out":"0x26cba360","size":2097152,"alignment":2097152,"addr":"0x0","flags":0} | 预留 CUDA 虚拟地址空间。ptr_out 返回地址，size 是范围大小，alignment 是对齐要求。 |
| cuKernelGetFunction                       | 958069   | {"pFunc_out":"0x7ffd8cd72a98","kernel":"0x22620000"}         | 从 kernel 对象获取函数句柄。pFunc_out 返回函数指针。         |
| cuLibraryLoadFromFile                     | 181      | {"library_out":"0x7ffd8cd722e0","fileName":"/root/.cache/deep_gemm/cache/kernel.transpose_and_pack_fp32_into_ue8m0.ec5749bfebd586ccf8e79219a5a84ced/kernel.cubin","jitOptions":"(nil)","jitOptionsValues":"(nil)","numJitOptions":0,"libraryO... | 从文件加载 CUDA library。fileName 是 cubin/library 路径，library_out 返回库对象。 |
| cuKernelSetAttribute                      | 765664   | {"attrib":8,"val":82240,"kernel":"0x22620000","dev":0}       | 设置 kernel 属性。attrib 指属性类型，val 是属性值，kernel 是目标 kernel。 |
| cudaHostAlloc                             | 120      | {"pHost_out":"0x2798bc30","size":4,"flags":2}                | 申请页锁定 Host 内存。pHost_out 返回主机地址，size 为大小，flags 控制属性。 |
| cudaMemsetAsync                           | 3720     | {"devPtr":"0x7c7372e00000","value":0,"count":2432,"stream":"0x1fb9af00"} | 异步设置 GPU 内存。devPtr 是目标地址，value 是填充值，count 是字节数量，stream 指执行流。 |
| cuLaunchKernel                            | 3052330  | {"func":"0x268c7cb0","grid":[1,1,1],"block":[128,1,1],"sharedMemBytes":0,"stream":"(nil)","kernelParams":"0x7ffd8cd72a30","extra":"(nil)"} | CUDA Driver API kernel 启动接口。func 为 kernel 函数，grid/block 定义启动尺寸，sharedMemBytes 为共享内存，kernelParams 或 extra 保存参数。 |
| cudaEventSynchronize                      | 4368     | {"event":"0x4e3c6750"}                                       | 等待指定 event 完成。event 是同步目标。                      |
| cudaMemPoolCreate                         | 8        | {"memPool_out":"0x2798d570","poolProps":"0x7ffd8cd72c70"}    | 创建 CUDA 内存池。memPool_out 返回内存池对象，poolProps 描述配置。 |
| cudaMemGetInfo                            | 50       | {"free_out":"0x7ffd8cd73718","total_out":"0x7ffd8cd73720"}   | 查询设备内存信息。free_out 返回空闲显存，total_out 返回总显存。 |
| cuGetProcAddress_v2                       | 1119722  | {"symbol":"cuMemcpy","pfn_out":"0x76b849d30610","cudaVersion":4000,"flags":0,"symbolStatus_out":"(nil)"} | 根据符号名称获取 CUDA Driver API 地址。symbol 是函数名，funcPtr_out 返回函数指针。 |
| cudaStreamCreateWithFlags                 | 20       | {"pStream_out":"0x26cba790","flags":1}                       | 创建 CUDA stream。pStream_out 返回 stream，flags 设置属性。  |
| cuMemRelease                              | 4        | {"handle":597593216}                                         | 释放 CUDA 虚拟内存分配句柄。handle 为待释放句柄。            |
| cuLibraryGetKernel                        | 1256     | {"pKernel_out":"0x7ffd8cd721e8","library":"0x26a20570","name":"_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIiEESt5arrayIPcLm1EEEEviT0_T1_"} | 从 library 中获取 kernel。pKernel_out 返回 kernel 对象，name 指定 kernel 名称。 |
| cudaKernelSetAttributeForDevice           | 64       | {"kernel":"0x627a5310","attr":8,"value":232448,"device":0}   | 针对指定设备设置 kernel 属性。kernel 为目标 kernel，attr/value 为属性和值。 |
| cuMemGetAllocationGranularity             | 2480     | {"granularity_out":"0x7ffd8cd72d88","prop":"0x7ffd8cd72da0","option":0} | 查询 CUDA 内存分配粒度。granularity_out 返回最小分配单位。   |
| cudaLibraryLoadData                       | 8        | {"library_out":"0x7ffd8cd71c28","code":"0x7c71aa005170","jitOptions":"(nil)","jitOptionsValues":"(nil)","numJitOptions":0,"libraryOptions":"(nil)","libraryOptionValues":"(nil)","numLibraryOptions":0} | Runtime API 版本的 library 加载接口。code 指向代码数据，library_out 返回对象。 |
| cuModuleLoad                              | 4        | {"module_out":"0x7ffd8cd6e238","fname":"/tmp/torchinductor_root/triton/0/W3SLRO5SZG6QCNW7YIXHGQOEVKPQ5ULTWNRU5AD72U27KPXNX2AA/triton_poi_fused_add_bitwise_and_bitwise_not_bitwise_or_ge_lt_mul_sub_0.cubin"} | 从文件加载 CUDA module。fname 是 cubin/PTX 文件路径。        |
| cudaMemcpy                                | 12       | {"dst":"0x7c7318004c80","src":"0x7ffd8cd72910","count":64,"kind":1,"kindName":"cudaMemcpyHostToDevice"} | 同步内存拷贝。dst/src 是目标和源地址，count 是字节数，kind 指方向。 |
| cudaMemset                                | 12       | {"devPtr":"0x7c7318000000","value":0,"count":18944}          | 同步设置 GPU 内存。devPtr 是地址，value 是填充值，count 是字节数。 |
| cudaDeviceSynchronize                     | 0        | {}                                                           | 等待整个 CUDA device 上所有任务完成。无参数表示全设备同步。  |
| cudaOccupancyMaxActiveClusters            | 120      | {"numClusters_out":"0x7ffd8cd72a04","func":"0x7c77cf32fb30","launchConfig":"0x7ffd8cd72a10"} | 查询 cluster launch 最大活动 cluster 数。func 为 kernel，launchConfig 描述配置。 |
| cuLibraryGetKernelCount                   | 181      | {"count_out":"0x7ffd8cd722d8","library":"0x43294b60"}        | 查询 library 中 kernel 数量。count_out 返回数量。            |
| cuInit                                    | 28       | {"Flags":0}                                                  | 初始化 CUDA Driver。Flags 控制初始化选项。                   |
| cudaGetDeviceProperties                   | 96       | {"prop_out":"0x2c5c9b40","device":0}                         | 获取 GPU 属性。prop_out 返回设备信息，device 是设备编号。    |
| cudaOccupancyAvailableDynamicSMemPerBlock | 4        | {"dynamicSmemSize_out":"0x7ffd8cd72a08","func":"0x7c77cf32fb30","numBlocks":1,"blockSize":1} | 参数字段描述：Sample params 展示该 CUDA API 调用时的关键参数，包括句柄、地址、尺寸、执行配置或控制选项，用于分析运行时行为。 |
| cuLibraryEnumerateKernels                 | 181      | {"kernels_out":"0x7ffd8cd722f0","numKernels":1,"library":"0x43294b60"} | 枚举 library 中所有 kernel。kernels_out 返回列表。           |
| cudaMemPoolSetAttribute                   | 8        | {"memPool":"0x23907110","attr":4,"value":"0x7ffd8cd72c58"}   | 设置 CUDA memory pool 属性。memPool 是目标池，attr/value 设置属性。 |
| cuModuleGetLoadingMode                    | 26       | {"mode_out":"0x7ffc2d3e23cc"}                                | 查询 module 加载模式。mode_out 返回模式。                    |
| cudaLibraryGetKernel                      | 32       | {"pKernel_out":"0x7c71aa00a000","library":"0x6247d990","name":"kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrbf16gmemalign16o1024i64div81_tensorptrbf16gmemalign16o10241_tensorptrbf16gmemalign16o1024i... | 从 runtime library 获取 kernel。pKernel_out 返回 kernel，name 为 kernel 名称。 |