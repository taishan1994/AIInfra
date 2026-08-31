| 功能                        | DeepSeek-V4-Pro | 启动参数                            | 主要依赖/用途                                                                             |
| ------------------------- | --------------- | ------------------------------- | ----------------------------------------------------------------------------------- |
| TBO（Two-Batch Overlap）    | **支持**          | `--enable-two-batch-overlap`    | 主要配合 EP + DeepEP，将一个 batch 拆成两个 micro-batch，重叠 Attention、MoE Dispatch/Combine 与专家计算 |
| SBO（Single-Batch Overlap） | **支持**          | `--enable-single-batch-overlap` | 在单个 batch 内重叠共享专家计算和 DeepEP 通信，更适合低延迟或 batch 较小时                                    |

帮我做TBO和SBO的消融对比实验。