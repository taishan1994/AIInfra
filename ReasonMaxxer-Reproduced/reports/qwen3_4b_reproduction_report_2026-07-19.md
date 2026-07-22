# ReasonMaxxer 基于 Qwen3-4B 的复现报告

日期：2026-07-19

## 1. 复现目标

本次工作的目标是：

- 在本地 Qwen3-4B 权重上完整跑通 ReasonMaxxer 训练流程；
- 从采样、生成、打分、筛选、训练、holdout 选 checkpoint，到公开测试集评测全部复现；
- 比较训练前基座模型与训练后最佳 checkpoint 在公开测试集上的表现差异。

目标模型：

- 基座模型：`/nfs/FM/gongoubo/checkpoints/Qwen/Qwen3-4B`
- 训练运行名：`qwen3_4b_reasonmaxxer_tau1p4_r32_lr1e4_ep2_seed42`

最终判断标准：

- 训练后模型是否在 `math500`、`gsm8k`、`AIME 2024`、`AIME 2025` 上相对训练前模型取得提升。

## 2. 环境与资源

工作目录：

- `/nfs/FM/gongoubo/new_project/github/ReasonMaxxer`

硬件环境：

- 4 x NVIDIA A800-SXM4-80GB

关键本地资源：

- Qwen3-4B 权重：`/nfs/FM/gongoubo/checkpoints/Qwen/Qwen3-4B`
- SimpleRL 训练数据：
  `/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/data/external/simpleRL/simplelr_abel_level3to5/train.parquet`

为避免远端下载不稳定，在复现过程中额外构建了本地 benchmark 缓存：

- [math500_cached.json](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/data/math500_cached.json)
- [gsm8k_cached.json](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/data/gsm8k_cached.json)
- [aime24.json](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/data/benchmarks/aime24.json)
- [aime25.json](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/data/benchmarks/aime25.json)

## 3. 为复现做的代码修改

### 3.1 新增 Qwen3-4B 实验脚本

在 [examples/qwen3_4b](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/examples/qwen3_4b) 下新增：

- `01_sample_300.sh`
- `02_generate_score_3x100x20.sh`
- `03_select_mid50_trim80.sh`
- `04_train_tau1p4.sh`
- `05_make_holdout60.sh`
- `06_eval_holdout60.sh`
- `07_eval_fullsuite.sh`
- `08_eval_base_fullsuite.sh`

这些脚本对应完整实验流水线：

1. 采样 300 道题
2. 生成 3 路 rollout
3. 熵打分
4. 中段难度筛选
5. 训练 LoRA
6. holdout 选 checkpoint
7. 训练后公开测试集评测
8. 训练前公开测试集评测

### 3.2 新增工具脚本

新增：

- [make_holdout_split.py](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/scripts/make_holdout_split.py)
- [audit_rollouts.py](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/scripts/audit_rollouts.py)
- [summarize_eval_results.py](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/scripts/summarize_eval_results.py)
- [compare_eval_runs.py](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/scripts/compare_eval_runs.py)

用途分别为：

- 构造 holdout60；
- 审计 rollout 数据质量；
- 将评测 JSON 汇总成 pass@1 表格；
- 自动生成训练前后对比表。

### 3.3 修改的核心文件

修改了 [reasonmaxxer/eval_lib.py](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/reasonmaxxer/eval_lib.py)：

- 支持 `VLLM_ENFORCE_EAGER`
- 默认设置：
  - `HF_HUB_DISABLE_XET=1`
  - `HF_ENDPOINT=https://hf-mirror.com`
- 对 `math500` 与 `gsm8k` 采用“本地缓存优先”加载逻辑

修改了 [reasonmaxxer/config.py](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/reasonmaxxer/config.py)：

- 注册 `math500`、`gsm8k`
- 增加 `aime25`
- 将 `aime24`、`aime25`、`math500`、`gsm8k` 纳入本地 benchmark 文件映射

修改了 [reasonmaxxer/answer_extraction.py](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/reasonmaxxer/answer_extraction.py)：

- 让 `aime25` 复用 `aime24` 的整数答案抽取逻辑

修改了 [scripts/generate_rollouts.py](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/scripts/generate_rollouts.py)：

- 增加 `aime25` 作为支持数据集

修改了 [scripts/eval_checkpoints.py](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/scripts/eval_checkpoints.py)：

- 修正早期无效参数问题
- 支持更新 checkpoint 指标 CSV

修改了 [scripts/summarize_eval_results.py](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/scripts/summarize_eval_results.py)：

- 增加 `aime25` 数据集识别

修改了评测脚本：

- [06_eval_holdout60.sh](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/examples/qwen3_4b/06_eval_holdout60.sh)
- [07_eval_fullsuite.sh](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/examples/qwen3_4b/07_eval_fullsuite.sh)
- [08_eval_base_fullsuite.sh](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/examples/qwen3_4b/08_eval_base_fullsuite.sh)

主要变化：

- 统一支持 eager 模式
- 支持参数化 `DATASETS`、`BATCH_SIZE`、`MAX_TOKENS`、`SEED`
- holdout 改为基于本地 sampled records 评测

## 4. 复现流程

### 4.1 从 SimpleRL 训练集采样 300 题

执行：

- [01_sample_300.sh](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/examples/qwen3_4b/01_sample_300.sh)

输出目录：

- [outputs/qwen3_4b_default/sampled_records](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/sampled_records)

关键文件：

- [records_l345_300.json](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/sampled_records/records_l345_300.json)

采样策略：

- level 3：100 题
- level 4：100 题
- level 5：100 题
- 合计：300 题

### 4.2 生成 3 x 100 x 20 rollout

执行：

- [02_generate_score_3x100x20.sh](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/examples/qwen3_4b/02_generate_score_3x100x20.sh)

设置：

- 3 个 shard
- 每个 shard 100 题
- 每题 20 个 generation

输出：

- `outputs/qwen3_4b_default/gen/qwen3_4b_shard0_n20.json`
- `outputs/qwen3_4b_default/gen/qwen3_4b_shard1_n20.json`
- `outputs/qwen3_4b_default/gen/qwen3_4b_shard2_n20.json`

对应 entropy 打分输出位于：

- `outputs/qwen3_4b_default/score/`

### 4.3 rollout 数据审计

执行：

- [audit_rollouts.py](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/scripts/audit_rollouts.py)

审计结果：

- 总生成数：`6000`
- 总体正确率：约 `0.7332`
- `null_extract = 61`
- `empty_text = 0`
- `long_text_count = 113`

分 shard 观察：

- shard0 正确率：约 `0.8255`
- shard1 正确率：约 `0.8095`
- shard2 正确率：约 `0.5645`

结论：

- 主链路输出结构是正确的
- 主要问题不是模板损坏，而是少量题目出现长推理发散
- 由于后续中段筛选会裁掉部分尾部样本，因此没有为此重跑生成

### 4.4 中段难度筛选

执行：

- [03_select_mid50_trim80.sh](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/examples/qwen3_4b/03_select_mid50_trim80.sh)

输出：

- [selected_problem_ids.json](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/selection/selected_problem_ids.json)
- [selected_rollouts_trim80_entropy.json](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/selection/selected_rollouts_trim80_entropy.json)

筛选结果：

- 入选问题数：`50`
- 保留 rollout 数：`800`

### 4.5 训练数据准备与训练

执行：

- [04_train_tau1p4.sh](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/examples/qwen3_4b/04_train_tau1p4.sh)

训练数据输出：

- [target_ids_tau1p4.json](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/train/target_ids_tau1p4.json)
- [processed_tau1p4.json](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/train/processed_tau1p4.json)
- [training_examples_tau1p4.json](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/train/training_examples_tau1p4.json)
- [training_stats_tau1p4.json](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/train/training_stats_tau1p4.json)

训练数据统计：

- `n_target_problems = 49`
- `n_rollouts_target = 800`
- 标签数量：
  - `+1 = 466`
  - `-1 = 334`
- 每条 rollout 的平均 decision positions：`25.545`
- 平均 decision fraction：`0.02318`
- 平均 decision entropy：`1.7708`

模型训练信息：

- 可训练参数量：约 `23.59M`
- 可训练参数占比：约 `0.5831%`

checkpoint 目录：

- [qwen3_4b_reasonmaxxer_tau1p4_r32_lr1e4_ep2_seed42](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/checkpoints/qwen3_4b_reasonmaxxer_tau1p4_r32_lr1e4_ep2_seed42)

保存的关键 checkpoint：

- `epochf_0p15`
- `epochf_0p3`
- `epochf_0p45`
- `epochf_0p6`
- `epochf_0p75`
- `epochf_0p9`
- `epoch_1`
- `epochf_1p05`
- `epochf_1p2`
- `epochf_1p35`
- `epoch_2`
- `final`

### 4.6 构造 holdout60 并选择最佳 checkpoint

执行：

- [05_make_holdout60.sh](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/examples/qwen3_4b/05_make_holdout60.sh)
- [06_eval_holdout60.sh](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/examples/qwen3_4b/06_eval_holdout60.sh)

关键输出：

- [holdout60_ids_tau1p4.json](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/train/holdout60_ids_tau1p4.json)
- [holdout60_summary.csv](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/eval/qwen3_4b_reasonmaxxer_tau1p4_r32_lr1e4_ep2_seed42_holdout60_n1/holdout60_summary.csv)
- [checkpoint_metrics.with_holdout60.csv](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/checkpoints/qwen3_4b_reasonmaxxer_tau1p4_r32_lr1e4_ep2_seed42/checkpoint_metrics.with_holdout60.csv)

holdout60 结果：

- `epochf_0p15`: `0.7167`
- `epochf_0p3`: `0.8000`
- `epochf_0p45`: `0.8000`
- `epochf_0p6`: `0.6833`
- `epochf_0p75`: `0.7500`
- `epochf_0p9`: `0.7833`
- `epoch_1`: `0.7333`
- `epochf_1p05`: `0.7667`
- `epochf_1p2`: `0.6833`
- `epochf_1p35`: `0.6167`
- `epoch_2`: `0.6000`

最佳 checkpoint：

- `epochf_0p3` 与 `epochf_0p45` 并列第一，都是 `0.8000`
- 最终选择更早的 [epochf_0p3](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/checkpoints/qwen3_4b_reasonmaxxer_tau1p4_r32_lr1e4_ep2_seed42/epochf_0p3)，以减少后期训练漂移风险

## 5. 公开测试集评测

### 5.1 评测设置

训练后模型评测脚本：

- [07_eval_fullsuite.sh](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/examples/qwen3_4b/07_eval_fullsuite.sh)

训练前基座模型评测脚本：

- [08_eval_base_fullsuite.sh](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/examples/qwen3_4b/08_eval_base_fullsuite.sh)

统一解码配置：

- `num_generations = 1`
- `temperature = 0.6`
- `top_p = 0.95`
- `max_tokens = 8192`
- `prompt_style = auto`
- `stop_profile = auto`
- `qwen3_enable_thinking = false`

### 5.2 math500 与 gsm8k 对比

汇总文件：

- 训练后：
  [trained_summary.csv](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/final_eval/trained_summary.csv)
- 训练前：
  [base_summary.csv](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/base_eval/base_summary.csv)
- 对比：
  [base_vs_trained_summary.csv](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/compare/base_vs_trained_summary.csv)

最终结果：

| 数据集 | 训练前 pass@1 | 训练后 pass@1 | 提升 |
|---|---:|---:|---:|
| `math500` | `0.7760` | `0.7940` | `+0.0180` |
| `gsm8k` | `0.8438` | `0.8522` | `+0.0083` |

原始结果文件：

- 训练后 `math500`：
  [epochf_0p3_math500.json](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/final_eval/epochf_0p3_math500.json)
- 训练后 `gsm8k`：
  [epochf_0p3_gsm8k.json](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/final_eval/epochf_0p3_gsm8k.json)
- 训练前 `math500`：
  [qwen3_4b_base_math500.json](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/base_eval/qwen3_4b_base_math500.json)
- 训练前 `gsm8k`：
  [qwen3_4b_base_gsm8k.json](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/base_eval/qwen3_4b_base_gsm8k.json)

### 5.3 AIME 2024 与 AIME 2025 对比

在本次报告补充阶段，新增了本地 benchmark：

- [aime24.json](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/data/benchmarks/aime24.json)
- [aime25.json](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/data/benchmarks/aime25.json)

汇总文件：

- 训练后：
  [trained_aime_summary.csv](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/aime_eval/trained_aime_summary.csv)
- 训练前：
  [base_aime_summary.csv](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/aime_eval/base_aime_summary.csv)
- 对比：
  [base_vs_trained_aime_summary.csv](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/aime_eval/base_vs_trained_aime_summary.csv)

最终结果：

| 数据集 | 训练前 pass@1 | 训练后 pass@1 | 提升 |
|---|---:|---:|---:|
| `AIME 2024` | `0.2000` | `0.2333` | `+0.0333` |
| `AIME 2025` | `0.1667` | `0.2667` | `+0.1000` |

补充 `avg@8` 结果：

这里的 `avg@8` 定义为：每题进行 `8` 次独立生成，先计算该题 `8` 次生成中的正确率，再对全部题目求平均。

| 数据集 | 训练前 avg@8 | 训练后 avg@8 | 提升 |
|---|---:|---:|---:|
| `AIME 2024` | `0.1958` | `0.2333` | `+0.0375` |
| `AIME 2025` | `0.1750` | `0.1792` | `+0.0042` |

等价题数：

- `AIME 2024`：
  - 训练前 `6/30`
  - 训练后 `7/30`
- `AIME 2025`：
  - 训练前 `5/30`
  - 训练后 `8/30`

原始结果文件：

- 训练后 `AIME 2024`：
  [epochf_0p3_aime24.json](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/aime_eval/trained/epochf_0p3_aime24.json)
- 训练后 `AIME 2025`：
  [epochf_0p3_aime25.json](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/aime_eval/trained/epochf_0p3_aime25.json)
- 训练前 `AIME 2024`：
  [qwen3_4b_base_aime24.json](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/aime_eval/base/qwen3_4b_base_aime24.json)
- 训练前 `AIME 2025`：
  [qwen3_4b_base_aime25.json](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/aime_eval/base/qwen3_4b_base_aime25.json)

`avg@8` 原始结果文件：

- 训练后 `AIME 2024`：
  [epochf_0p3_aime24_n8.json](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/aime_eval/trained/epochf_0p3_aime24_n8.json)
- 训练后 `AIME 2025`：
  [epochf_0p3_aime25_n8.json](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/aime_eval/trained/epochf_0p3_aime25_n8.json)
- 训练前 `AIME 2024`：
  [qwen3_4b_base_aime24_n8.json](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/aime_eval/base/qwen3_4b_base_aime24_n8.json)
- 训练前 `AIME 2025`：
  [qwen3_4b_base_aime25_n8.json](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/aime_eval/base/qwen3_4b_base_aime25_n8.json)

## 6. 结果一致性检查

对最终结果做了完整性检查。

### 6.1 math500 / gsm8k

- 训练后 `math500`：`500` 行，`500` 个唯一 problem_id
- 训练前 `math500`：`500` 行，`500` 个唯一 problem_id
- 训练后 `gsm8k`：`1319` 行，`1319` 个唯一 problem_id
- 训练前 `gsm8k`：`1319` 行，`1319` 个唯一 problem_id

每题都是 `1` 次 generation，没有重复题目。

答案提取失败数量：

- 训练后 `math500`：`2`
- 训练前 `math500`：`3`
- 训练后 `gsm8k`：`0`
- 训练前 `gsm8k`：`0`

### 6.2 AIME 2024 / 2025

- 训练后 `AIME 2024`：`30` 行，`30` 个唯一 problem_id
- 训练前 `AIME 2024`：`30` 行，`30` 个唯一 problem_id
- 训练后 `AIME 2025`：`30` 行，`30` 个唯一 problem_id
- 训练前 `AIME 2025`：`30` 行，`30` 个唯一 problem_id

每题都是 `1` 次 generation，没有重复题目。

答案提取失败数量：

- 训练后 `AIME 2024`：`0`
- 训练前 `AIME 2024`：`0`
- 训练后 `AIME 2025`：`0`
- 训练前 `AIME 2025`：`0`

结论：

- 最终对比结果没有缺题、重复题或半写文件的问题
- 汇总数字与原始 JSON 一致

## 7. 复现中遇到的主要问题与修复

### 7.1 vLLM compile cache / 并行冷启动不稳定

问题：

- `vLLM 0.9.2 + torch.compile` 在并行生成时会出现缓存不稳定或冷启动异常

修复：

- 在 [eval_lib.py](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/reasonmaxxer/eval_lib.py) 中加入 eager 模式支持
- 在长流程脚本中启用 `VLLM_ENFORCE_EAGER=1`

影响：

- 吞吐略低
- 但稳定性明显提高，适合长作业复现

### 7.2 Hugging Face Xet / CAS 下载失败

问题：

- `math500`、`gsm8k` 等数据集在当前环境下会随机命中 `xet/CAS` 下载错误

修复：

- 默认设置：
  - `HF_HUB_DISABLE_XET=1`
  - `HF_ENDPOINT=https://hf-mirror.com`
- 将公开 benchmark 落为本地 JSON 缓存

影响：

- 后续所有评测都不再依赖不稳定远端下载

### 7.3 holdout ID 命名空间不一致

问题：

- `holdout60_ids_tau1p4.json` 来自 sampled SimpleRL 的 `problem_id`
- 早期版本 holdout 评测却直接去过滤远端 `math500`
- 导致过滤结果为 0 条

修复：

- `holdout60` 改为基于本地 [records_l345_300.json](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/sampled_records/records_l345_300.json) 做评测

影响：

- holdout 评测语义正确，checkpoint 选择有效

### 7.4 AIME 2025 原始仓库未接入

问题：

- 仓库原生只支持 `aime24`，不支持 `aime25`

修复：

- 增加 `aime25` 到配置、答案抽取和 rollout 脚本
- 使用公开数据源构建本地 [aime25.json](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/data/benchmarks/aime25.json)

影响：

- 可以直接进行 `AIME 2024/2025` 的训练前后同口径对比

## 8. 总结

从这次复现的最终结果看，ReasonMaxxer 在本地 Qwen3-4B 上是有效的，但提升幅度偏“稳定小幅提升”，而不是非常激进的大涨。

最终采用的训练后模型：

- [epochf_0p3](/nfs/FM/gongoubo/new_project/github/ReasonMaxxer/outputs/qwen3_4b_default/checkpoints/qwen3_4b_reasonmaxxer_tau1p4_r32_lr1e4_ep2_seed42/epochf_0p3)

训练前后对比结果总结：

- `math500`：`0.7760 -> 0.7940`，提升 `+0.0180`
- `gsm8k`：`0.8438 -> 0.8522`，提升 `+0.0083`
- `AIME 2024`：`0.2000 -> 0.2333`，提升 `+0.0333`
- `AIME 2025`：`0.1667 -> 0.2667`，提升 `+0.1000`
- `AIME 2024 avg@8`：`0.1958 -> 0.2333`，提升 `+0.0375`
- `AIME 2025 avg@8`：`0.1750 -> 0.1792`，提升 `+0.0042`

整体判断：

- 训练后的 checkpoint 在四个评测集上都优于训练前基座模型；
- 对 `AIME 2025` 的提升最明显；
- 对 `math500`、`gsm8k` 的提升较小但稳定；
- 从工程角度看，本次复现的主要工作量不在训练本身，而在于让生成、评测、benchmark 加载在本地环境下稳定可重复。
