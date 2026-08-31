# 总体目标
【什么机器】上【可选：基本配置，可以是1P1D】 PD分离部署【什么模型】

# 基础管径
机器列表：
【机器1】(本机)：xxxx 密码：xxxx 容器：xxxx 
【机器2】：xxxx 密码：xxxx 容器：xxxx

【在4卡上部署prefill，在4卡上部署decode】

【baseline】
prefill：
```shell
PYTHONUNBUFFERED=1 \
SGLANG_DEFAULT_THINKING=1 \
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
SGLANG_DSV4_REASONING_EFFORT=max \
SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1 \
SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN=1 \
SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW=1 \
SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1 \
SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0 \
SGLANG_RADIX_FORCE_MISS=1 \
python3 -m sglang.launch_server --model-path 模型地址 \
--served-model-name 模型名称 \
--trust-remote-code \
--host 0.0.0.0 \
--port 30000 \
--tp-size 4 \
--dp-size 1 \
--ep-size 1 \
--disaggregation-mode prefill \
--disaggregation-transfer-backend mooncake \
--moe-runner-backend flashinfer_mxfp4 \
--disable-flashinfer-autotune \
--mem-fraction-static 0.9 \
--swa-full-tokens-ratio 0.1 \
--disable-radix-cache
```

decode：
```shell
PYTHONUNBUFFERED=1 \
SGLANG_DEFAULT_THINKING=1 \
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
SGLANG_DSV4_REASONING_EFFORT=max \
SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1 \
SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN=1 \
SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW=1 \
SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1 \
SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0 \
SGLANG_RADIX_FORCE_MISS=1 \
python3 -m sglang.launch_server --model-path 模型地址 \
--served-model-name 模型名称 \
--trust-remote-code \
--host 0.0.0.0 \
--port 30001 \
--tp-size 4 \
--dp-size 4 \
--ep-size 4 \
--enable-dp-attention \
--disable-flashinfer-autotune \
--mem-fraction-static 0.9 \
--swa-full-tokens-ratio 0.1 \
--disaggregation-mode decode \
--disaggregation-transfer-backend mooncake \
--moe-a2a-backend deepep \
--max-running-requests 1024 \
--deepep-mode low_latency \
--deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}' \
```

router：
```shell
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://10.1.17.13:30000 \
  --decode http://10.1.17.13:30001 \
  --host 0.0.0.0 --port 13784 \
  --disable-circuit-breaker \
  --health-check-interval-secs 999999
```

# 详细目标
prefill需要尝试的技术：
- megamoe
- deepep：注意ep和tp应该保持一致
- hicache

decode需要尝试的技术：
- dspark
- megamoe 
- waterfill
- lplp
- tbo或者sbo
- mtp 
- hisparse
- fp4-indexer

## 特别注意
- 每次实验之前，先请求一条问“你是谁？”来确认服务没有问题。
- 每一个技术都在readmes下面存放着说明markdown，你可以参考。你先完成baseline的测试，再逐步进行尝试。
- 你要先对每一个技术单独验证，最后再将他们整合到一起。
- decode 必须使用cuda graph。
- PRONLEM.MD和SOLUTION.MD是一些遇到的问题和解决方法，你可以进行参考。
- refrence下是一些经验，你也可以参考。

# 测试benchmark

使用InferenceX进行测试：

测试配置如下：
ISL	OSL	Concurrency  
1024	1024	1
1024	1024	16
1024	1024	256
1024	1024	512
8192	1024	1
8192	1024	16
8192	1024	256
8192	1024	512

num_prompts为Concurrency*10。

# 输出配置
- 每完成一项就需要总结车工报告。注意：如果遇到环境和部署问题，自行根据参考资料和sglang源码帮我解决。
- 报告中要记录部署的参数、指令，服务的日志，测试的结果。如果遇到问题，要记录问题和解决方法，并进行汇总。这些日志要存储到每一次单独实验的文件夹中，而不是只复用一个启动命令bash。  
- 按日期、任务类型存放文件。

以下是是一个可参考的目录结构例子：
├── readmes/
├──├── reference/
├──├── deepseek-v4
├──├── hy3
├── logs/
├── index.md
├──├── xxxx/xx/xx/
├──├──├── opt_name/
├──├──├──├── prefill/
├──├──├──├── decode/
├──├──├──├── router/
├──├──├──├── results/
├──├──├──├── summary.md
├── AGENTS.MD
├── README.MD

prefill/decode/router下面记录的是服务部署的日志
resuls/下面记录测试的结果，包括你是谁验证以及InferenceX的测试结果
sumamry.md 里面要记录部署的具体指令，包括环境变量，以及每轮优化的过程和结果
index.md是所有summary.md的索引，简单总结了每一轮的实验结果，可快速追溯
readmes/下面除了有每一个技术科参考的，你也可以新建模型文件夹，在其下面记录实验记录以及总结经验