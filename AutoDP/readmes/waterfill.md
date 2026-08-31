# 参考
- deepseekv4+deepep+waterfill：https://github.com/sgl-project/sglang/pull/25391
- megamoe+waterfill：https://github.com/sgl-project/sglang/pull/27350
- LMSYS博客：https://www.lmsys.org/blog/2026-06-26-waterfill-lplb
- eplb+waterfill tips：https://github.com/sgl-project/sglang/pull/27049/
- dynamic eplb+waterfill：https://github.com/sgl-project/sglang/pull/27150
- DSV4 shared expert fusion for DeepEP and MegaMOE：https://github.com/sgl-project/sglang/pull/27349
- DeepSeek-V4 EPLB Waterfill tips- #27049：https://github.com/sgl-project/sglang/pull/27049

# 参考实现
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
python3 -m sglang.launch_server --model-path /data/ssd2/checkpoints/deepseek-ai/DeepSeek-V4-Pro \
--served-model-name deepseek-ai/DeepSeek-V4-Pro \
--trust-remote-code \
--tool-call-parser deepseekv4 \
--host 0.0.0.0 \
--port 30001 \
--tp-size 8 \
--dp-size 8 \
--ep-size 8 \
--enable-dp-attention \
--disable-flashinfer-autotune \
--mem-fraction-static 0.9 \
--swa-full-tokens-ratio 0.1 \
--moe-a2a-backend deepep \
--max-running-requests 1024 \
--deepep-mode auto \
--deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}' \
--enable-waterfill
