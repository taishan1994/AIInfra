目前未成功完成的：

1. prefill+hicache失败
失败点不是参数解析或权重加载；prefill 已进入 HiCache / Unified RadixTree 初始化并开始分配 host KV pool，随后 rank0 scheduler 被 SIGKILL：
RuntimeError: Rank 0 scheduler died during initialization (exit code: -9).
按日志，hicache_ratio=2.0 下每个 TP rank 分配：
- swa pool：81.93 GB host memory
- deepseek_v4_c4 pool：100.03 GB host memory
TP=4 合计约 4 * (81.93 + 100.03) = 727.84 GB host memory，仅 HiCache host pool 部分就接近 728GB。失败时日志给出的 exit code: -9 与 OS OOM killer / host memory pressure 表现一致。
后续可单独降低 host cache 配置重试，例如降低 --hicache-ratio 或改用显式 --hicache-size。

2. decode+dspark失败
DSpark 正式性能实验应先准备 profiled SPS table；否则 DSpark 的 SPS 调度收益无法体现。
当前 PD 分离下 DSpark 崩溃不是 SPS table 缺失直接导致。缺 SPS table 时，代码路径应退化为 flat / verify-all，表现为没有调度收益，而不是请求崩溃。
当前失败点是 PD decode 路径中首个请求进入 decode 时 batch.spec_info 为 None，但 DFlash-family/DSpark 的 spec_prepare_for_decode() 直接调用 batch.spec_info.prepare_for_decode(batch)，触发：
AttributeError: 'NoneType' object has no attribute 'prepare_for_decode

3. 没有使用decode_baseline+hispark，而是使用了megamoe+hispark

4. lplb失败，这个已经明确了deepseek-v4是不支持的。

5. 目前已跑的megamoe+waterfill是不如单单使用megamoe的，没有体现出waterfill的效果，而从waterfill的lmsys报告来看，其无论是单独使用，还是结合eplb使用，都是有效的。
Configuration	Baseline	Waterfill	Gain
No EPLB	45,951 tok/s	47,876 tok/s	+4.19%
Static EPLB, red0	49,253 tok/s	51,677 tok/s	+4.92%
Static EPLB, red16	50,006 tok/s	51,655 tok/s	+3.30%
Static EPLB, red32	50,167 tok/s	51,813 tok/s	+3.28%

6. deocdeo+tbo或sbo
我看你decode使用的是normal模式的deepep，但是decode一般使用的是low_latency，这个你需要再验证一下。

7. decode+mtp
最终结果明显有问题，total throughput异常偏大，而且我发现日志里面有报错，重新尝试。

8. decode+fp4_indexer
8192输入下的case并没有全部完成，你遇到了问题，请分析一下是什么问题，并且解决掉它。