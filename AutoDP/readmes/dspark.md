# 参考
- LMSYS博客：https://www.lmsys.org/blog/2026-07-06-dspark-sglang
- deepseek-v4 PD分离使用DSpark：https://github.com/sgl-project/sglang/pull/31466，https://github.com/sgl-project/sglang/pull/33411
- b200上pd分离，dspark结合hicache使用：https://github.com/sgl-project/sglang/pull/33204
- PP+PD+Dspark：https://github.com/sgl-project/sglang/pull/33204，https://github.com/sgl-project/sglang/pull/32793
- dspark支持deepep和deepgemm：https://github.com/sgl-project/sglang/pull/31868，https://github.com/sgl-project/sglang/pull/30513
- Fix DSpark and DP/EP：https://github.com/sgl-project/sglang/pull/33098
- dspark loadmap：https://github.com/sgl-project/sglang/issues/30344

reference/下可能有改技术的成功案例，你可以参考。

# 可选
如果以下已经做了，跳过即可：

不包含dspark的权重：cd /data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Pro 里面包含了deepseek-v4-pro+mtp的权重，mtp主要是第64个safetensor。

另外，我下载了dspark的权重，在/data/ssd1/checkpoints/deepseek-ai/DeepSeek-V4-Pro/dspark下。

你先建一个DeepSeek-V4-Pro-dpsark，然后deepseek-v4-pro其余文件和配置就用软连接，帮我新建一个model.safetensors.index.json，里面删除掉mtp的权重配置，并加入dpsark的权重配置。

最后参考链接，帮我完成单机B200部署dspark的对比实验（使用和不使用）。

同理对于：/data/ssd2/checkpoints/deepseek-ai/DeepSeek-V4-Flash





