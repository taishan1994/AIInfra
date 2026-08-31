# 将 Qwen3.8-27B 适配到 DeepSeek Harness：从部署到多模态验证

## 引言

DeepSeek Harness 支持通过 OpenAI-compatible API 接入外部大模型。本文记录将 Qwen3.8-27B 部署到 SGLang，并接入 DeepSeek Harness 的完整过程。

这次适配并不是简单修改一个模型名称，过程中还遇到了两个实际问题：

1. Qwen 服务运行一段时间后出现 KV Cache OOM，导致回答输出到一半中断；
2. Qwen 虽然是多模态模型，但 Harness 默认只把模型声明为文本模型，页面无法正常发送图片。

最终，我们完成了 Qwen3.8-27B 的文本、思考、工具调用和图片输入适配。

## 一、整体架构

完整调用链路如下：

~~~text
浏览器
  ↓ WebSocket
DeepSeek Harness Web
  ↓ session.prompt
Host API Proxy
  ↓
llm-pi-ai
  ↓ OpenAI-compatible HTTP API
SGLang
  ↓
Qwen3.8-27B
~~~

图片请求的链路稍有不同：

~~~text
浏览器上传图片
  ↓
Harness 转换为 base64 image content
  ↓
session.prompt
  ↓
OpenAI-compatible multimodal message
  ↓
Qwen3.8-27B 图像理解
~~~

## 二、SGLang 部署

Qwen 服务部署在：

~~~text
服务器：192.168.16.19
端口：30000
模型路径：/nfs/FM/checkpoints/Qwen/Qwen3.8-27B
~~~

最初使用的启动命令如下：

~~~bash
sglang serve \
  --trust-remote-code \
  --model-path /nfs/FM/checkpoints/Qwen/Qwen3.8-27B \
  --served-model-name Qwen3.8-27B \
  --context-length 260000 \
  --kv-cache-dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 2048 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --mamba-full-memory-ratio 4.13 \
  --host 0.0.0.0 \
  --port 30000 \
  --api-key '替换为真实密钥'
~~~

参数作用：

| 参数 | 作用 |
|---|---|
| trust-remote-code | 允许加载模型自定义代码 |
| model-path | 本地模型目录 |
| served-model-name | API 中使用的模型名称 |
| context-length | 理论最大上下文长度 |
| kv-cache-dtype | KV Cache 数据类型 |
| mem-fraction-static | 用于模型和缓存的显存比例 |
| reasoning-parser | 解析 Qwen 思考内容 |
| tool-call-parser | 解析工具调用 |
| speculative-algorithm | 启用推测解码 |
| mamba-full-memory-ratio | Mamba 状态池与 KV Cache 的内存比例 |
| host/port | 服务监听地址和端口 |

## 三、接入 Harness

在 /root/.dsh/settings.yaml 中加入 Qwen Provider：

~~~yaml
llm-pi-ai:
  providers:
    qwen38:
      displayName: qwen
      apiKeyEnv: QWEN38_API_KEY
      api: openai-completions
      baseURL: http://192.168.16.19:30000/v1
      streamDelayMs: 25
      models:
        - id: Qwen3.8-27B
          name: Qwen3.8-27B
          contextWindow: 28672
          maxTokens: 2048
          input:
            - text
            - image
~~~

密钥通过环境变量或受保护的凭证文件保存：

~~~bash
export QWEN38_API_KEY='替换为真实密钥'
~~~

这里没有将 Qwen 设置为全局默认模型，因此 MiniMax 仍然可以继续作为默认模型。用户可以在页面模型选择器中选择：

~~~text
qwen / Qwen3.8-27B
~~~

## 四、为什么需要设置 contextWindow 和 maxTokens

SGLang 启动时设置了：

~~~bash
--context-length 260000
~~~

但这只是模型允许的理论上限，并不代表 GPU 实际拥有 260K token 的 KV Cache 容量。

实际运行时出现了：

~~~text
full token: 31783
full token usage: 1.00
KV cache pool is full
~~~

随后服务端报错：

~~~text
Out of memory even after retracting all other requests in the decode batch.
Aborting the last request.
~~~

Harness 中的配置：

~~~yaml
contextWindow: 28672
maxTokens: 2048
~~~

作用是让 Harness 提前知道这个部署的安全上下文范围，并限制单次输出，避免请求继续接近服务端的实际 32K 缓存边界。

需要注意，已经很长的旧会话仍然保存着历史上下文。修改配置后最好新建会话测试。

## 五、KV Cache OOM 的定位过程

Harness 会话日志中可以看到：

~~~text
assistant finish:
Out of memory even after retracting all other requests in the decode batch.
~~~

随后出现：

~~~text
Connection error.
~~~

这说明回答中断不是前端截断，而是：

~~~text
Qwen 生成
  ↓
KV Cache 使用率达到 100%
  ↓
SGLang retract_decode
  ↓
服务端删除请求状态
  ↓
Harness 收到 OOM
  ↓
请求结束或重试
~~~

其中：

~~~text
Received output ... but the state was deleted in TokenizerManager
~~~

是请求被服务端删除后的后续日志，不是根因。

## 六、SGLang OOM 的解决思路

Qwen3.8-27B 是带有 Mamba/状态缓存特征的模型。原始配置中的：

~~~bash
--mamba-full-memory-ratio 4.13
--mamba-ssm-dtype float32
--mamba-radix-cache-strategy extra_buffer
~~~

会占用较多状态内存，进而压缩 KV Cache。

建议先使用更保守的单请求配置：

~~~bash
sglang serve \
  --trust-remote-code \
  --model-path /nfs/FM/checkpoints/Qwen/Qwen3.8-27B \
  --served-model-name Qwen3.8-27B \
  --context-length 32768 \
  --kv-cache-dtype bfloat16 \
  --mem-fraction-static 0.90 \
  --chunked-prefill-size 2048 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --mamba-full-memory-ratio 0.5 \
  --mamba-ssm-dtype bfloat16 \
  --mamba-radix-cache-strategy no_buffer \
  --host 0.0.0.0 \
  --port 30000 \
  --api-key '替换为真实密钥'
~~~

排查阶段先移除推测解码参数：

~~~bash
--speculative-algorithm EAGLE
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
~~~

确认稳定后，再逐项恢复并观察：

- KV Cache 使用率；
- Mamba 状态池使用率；
- 单请求最大上下文；
- 首 token 延迟；
- 生成速度；
- 是否再次出现 retract_decode。

SGLang 的 Mamba Cache 文档说明，Mamba 状态池和 KV Cache 池共享有限的静态显存；mamba-full-memory-ratio 越高，分配给 Mamba 状态池的比例越大，KV Cache 空间就越少。[SGLang Server Arguments](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md)

## 七、流式输出适配

Qwen 通过 SGLang 返回 reasoning_content 和普通 content。Harness 的 llm-pi-ai 适配器会将其转换为：

~~~text
reasoning-delta
text-delta
tool-call-delta
finish
~~~

为了处理网关将大段内容集中返回的情况，adapter.ts 增加了大块拆分逻辑：

~~~ts
async function* paceChunks(
  chunks: AsyncIterable<StreamChunk>,
  minimumDelayMs: number,
): AsyncIterable<StreamChunk> {
  const pieceSize = 12

  for await (const chunk of chunks) {
    if (
      (chunk.type === 'text-delta' || chunk.type === 'reasoning-delta')
      && chunk.text.length > pieceSize
    ) {
      for (
        let offset = 0;
        offset < chunk.text.length;
        offset += pieceSize
      ) {
        yield {
          ...chunk,
          text: chunk.text.slice(offset, offset + pieceSize),
        }

        await new Promise<void>(resolve =>
          setTimeout(resolve, minimumDelayMs),
        )
      }
    } else {
      yield chunk
    }
  }
}
~~~

这样可以避免模型已经产生响应，但前端最后一次性显示完整文本。

## 八、图片能力验证

Qwen3.8-27B 的图片能力不是只根据模型名称判断，而是通过真实请求验证。

测试请求采用 OpenAI-compatible 的 image_url 格式：

~~~json
{
  "model": "Qwen3.8-27B",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "请识别这张图片的主要颜色，只回答颜色名称。"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,..."
          }
        }
      ]
    }
  ],
  "max_tokens": 64
}
~~~

实际验证结果：

~~~text
HTTP 200
model: Qwen3.8-27B
image_tokens: 64
~~~

模型返回的 reasoning_content 能正确识别图片主要颜色。image_tokens 为 64 说明图片已经被模型处理，而不是被接口忽略。

因此，在 Harness 的模型配置中声明：

~~~yaml
input:
  - text
  - image
~~~

Harness 本身支持：

- PNG；
- JPEG；
- WebP；
- GIF。

前端负责把图片转换为 base64，Host API Proxy 负责校验图片格式，LLM Provider 再把图片转换成模型 API 所需的多模态消息。

## 九、图片能力的限制

虽然 Qwen 主模型支持图片，但不是所有 Harness 子代理都支持图片。

当前需要区分：

~~~text
主模型 Qwen3.8-27B：支持图片
MiniMax 文本模型：取决于具体模型和网关
Codex/Claude 子代理：当前不支持图片继续对话
~~~

如果模型配置中没有 input: image，页面可能仍然允许选择图片，但提交时会被模型能力检查拒绝。

## 十、验证方式

检查 Qwen 服务：

~~~bash
ss -ltnp | grep 30000
curl -sS http://192.168.16.19:30000/v1/models
~~~

检查 Harness 服务：

~~~bash
ss -ltnp | grep 13080
curl -I http://127.0.0.1:13080
~~~

直接测试文本：

~~~bash
curl -sS \
  -H "Authorization: Bearer $QWEN38_API_KEY" \
  -H "Content-Type: application/json" \
  http://192.168.16.19:30000/v1/chat/completions \
  -d '{
    "model": "Qwen3.8-27B",
    "messages": [{"role": "user", "content": "你是谁？"}]
  }'
~~~

前端测试：

1. 强制刷新页面；
2. 选择 qwen / Qwen3.8-27B；
3. 先发送纯文本；
4. 再拖入一张 PNG 或 JPEG；
5. 询问图片中的颜色、文字或物体；
6. 检查是否出现正常回答。

## 十一、常见问题

### 1. 端口 30000 拒绝连接

说明 SGLang 没有运行或进程因 OOM 退出：

~~~bash
ss -ltnp | grep 30000
ps aux | grep sglang
~~~

### 2. 回答输出到一半消失

优先查看是否出现：

~~~text
KV cache pool is full
Out of memory
retract_decode
Connection error
~~~

如果出现，应该先降低上下文和输出上限，调整 Mamba/KV Cache 内存比例。

### 3. 图片上传后模型报不支持

检查模型配置是否包含：

~~~yaml
input:
  - text
  - image
~~~

同时确认实际 SGLang 服务的模型确实是多模态版本。

### 4. 修改配置后页面没有变化

重新启动 Harness：

~~~bash
pnpm dsh web --host 0.0.0.0 --port 13080
~~~

如果修改了客户端源码，还要重新构建：

~~~bash
pnpm run build:web
~~~

浏览器执行 Ctrl + F5。

## 十二、最终运行流程

先启动 Qwen：

~~~bash
sglang serve \
  --trust-remote-code \
  --model-path /nfs/FM/checkpoints/Qwen/Qwen3.8-27B \
  --served-model-name Qwen3.8-27B \
  --context-length 32768 \
  --kv-cache-dtype bfloat16 \
  --mem-fraction-static 0.90 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --mamba-full-memory-ratio 0.5 \
  --mamba-ssm-dtype bfloat16 \
  --mamba-radix-cache-strategy no_buffer \
  --host 0.0.0.0 \
  --port 30000 \
  --api-key '替换为真实密钥'
~~~

再启动 Harness：

~~~bash
export QWEN38_API_KEY='替换为真实密钥'
cd /nfs/FM/gongoubo/new_project/github/harness/deepseek-harness
pnpm dsh web --host 0.0.0.0 --port 13080
~~~

访问：

~~~text
http://服务器IP:13080
~~~

## 结语

Qwen3.8-27B 适配 DeepSeek Harness 的关键，不只是把 API 地址接通，而是让模型能力和 Harness 能力声明保持一致：

- 文本模型需要声明 text；
- 多模态模型需要声明 text 和 image；
- 大上下文模型必须考虑实际 KV Cache；
- Mamba 模型需要合理分配状态池和 KV Cache；
- 流式输出要同时检查模型、适配器、WebSocket 和前端；
- 长会话 OOM 后应新建会话，而不是继续复用已经接近上限的历史。

完成这些适配后，Qwen3.8-27B 就可以作为 DeepSeek Harness 的主模型，支持文本、思考、工具调用以及图片理解。

