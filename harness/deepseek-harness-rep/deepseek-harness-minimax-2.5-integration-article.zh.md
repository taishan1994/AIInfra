# 将 MiniMax-M2.5 集成到 DeepSeek Harness：一次完整的排查记录

## 写在前面

把一个模型接入 DeepSeek Harness，看起来只是配置 API 地址、模型名称和密钥，实际还要处理模型消息格式、思考内容、流式事件、工具调用索引以及浏览器渲染。

本文记录 MiniMax-M2.5 的集成过程，重点介绍遇到的问题和解决方法。相关修复同样适用于其他 OpenAI-compatible 模型网关。

## 一、先看完整链路

~~~text
浏览器
  ↓ WebSocket
DeepSeek Harness Web
  ↓ session.prompt
Host API Proxy
  ↓
llm-pi-ai
  ↓ pi-ai
MiniMax OpenAI-compatible API
~~~

模型返回时则反向经过：

~~~text
MiniMax SSE
  → pi-ai AssistantMessageEvent
  → Harness StreamChunk
  → assistant/chunk SessionEvent
  → WebSocket
  → 浏览器 UI
~~~

页面不流式显示时，问题可能来自上游 API、LLM 适配器、WebSocket 或前端渲染层，不能只看其中一层。

## 二、配置 MiniMax

在服务器上设置密钥：

~~~bash
export MINIMAX25_API_KEY='替换为真实密钥'
~~~

不要将真实密钥写进代码、Git 或文章。

在 /root/.dsh/settings.yaml 中配置：

~~~yaml
llm-pi-ai:
  providers:
    minimax25:
      displayName: minimax
      apiKeyEnv: MINIMAX25_API_KEY
      api: openai-completions
      baseURL: http://192.168.11.18:30055/v1
      streamDelayMs: 25
      compat:
        thinkingFormat: string-thinking
      models:
        - id: MiniMax-M2.5
          name: MiniMax-M2.5

agent-default-model:
  provider: minimax25
  model: MiniMax-M2.5
~~~

其中，apiKeyEnv 指定密钥来源，api 选择 OpenAI-compatible 适配器，baseURL 是网关地址，streamDelayMs 用于处理网关批量返回，thinkingFormat 用于兼容字符串形式的思考内容。

配置前先直接验证 API：

~~~bash
curl -sS --max-time 15 \
  -H "Authorization: Bearer $MINIMAX25_API_KEY" \
  -H "Content-Type: application/json" \
  http://192.168.11.18:30055/v1/chat/completions \
  -d '{
    "model": "MiniMax-M2.5",
    "stream": true,
    "messages": [{"role": "user", "content": "你好"}]
  }'
~~~

## 三、问题一：API 正常，页面却提示 fail to fetch

服务器部署时有两条网络连接：

~~~text
浏览器 → Harness:13080
服务器 → MiniMax:30055
~~~

浏览器能打开 Harness 页面，并不代表服务器能访问 MiniMax；服务器能访问 MiniMax，也不代表浏览器能访问 Harness。

检查命令：

~~~bash
ss -ltnp | grep 13080
curl -I http://127.0.0.1:13080
curl -sS http://xxxx:30055/v1/models
~~~

浏览器中的 127.0.0.1 指向本地电脑，不是远程服务器。远程访问要使用：

~~~text
http://服务器IP:13080
~~~

## 四、问题二：服务器上启动后本地访问不了

项目原先禁止使用 --host 0.0.0.0。服务器部署时需要让服务监听外部网卡。

修改 packages/bundle/web-app/cordis.patch.yml：

~~~yaml
config:
  host: !!js ctx.webStartup.host ?? '0.0.0.0'
  port: !!js ctx.webStartup.port ?? 13080
~~~

同时删除 packages/bundle/web-app/src/startup.ts 中拒绝 0.0.0.0 的参数校验。

启动：

~~~bash
pnpm dsh web --host 0.0.0.0 --port 13080
~~~

确认监听：

~~~bash
ss -ltnp | grep 13080
~~~

预期为：

~~~text
0.0.0.0:13080
~~~

注意：0.0.0.0 会让服务暴露到网络，生产环境还需要防火墙、HTTPS、鉴权和反向代理。

## 五、问题三：页面把 think 内容和最终回答混在一起

模型可能返回：

~~~text
<think>
模型的思考内容
</think>
最终回答
~~~

如果直接把它当普通文本，页面就会原样展示标签。

在 packages/llm/llm-pi-ai/src/stream.ts 中增加状态机，把它转换成 reasoning block 和 text block：

~~~ts
const THINK_OPEN = '<think>'
const THINK_CLOSE = '</think>'

type EmbeddedThinkingState = {
  index: number
  mode: 'undecided' | 'text' | 'thinking'
  pending: string
  thinking: string
  text: string
  thinkingIndex: number
  textIndex: number
  activeTextIndex: number
  closeTail: string
}
~~~

不能简单使用 split('</think>')，因为标签可能被拆在多个事件中：

~~~text
事件一：<thi
事件二：nk>正在思考
事件三：</think>最终答案
~~~

状态机先缓存前缀，确认进入 thinking 模式后输出 reasoning-delta，遇到结束标签再创建 text block。

## 六、问题四：多轮对话出现 replay state 错误

修复显示后，可能出现：

~~~text
invalid pi-ai replay state:
block count does not match assistant content
~~~

原因是原始消息只有一个 text block，但 Harness 已经拆成 reasoning 和 text 两个 block。若继续按照原始消息生成 replay state，下一轮恢复上下文时数量就不一致。

在 done 事件中先规范化：

~~~ts
const replayMessage = normalizeEmbeddedThinkingMessage(event.message)

yield {
  type: 'finish',
  reason: mapStopReason(replayMessage, contextWindow),
  ...replayMessage === event.message
    ? { replayState: toPiReplayState(replayMessage) }
    : {},
}
~~~

这说明消息显示结构和历史恢复结构必须一起修改。

## 七、问题五：思考块导致工具调用索引错乱

Provider 原始索引可能是：

~~~text
text[0] → tool-call[1]
~~~

插入 reasoning 后，正确索引应该是：

~~~text
reasoning[0] → text[1] → tool-call[2]
~~~

在 stream.ts 中统一计算输出索引：

~~~ts
const syntheticThinkingIndices = new Set<number>()

const outputIndex = (providerIndex: number): number =>
  providerIndex
  + [...syntheticThinkingIndices]
    .filter(index => index <= providerIndex).length
~~~

toolcall_start、toolcall_delta、toolcall_end 全部通过 outputIndex 计算索引。

## 八、问题六：stream=true，页面仍然一次性显示

通过 WebSocket 记录时间戳发现：

~~~text
block-start：约 690ms
block-end：约 5668ms
text-delta：约 5671ms，长度 285
~~~

这说明 WebSocket 并非完全没有分帧，而是 LLM 适配器最终只生成了一个 285 字符的大块。

部分 OpenAI-compatible 网关虽然接受 stream=true，但 pi-ai 可能收到完整 text_end。因此在 packages/llm/llm-pi-ai/src/adapter.ts 增加拆分器：

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
      for (let offset = 0; offset < chunk.text.length; offset += pieceSize) {
        yield {
          ...chunk,
          text: chunk.text.slice(offset, offset + pieceSize),
        }

        if (minimumDelayMs > 0) {
          await new Promise<void>(resolve =>
            setTimeout(resolve, minimumDelayMs),
          )
        }
      }
    } else {
      yield chunk
    }
  }
}
~~~

这不是重新生成回答，而是把已经收到的一个大块拆成多个 Harness 增量块。

## 九、问题七：传输层和前端仍可能合并

在 packages/client/connection/src/websocket-downlink.ts 中，每次发送后让出事件循环：

~~~ts
function yieldToSocket(): Promise<void> {
  return new Promise(resolve => setImmediate(resolve))
}

for await (const frame of frames) {
  await send(socket, frame)
  await yieldToSocket()
}
~~~

在 packages/client/ui-conversation/src/client/conversation-nodes/assistant.ts 中，把 assistant chunk 从动画帧更新改为立即更新：

~~~ts
publication: (match) => {
  if (match.event.type === 'step/start') return 'none'
  if (match.event.type !== 'assistant/chunk') return 'immediate'

  const type = match.event.data.chunk.type
  return type === 'usage' || type === 'finish'
    ? 'none'
    : 'immediate'
}
~~~

## 十、前端源码修改后必须重新构建

dsh web 使用构建后的静态前端。修改客户端 TypeScript 后只重启 Node 服务是不够的，浏览器可能仍然加载旧 bundle。

完整流程：

~~~bash
pnpm run build:web
pnpm dsh web --host 0.0.0.0 --port 13080
~~~

浏览器执行 Ctrl + F5。

## 十一、Codex 和 Claude 子代理

在 packages/bundle/web-app/cordis.patch.yml 中加入：

~~~yaml
- insert:
    - id: subagent-codex
      name: '@deepseek-ai/dsh-subagent-codex'
    - id: subagent-claude-code
      name: '@deepseek-ai/dsh-subagent-claude-code'
~~~

MiniMax 是主模型，Codex 和 Claude 是被 Harness 调度的子代理。

## 十二、验证结果

LLM 测试：

~~~bash
pnpm vitest run \
  packages/llm/llm-pi-ai/tests/convert.spec.ts \
  packages/llm/llm-pi-ai/tests/adapter.spec.ts
~~~

结果：2 个测试文件、116 个测试通过。

类型检查：

~~~bash
pnpm exec tsc --noEmit -p packages/llm/llm-pi-ai/tsconfig.json
~~~

WebSocket 测试：

~~~bash
pnpm vitest run \
  packages/client/connection/tests/websocket-downlink.host.spec.ts
~~~

结果：1 个测试文件、8 个测试通过。

真实 WebSocket 测试观察到：

~~~text
文本块数量：12
连续文本块间隔：约 25–27ms
~~~

这证明回答已经被拆成多个文本事件，而不是最后一次性发送。

## 十三、最终启动命令

~~~bash
cd /xxx/harness/deepseek-harness
export MINIMAX25_API_KEY='替换为真实密钥'
pnpm run build:web
pnpm dsh web --host 0.0.0.0 --port 13080
~~~

访问：

~~~text
http://服务器IP:13080
~~~

## 结语

这次集成最重要的经验不是某一行配置，而是要用完整链路思考问题：

1. 先直接验证模型 API；
2. 再确认服务器监听和浏览器访问；
3. 用时间戳判断数据在哪一层被合并；
4. 思考内容转换必须同步处理索引和 replay state；
5. stream=true 不一定意味着应用层得到细粒度增量；
6. 前端源码修改后必须重新构建静态包。

这些方法同样适用于其他 OpenAI-compatible 模型和网关。

