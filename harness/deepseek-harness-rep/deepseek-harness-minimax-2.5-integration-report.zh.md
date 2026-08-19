# DeepSeek Harness 集成 MiniMax-M2.5 完整报告

## 1. 集成结果

MiniMax-M2.5 已通过 OpenAI-compatible API 接入 DeepSeek Harness。当前链路为：

~~~text
浏览器 UI
  -> WebSocket /api/events.mux
  -> client/connection
  -> client/runtime + ui-conversation
  -> Host API Proxy
  -> llm-pi-ai adapter
  -> pi-ai streamSimple()
  -> MiniMax-M2.5 API
~~~

已完成：

- Provider 和默认模型配置；
- 环境变量 API 密钥引用；
- OpenAI Completions 协议适配；
- <think> 思考内容解析；
- replay state 和多轮对话修复；
- WebSocket 逐帧发送；
- 前端 assistant/chunk 立即刷新；
- 大块 text_end 拆分为可见流式块；
- Web 默认监听 0.0.0.0:13080；
- Codex 和 Claude 子代理插件集成。

API 密钥没有写入本文。已经暴露过的密钥应撤销并重新生成。

## 2. MiniMax 配置

### 2.1 环境变量

在启动 Harness 的同一个 shell 中：

~~~bash
export MINIMAX25_API_KEY='替换为真实密钥'
~~~

### 2.2 OpenAI-compatible API 验证

~~~bash
curl -sS --max-time 15 \
  -H "Authorization: Bearer $MINIMAX25_API_KEY" \
  -H "Content-Type: application/json" \
  http://192.168.11.18:30055/v1/chat/completions \
  -d '{
    "model": "MiniMax-M2.5",
    "stream": true,
    "messages": [{"role": "user", "content": "你好，请简单介绍自己。"}]
  }'
~~~

配置参数：

- API 根地址：http://192.168.11.18:30055/v1
- Provider：minimax25
- API 类型：openai-completions
- 模型：MiniMax-M2.5

### 2.3 /root/.dsh/settings.yaml

当前配置：

~~~yaml
ui-onboarding:
  welcomeNoticeVersion: 2026-08-13.1

agent-presets:
  default: standard

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

permission:
  defaultPreset: danger-full-access
~~~

字段说明：

| 字段 | 作用 |
|---|---|
| apiKeyEnv | 从环境变量读取 API 密钥 |
| api | 使用 pi-ai 的 OpenAI Completions 适配器 |
| baseURL | MiniMax API 地址 |
| streamDelayMs | 网关批量返回时的可见输出间隔 |
| thinkingFormat | 兼容字符串思考格式 |
| agent-default-model | 默认 Provider 和模型 |

streamDelayMs 不是 API 必填字段。当前设置为 25ms，是为了让大块响应在页面上肉眼可见；网关稳定逐块输出时可以设置为 0。

## 3. 源码修改

### 3.1 packages/llm/llm-pi-ai/src/config.ts

新增 Provider 字段：

~~~ts
export interface PiAiProviderProfile {
  // existing fields
  streamDelayMs?: number
}

export interface ResolvedPiAiProviderProfile {
  // existing fields
  streamDelayMs: number
}
~~~

Schema 新增：

~~~ts
streamDelayMs: z.natural().max(1000).default(0),
~~~

resolveProfiles 新增：

~~~ts
const streamDelayMs = source.streamDelayMs ?? 0

return {
  // existing fields
  streamDelayMs,
}
~~~

约束为 0 到 1000ms，缺省为 0。

### 3.2 packages/llm/llm-pi-ai/src/adapter.ts

#### Provider 事件节奏

~~~ts
async function* paceEvents(
  events: AsyncIterable<AssistantMessageEvent>,
  minimumDelayMs: number,
): AsyncIterable<AssistantMessageEvent> {
  let nextAt = 0

  for await (const event of events) {
    const waitMs = Math.max(0, nextAt - Date.now())
    if (waitMs > 0) {
      await new Promise<void>(resolve => setTimeout(resolve, waitMs))
    }

    yield event
    nextAt = Date.now() + minimumDelayMs
  }
}
~~~

#### 大块响应拆分

某些 OpenAI-compatible 网关虽然使用 stream=true，但 pi-ai 可能收到一个大的 text_end 或 reasoning_end 内容。新增：

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
          await new Promise<void>(resolve => setTimeout(resolve, minimumDelayMs))
        }
      }
    } else {
      yield chunk
      if (minimumDelayMs > 0) {
        await new Promise<void>(resolve => setTimeout(resolve, minimumDelayMs))
      }
    }
  }
}
~~~

适配器主流程改为：

~~~ts
const pacedEvents = profile.streamDelayMs > 0
  ? paceEvents(events, profile.streamDelayMs)
  : events

const chunks = toStreamChunks(pacedEvents, model.contextWindow)

const pacedChunks = profile.streamDelayMs > 0
  ? paceChunks(chunks, profile.streamDelayMs)
  : chunks

const iterator = pacedChunks[Symbol.asyncIterator]()
~~~

消费者终止时：

~~~ts
await iterator.return?.(undefined)
~~~

### 3.3 packages/llm/llm-pi-ai/src/stream.ts

#### <think> 思考转换

新增：

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

text_start 时建立状态：

~~~ts
case 'text_start':
  embeddedThinking.set(
    event.contentIndex,
    createEmbeddedThinkingState(outputIndex(event.contentIndex)),
  )
  break
~~~

text_delta 的处理逻辑：

1. 先暂存内容，判断是否以 <think> 开始；
2. 进入 thinking 模式后输出 reasoning-delta；
3. 找到 </think> 后结束 reasoning block；
4. 后续内容转成 text-delta；
5. 没有思考标记时按普通文本输出。

核心形式：

~~~ts
if (state.mode === 'thinking') {
  const combined = state.closeTail + state.pending
  const closeAt = combined.indexOf(THINK_CLOSE)

  if (closeAt === -1) {
    const safe = combined.slice(
      0,
      Math.max(0, combined.length - THINK_CLOSE.length + 1),
    )
    state.closeTail = combined.slice(safe.length)

    if (safe) {
      state.thinking += safe
      yield {
        type: 'reasoning-delta',
        index: state.thinkingIndex,
        text: safe,
      }
    }
  } else {
    yield {
      type: 'block-end',
      index: state.thinkingIndex,
      block: { type: 'reasoning', text: state.thinking },
    }

    state.mode = 'text'
    yield {
      type: 'block-start',
      index: state.textIndex,
      blockType: 'text',
    }
  }
}
~~~

#### 索引冲突修复

如果原始 Provider block 为：

~~~text
text[0] -> tool-call[1]
~~~

插入 reasoning 后必须变为：

~~~text
reasoning[0] -> text[1] -> tool-call[2]
~~~

实现：

~~~ts
const syntheticThinkingIndices = new Set<number>()

const outputIndex = (providerIndex: number): number =>
  providerIndex
  + [...syntheticThinkingIndices]
    .filter(index => index <= providerIndex).length
~~~

toolcall_start、toolcall_delta、toolcall_end 全部使用 outputIndex。

#### replay state 修复

网关返回的 <think> 字符串不是 pi-ai 原生 replay block。结束时：

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

这样避免多轮对话报：

~~~text
invalid pi-ai replay state: block count does not match assistant content
~~~

#### 原生 thinking_end 补增量

部分 Provider 只有 thinking_end 的完整内容，没有 thinking_delta。当前代码维护累计内容，并在结束时只补未发送的部分：

~~~ts
const nativeThinking = new Map<number, string>()

const emitted = nativeThinking.get(event.contentIndex) ?? ''
const remaining = event.content.startsWith(emitted)
  ? event.content.slice(emitted.length)
  : event.content

if (remaining) {
  yield {
    type: 'reasoning-delta',
    index: outputIndex(event.contentIndex),
    text: remaining,
  }
}
~~~

### 3.4 packages/client/connection/src/websocket-downlink.ts

每个 WebSocket frame 发送后主动让出 Node 事件循环：

~~~ts
function yieldToSocket(): Promise<void> {
  return new Promise(resolve => setImmediate(resolve))
}
~~~

发送循环：

~~~ts
for await (const frame of frames) {
  await send(socket, frame)
  await yieldToSocket()
}
~~~

### 3.5 packages/client/ui-conversation/src/client/conversation-nodes/assistant.ts

原来 assistant/chunk 使用 animation-frame 批量刷新，修改为立即刷新：

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

### 3.6 Web 监听地址和端口

文件：packages/bundle/web-app/cordis.patch.yml

~~~yaml
config:
  host: !!js ctx.webStartup.host ?? '0.0.0.0'
  port: !!js ctx.webStartup.port ?? 13080
~~~

文件：packages/bundle/web-app/src/startup.ts

删除了原先禁止 --host 0.0.0.0 的校验。现在可以：

~~~bash
pnpm dsh web --host 0.0.0.0 --port 13080
~~~

### 3.7 Codex 和 Claude 子代理

在 packages/bundle/web-app/cordis.patch.yml 中加入：

~~~yaml
- insert:
    - id: subagent-codex
      name: '@deepseek-ai/dsh-subagent-codex'
    - id: subagent-claude-code
      name: '@deepseek-ai/dsh-subagent-claude-code'
~~~

主模型和子代理是独立链路：

- MiniMax-M2.5：主 LLM；
- Codex/Claude：子代理插件。

## 4. 构建和启动

### 构建前端

dsh web 使用构建后的静态前端，客户端源码修改后必须重新构建：

~~~bash
cd /nfs/FM/gongoubo/new_project/github/harness/deepseek-harness
pnpm run build:web
~~~

### 启动服务

~~~bash
export MINIMAX25_API_KEY='替换为真实密钥'
pnpm dsh web --host 0.0.0.0 --port 13080
~~~

也可以使用默认配置：

~~~bash
pnpm dsh web
~~~

### 检查监听

~~~bash
ss -ltnp | grep 13080
~~~

预期：

~~~text
0.0.0.0:13080
~~~

访问：

~~~text
http://服务器IP:13080
~~~

构建或重启后，浏览器执行 Ctrl + F5。

## 5. 流式验证

真实 WebSocket 测试结果：

~~~text
文本块数量：12
首个文本块：约 4371ms
最后文本块：约 4659ms
文本块间隔：约 25–27ms
~~~

说明：

- 首次延迟主要是模型思考时间；
- 最终文本不是一次性 WebSocket 消息；
- 文本块按约 25ms 间隔抵达；
- MiniMax 和 Claude 共用 llm-pi-ai 适配逻辑。

典型事件顺序：

~~~text
block-start
text-delta
text-delta
text-delta
block-end
usage
finish
~~~

## 6. 测试记录

LLM 测试：

~~~bash
pnpm vitest run \
  packages/llm/llm-pi-ai/tests/convert.spec.ts \
  packages/llm/llm-pi-ai/tests/adapter.spec.ts
~~~

结果：

~~~text
Test Files  2 passed
Tests       116 passed
~~~

类型检查：

~~~bash
pnpm exec tsc --noEmit -p packages/llm/llm-pi-ai/tsconfig.json
~~~

结果：通过。

WebSocket 测试：

~~~bash
pnpm vitest run \
  packages/client/connection/tests/websocket-downlink.host.spec.ts
~~~

结果：

~~~text
Test Files  1 passed
Tests       8 passed
~~~

前端构建：

~~~bash
pnpm run build:web
~~~

结果：成功。

## 7. 故障排查

### fail to fetch

检查：

- 浏览器能否访问服务器 IP 的 13080；
- 服务是否监听 0.0.0.0；
- baseURL 是否能从服务器访问；
- MINIMAX25_API_KEY 是否存在于启动 shell；
- 网关是否需要 /v1；
- 防火墙是否放行 13080。

### 页面仍一次性显示

~~~bash
pnpm run build:web
pnpm dsh web --host 0.0.0.0 --port 13080
ss -ltnp | grep 13080
grep -n "streamDelayMs\|baseURL\|MiniMax" /root/.dsh/settings.yaml
~~~

然后浏览器执行 Ctrl + F5。

### YAML 启动失败

streamDelayMs 必须和 baseURL 同级：

~~~yaml
      baseURL: http://192.168.11.18:30055/v1
      streamDelayMs: 25
~~~

### invalid pi-ai replay state

检查 stream.ts 的：

- synthetic reasoning 索引偏移；
- 工具调用 outputIndex；
- <think> 消息规范化；
- synthetic thinking 不携带原始 replay state。

## 8. 安全建议

- 不要在 Git、报告、截图或聊天中保存真实 API 密钥；
- 已暴露密钥应立即撤销并重新生成；
- 0.0.0.0 会将服务暴露到网络，不等于安全；
- 对外服务建议使用 HTTPS、反向代理、鉴权和防火墙；
- API 密钥仅放在环境变量或受权限保护的 secrets 文件中。

## 9. 最终命令

~~~bash
cd /nfs/FM/gongoubo/new_project/github/harness/deepseek-harness
export MINIMAX25_API_KEY='替换为真实密钥'
pnpm run build:web
pnpm dsh web --host 0.0.0.0 --port 13080
~~~

