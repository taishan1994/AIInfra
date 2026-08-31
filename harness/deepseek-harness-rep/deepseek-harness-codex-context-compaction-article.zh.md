# Codex 上下文压缩技术解析：从上下文压力检测到可恢复摘要

> 本文基于 deepseek-harness 当前源码整理，重点分析 Harness 中服务于 Codex 类编程 Agent 的上下文压缩机制。这里的 Codex 既可以指 Harness 的主编程 Agent 工作流，也可以指通过 `codex app-server --stdio` 接入的外部 Codex 子代理。两者需要分开理解：Harness 的会话压缩由本项目管理；外部 Codex 进程内部如何压缩，Harness 不会直接改写。

## 一、为什么编程 Agent 必须做上下文压缩

一次完整的编程任务通常会产生大量上下文：

- 系统提示词和工具定义；
- 用户的多轮需求；
- Agent 的分析和最终回答；
- 文件读取、搜索、补丁和命令执行结果；
- 工具调用与工具结果的配对记录；
- 图片、错误日志和模型路由信息。

这些内容最终都会进入下一次模型请求。即使模型声明了很大的 context window，实际可用空间仍会受到系统提示词、工具 schema、KV cache、并发请求和输出预算影响。

如果历史一直原样累积，会出现三类问题：

1. 请求超过模型上下文窗口，服务端返回 context overflow；
2. KV cache 被长历史占满，出现 OOM 或请求被撤回；
3. 历史过长导致模型注意力下降，反而更容易忘记当前任务。

因此，Codex 类型的编程 Agent 不能只保存“聊天记录”，还需要维护一份适合继续推理的“模型可见上下文”。

## 二、总体架构

deepseek-harness 将上下文压缩拆成几个职责明确的组件：

| 组件 | 作用 |
| --- | --- |
| dsh-token-meter | 估算当前请求和会话 surface 的 token 压力 |
| dsh-compaction | 定义压缩服务接口、事件和边界规则 |
| dsh-compaction-basic | 实现自动压缩、保留策略和摘要调用 |
| dsh-command-compact | 提供用户侧的 /compact 命令 |
| session surface | 维护模型当前可见的消息序列 |
| LLM adapter | 提供具体模型的 contextWindow，并执行摘要请求 |
| UI projection | 展示 contextPressure、contextBreakdown 和压缩记录 |

核心关系可以概括为：

~~~text
会话日志
   │
   ├── session surface：推导模型当前可见消息
   ├── token-meter：测量下一个请求的上下文压力
   └── compaction-basic：判断是否需要压缩
           │
           ├── 可选 tool-result-pruner：先缩小超大工具结果
           ├── 选择可压缩的旧消息范围
           ├── 调用摘要模型
           └── 用一个 checkpoint user message 替换旧范围
~~~

这种设计的关键点是：原始日志仍然保留，但被压缩的旧消息不再出现在后续模型请求的 surface 中。

## 三、Token 压力是如何计算的

### 3.1 模型容量由路由适配器提供

压缩策略不会假定所有模型共享同一个上下文长度。它首先根据最新持久化的 provider/model 路由，调用：

~~~ts
ctx.llm.resolveModelInfo(provider, model)
~~~

获取该模型的 contextWindow。

例如，一个模型的上下文窗口为 32768，默认配置为：

~~~yaml
thresholdRatio: 0.8
retainRatio: 0.16
~~~

那么：

~~~text
压缩阈值 = floor(32768 × 0.8) = 26214
近期原文保留预算 = floor(32768 × 0.16) = 5242
~~~

达到阈值后，系统会优先保留最近的消息，把较早的历史交给摘要流程。

### 3.2 token-meter 使用回放感知的测量

dsh-token-meter 的核心接口是：

~~~ts
ctx.tokenMeter.measure(session)
~~~

返回的数据包含：

- totalTokens：请求 envelope 和当前 surface 的总估算压力；
- surfaceTokens：当前模型可见消息的估算总量；
- nodes：每个 surface 节点及其 token 价格；
- provider usage：如果请求与历史 envelope 完全匹配，可以复用提供方报告的真实用量。

没有精确提供方用量时，项目使用固定启发式：

~~~text
文本 token 数 ≈ 字符数 / 4 + 角色、内容块和请求字段的结构开销
~~~

这不是精确 tokenizer，因此 CJK 文本、JSON schema 和工具定义可能被低估。它更适合做趋势判断和压缩门控，而不是代替服务端真实计费。

### 3.3 projectedTokens 比单纯的最后一次 usage 更实用

流式请求结束后，提供方可能只报告上一次请求的 prompt token。若此后用户又追加了消息，仅查看上一次 pressureTokens 会滞后。

token-meter 因此维护 projectedTokens：

~~~text
projectedTokens
= 最近一次提供方压力锚点
+ 自该锚点以来 surface 的增量或减量
~~~

压缩替换旧消息时，projectedTokens 会立即下降。这样 UI 可以更快反映压缩效果，而不必等待下一整轮模型请求。

不过，自动压缩使用的是 token-meter 的 measure()，不是 UI 上展示的近似占用率。这样可以避免把一个面向用户的投影数字误当作严格的门控依据。

## 四、压缩何时触发

### 4.1 自动压力压缩

dsh-compaction-basic 默认注册 agent/pre-step listener。每次 Agent 准备执行下一步时，系统会：

1. 测量当前会话；
2. 解析当前 provider/model 的 contextWindow；
3. 根据 thresholdRatio 计算阈值；
4. 判断 totalTokens 是否达到阈值；
5. 如有工具结果剪枝器，先做不依赖模型的缩减；
6. 再次测量；
7. 仍然超阈值时，选择旧历史并生成摘要。

因此，压缩发生在“下一步模型请求之前”，而不是等请求已经超限后才被动处理。

### 4.2 上下文溢出恢复

如果模型服务端已经返回规范化的 CONTEXT_WINDOW_EXCEEDED，自动流程会进入第二条恢复路径：

1. 跳过普通 threshold 检查；
2. 先尝试剪枝工具结果；
3. 选择一个尽可能大的、工具配对平衡的旧范围；
4. 完成一次压缩；
5. 只要 surface.replaceGeneration 前进，就允许重试原请求。

这里的关键是“是否产生了持久的 surface 替换”，而不是摘要函数是否完全没有抛错。即使剪枝已经落地、后续摘要失败，系统也可以从已经缩小的 surface 继续恢复。

### 4.3 手动 /compact

dsh-command-compact 提供：

~~~text
/compact
~~~

手动压缩不要求已经达到自动阈值，但要求 Agent 处于可维护状态。它会选择一段有效且工具配对平衡的旧历史，生成独立的 turn: null 压缩事务，并在持久化 flush 完成后返回。

手动压缩不是模型工具，也不会把 /compact 文本发送给模型。

## 五、一次压缩事务的完整生命周期

成功压缩不是简单地把数组 splice 掉，而是一个可恢复的持久化事务：

~~~text
compaction/start
       │
       ├── 锁定本次压缩
       ├── 记录待压缩范围
       ├── 调用摘要模型
       ├── 校验摘要确实小于原范围
       ├── 写入 compaction/summary
       ├── 写入带 surface replace 的 user/message
       └── 写入 compaction/end
~~~

### 5.1 compaction/start：先写锁，再让出控制权

start 事件是日志中的持久锁。它在摘要调用开始前写入，因此异步摘要期间如果进程崩溃，恢复逻辑能识别一个未闭合的压缩操作。

压缩中的请求不能与另一个压缩并发修改同一 surface。

### 5.2 摘要调用：复用原请求前缀

默认摘要器位于 packages/compaction/compaction-basic/src/summarizer.ts。它会回放：

- 会话原有的 system prompt；
- 原有工具 schema；
- 待压缩范围内的消息；
- 最后追加一条压缩指令。

摘要指令要求模型输出固定 Markdown 结构：

~~~markdown
## Primary Request and Intent
## Key Technical Concepts
## Files and Code
## Errors and Fixes
## Pending Jobs
## Current Work
## Next Step
## Critical Context
~~~

这种设计有两个目的：

1. 让下一个模型快速恢复用户目标、文件、错误和待办；
2. 让摘要具备稳定结构，避免只留下模糊的自然语言概括。

摘要调用通过 ctx.llm.stream() 直接发起，并设置：

~~~ts
purpose: 'compaction'
~~~

它不是 Agent loop 的普通步骤，所以不会新增工具调用，也不会把摘要模型的私有推理写进会话 surface。

此外，摘要器会尽量复用会话原有的系统提示词、工具和消息前缀，以提高提供方 KV cache 的复用率。压缩本身仍然会从第一个被替换的历史 token 起造成 cache 失效，这是替换历史的必然代价。

### 5.3 compaction/summary：保存摘要和可重建信息

summary 事件保存：

- compactionId；
- 摘要内容；
- 被遮蔽的 range 和 seq；
- 被替换内容的估算 token 数；
- 生成摘要的 provider/model；
- maxTokens 和 provider usage；
- 必要时的 rawOutput。

这样做的意义是：以后分析会话时，可以知道“是哪一个模型、以什么预算生成了这个摘要”，而不是只看到一段无法追溯来源的文本。

### 5.4 surface replace：用一个检查点替换旧历史

真正改变模型可见上下文的是紧随其后的 user/message：

~~~ts
surfaceOp: { op: 'replace', start, end }
~~~

它携带：

~~~text
<compacted-summary>
摘要内容
</compacted-summary>
~~~

原始事件不会被物理删除，而是从 surface 中被 shadow。后续 deriveMessages() 只会把检查点和未被替换的近期内容交给模型。

这是“日志”和“模型上下文”分离的关键：

- 日志用于审计、恢复和调试；
- surface 用于生成下一次模型请求。

### 5.5 compaction/end：释放锁

当替换成功后，系统追加 end 事件释放压缩锁。失败时也会尽量写入带 error 的 end。

如果 end 本身写入失败，未匹配的 start 会继续存在，故意保持为阻塞信号。系统不会伪装成压缩已经正常完成。

## 六、为什么压缩边界必须考虑工具调用配对

编程 Agent 的历史不是普通文本序列。典型结构是：

~~~text
assistant: tool_call
tool: tool_result
assistant: next response
~~~

如果压缩边界切在 tool_call 和 tool_result 中间，下一轮模型可能看到一个没有结果的调用，或者看到工具结果却找不到对应调用。这会造成：

- tool_call/tool_result 数量不匹配；
- provider 拒绝请求；
- Agent 误以为工具还没有执行；
- 多轮继续时出现 replay state 错误。

dsh-compaction 提供 toolPairingBalancedBefore() 和 toolPairingBalancedAfter()，dsh-compaction-basic 在选择范围时使用这些规则，将边界调整到工具调用和结果都闭合的位置。

这也是解决“多轮问答报 invalid replay state”类问题的基础思路：压缩不能只按字符长度截断，必须保持消息协议和工具事件的结构完整。

## 七、摘要失败、并发变化和重试

### 7.1 摘要必须真的变小

region.ts 会估算带有 checkpoint 框架的摘要大小。如果摘要不小于被遮蔽的历史，会拒绝提交：

~~~text
summary is not smaller than the shadowed content
~~~

否则压缩可能越压越大。

### 7.2 摘要期间 surface 发生变化

自动压缩要求整个 surface 在摘要期间保持不变。若另一个操作追加或替换了上下文，系统会判定当前摘要基于过期快照，不提交替换。

手动压缩的规则更细：它只要求被选中的 span 仍然稳定，压缩期间追加到选定范围之外的上下文可以保留。这样可以兼容空闲 Agent 上的上下文注入。

### 7.3 失败时保留日志，不破坏原历史

常见失败处理是：

- start 已写入；
- 摘要失败；
- 写入带 error 的 end；
- 原 surface 保持不变。

这比直接删除失败记录更安全，因为后续可以知道曾经尝试压缩，也不会把未验证的摘要暴露给模型。

## 八、与长上下文模型和 OOM 的关系

上下文窗口和显存中的 KV cache 不是同一个概念。

例如 Qwen3.8-27B 可能配置了很大的 context-length，但实际可承载的 token 数还取决于：

- GPU 显存；
- KV cache dtype；
- mamba 状态；
- batch 中的并发请求；
- speculative decoding；
- 生成长度；
- 服务端保留比例。

因此，下面两种情况都可能出现：

1. API 声明的 contextWindow 还没有达到，但服务端 KV cache 已满；
2. 请求已经接近服务端真实上限，产生 OOM 或 retract。

上下文压缩可以减少后续 prompt 和 KV cache 压力，但不能解决固定的 system prompt、工具 schema、模型权重或单条不可拆分的大消息。对于 Qwen/MiniMax 这类模型，建议同时控制：

- Harness 侧的 contextWindow 和压缩阈值；
- 服务端的 context-length；
- maxTokens；
- 工具返回的长度；
- 图片分辨率和图片 token；
- 并发和 speculative decoding 参数。

在本项目中，压缩不会缩减 system prompt 和 tools，因此如果工具 schema 本身很大，需要单独优化工具注册和返回内容。

## 九、Codex 子代理接入时要注意的边界

packages/subagent/subagent-codex/src/index.ts 负责启动外部：

~~~text
codex app-server --stdio
~~~

这条链路的主要职责是进程管理、请求转发和子代理生命周期。它不等于把外部 Codex 的内部上下文状态转换成 Harness 的 session surface。

因此要区分：

- Harness 主 Agent：可以使用 dsh-token-meter 和 dsh-compaction-basic 做会话级压缩；
- 外部 Codex app-server：其内部上下文压缩由外部 Codex 自己决定，Harness 只能看到接入层允许暴露的请求和结果。

如果需要统一压缩策略，最稳定的做法是在 Harness 主会话边界做摘要，并把必要的任务状态以结构化内容传给子代理，而不是尝试从外部 Codex 进程中截断内部 replay 状态。

## 十、如何排查压缩是否生效

可以按以下顺序排查：

1. 确认基础插件已加载：
   - dsh-token-meter；
   - dsh-compaction-basic；
   - 如需手动命令，再加载 dsh-command-compact。

2. 检查模型路由是否提供 contextWindow：
   - 压缩策略按实际 provider/model 查找；
   - 没有容量信息时，自动压力压缩会警告并继续，不会盲目裁剪。

3. 观察 compaction 事件：
   - compaction/start；
   - compaction/summary；
   - user/message 的 surface replace；
   - compaction/end。

4. 观察日志中的压缩结果：
   - shadowed surface nodes；
   - shadowed range；
   - estimated token count。

5. 检查工具配对：
   - 不要在 assistant tool call 和 tool result 之间人为截断；
   - 如果出现 invalid replay state，先检查 surface replacement 是否覆盖了完整工具对。

6. 对比压缩前后的测量：
   - ctx.tokenMeter.measure(session).totalTokens；
   - surfaceTokens；
   - contextPressure projectedTokens；
   - contextBreakdown.messageTokens。

## 十一、总结

Codex 上下文压缩的本质不是“删除旧消息”，而是建立一个可恢复的上下文重写系统：

- 用模型路由提供的 contextWindow 判断容量；
- 用 token-meter 测量当前 surface 和下一个请求的压力；
- 先尝试不依赖模型的工具结果剪枝；
- 在工具调用配对安全的边界选择旧历史；
- 通过一次结构化摘要调用把旧历史浓缩成 checkpoint；
- 用 surface replace 让模型只看到摘要和近期上下文；
- 原始日志继续保留，支持恢复、审计和调试；
- 用 start/summary/end 事件保证事务、锁和失败可见；
- 对 provider overflow 支持压缩后重试；
- 对外部 Codex 子进程保持边界，不假设能控制其内部 replay 状态。

这套机制最终解决的不是“如何让一次请求变短”，而是“如何让一个长时间运行的编程 Agent 在不丢失任务状态的情况下持续工作”。这也是它比简单截断更适合 Codex、工具调用和多轮代码任务的原因。


