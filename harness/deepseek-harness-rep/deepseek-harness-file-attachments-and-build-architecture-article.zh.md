# DeepSeek Harness 文件上传解析与项目构建实践

## 一、为什么需要文件上传解析

普通聊天只能处理用户输入的文字。当用户希望让模型阅读 PDF、Word、Excel 或代码文件时，前端需要完成三件事：

1. 选择或拖拽文件；
2. 在浏览器端解析文件内容；
3. 把解析后的文本和用户问题一起发送给模型。

DeepSeek Harness 的实现遵循“一切皆插件”的思路。文件解析能力不直接堆进聊天组件，而是拆成独立附件插件，再由 `ui-conversation` 统一编排。

## 二、整体处理流程

```text
文件选择/拖拽
      ↓
InputBar
      ↓
ConversationController
      ↓
文件类型插件
      ↓
统一文本内容
      ↓
Session.prompt()
      ↓
模型回答
```

图片和文本文件的处理方式不同：

- 图片转换为 Base64 图片块，交给支持视觉输入的模型；
- TXT、Markdown、CSV、代码文件直接读取文本；
- PDF、DOC/DOCX、XLS/XLSX 先解析成文本，再作为上下文发送。

## 三、各类文件的解析实现

### 1. 图片

图片沿用 Harness 原有的附件协议，最终发送为：

```ts
{
  type: 'image',
  mediaType: 'image/png',
  data: base64Data
}
```

图片不会被转换成文本，因此模型必须支持视觉输入。

### 2. 文本和代码文件

实现位于：

```text
packages/attachment/attachment-document/src/index.ts
```

支持 TXT、Markdown、CSV、JSON、XML、HTML，以及常见代码文件。核心逻辑是浏览器原生的：

```ts
const text = await file.text()
```

文本文件的限制由 `DOCUMENT_MAX_BYTES` 控制，目前统一为 20MB。

### 3. PDF

实现位于：

```text
packages/attachment/attachment-pdf/src/index.ts
```

PDF 使用 `pdfjs-dist` 提取每一页的文本层：

```ts
const pdf = await getDocument({
  data: new Uint8Array(await file.arrayBuffer())
}).promise
```

解析结果会添加页码标记：

```text
[Page 1]
第一页文本

[Page 2]
第二页文本
```

扫描版 PDF 通常没有文本层，因此当前实现会提示需要 OCR。PDF Worker 由 `apps/web/vite.config.ts` 在构建时输出为：

```text
apps/web/dist/pdf.worker.mjs
```

同时，`packages/host/frontend-static/src/index.ts` 将 `.mjs` 映射为 `text/javascript`，否则浏览器会因 MIME 类型为 `application/octet-stream` 而拒绝执行 Worker。

### 4. DOC 和 DOCX

实现位于：

```text
packages/attachment/attachment-word/src/index.ts
```

DOCX 使用 Mammoth 的浏览器构建：

```ts
import mammoth from 'mammoth/mammoth.browser.js'

const result = await mammoth.extractRawText({ arrayBuffer })
```

旧式 `.doc` 是二进制格式，浏览器没有原生解析接口，因此插件采用 best-effort 文本提取。复杂排版、图片、嵌入对象不保证完整还原，实际使用中建议优先转换为 DOCX 或 PDF。

### 5. CSV、XLS 和 XLSX

实现位于：

```text
packages/attachment/attachment-spreadsheet/src/index.ts
```

表格文件使用 SheetJS 解析，每个工作表转换为带名称的 CSV 文本：

```ts
const workbook = XLSX.read(await file.arrayBuffer(), {
  type: 'array',
  cellDates: true
})

const text = workbook.SheetNames.map(name =>
  `[Sheet: ${name}]\n${XLSX.utils.sheet_to_csv(workbook.Sheets[name])}`
).join('\n\n')
```

这样模型接收到的不是二进制 Excel 文件，而是结构清晰的表格文本。

## 四、前端为什么需要解析状态

解析大型 PDF 或 Excel 不是同步操作。如果用户在解析完成前点击发送，可能出现以下问题：

- 发送了不完整的文本；
- 解析 Promise 和会话提交发生竞态；
- 页面看起来没有反应；
- 解析失败后仍然提交空内容。

因此 `ComposerAttachment` 增加了本地状态：

```ts
parseStatus: 'parsing' | 'ready' | 'error'
parseProgress?: number
parsedText?: string
parseError?: string
```

文件选中后立即开始解析。PDF 按页更新进度，XLSX 在读取和转换阶段更新进度。解析期间附件缩略图显示百分比，发送按钮被禁用；只有 `parseStatus === 'ready'` 时才允许提交。

相关文件：

```text
packages/client/ui-conversation/src/client/contract/slots.ts
packages/client/ui-conversation/src/client/service.ts
packages/client/ui-conversation/src/client/skeleton/InputBar.tsx
packages/client/ui-attachment/src/AttachmentRail.tsx
```

## 五、统一发送格式

在 `ConversationController.serializeAttachments()` 中，图片和文档被转换成模型请求内容：

```ts
if (attachment.kind === 'image') {
  blocks.push({ type: 'image', data, mediaType })
} else {
  blocks.push({
    type: 'text',
    text: `[Attached document: ${name}]\n${parsedText}\n[/Attached document]`
  })
}
```

因此后端模型只需要处理标准文本块和图片块，不需要知道浏览器具体使用了哪个解析库。

## 六、集成过程中遇到的问题

### 1. Node 模块被带入浏览器

PDF.js、Mammoth 和 SheetJS 都有 Node 入口。如果直接导入默认入口，可能出现：

```text
require("url") missed the module table
require("stream") missed the module table
```

解决方式是选择浏览器入口，并把纯浏览器依赖内联到客户端插件：

- Mammoth：`mammoth.browser.js`；
- SheetJS：`xlsx/dist/xlsx.full.min.js`；
- PDF.js：浏览器构建入口；
- `packages/client/tsdown.client.ts` 将附件解析包加入 `INLINE_SAFE`。

### 2. PDF Worker MIME 类型错误

静态服务器原本只识别 `.js`，`.mjs` 会返回 `application/octet-stream`。浏览器会报：

```text
Expected a JavaScript-or-Wasm module script
```

解决方式是：

```ts
'.mjs': 'text/javascript; charset=utf-8'
```

### 3. 进度实际变化但界面仍显示 0%

解析状态是对象内部字段变化，React 的附件列表又被 `useMemo` 缓存，导致对象已更新但缩略图没有重算。最终通过 `attachmentRevision` 触发附件列表重新计算。

## 七、DeepSeek Harness 插件是如何构建的

DeepSeek Harness 的插件不是简单的一个前端组件，而是一个可以被 Cordis Loader 发现、加载和销毁的独立 workspace package。一个典型插件至少包含：

```text
packages/<domain>/<plugin>/
├── package.json
├── tsconfig.json
├── tsdown.config.ts
├── src/index.ts
├── src/invariant.ts
└── README.zh.md
```

如果插件有浏览器界面，还会增加：

```text
src/client/index.ts
src/client/apply.ts
src/client/*.tsx
```

### 1. 从 package.json 声明插件身份

以 PDF 附件插件为例：

```json
{
  "name": "@deepseek-ai/dsh-attachment-pdf",
  "version": "0.1.0-rc.5",
  "type": "module",
  "main": "lib/index.js",
  "types": "lib/types/index.d.ts",
  "exports": {
    ".": {
      "types": "./lib/types/index.d.ts",
      "default": "./lib/index.js"
    }
  },
  "dependencies": {
    "pdfjs-dist": "^5.4.54"
  }
}
```

这里最重要的是 `main`、`types` 和 `exports`。源码不会直接被生产环境加载，构建后的 `lib/index.js` 才是 Host Loader 使用的入口，`lib/types` 则是 TypeScript project reference 消费的类型入口。

### 2. 用 tsconfig 接入 workspace 类型图

```json
{
  "extends": "../../../tsconfig.base.client.json",
  "compilerOptions": {
    "rootDir": "src",
    "outDir": "lib/types"
  },
  "include": ["src"]
}
```

大型插件不会通过复制文件来解决依赖，而是通过 `references` 接入其他 workspace package。这样 `tsc -b` 可以按依赖顺序构建，并在跨包 API 变化时及时报错。

### 3. 用 tsdown 声明构建类型

附件解析插件是浏览器安全的 Client Library，因此使用：

```ts
import { clientLibrary } from '../../client/tsdown.client.ts'

export default clientLibrary(
  '@deepseek-ai/dsh-attachment-pdf',
  ['lib/types/index.js'],
)
```

`clientLibrary()` 会根据当前构建面生成对应配置。完整 UI 插件通常使用 `clientBundle()`，同时生成 Node/Host 侧库和浏览器 Client bundle：

```ts
export default clientBundle(
  '@deepseek-ai/dsh-client-ui-conversation',
  ['lib/types/index.js', 'lib/types/invariant.js'],
  { hostPhase: true },
)
```

### 4. Cordis 插件通过 inject 和 apply 组织运行时

一个带服务的 Host 插件通常会声明依赖并注册服务：

```ts
import { Service } from '@deepseek-ai/cordis'
import type { Context } from '@deepseek-ai/cordis'

export const inject = ['sessions', 'locale']

export class ExampleService extends Service {
  constructor(ctx: Context) {
    super(ctx, 'example')
  }
}

export function apply(ctx: Context): void {
  ctx.plugin(ExampleService)
}
```

`inject` 表示插件启动前必须存在的服务；`apply(ctx)` 是插件真正注册行为的地方。若依赖服务尚未出现，Loader 会让插件保持 pending，而不是让插件在错误状态下运行。

UI 插件通常将 Cordis 服务和 React Slot 分开：

```ts
export const inject = [
  'slots',
  'sessions',
  'locale',
  'connection',
]

export function apply(ctx: Context): void {
  ctx.slots.register({
    name: 'conversation.input.left',
    id: 'example-control',
    render: ExampleControl,
  })
}
```

这样 UI 插件不需要修改主页面，只需占用公开 Slot。不同插件之间通过 Cordis service、Slot 和类型 contract 协作，而不是直接复制对方的运行时对象。

### 5. Client 插件如何被打包和加载

浏览器插件不是普通的 `<script>`。`packages/client/tsdown.client.ts` 会把 Client 代码包装成模块加载器工厂，形式类似：

```js
window.__ModuleLoader__.load({
  id: '@deepseek-ai/dsh-client-ui-conversation',
  factory: (require) => {
    // plugin bundle
    return module.exports
  }
})
```

插件内部依赖通过注入的 `require` 查找模块表。平台模块，例如 `connection`、`runtime`、`slots`，由 Web Seed 预先注册；普通浏览器依赖则必须被内联。

因此 `packages/client/tsdown.client.ts` 中存在 Client bundle purity 规则：

```ts
export const INLINE_SAFE = /^@deepseek-ai\/dsh-(
  host-apiproxy|session|llm|tools|brand|
  attachment-document|attachment-pdf|
  attachment-word|attachment-spreadsheet
)(\/|$)/
```

附件解析包属于无共享运行时状态的浏览器安全层，可以内联。如果错误地把 Node-only 依赖外置，就会出现：

```text
require("stream") missed the module table
require("url") missed the module table
```

这也是为什么 DOCX 使用 `mammoth.browser.js`，XLSX 使用 `xlsx/dist/xlsx.full.min.js`，而不是直接导入它们的 Node 默认入口。

### 6. 一个完整的插件构建过程

以本次新增的表格解析插件为例，实际构建链路是：

```text
attachment-spreadsheet/src/index.ts
              ↓ tsc -b
attachment-spreadsheet/lib/types/index.d.ts
              ↓ tsdown
attachment-spreadsheet/lib/index.js
              ↓ ui-conversation 内联
ui-conversation/lib/client.js
              ↓ Vite
apps/web/dist/assets/index-*.js
              ↓ dsh web
浏览器 ModuleLoader
```

对应命令：

```bash
pnpm install
pnpm run build:lib:client
pnpm --filter @deepseek-ai/dsh-web-frontend run build
```

如果是一个同时拥有 Host 和 Client 代码的完整插件，则执行：

```bash
pnpm run build:lib:host
pnpm run build:lib:client
```

### 7. 插件如何进入最终 Web 组合

构建单个 package 还不代表浏览器会加载它。最终组合通常还需要：

1. 在 Web bundle 的插件 roster 或 `cordis.patch.yml` 中声明插件；
2. Host 侧 Loader 读取插件的 `lib/index.js`；
3. Client 侧根据 manifest 请求 `lib/client.js`；
4. ModuleLoader 注册工厂；
5. 依赖满足后执行 `apply(ctx)`；
6. 插件注册服务、Slot 或 UI 组件。

这解释了为什么“代码已经构建成功”但页面仍可能出现 `pending`：构建成功只说明文件生成了，插件是否激活还取决于 roster、依赖服务和模块表是否完整。

## 八、DeepSeek Harness 一般如何构建

DeepSeek Harness 是 pnpm workspace，多数能力以独立 package 存在。常见构建流程如下：

### 1. 安装依赖

```bash
pnpm install
```

### 2. 构建 Host 侧包

Host 侧主要运行在 Node.js，负责配置、会话、服务和文件系统能力：

```bash
pnpm run build:lib:host
```

### 3. 构建 Client 侧包

Client 侧生成浏览器插件闭包，并通过模块表加载：

```bash
pnpm run build:lib:client
```

构建过程同时会执行 TypeScript project references 和 tsdown 打包。

### 4. 构建 Web 前端

```bash
pnpm --filter @deepseek-ai/dsh-web-frontend run build
```

输出位于：

```text
apps/web/dist
```

### 5. 启动 Web 服务

服务器部署时使用：

```bash
pnpm dsh web --host 0.0.0.0 --port 13080
```

浏览器访问：

```text
http://服务器IP:13080
```

修改源码后，推荐完整执行：

```bash
pnpm install
pnpm run build:lib:host
pnpm run build:lib:client
pnpm --filter @deepseek-ai/dsh-web-frontend run build
pnpm dsh web --host 0.0.0.0 --port 13080
```

## 九、总结

文件上传能力的关键不是把所有解析代码写进聊天组件，而是建立清晰的插件边界：

- 每种文件类型由独立插件负责；
- `ui-conversation` 负责生命周期、状态和发送编排；
- `ui-attachment` 负责通用附件展示；
- 构建系统负责把浏览器安全的解析依赖内联；
- Web 静态服务器负责正确提供 Worker 和模块 MIME 类型。

这种结构使新增 PPTX、图片 OCR 或更多表格格式时，只需要增加新的附件插件，再接入统一的解析接口，不必重写整个聊天页面。

## 十、关键文件的具体代码

下面代码均来自本次实际实现的核心逻辑，省略了与功能无关的注释和样式。

### 1. 表格解析插件完整核心代码

文件：`packages/attachment/attachment-spreadsheet/src/index.ts`

```ts
import XLSX from 'xlsx/dist/xlsx.full.min.js'

export const SPREADSHEET_MAX_BYTES = 20 * 1024 * 1024

export class SpreadsheetParseError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'SpreadsheetParseError'
  }
}

export function isSpreadsheetFile(file: File): boolean {
  const name = file.name.toLowerCase()
  return file.type === 'text/csv'
    || file.type === 'application/vnd.ms-excel'
    || file.type === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    || name.endsWith('.csv')
    || name.endsWith('.xls')
    || name.endsWith('.xlsx')
}

export async function readSpreadsheetFile(
  file: File,
  onProgress?: (value: number) => void,
): Promise<string> {
  if (!isSpreadsheetFile(file)) {
    throw new SpreadsheetParseError(`不是 CSV/XLS/XLSX 文件：${file.name}`)
  }
  if (file.size > SPREADSHEET_MAX_BYTES) {
    throw new SpreadsheetParseError('表格文件不能超过 20MB')
  }

  try {
    onProgress?.(10)
    const workbook = XLSX.read(await file.arrayBuffer(), {
      type: 'array',
      cellDates: true,
    })

    const sheets = workbook.SheetNames.map((name) => {
      const sheet = workbook.Sheets[name]
      if (sheet === undefined) return ''
      return `[Sheet: ${name}]\n${XLSX.utils.sheet_to_csv(sheet, {
        blankrows: false,
      })}`
    }).filter(text => text.trim() !== '')

    onProgress?.(100)
    if (sheets.length === 0) {
      throw new SpreadsheetParseError('表格中没有可提取的内容')
    }
    return sheets.join('\n\n')
  } catch (error) {
    if (error instanceof SpreadsheetParseError) throw error
    throw new SpreadsheetParseError(
      `表格解析失败：${error instanceof Error ? error.message : String(error)}`,
    )
  }
}
```

浏览器构建必须使用：

```ts
import XLSX from 'xlsx/dist/xlsx.full.min.js'
```

不要直接使用：

```ts
import * as XLSX from 'xlsx'
```

后者可能把 Node 的 `stream` 模块带进浏览器插件。

### 2. PDF 分页解析代码

文件：`packages/attachment/attachment-pdf/src/index.ts`

```ts
import { GlobalWorkerOptions, getDocument } from 'pdfjs-dist'

export const PDF_MAX_BYTES = 20 * 1024 * 1024

if (typeof window !== 'undefined') {
  GlobalWorkerOptions.workerSrc = '/pdf.worker.mjs'
}

export async function readPdfFile(
  file: File,
  onProgress?: (value: number) => void,
): Promise<string> {
  if (file.size > PDF_MAX_BYTES) {
    throw new Error('PDF 文件不能超过 20MB')
  }

  const pdf = await getDocument({
    data: new Uint8Array(await file.arrayBuffer()),
  }).promise

  const pages: string[] = []
  for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber++) {
    const page = await pdf.getPage(pageNumber)
    const content = await page.getTextContent()
    const text = content.items
      .map(item => 'str' in item ? item.str : '')
      .join(' ')
      .replace(/\s+/g, ' ')
      .trim()

    if (text) pages.push(`[Page ${pageNumber}]\n${text}`)
    onProgress?.(Math.round(pageNumber / pdf.numPages * 100))
  }

  if (pages.length === 0) {
    throw new Error('PDF 中没有可提取的文本，扫描件需要 OCR')
  }
  return pages.join('\n\n')
}
```

### 3. DOCX 浏览器解析代码

文件：`packages/attachment/attachment-word/src/index.ts`

```ts
import mammoth from 'mammoth/mammoth.browser.js'

export async function readWordFile(file: File): Promise<string> {
  const name = file.name.toLowerCase()
  const bytes = await file.arrayBuffer()

  if (name.endsWith('.docx')) {
    const result = await mammoth.extractRawText({ arrayBuffer: bytes })
    if (!result.value.trim()) throw new Error('文档中没有可提取的正文')
    return result.value.trim()
  }

  // 旧式 .doc 只能做 best-effort 二进制文本提取
  return extractLegacyDocText(new Uint8Array(bytes))
}
```

### 4. ConversationController 统一解析和缓存

文件：`packages/client/ui-conversation/src/client/service.ts`

```ts
async prepareAttachments(
  attachments: readonly ComposerAttachment[],
  notify?: (text: string) => void,
): Promise<void> {
  await Promise.all(attachments
    .filter(attachment => attachment.kind === 'document')
    .map(async (attachment) => {
      try {
        attachment.parseStatus = 'parsing'
        attachment.parseProgress = 0

        const onProgress = (value: number): void => {
          attachment.parseProgress = value
        }

        attachment.parsedText = isPdfFile(attachment.file)
          ? await readPdfFile(attachment.file, onProgress)
          : isWordFile(attachment.file)
            ? await readWordFile(attachment.file)
            : isSpreadsheetFile(attachment.file)
              ? await readSpreadsheetFile(attachment.file, onProgress)
              : await readDocumentFile(attachment.file)

        attachment.parseProgress = 100
        attachment.parseStatus = 'ready'
      } catch (error) {
        attachment.parseStatus = 'error'
        attachment.parseError = error instanceof Error
          ? error.message
          : String(error)
        notify?.(attachment.parseError)
      }
    }))
}
```

发送时优先使用已经解析好的缓存：

```ts
private async serializeAttachments(
  attachments: readonly ComposerAttachment[],
): Promise<PromptContent[]> {
  const blocks: PromptContent[] = []

  for (const attachment of attachments) {
    if (attachment.kind === 'image') {
      blocks.push({
        type: 'image',
        mediaType: imageMediaType(attachment.file.type),
        data: bytesToBase64(new Uint8Array(
          await attachment.file.arrayBuffer(),
        )),
      })
      continue
    }

    if (attachment.parseStatus === 'parsing') {
      throw new Error('附件仍在解析，请稍候')
    }
    if (attachment.parseStatus === 'error') {
      throw new Error(attachment.parseError ?? '附件解析失败')
    }

    blocks.push({
      type: 'text',
      text: `[Attached document: ${attachment.file.name}]\n`
        + `${attachment.parsedText ?? ''}\n`
        + '[/Attached document]',
    })
  }
  return blocks
}
```

### 5. 附件状态类型

文件：`packages/client/ui-conversation/src/client/contract/slots.ts`

```ts
export interface ComposerAttachment {
  kind: 'image' | 'document'
  id: DraftAttachmentId
  file: File
  previewUrl: string
  parseStatus?: 'parsing' | 'ready' | 'error'
  parseProgress?: number
  parsedText?: string
  parseError?: string
}
```

### 6. 文件选择器和提交禁用

文件：`packages/client/ui-conversation/src/client/skeleton/InputBar.tsx`

```tsx
<input
  ref={fileInputRef}
  type="file"
  multiple
  accept="image/*,.txt,.md,.csv,.xls,.xlsx,.pdf,.doc,.docx"
  onChange={event => {
    const files = Array.from(event.currentTarget.files ?? [])
    intakeImages(files)
    event.currentTarget.value = ''
  }}
/>
```

发送按钮根据解析状态锁定：

```tsx
const parsingAttachments = attachments.some(
  attachment => attachment.kind === 'document'
    && attachment.parseStatus === 'parsing',
)

<button
  type="button"
  disabled={empty || disabled || machineBusy || parsingAttachments}
  onClick={onPrimary}
>
  发送
</button>
```

### 7. 插件构建文件

文件：`packages/attachment/attachment-spreadsheet/tsdown.config.ts`

```ts
import { clientLibrary } from '../../client/tsdown.client.ts'

export default clientLibrary(
  '@deepseek-ai/dsh-attachment-spreadsheet',
  ['lib/types/index.js'],
)
```

文件：`packages/client/tsdown.client.ts`：

```ts
export const INLINE_SAFE = /^@deepseek-ai\/dsh-(
  attachment-document|attachment-pdf|
  attachment-word|attachment-spreadsheet
)(\/|$)/
```

### 8. PDF Worker 构建和 MIME 配置

文件：`apps/web/vite.config.ts`：

```ts
function emitPdfWorker(): Plugin {
  return {
    name: 'dsh-emit-pdf-worker',
    generateBundle() {
      this.emitFile({
        type: 'asset',
        fileName: 'pdf.worker.mjs',
        source: readFileSync(
          require.resolve('pdfjs-dist/build/pdf.worker.mjs'),
        ),
      })
    },
  }
}
```

文件：`packages/host/frontend-static/src/index.ts`：

```ts
const MIME: Record<string, string> = {
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.json': 'application/json',
}
```

### 9. 最终构建命令

```bash
# 安装新增的 workspace 依赖
pnpm install

# 构建 Host 侧插件
pnpm run build:lib:host

# 构建 Client 侧插件和浏览器 bundle
pnpm run build:lib:client

# 构建 Vite Web 前端和 PDF Worker
pnpm --filter @deepseek-ai/dsh-web-frontend run build

# 监听 0.0.0.0，使用 13080 端口
pnpm dsh web --host 0.0.0.0 --port 13080
```
