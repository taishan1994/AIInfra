# pnpm 安装说明

本文记录在 Ubuntu 20.04 x86_64 环境中，为 DeepSeek Harness 安装 pnpm 的完整流程。

## 1. 项目版本要求

项目根目录 `package.json` 要求：

```json
{
  "packageManager": "pnpm@11.7.0",
  "engines": {
    "node": "^22.19.0 || >=24.0.0"
  }
}
```

因此本次使用：

```text
Node.js v22.19.0
pnpm 11.7.0
```

## 2. 检查现有环境

```bash
command -v node || true
command -v npm || true
command -v corepack || true
command -v pnpm || true

node --version 2>/dev/null || true
npm --version 2>/dev/null || true
corepack --version 2>/dev/null || true
pnpm --version 2>/dev/null || true
```

如果系统尚未安装 Node.js，以上命令不会输出对应版本。

## 3. 安装 Node.js

Ubuntu 20.04 默认 APT 源可能没有满足项目要求的 Node.js 版本。本次使用 Node.js 官方二进制包安装。

### 3.1 下载并安装

```bash
set -e

node_version='22.19.0'
archive="/tmp/node-v${node_version}-linux-x64.tar.xz"
install_dir="/usr/local/lib/node-v${node_version}"

curl -fsSL \
  "https://nodejs.org/dist/v${node_version}/node-v${node_version}-linux-x64.tar.xz" \
  -o "$archive"

mkdir -p /usr/local/lib
tar -xJf "$archive" -C /usr/local/lib

if [ ! -e "$install_dir" ]; then
  mv "/usr/local/lib/node-v${node_version}-linux-x64" "$install_dir"
fi

ln -sfn "$install_dir" /usr/local/lib/node-current
ln -sfn /usr/local/lib/node-current/bin/node /usr/local/bin/node
ln -sfn /usr/local/lib/node-current/bin/npm /usr/local/bin/npm
ln -sfn /usr/local/lib/node-current/bin/npx /usr/local/bin/npx
ln -sfn /usr/local/lib/node-current/bin/corepack /usr/local/bin/corepack
```

上述命令会将 Node.js 安装到 `/usr/local/lib/node-v22.19.0`，并将命令链接到 `/usr/local/bin`。

### 3.2 验证 Node.js

```bash
node --version
npm --version
corepack --version
```

预期结果类似：

```text
v22.19.0
10.9.3
0.34.0
```

## 4. 安装并启用 pnpm

项目使用 Corepack 管理 pnpm 版本：

```bash
corepack enable
corepack prepare pnpm@11.7.0 --activate
```

验证版本：

```bash
pnpm --version
```

预期结果：

```text
11.7.0
```

## 5. 安装项目依赖

进入项目根目录：

```bash
cd /nfs/FM/gongoubo/new_project/github/harness/deepseek-harness
```

安装 workspace 依赖：

```bash
pnpm install
```

该项目使用 pnpm workspace 管理以下内容：

- `packages/*/*`
- `apps/*`
- `examples`
- `native/landlock-run`
- `python/sdk-runtime`
- `vendor/*`
- `website`

## 6. 安装后验证

```bash
node --version
pnpm --version
pnpm list --depth 0
```

也可以执行项目的基础类型检查或构建：

```bash
pnpm run typecheck
```

完整构建命令为：

```bash
pnpm run build
```

## 7. 常用 pnpm 命令

```bash
# 安装依赖
pnpm install

# 运行 CLI
pnpm dsh --profile headless "hello"

# 构建库和 Web 前端
pnpm run build

# 运行测试
pnpm run test

# 运行类型检查
pnpm run typecheck

# 运行代码检查
pnpm run lint

# 构建文档站点
pnpm run website:build
```

## 8. 版本升级

如果项目未来修改了根目录 `package.json` 中的 `packageManager`，应以该字段为准。例如：

```bash
corepack prepare pnpm@<项目指定版本> --activate
pnpm --version
```

不要在项目中随意使用全局最新 pnpm，因为 pnpm 主版本变化可能影响 lockfile、workspace 解析和安装脚本策略。

