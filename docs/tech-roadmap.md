# Agent Learning — 技术路线图

## 架构演进总览

```
Phase 1（当前）                  Phase 2（未来）                    Phase 3（远期）
小作坊                          公司                             生态
┌──────────────┐         ┌──────────────────────┐        ┌──────────────────────┐
│ 一个 Agent    │         │ 主 Agent（管理者/接待） │        │ 多组织 Agent 网络     │
│ 16 工具       │  ──→   │ ├─ 听懂用户需求        │  ──→  │ ANP 服务发现          │
│ 0 Skill       │         │ ├─ 了解员工擅长        │        │ 跨团队协作            │
│ 全自己干       │         │ ├─ 分派任务            │        │ ...                  │
└──────────────┘         │ └─ 整合结果            │        └──────────────────────┘
                          │                       │
                          │ 员工 Agent             │
                          │ ├─ 专属工具            │
                          │ ├─ 专属 Skill          │
                          │ └─ 互不交叉            │
                          └──────────────────────┘
```

---

## Phase 1：MCP — 工具进程隔离（当前阶段）

### 目标

将现有工具中的高风险操作（网络访问、文件读写、命令执行）从 Agent 进程拆出，
封装为独立的 MCP Server 进程。工具变为可独立部署、跨语言复用、权限隔离的服务单元。

### 内容

- 14 个高风险工具按功能域拆为 6 个 MCP Server（stdio 传输）
- 2 个低风险工具（calculator、timer_set）保留进程内
- 引擎通过 `ToolRegistry` 统一管理本地工具 + MCP 工具
- 现有 REST API / WebSocket / Tool dataclass 向后兼容

### 不做

- A2A、ANP
- Agent 间协作
- Skill 系统

详见：[MCP 集成设计文档](superpowers/specs/2026-06-17-mcp-integration-design.md)

---

## Phase 2：A2A — Agent 间协作（MCP 完成后评估）

### 前置条件（全部满足后再启动）

1. MCP 工具系统稳定运行
2. 项目中出现**至少 2 个职责不同的 Agent**（例如：主接待 Agent + 代码执行 Agent）
3. Agent 拥有的工具互不交叉，有明确的职责边界
4. 存在至少一个**跨运行时**的 Agent（例如 ClaudeCode CLI 子进程）

### A2A 的核心价值

**A2A 的价值不是让 Agent 协作更快，而是：**

1. **工具与 Skill 不混淆**
   - 没有 A2A：LLM 看到的工具列表扁平化，分不清"调 calculator 马上出结果"和"委派给另一个 Agent 需要几十秒返回完整推理"
   - 有 A2A：Agent Card 作为独立的"元工具"，与原子工具分层清晰

2. **Agent 间边界干净**
   - 每个 Agent 是黑盒服务，内部实现（工具、LLM 后端、推理策略）对调用方透明
   - 换 LLM 后端、拆到独立进程——Card URL 不变，调用方一行不改

3. **职责不交叉**
   - 新工具/新 Skill 只分配给专属 Agent
   - 管理 Agent 负责路由，不拥有执行工具

### A2A 协议设计方向

#### Agent Card（能力声明）

每个 Agent 启动时注册一张 Card：

```
┌─────────────────────────────────────────┐
│ Agent: code-executor                    │
│ 通信方式: POST /a2a/task                │
│ 拥有工具: read_file, execute_command,   │
│           edit_file, grep_files          │
│ 拥有 Skill: 代码重构, 测试生成            │
│ 预期行为: 收到任务 → 多步执行 → 返回结果   │
│ 超时预期: ~60s                           │
│ LLM 后端: Claude (claude-code CLI)       │
└─────────────────────────────────────────┘
```

#### 任务委派

```
主 Agent（DeepSeek）                  员工 Agent（ClaudeCode CLI）
  │                                         │
  │ "重构 backend/tools.py"                  │
  │ 自己有重构 Skill? ❌                      │
  │ 查 Agent Card → code-executor 有 ✅      │
  │                                         │
  ├── POST /a2a/task ──────────────────────>│
  │   { task: "...", context: {...} }        │
  │                                         ├── 子进程开始执行
  │                                         ├── 多步推理 + 工具调用
  │<── Response ────────────────────────────┤
  │   { status, result, plan, steps }        │
  │                                         │
  ├── 反思验证（元数据 + LLM 反思）           │
  ├── 整合 → 最终回答                        │
```

#### 架构

```
主 Agent（管理者/接待）                  员工 Agent（执行者）
┌────────────────────┐          ┌────────────────────┐
│ 工具: web_search    │          │ 工具: read_file     │
│       stock_query   │  A2A    │       execute_cmd   │
│       calculator    │ ◄─────> │       edit_file     │
│                    │  Task   │       grep_files    │
│ Skill: 需求理解     │  Response│                    │
│        任务路由     │          │ Skill: 代码重构      │
│        结果整合     │          │        测试生成      │
└────────────────────┘          └────────────────────┘
  工具不交叉                      工具不交叉
  Skill 不交叉                    Skill 不交叉
```

### 关键设计原则

1. **工具不交叉** — 每个工具/只属于一个 Agent。新增工具时明确归属
2. **Skill 跟随 Agent** — Agent 的 Skill 是它的专业知识域，不像工具那样可独立部署
3. **Card 发现 = 工具列表扩展** — 不需要单独的 RAG 管道。远程 Agent 的能力以"元工具"形式注册到主 Agent 的工具列表，LLM 按需选择
4. **验证靠元数据 + 反思** — 返回结果带 plan/steps/sources/confidence，主 Agent 反思验证
5. **失败兜底** — 委派失败 → 带提示重试 → 重试耗尽 → 主 Agent 自行处理
6. **委派是优化路径，不是唯一路径**

---

## Phase 3：ANP — 多组织网络（远期）

### 前置条件

- 多个独立部署的 Agent 服务
- 需要跨组织、跨团队的服务发现
- 需要去中心化的 Agent 路由

### 方向

- 服务注册与发现
- 去中心化 Agent 路由
- Agent 信誉系统

**当前不做，技术栈待定。**

---

## 工具与 Skill 分类原则

### 当前约定（Phase 1）

| 类别 | 判断标准 | 部署方式 |
|------|---------|---------|
| 高风险工具 | 网络访问、文件读写、系统调用 | MCP Server（独立进程） |
| 低风险工具 | 纯计算、无副作用 | Agent 进程内 |

### 未来约定（Phase 2+）

| 类别 | 归属 | 添加方式 |
|------|------|---------|
| 共享工具 | 所有 Agent 可用 | 注册到 ToolRegistry |
| 专属工具 | 只属于一个员工 Agent | 注册到该 Agent 的 MCP Server |
| Skill | 只属于一个员工 Agent | 注册到该 Agent 的 Skill 系统 |
| Agent Card | 主 Agent 可见 | 注册到 AgentRegistry |

---

## 版本记录

| 日期 | 内容 | 作者 |
|------|------|------|
| 2026-06-17 | 初始版本：Phase 1 MCP 路线确认，Phase 2 A2A 方向记录 | Xin-Zho + Claude |
