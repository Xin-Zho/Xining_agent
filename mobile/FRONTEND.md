# 前端说明

## 1. 项目定位

`mobile/` 是这个项目的前端应用目录，技术栈为：

- Expo SDK 56
- React Native 0.85
- Expo Router
- Zustand
- Axios

当前前端同时支持：

- Web 调试运行
- iOS / Android 端运行

入口由 `expo-router/entry` 提供，路由文件集中在 `app/`，业务实现集中在 `src/`。

## 2. 目录结构

```text
mobile/
├─ app/                  路由层，页面入口
│  ├─ _layout.tsx        根布局，挂载 AuthProvider，决定进认证流还是主应用流
│  ├─ (auth)/            认证相关页面
│  │  ├─ _layout.tsx     认证路由组，已登录时重定向到首页
│  │  ├─ login.tsx       登录页
│  │  └─ register.tsx    注册页
│  ├─ (tabs)/            主应用 Tab 路由组
│  │  ├─ _layout.tsx     Tab 配置，未登录时重定向到登录页
│  │  ├─ index.tsx       对话列表页
│  │  └─ tasks.tsx       Agent 任务列表页
│  ├─ chat/[id].tsx      单个对话详情页
│  └─ agent/
│     ├─ new.tsx         新建 Agent 任务页
│     └─ [id].tsx        Agent 任务详情页
├─ src/
│  ├─ api/               后端接口封装
│  ├─ components/        可复用 UI 组件
│  ├─ contexts/          React Context，当前主要是认证上下文
│  ├─ hooks/             业务 Hook，当前主要是 Agent WebSocket 状态管理
│  ├─ store/             Zustand 状态
│  └─ utils/             工具函数和跨平台适配
├─ assets/               图标与静态资源
├─ app.json              Expo 应用配置
├─ package.json          前端依赖与脚本
└─ tsconfig.json         TypeScript 配置
```

## 3. 前端分层

当前代码可以拆成四层：

### 3.1 路由层

由 `app/` 目录承担，负责页面入口和导航边界。

- `app/_layout.tsx`
  负责全局入口，读取认证状态，决定进入：
  - `(auth)` 认证流
  - `(tabs)` 主应用流
  - `chat/[id]`、`agent/new`、`agent/[id]` 等详情页

- `app/(auth)/_layout.tsx`
  负责认证页面组。如果用户已登录，直接重定向到 `/`。

- `app/(tabs)/_layout.tsx`
  负责主应用底部 Tab。如果未登录，直接重定向到 `/login`。

这一层的职责是“页面结构和路由守卫”，不处理实际业务请求。

### 3.2 页面层

由 `app/**/*.tsx` 里的页面文件承担，负责：

- 页面状态组织
- 调用 `src/api`
- 调用 `src/hooks`
- 渲染 `src/components`

页面职责如下：

- `app/(auth)/login.tsx`
  用户登录入口，调用 `useAuth().login()`。

- `app/(auth)/register.tsx`
  用户注册入口，调用 `useAuth().register()`。

- `app/(tabs)/index.tsx`
  对话列表页，支持：
  - 拉取对话列表
  - 新建对话
  - 跳转聊天页
  - 退出登录

- `app/chat/[id].tsx`
  单个对话详情页，支持：
  - 加载历史消息
  - 发送消息
  - 显示加载失败 / 重试状态

- `app/(tabs)/tasks.tsx`
  Agent 任务列表页，显示任务状态、时间和 token 数据。

- `app/agent/new.tsx`
  新建 Agent 任务页，输入自然语言描述后创建任务并跳转详情页。

- `app/agent/[id].tsx`
  Agent 任务详情页，负责：
  - 拉取任务详情
  - 连接 WebSocket 获取实时步骤更新
  - 展示最终结果、错误、任务步骤和进度

### 3.3 数据访问层

由 `src/api/` 承担。

- `client.ts`
  Axios 基础客户端，负责：
  - 计算 `API_BASE`
  - 注入 JWT 请求头
  - 统一处理 `401`
  - 清除本地认证信息

- `auth.ts`
  登录 / 注册接口。

- `conversations.ts`
  对话列表、创建对话、获取单个对话。

- `chat.ts`
  发送消息接口。

- `agent.ts`
  Agent 任务列表、详情、创建、删除，以及 WebSocket 地址拼接。

这一层只负责“和后端通信”，不负责页面跳转和 UI。

### 3.4 状态与基础设施层

由 `src/contexts/`、`src/store/`、`src/hooks/`、`src/utils/` 承担。

- `contexts/AuthContext.tsx`
  封装认证能力，暴露：
  - `login`
  - `register`
  - `logout`
  - `isLoading`

  同时负责：
  - 启动时从本地存储恢复登录状态
  - 注册全局 `401` 回调

- `store/index.ts`
  使用 Zustand 管理轻量全局状态。目前包含：
  - `useAuthStore`
  - `useConversationStore`
  - `useAgentTaskStore`

  其中实际使用最核心的是 `useAuthStore`。

- `hooks/useAgentTask.ts`
  Agent 实时任务状态管理核心，封装：
  - WebSocket 建连
  - 步骤流转
  - 完成 / 失败 / 取消状态
  - 最终答案和统计数据

- `utils/storage.ts`
  跨平台存储适配：
  - Web 使用 `localStorage`
  - Native 使用 `expo-secure-store`

- `utils/format.ts`
  日期、耗时、状态文案格式化。

- `utils/notify.ts`
  统一提示入口：
  - Web 使用 `window.alert`
  - Native 使用 `Alert.alert`

## 4. 页面流转关系

核心页面流转如下：

```text
未登录
  -> /login
  -> /register

登录成功
  -> /(tabs)/index
     -> /chat/[id]
     -> /agent/new

任务创建成功
  -> /agent/[id]

主应用 Tab
  -> /(tabs)/index    对话列表
  -> /(tabs)/tasks    任务列表
```

路由守卫的实际策略是：

- 根布局根据 `isAuthenticated` 决定走哪一组页面
- `(auth)` 页面组阻止已登录用户停留在登录页
- `(tabs)` 页面组阻止未登录用户进入主界面

## 5. 关键数据流

### 5.1 认证数据流

```text
登录页 / 注册页
  -> AuthContext.login / register
  -> auth API
  -> storage 保存 jwt + username
  -> useAuthStore.setAuth
  -> RootLayout 切换到已登录路由
```

### 5.2 对话数据流

```text
对话列表页
  -> listConversations()
  -> 点击“新对话”
  -> createConversation()
  -> router.push('/chat/{id}')

聊天页
  -> getConversation(id)
  -> sendMessage(conversationId, text)
  -> UI 追加 user / assistant 消息
```

### 5.3 Agent 任务数据流

```text
新建任务页
  -> createAgentTask(description)
  -> router.replace('/agent/{id}')

任务详情页
  -> getAgentTask(id) 获取已有详情
  -> useAgentTask(id) 建立 WebSocket
  -> 接收 step_start / step_complete / task_complete 等事件
  -> UI 更新步骤、状态、最终答案
```

## 6. 当前组件边界

`src/components/` 目前是纯展示组件为主：

- `ChatInput.tsx`
  聊天输入框和发送按钮。

- `MessageBubble.tsx`
  对话消息气泡。

- `AgentProgressBar.tsx`
  Agent 步骤完成进度条。

- `AgentStepCard.tsx`
  Agent 单步详情卡片。

- `MarkdownRenderer.tsx`
  渲染 Agent 最终结果中的 Markdown 内容。

这些组件总体上保持了“展示优先”，复杂逻辑仍主要留在页面和 Hook 中。

## 7. 依赖关系总结

前端的主依赖方向是单向的：

```text
app/*
  -> src/api/*
  -> src/hooks/*
  -> src/contexts/*
  -> src/components/*
  -> src/utils/*

src/contexts/*
  -> src/api/*
  -> src/store/*
  -> src/utils/*

src/hooks/*
  -> src/api/*
  -> src/utils/*
```

这意味着当前结构总体还算清晰：

- 路由和页面在上层
- API、Hook、Store 在中层
- 工具函数和 UI 组件在底层

## 8. 当前结构的优点

- 目录规模小，入口清晰，适合继续快速开发。
- `expo-router` 让页面和路由结构一一对应，定位成本低。
- API 层和页面层已经分开，没有把请求逻辑全部塞进组件。
- 认证恢复、401 处理、WebSocket 实时状态都已经有独立封装。
- Web 和 Native 的差异主要被收敛在 `storage.ts` 和 `notify.ts`。

## 9. 当前结构的限制

当前结构可以工作，但后续复杂度上来后，会遇到这些限制：

- `useConversationStore` 和 `useAgentTaskStore` 定义了，但页面仍多用本地 state，状态来源不够统一。
- 页面文件同时承担请求、状态拼装和渲染，复杂页面会继续变重。
- 错误提示策略还不完全统一，部分页面仍直接使用 `Alert.alert`。
- API 返回类型和页面消费类型之间仍有少量耦合，尤其是 Agent 步骤数据。
- 还没有专门的“页面级容器组件 / 业务 service 层”。

## 10. 后续建议

如果后续继续扩展前端，建议按这个方向演进：

1. 统一错误提示
   把页面里的 `Alert.alert` 逐步收敛到 `utils/notify.ts`。

2. 收敛页面状态
   对“对话列表 / 任务列表 / 当前任务详情”这类跨页面状态，明确哪些放本地 state，哪些放 Zustand。

3. 拆分 Agent 详情页
   `app/agent/[id].tsx` 可以继续拆成：
   - 头部状态区
   - 步骤列表区
   - 最终结果区

4. 引入更明确的业务层
   如果后续会增加更多接口和数据转换，可以在 `src/services/` 增加一层，避免页面直接拼装过多 API 结果。

5. 补前端回归文档
   后续可以补一份页面验证清单，例如：
   - 登录
   - 退出
   - 新建对话
   - 发送消息
   - 新建 Agent 任务
   - 任务完成 / 失败 / 取消

## 11. 结论

当前前端结构属于“轻量但清晰”的 Expo Router 项目：

- `app/` 负责路由和页面入口
- `src/api/` 负责请求
- `src/contexts/` 和 `src/store/` 负责认证与全局状态
- `src/hooks/` 负责实时任务逻辑
- `src/components/` 负责复用 UI
- `src/utils/` 负责跨平台差异和格式化

对于当前规模，这个结构是合理的；如果接下来要继续扩功能，最值得优先治理的是：

- 状态统一
- 错误处理统一
- Agent 详情页继续拆分
