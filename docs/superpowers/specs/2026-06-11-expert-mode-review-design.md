# 专家模式主副 Agent 协作 — 设计

## 概述

将"主 Agent 产出 → 副 Agent 审校补充 → 传回主 Agent 继续执行"的协作模式嵌入专家模式（Plan-Solve）。

核心流程：

```
用户提问
    │
    ▼
主 Agent 分析 → 产出初始方案/任务清单
    │
    ▼
副 Agent 审校 → 三步方法论 → 结构化审校清单
    │
    ▼
主 Agent 消化反馈 → 修正方案 → 逐步执行 → 汇总
```

---

## 审校方法论（副 Agent System Prompt 核心）

副 Agent 审校方案时，按以下三步走：

### 第一步：读方案找锚点
- 把方案的每个步骤映射到具体目标
- 检查方案中是否存在模糊描述（"分析数据"——分析什么？怎么分析？）
- 映射不上 = 方案有漏洞

### 第二步：跑状态机穷举
- 对方案涉及的每个实体，枚举其可能状态和转换
- 检查方案是否覆盖了异常分支（失败/超时/空结果/权限不足）
- 找出方案只画了正常路径但缺失的异常路径

### 第三步：查时序和依赖
- 步骤的执行顺序是否正确？有依赖遗漏吗？
- 并行步骤之间有没有隐含的数据竞争？
- 有没有"步骤 B 需要步骤 A 的结果，但方案把 B 排在 A 前面"的情况？

---

## 架构

```
┌─────────────────────────────────────────────────────┐
│                  Expert Mode Engine                   │
│                                                       │
│  ┌──────────────────┐    ┌───────────────────────┐  │
│  │  主 Agent         │    │  副 Agent（审校者）     │  │
│  │  独立 messages    │    │  独立 messages          │  │
│  │  Plan-Solve 流程  │    │  只含方案 + 审校 Prompt  │  │
│  └──────┬───────────┘    └──────────┬────────────┘  │
│         │                           │                │
│         │  1. 产出方案               │                │
│         │─────────────────────────→│                │
│         │                           │ 2. 三步审校     │
│         │  3. 返回结构化审校清单     │                │
│         │←─────────────────────────│                │
│         │                           │                │
│         │  4. 消化审校 → 修正方案    │                │
│         │  5. 逐步执行              │                │
│         │  6. 汇总输出              │                │
│                                                       │
└─────────────────────────────────────────────────────┘
```

### 引擎隔离

- 主 Agent：`engine.py` 现有 Plan-Solve 引擎，完整上下文（用户问题 + 对话历史）
- 副 Agent：新建独立 engine 实例，独立 messages 列表，只含两个消息：
  1. System Prompt（审校方法论 + 输出格式要求）
  2. User 消息（主 Agent 产出的方案全文）
- 副 Agent 不接触用户原始问题、不接触对话历史、不接触主 Agent 的思考过程

---

## 副 Agent 审校产出格式

结构化 JSON，四个字段：

```json
{
  "missing_steps": [
    {
      "description": "遗漏的具体步骤",
      "why_important": "为什么这个步骤必不可少",
      "suggested_insert_after": "建议插在哪个步骤之后"
    }
  ],
  "flawed_logic": [
    {
      "step_reference": "有问题的步骤描述",
      "issue": "逻辑哪里有问题",
      "suggested_fix": "建议怎么修正"
    }
  ],
  "boundary_gaps": [
    {
      "scenario": "未被覆盖的边界情况",
      "impact": "如果发生会影响什么",
      "suggested_handling": "建议如何处理"
    }
  ],
  "suggestions": [
    {
      "aspect": "优化方向（性能/可读性/鲁棒性/用户体验）",
      "current_approach": "当前方案的做法",
      "alternative": "替代方案或改进点"
    }
  ]
}
```

四个字段均可为空数组。如果全部为空，表示副 Agent 认为方案没有结构性问题。

---

## 主 Agent 消化审校

副 Agent 的审校结果以 system message 形式注入主 Agent 的 messages：

```
[system] 方案审校结果如下。请逐条判断是否采纳，修正方案后继续执行。

审校清单：
- missing_steps: [...]
- flawed_logic: [...]
- boundary_gaps: [...]
- suggestions: [...]
```

主 Agent 在下一轮 Thought 中：
1. 逐条回应审校意见（采纳/拒绝/部分采纳 + 理由）
2. 修正方案/任务清单
3. 给出回应后继续执行阶段

---

## 用户可见性：半透明

- 默认视图：主 Agent 产出方案后，用户看到一个 **"方案已优化"** 标记（可展开）
- 展开后：显示副 Agent 的审校清单 + 主 Agent 的逐条回应
- 主流程步骤卡片不展示副 Agent 的内部工作，保持界面简洁
- 实现：StepListItem union 类型中新增 `{ kind: "review_banner", data: ReviewEntry }`

---

## 改动清单

| 文件 | 改动 |
|------|------|
| `backend/agent/plan_solve_engine.py` | 规划阶段后插入审校步骤；消化审校结果后修正方案；恢复执行 |
| `backend/agent/review_prompt.py` | 新建：副 Agent 的 System Prompt（三步方法论 + JSON 输出格式） |
| `backend/agent/engine.py` | 支持创建独立副 Agent 实例（共享 tool registry 但不共享 messages） |
| `mobile/src/components/ReviewBanner.tsx` | 新建："方案已优化"可展开标记组件 |
| `mobile/app/agent/[id].tsx` | 步骤列表支持 review_banner 类型 |
