# Calculate Agent

> **Branch:** `calculate_agent` | **Status:** 7/10 Tasks 完成

面向本科生及研究生的 **化学物理计算 Agent**。基于通用 Agent 底座改造——本地 Ollama 大模型 + sympy 符号计算 + 16 个科学工具 + 科学知识库 + 验证链。

---

## 当前架构

```
                        ┌── Browser (React SPA) ──────┐
                        └──────────────┬───────────────┘
                                       │ HTTP/SSE
                        ┌──────────────▼───────────────┐
                        │   FastAPI (server.py)         │
                        │   ┌───────────────────────┐   │
                        │   │  ReAct Engine          │   │
                        │   │  compute-first prompt   │   │
                        │   └───────────┬───────────┘   │
                        │               │               │
                        │   ┌───────────▼───────────┐   │
                        │   │  16 local tools         │   │
                        │   │  (zero MCP dependency)  │   │
                        │   │                         │   │
                        │   │  calculator (sympy)     │   │
                        │   │  chemistry ×5           │   │
                        │   │  physics ×5             │   │
                        │   │  search ×2 + ingest ×1  │   │
                        │   │  timer ×1 + KB ×2       │   │
                        │   └─────────────────────────┘   │
                        └──────────────┬───────────────────┘
                                       │
                ┌──────────────────────┼──────────────────────┐
                │                                             │
        ┌───────▼──────┐                           ┌──────────▼──────────┐
        │  Ollama       │                           │  ChromaDB            │
        │  qwen2.5:7b   │                           │  science_kb          │
        │  4.7 GB       │                           │  user_uploads        │
        │  (from GGUF)  │                           │  rag_documents       │
        └──────────────┘                           └─────────────────────┘
```

---

## 已完成功能（7/10 Tasks）

### Phase 1：基础设施 ✅

| Task | 内容 | 测试 |
|------|------|------|
| 1. Ollama 客户端 | `llm_client.py` 重写，OpenAI 兼容端点 | 6/6 PASS |
| 2. sympy 计算器 | diff/integrate/solve/evalf/unit，代码注入拦截 | 11/11 PASS |
| 2.5 科学 prompt | compute → verify → search → output，强制工具调用 | 已验证 |
| 3. 前端 | React SPA (Vite+TypeScript+shadcn/ui) | 可用 |

### Phase 2：领域能力 ✅

| Task | 内容 | 测试 |
|------|------|------|
| 4. Chemistry | balance_equation, element_lookup, solution_chem, kinetics, electrochem | 9/9 PASS |
| 5. Physics | mechanics, electromagnetism, quantum, thermodynamics, error_propagation | 6/6 PASS |
| 6. 科学知识库 | 公式感知分块，3 collections（science_kb/user_uploads/rag_documents），8 来源类型，5 级置信度 | 13/13 PASS |

### 额外改动

| 改动 | 说明 |
|------|------|
| MCP 全砍 | anyio/Python 3.12 不兼容，16 个工具全改本地 |
| stock_query 删除 | 计算 Agent 不需要股票 |
| 简单路径移除 | 所有问题走完整 ReAct 循环 |
| 硬编码清理 | 全部 `deepseek-v4-pro` → `OLLAMA_MODEL`，7 处修完 |
| .env 加载 | load_dotenv + shell export |

---

## 工具清单（16 个）

| 工具 | 来源 | 功能 |
|------|------|------|
| calculator | sympy | 符号微分/积分/求解/级数，单位换算 |
| balance_equation | chemistry | 化学方程式配平（线性代数） |
| element_lookup | chemistry/mendeleev | 元素属性、分子量 |
| solution_chem | chemistry | pH 计算（强酸/弱酸 + Ka 表） |
| kinetics | chemistry | 反应动力学（0/1/2 级，半衰期） |
| electrochem | chemistry | Nernst 方程 + 标准电极电势 |
| mechanics | physics | 运动学（5 选 3 求其余） |
| electromagnetism | physics | 库仑力 |
| quantum | physics | 量子力学（仅解析可解：势阱/谐振子/氢原子） |
| thermodynamics | physics | 卡诺循环效率 |
| error_propagation | physics | 误差传递 |
| rag_search | KB | 搜索知识库（all/science_kb/user_uploads） |
| rag_ingest | KB | 文档摄入知识库 |
| web_search | tools | 多引擎网页搜索 |
| web_fetch | tools | 网页抓取 + SSRF 防护 |
| timer_set | tools | 异步延时 |

---

## 核心能力验证

| # | 测试问题 | 预期结果 | 实测 |
|---|---------|---------|------|
| 1 | `diff(x^3+sin(x), x)` | 3x²+cos(x) | ✅ 11/11 pytest |
| 2 | `Fe + Cl2 -> FeCl3 配平` | 2Fe+3Cl₂→2FeCl₃ | ✅ 9/9 pytest |
| 3 | `0.1M HCl 的 pH` | ~1.0 | ✅ 9/9 pytest |
| 4 | `氢原子基态能量` | -13.6 eV | ✅ LLM 回答正确 |
| 5 | `两个1μC电荷相距1m的力` | ~8.99×10⁻³ N | ✅ 6/6 pytest |
| 6 | `氦原子基态能量` | 拒绝回答 | ✅ He → "not analytically solvable" |

**工具层**：32 个测试全过。  
**Agent 端到端**：模型能调工具，受限于 qwen2.5:7b 推理速度（RTX 4060 8GB，首次 30s+）。

---

## 待做（3/10 Tasks）

| Task | 内容 | 重要性 | 预计工作量 |
|------|------|--------|-----------|
| **3a. RAG 增强** | 自动主题标签 + 置信度分级排序 + 公式索引 | 中 | 2d |
| **3b. Verify Server** | 量纲检查 + 数量级检查 + 符号回代验证 | 高 | 2d |
| **3c. 验证链** | 引擎 `_verify_claims()` 钩子，答案自动附带验证报告 `[✓]/[!]` | 高 | 1-2d |
| 前端 LaTeX 渲染 | React 版配 KaTeX（老 H5 已有） | 中 | 0.5d |
| 知识库内容填充 | 从 IUPAC/NIST/Wikipedia 爬取科学文献入库 | 中 | 1-2d |
| 用户上传界面 | 前端文献上传 + 入库 user_uploads | 低 | 1d |

### 优先级

```
Phase 3b (Verify) ──┬── 并行 ──> Phase 3c (验证链)
Phase 3a (RAG增强) ──┘
```

---

## 快速开始

### 前置条件

- WSL2 Ubuntu + Python 3.12
- Ollama + 模型（qwen2.5:7b 推荐，~4.7GB）
- Node.js（前端构建）

### 环境

```bash
cd /mnt/d/agent_learning
source venv_linux/bin/activate
export $(grep -v '^#' .env | xargs)
```

### 后端

```bash
bash start_server.sh
# → http://localhost:8000/app
# → API docs: http://localhost:8000/docs
# 账号: test / test123
```

### 前端（开发模式）

```bash
cd web/react
npm run dev
```

### 导入 GGUF 模型

```bash
ollama create qwen2.5:7b -f Modelfile
```

---

## 已知问题

| 问题 | 状态 | 方案 |
|------|------|------|
| qwen2.5:7b 推理慢（首次 30s+） | 使用中 | 换更小模型或 GPU 升级 |
| React 前端 LaTeX 未渲染 | 待修 | App.tsx useEffect + KaTeX CDN |
| MCP Server 全挂（anyio 兼容性） | 已隔离 | 16 工具全改本地，不再依赖 MCP |
| Ollama 注册表被墙 | 已绕过 | 从 ModelScope/浏览器下载 GGUF 手动导入 |

---

## 项目结构

```
calculate_agent/
├── backend/
│   ├── server.py                    # FastAPI 入口
│   ├── llm_client.py                # Ollama 客户端（OpenAI兼容）
│   ├── dependencies.py              # 全局单例
│   ├── agent/
│   │   ├── engine.py                # ReAct 引擎
│   │   ├── tools.py                 # 16 个本地工具
│   │   └── plan_solve_engine.py    # Plan-Solve 引擎
│   ├── memory/
│   │   ├── science_kb_ingest.py    # 科学知识库摄入 + 检索
│   │   └── embedding.py            # BGE 嵌入模型
│   └── protocols/mcp/              # MCP 层（已禁用）
├── web/
│   ├── react/                       # React 前端（Vite+TS+shadcn/ui）
│   └── static/                      # 前端构建产物
├── tests/                           # 32 个 pytest
├── start_server.sh                  # 一键启动脚本
├── Modelfile                        # Ollama 模型导入配置
└── .env                             # 环境变量
```
