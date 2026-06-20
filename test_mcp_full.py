#!/usr/bin/env python3
"""MCPClientManager 完整测试 v2 — 手动 context 管理"""
import os
import sys
import asyncio
import json

os.environ['AGENT_PROJECT_ROOT'] = 'D:/agent_learning'
sys.path.insert(0, 'D:/agent_learning')

from backend.protocols.mcp import ToolRegistry, MCPClientManager
from backend.agent.tools import LOCAL_TOOLS


async def test_full_lifecycle():
    # ── Step 1: 注册本地工具 ──
    print("1. 创建 ToolRegistry + 注册本地工具...")
    registry = ToolRegistry()
    for tool in LOCAL_TOOLS:
        registry.register_local(tool)
    print(f"   {len(registry._local_tools)} 个本地工具: {[t.name for t in registry._local_tools]}")

    # ── Step 2: 初始化 ToolRegistry (内含 connect_all) ──
    print("2. 初始化 ToolRegistry (含 connect_all + 发现 MCP 工具 → freeze)...")
    manager = MCPClientManager(project_root='D:/agent_learning')

    try:
        await registry.initialize(manager)
        all_tools = registry.get_all_tools()
        print(f"   初始化成功! {len(all_tools)} 个工具:")
        for t in all_tools:
            print(f"     - {t.name} (confirm={t.require_confirmation})")
    except Exception as e:
        print(f"   初始化失败: {e}")
        import traceback; traceback.print_exc()
        await manager.shutdown()
        return

    # ── Step 3: 验证工具调用 ──
    print("4. 验证 call_tool...")
    # calculator (本地)
    from backend.agent.tools import set_current_user
    set_current_user(1)

    calc = next(t for t in all_tools if t.name == 'calculator')
    result = await calc.handler(expression='2+3*5')
    print(f"   calculator(2+3*5) = {result}")

    # grep_files (MCP)
    grep = next(t for t in all_tools if t.name == 'grep_files')
    try:
        result = await grep.handler(pattern='calculator', glob='*.py', path='backend/agent')
        print(f"   grep_files: {json.dumps(result, ensure_ascii=False)[:150]}...")
    except Exception as e:
        print(f"   grep_files 错误: {e}")

    # ── Step 4: shutdown ──
    print("5. Shutdown...")
    await registry.shutdown()
    print("   shutdown 完成!")

    print("\n=== 全生命周期测试完成 ===")


if __name__ == "__main__":
    asyncio.run(test_full_lifecycle())