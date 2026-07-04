#!/usr/bin/env python3
"""MCP Server: network — web_search, web_fetch, stock_query"""
import os
import sys

# §4.2: 环境变量传递 PROJECT_ROOT
PROJECT_ROOT = os.environ.get("AGENT_PROJECT_ROOT")
if not PROJECT_ROOT:
    raise RuntimeError("AGENT_PROJECT_ROOT environment variable required")

BING_API_KEY = os.environ.get("BING_API_KEY", "").strip()
BING_ENABLED = bool(BING_API_KEY)

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool as MCPToolType, TextContent

server = Server("network")

# ── 工具定义 ──────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools():
    return [
        MCPToolType(
            name="web_search",
            description="智能搜索引擎，自动感知时效需求。查行情/新闻/最新信息自动开启24h过滤，无结果自动回退。需要精确信息时配合 web_fetch 抓取详情页。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词。英文更精准，中文亦可。"},
                    "max_results": {"type": "integer", "description": "返回结果数，默认5，最多10"},
                    "fresh": {"type": "string", "description": "时效过滤: d(24h内) / w(一周) / m(一月)。查实时信息用d，留空自动判断"},
                    "user_id": {"type": "integer", "description": "internal: user id"},
                },
                "required": ["query"],
            },
        ),
        MCPToolType(
            name="web_fetch",
            description="抓取指定URL的网页内容，返回页面文本。用于获取搜索结果的详情页面。",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "要抓取的网页URL"},
                    "user_id": {"type": "integer", "description": "internal: user id"},
                },
                "required": ["url"],
            },
        ),
        MCPToolType(
            name="stock_query",
            description="查询A股实时行情。action='top'涨幅榜/'down'跌幅榜/'volume'成交量榜。市场: a=A股/kcb=科创板/cyb=创业板。返回实时价格、涨跌幅、成交量。",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "查询类型: top(涨幅榜), down(跌幅榜), volume(成交量榜)"},
                    "market": {"type": "string", "description": "市场: a(A股), kcb(科创板), cyb(创业板)"},
                    "count": {"type": "integer", "description": "返回数量，默认10"},
                    "user_id": {"type": "integer", "description": "internal: user id"},
                },
                "required": ["action"],
            },
        ),
    ]


# ── Handler ──────────────────────────────────────────────────────────

async def handle_web_search(query: str, max_results: int = 5, fresh: str = "", **kwargs) -> dict:
    """智能搜索：自动感知时效需求，优先返回最新结果。"""
    from datetime import datetime, timedelta, timezone

    user_id = kwargs.get("user_id", 0)

    CST = timezone(timedelta(hours=8))
    now = datetime.now(CST)

    target_date = now
    if "前天" in query:
        target_date = now - timedelta(days=2)
    elif "昨天" in query or "昨日" in query:
        target_date = now - timedelta(days=1)
    elif "今天" in query or "今日" in query:
        target_date = now
    elif "明天" in query or "明日" in query:
        target_date = now + timedelta(days=1)
    elif "上周" in query:
        target_date = now - timedelta(weeks=1)
    elif "上个月" in query or "上月" in query:
        target_date = now - timedelta(days=30)
    elif "本周" in query:
        target_date = now

    time_keywords = ["今日", "今天", "昨天", "昨日", "最新", "实时", "刚刚",
                     "涨幅", "跌", "股票", "行情", "新闻", "发布", "公布",
                     "today", "latest", "news", "stock", "breaking", "本周", "这周"]
    if not fresh:
        for kw in time_keywords:
            if kw in query:
                fresh = "d"
                break

    if fresh == "d" and abs((target_date - now).days) > 1:
        fresh = "w"

    date_str = target_date.strftime("%Y年%m月%d日")
    has_time_word = any(kw in query for kw in
        ["昨天", "昨日", "今日", "今天", "最新", "前天", "明天", "上周", "本周", "这周"])
    if has_time_word:
        query = f"{date_str} {query}"

    items = []

    # ── 百度搜索（国内优先）───
    try:
        import urllib.request as _ureq, re as _re
        baidu_url = f"https://www.baidu.com/s?wd={_ureq.quote(query)}&rn={max_results}"
        req = _ureq.Request(baidu_url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        with _ureq.urlopen(req, timeout=6) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        for m in _re.finditer(r'<h3[^>]*>.*?<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, _re.DOTALL):
            url = m.group(1)
            title = _re.sub(r'<[^>]+>', '', m.group(2)).strip()
            if url.startswith("http") and title and "百度" not in title:
                items.append({"title": title, "url": url, "snippet": "", "source": "baidu"})
                if len(items) >= max_results: break
    except Exception:
        pass

    # ── cn.bing.com 抓取兜底（国内直连，免 API Key）───
    if not items:
        try:
            import urllib.request as _ureq_bing
            cn_bing_url = f"https://cn.bing.com/search?q={_ureq_bing.quote(query)}&count={max_results}&setlang=zh-cn"
            bing_req = _ureq_bing.Request(cn_bing_url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            with _ureq_bing.urlopen(bing_req, timeout=8) as resp:
                bing_html = resp.read().decode("utf-8", errors="ignore")
            for m in _re.finditer(r'<li[^>]*class="[^"]*b_algo[^"]*"[^>]*>.*?<h2[^>]*>.*?<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', bing_html, _re.DOTALL):
                url = m.group(1)
                title = _re.sub(r'<[^>]+>', '', m.group(2)).strip()
                if url.startswith("http") and title:
                    items.append({"title": title, "url": url, "snippet": "", "source": "cn-bing"})
                    if len(items) >= max_results: break
        except Exception:
            pass

    # ── 搜狗抓取兜底 ──────────
    if not items:
        try:
            import urllib.request as _ureq_sg
            sogou_url = f"https://www.sogou.com/web?query={_ureq_sg.quote(query)}"
            sg_req = _ureq_sg.Request(sogou_url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            with _ureq_sg.urlopen(sg_req, timeout=8) as resp:
                sg_html = resp.read().decode("utf-8", errors="ignore")
            for m in _re.finditer(r'<a[^>]*href="([^"]+)"[^>]*id="[^"]*result[^"]*"[^>]*>(.*?)</a>', sg_html, _re.DOTALL):
                url = m.group(1)
                title = _re.sub(r'<[^>]+>', '', m.group(2)).strip()
                if url.startswith("http") and title and "sogou.com" not in url:
                    items.append({"title": title, "url": url, "snippet": "", "source": "sogou"})
                    if len(items) >= max_results: break
        except Exception:
            pass

    # ── Bing API 兜底（需 BING_API_KEY）──────────
    if not items and BING_ENABLED:
        try:
            import urllib.request as _ureq2
            bing_url = f"https://api.bing.microsoft.com/v7.0/search?q={_ureq2.quote(query)}&count={max_results}&mkt=zh-CN"
            bing_req = _ureq2.Request(bing_url, headers={
                "Ocp-Apim-Subscription-Key": BING_API_KEY,
            })
            with _ureq2.urlopen(bing_req, timeout=8) as resp:
                bing_data = json.loads(resp.read().decode("utf-8"))
            for page in (bing_data.get("webPages", {}).get("value", []) or []):
                items.append({
                    "title": page.get("name", ""),
                    "url": page.get("url", ""),
                    "snippet": (page.get("snippet", "") or "")[:300],
                    "source": "bing",
                })
                if len(items) >= max_results:
                    break
        except Exception:
            pass

    # ── DuckDuckGo 兜底 ────────
    if not items:
        try:
            from duckduckgo_search import DDGS
            kwargs_ddg = {"max_results": max_results, "backend": "html"}
            if fresh:
                kwargs_ddg["timelimit"] = fresh
            with DDGS() as ddgs:
                results = list(ddgs.text(query, **kwargs_ddg))
            for r in results:
                items.append({"title": r.get("title",""), "url": r.get("href",""),
                              "snippet": (r.get("body","") or "")[:300], "source": "ddg"})
        except Exception:
            pass

    if not items:
        return {"query": query, "results": [], "count": 0,
                "hint": "均无结果，换短关键词或用 web_fetch 直接抓取URL"}

    return {"query": query, "fresh": fresh or "auto", "search_date": date_str,
            "results": items[:max_results], "count": len(items)}


async def handle_web_fetch(url: str, **kwargs) -> dict:
    """抓取指定URL的网页内容"""
    # SSRF 防护：仅允许访问公网地址
    # 从 tools.py import 安全函数
    sys.path.insert(0, PROJECT_ROOT)
    from backend.agent.tools import _is_public_url
    is_safe, reason = _is_public_url(url)
    if not is_safe:
        return {"error": reason, "url": url}

    try:
        import httpx
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers={"User-Agent": "AI-Agent/1.0"})
            resp.raise_for_status()
            text = resp.text[:8000]
        return {"url": url, "content": text, "status_code": resp.status_code}
    except Exception as e:
        return {"error": str(e), "url": url}


async def handle_stock_query(action: str = "top", market: str = "a", count: int = 10, **kwargs) -> dict:
    """查询 A 股实时行情"""
    try:
        # §P2-12: SSRF 防护 — URL 白名单
        import httpx
        allowed_host = "vip.stock.finance.sina.com.cn"

        sort_map = {"top": "changepercent", "down": "changepercent", "volume": "volume"}
        order_map = {"top": 0, "down": 1, "volume": 0}
        market_nodes = {
            "a": "hs_a",
            "kcb": "kcb",
            "cyb": "cyb",
        }

        node = market_nodes.get(market, "hs_a")
        sort_field = sort_map.get(action, "changepercent")
        asc = order_map.get(action, 0)

        url = (
            f"http://{allowed_host}/quotes_service/api/json_v2.php/"
            f"Market_Center.getHQNodeData?"
            f"page=1&num={count}&sort={sort_field}&asc={asc}"
            f"&node={node}&symbol=&_s_r_a=auto"
        )

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://finance.sina.com.cn/",
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=headers)
            data = resp.json()

        stocks = []
        for item in data:
            stocks.append({
                "code": item.get("code", ""),
                "name": item.get("name", ""),
                "price": float(item.get("trade", 0)),
                "change_pct": float(item.get("changepercent", 0)),
                "change_amount": float(item.get("pricechange", 0)),
                "volume_hand": int(item.get("volume", 0)),
                "turnover_yuan": int(item.get("amount", 0)),
                "high": float(item.get("high", 0)),
                "low": float(item.get("low", 0)),
                "open": float(item.get("open", 0)),
                "pre_close": float(item.get("settlement", 0)),
            })

        if not data or len(stocks) == 0:
            return {"error": "stock_query仅覆盖A股（沪深/科创/创业板）",
                    "hint": "该股票不在A股范围。请立即改用 web_search 搜索美股/港股行情", "stocks": []}

        action_names = {"top": "涨幅榜", "down": "跌幅榜", "volume": "成交量榜"}
        return {
            "action": action_names.get(action, action),
            "market": market,
            "count": len(stocks),
            "stocks": stocks[:count],
        }
    except Exception as e:
        return {"error": str(e), "action": action,
                "hint": "新浪接口可能暂时不可用，建议用 web_search 搜索股票行情替代"}


TOOL_HANDLERS = {
    "web_search": handle_web_search,
    "web_fetch": handle_web_fetch,
    "stock_query": handle_stock_query,
}


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False))]

    try:
        result = await handler(**arguments)
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, ensure_ascii=False))]


import json

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())