#!/usr/bin/env python3
"""MCP Server: document — create_excel, create_docx, create_document"""
import os
import sys

# §4.2: 环境变量传递 PROJECT_ROOT
PROJECT_ROOT = os.environ.get("AGENT_PROJECT_ROOT")
if not PROJECT_ROOT:
    raise RuntimeError("AGENT_PROJECT_ROOT environment variable required")
DOWNLOADS_DIR = os.path.join(PROJECT_ROOT, "web", "static", "downloads")

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool as MCPToolType, TextContent

server = Server("document")


# ── 工具定义 ──────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools():
    return [
        MCPToolType(
            name="create_excel",
            description="生成真正的 .xlsx Excel 文件（可多Sheet、带表头样式、自动列宽）。适合报表、数据导出、表格。",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "文件名，如 'stock_report.xlsx'"},
                    "data_json": {"type": "string", "description": "JSON数据: {\"headers\":[\"列1\",\"列2\"],\"rows\":[[\"a\",1],[\"b\",2]]} 或 [{\"title\":\"Sheet名\",\"headers\":[...],\"rows\":[...]}]"},
                    "user_id": {"type": "integer", "description": "internal: user id"},
                },
                "required": ["filename", "data_json"],
            },
        ),
        MCPToolType(
            name="create_docx",
            description="生成 .docx Word 文档。传入 Markdown 格式内容，自动转换为标题/表格/列表/引用。适合报告、方案、说明书。",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "文件名，如 'report.docx'"},
                    "markdown_content": {"type": "string", "description": "Markdown 格式的文档内容（支持 #标题/|表格|/列表/**加粗**）"},
                    "user_id": {"type": "integer", "description": "internal: user id"},
                },
                "required": ["filename", "markdown_content"],
            },
        ),
        MCPToolType(
            name="create_document",
            description="创建可下载文件（Markdown表格、CSV、HTML、Python脚本等）。生成后返回下载链接给用户。适合做报表、数据汇总、文档。",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "文件名，如 'stock_report.md' 或 'data.csv'"},
                    "content": {"type": "string", "description": "文件内容。Markdown用|表格|、HTML用标签、CSV用逗号分隔"},
                    "file_type": {"type": "string", "description": "文件类型: md/csv/html/py/txt，默认md"},
                    "user_id": {"type": "integer", "description": "internal: user id"},
                },
                "required": ["filename", "content"],
            },
        ),
    ]


# ── Handler ──────────────────────────────────────────────────────────

async def handle_create_excel(filename: str, data_json: str, **kwargs) -> dict:
    """生成 .xlsx Excel 文件"""
    # lazy import
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    import json as _json

    os.makedirs(DOWNLOADS_DIR, exist_ok=True)

    safe_name = "".join(c for c in filename if c.isalnum() or c in "._- ()[]")
    if not safe_name.endswith('.xlsx'):
        safe_name += '.xlsx'
    filepath = os.path.join(DOWNLOADS_DIR, safe_name)

    try:
        wb = openpyxl.Workbook()
        ws = wb.active

        data = _json.loads(data_json)
        sheets_data = data if isinstance(data, list) else [{"title": data.get("title", "Sheet1"), "headers": data.get("headers", []), "rows": data.get("rows", [])}]

        header_font = Font(bold=True, color="FFFFFF", size=12)
        header_fill = PatternFill(start_color="5B6AF0", end_color="5B6AF0", fill_type="solid")
        thin_border = Border(
            left=Side(style='thin'), right=Side(style='thin'),
            top=Side(style='thin'), bottom=Side(style='thin')
        )

        for idx, sheet_data in enumerate(sheets_data):
            if idx > 0:
                ws = wb.create_sheet(title=sheet_data.get("title", f"Sheet{idx+1}"))
            else:
                ws.title = sheet_data.get("title", "Sheet1")

            headers = sheet_data.get("headers", [])
            for col, h in enumerate(headers, 1):
                cell = ws.cell(row=1, column=col, value=h)
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal='center')
                cell.border = thin_border

            for row_idx, row in enumerate(sheet_data.get("rows", []), 2):
                for col_idx, val in enumerate(row, 1):
                    cell = ws.cell(row=row_idx, column=col_idx, value=val)
                    cell.border = thin_border
                    cell.alignment = Alignment(vertical='center')

            for col in ws.columns:
                max_len = 0
                for cell in col:
                    if cell.value:
                        max_len = max(max_len, len(str(cell.value)))
                ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 40)

        wb.save(filepath)
        wb.close()
        size = os.path.getsize(filepath)

        from urllib.parse import quote as _quote
        encoded_url = "/api/download/" + _quote(safe_name, safe='/')
        return {
            "ok": True, "filename": safe_name, "file_type": "xlsx",
            "size_bytes": size,
            "download_url": encoded_url,
            "clickable_link": f"[📥 下载 {safe_name}]({encoded_url})",
            "_download_info": {
                "filename": safe_name,
                "filepath": filepath,
                "size_bytes": size,
            }
        }
    except Exception as e:
        return {"error": str(e), "hint": "data_json 格式: {\"headers\":[\"列1\",\"列2\"],\"rows\":[[\"a\",1],[\"b\",2]]} 或数组形式"}


async def handle_create_docx(filename: str, markdown_content: str, **kwargs) -> dict:
    """生成 .docx Word 文档（从 Markdown 转换）"""
    # lazy import
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    import re as _re

    os.makedirs(DOWNLOADS_DIR, exist_ok=True)

    safe_name = "".join(c for c in filename if c.isalnum() or c in "._- ()[]")
    if not safe_name.endswith('.docx'):
        safe_name += '.docx'
    filepath = os.path.join(DOWNLOADS_DIR, safe_name)

    try:
        doc = Document()

        # §P1-8: _create_docx._table 函数属性 → 改为局部变量 + finally 清理
        table_data: list = []

        for line in markdown_content.split('\n'):
            line = line.strip()
            if not line:
                doc.add_paragraph()
                continue

            if line.startswith('# ') or line.startswith('## ') or line.startswith('### '):
                level = line.count('#')
                doc.add_heading(line.lstrip('# ').strip(), level=min(level, 3))
            elif line.startswith('|') and '|' in line[1:]:
                if _re.match(r'\|[\s\-:|]+\|', line):
                    continue
                cells = [c.strip() for c in line.split('|')[1:-1]]
                if not table_data:
                    table_data = doc.add_table(rows=0, cols=len(cells))
                    table_data.style = 'Light Shading Accent 1'
                row = table_data.add_row()
                for i, cell_text in enumerate(cells):
                    row.cells[i].text = cell_text
            elif line.startswith('- ') or line.startswith('* '):
                doc.add_paragraph(line[2:], style='List Bullet')
            elif _re.match(r'^\d+[\.、]', line):
                doc.add_paragraph(_re.sub(r'^\d+[\.、]\s*', '', line), style='List Number')
            elif line.startswith('> '):
                p = doc.add_paragraph(line[2:])
                p.style = 'Intense Quote'
            elif line.startswith('---'):
                doc.add_paragraph('─' * 40)
            else:
                p = doc.add_paragraph(line)
                for run in p.runs:
                    if _re.search(r'\*\*(.+?)\*\*', run.text):
                        run.bold = True
                        run.text = _re.sub(r'\*\*(.+?)\*\*', r'\1', run.text)

        doc.save(filepath)
        size = os.path.getsize(filepath)

        from urllib.parse import quote as _quote
        encoded_url = "/api/download/" + _quote(safe_name, safe='/')
        return {
            "ok": True, "filename": safe_name, "file_type": "docx",
            "size_bytes": size,
            "download_url": encoded_url,
            "clickable_link": f"[📥 下载 {safe_name}]({encoded_url})",
            "_download_info": {
                "filename": safe_name,
                "filepath": filepath,
                "size_bytes": size,
            }
        }
    except Exception as e:
        return {"error": str(e), "hint": "提供 Markdown 格式的内容，会自动转换为 Word 文档"}


async def handle_create_document(filename: str, content: str, file_type: str = "md", **kwargs) -> dict:
    """创建可下载文件"""
    os.makedirs(DOWNLOADS_DIR, exist_ok=True)

    safe_name = "".join(c for c in filename if c.isalnum() or c in "._- ()[]")
    if not safe_name:
        safe_name = f"document.{file_type}"

    filepath = os.path.join(DOWNLOADS_DIR, safe_name)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        return {"error": str(e), "filename": safe_name}

    size = os.path.getsize(filepath)

    from urllib.parse import quote as _quote
    encoded_url = "/api/download/" + _quote(safe_name, safe='/')
    return {
        "ok": True, "filename": safe_name,
        "file_type": file_type,
        "size_bytes": size,
        "download_url": encoded_url,
        "clickable_link": f"[📥 下载 {safe_name}]({encoded_url})",
        "_download_info": {
            "filename": safe_name,
            "filepath": filepath,
            "size_bytes": size,
        }
    }


TOOL_HANDLERS = {
    "create_excel": handle_create_excel,
    "create_docx": handle_create_docx,
    "create_document": handle_create_document,
}


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    import json
    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False))]

    try:
        result = await handler(**arguments)
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, ensure_ascii=False))]


async def main():
    # 预热：提前导入重量级库，避免首次工具调用超时
    import openpyxl
    from docx import Document
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())