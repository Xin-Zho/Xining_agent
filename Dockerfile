# ============================================================
# Agent 学习项目 Docker 镜像
#
# 构建：
#   docker build -t agent-chat .
#
# 运行（需要 .env 文件）：
#   docker run -p 8000:8000 --env-file .env agent-chat
#
# 浏览器打开：http://localhost:8000
# ============================================================

FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 先复制依赖文件（利用 Docker 层缓存，代码变了不用重装依赖）
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "-m", "uvicorn", "web.backend:app", "--host", "0.0.0.0", "--port", "8000"]
