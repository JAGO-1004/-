FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . .

# 安装 Python 依赖
RUN pip install --no-cache-dir \
    fastapi uvicorn python-multipart \
    openai anthropic \
    torch transformers \
    Pillow

EXPOSE 8000

CMD ["uvicorn", "05_部署与闭环.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
