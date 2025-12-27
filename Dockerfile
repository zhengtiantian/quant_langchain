FROM python:3.12-slim

WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install -U pip && pip install -r requirements.txt

# 复制项目文件（包括 main.py）
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "main.py"]