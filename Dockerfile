# Kiro API Proxy - Docker 镜像
# 多阶段构建，优化镜像大小

FROM python:3.11-slim AS base

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 创建非 root 用户
RUN groupadd --gid 1000 kiro && \
    useradd --uid 1000 --gid kiro --shell /bin/bash --create-home kiro

WORKDIR /app

# ==================== 依赖安装阶段 ====================
FROM base AS dependencies

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# ==================== 最终镜像 ====================
FROM base AS final

# 从依赖阶段复制已安装的包
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# 复制项目文件
COPY --chown=kiro:kiro . .

# 创建配置目录
RUN mkdir -p /home/kiro/.kiro-proxy && \
    chown -R kiro:kiro /home/kiro/.kiro-proxy

# 切换到非 root 用户
USER kiro

# 暴露端口
EXPOSE 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8080/api/status', timeout=5)" || exit 1

# 设置数据卷
VOLUME ["/home/kiro/.kiro-proxy"]

# 启动命令
CMD ["python", "run.py", "serve", "-p", "8080"]
