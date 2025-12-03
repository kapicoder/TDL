# 使用一个合适的基础镜像（比如Debian或Ubuntu）
FROM ubuntu:22.04  

# 安装 QEMU 和必要的工具
RUN apt-get update && apt-get install -y \
    qemu-user-static \
    binfmt-support \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 下载并安装 qemu-arm-static
RUN curl -fsSL https://github.com/multiarch/qemu-user-static/releases/download/v6.2.0/qemu-arm-static-6.2.0-linux-x86_64 -o /usr/bin/qemu-arm-static \
    && chmod +x /usr/bin/qemu-arm-static

# 配置 Docker 在构建过程中使用 QEMU 进行跨架构支持
RUN [ "cross-build-start" ]

# 在这里添加你的构建命令，例如安装依赖或者构建应用程序
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    make \
    && rm -rf /var/lib/apt/lists/*

# 最后标志着跨平台构建过程结束
RUN [ "cross-build-end" ]

# 默认的容器命令
CMD ["bash"]
