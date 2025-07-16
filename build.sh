#!/bin/bash
# Linux/macOS 构建脚本

set -e  # 遇到错误立即退出

echo "🚀 开始构建图像识别系统..."

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装"
    exit 1
fi

# 检查pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 未安装"
    exit 1
fi

# 创建虚拟环境 (可选)
if [ "$1" = "--venv" ]; then
    echo "🔧 创建虚拟环境..."
    python3 -m venv build_env
    source build_env/bin/activate
    echo "✅ 虚拟环境已激活"
fi

# 安装依赖
echo "📦 安装构建依赖..."
pip3 install --upgrade pip
pip3 install pyinstaller

# 安装项目依赖
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
else
    echo "⚠️  requirements.txt 不存在，手动安装依赖..."
    pip3 install PyQt5 opencv-python numpy tensorflow scikit-learn scikit-image psutil
fi

# 运行构建脚本
echo "🔨 运行PyInstaller构建..."
python3 build.py

# 设置执行权限
if [ -d "dist/ImageRecognitionSystem" ]; then
    chmod +x dist/ImageRecognitionSystem/ImageRecognitionSystem
    chmod +x dist/ImageRecognitionSystem/*.sh
    echo "✅ 执行权限已设置"
fi

echo "🎉 构建完成！"
echo "📁 输出目录: $(pwd)/dist"

# 清理虚拟环境
if [ "$1" = "--venv" ]; then
    deactivate
    echo "🧹 虚拟环境已停用"
fi
