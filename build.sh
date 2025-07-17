#!/bin/bash

# 图像识别系统构建脚本 - Linux/macOS
# 使用方法: ./build.sh [--onefile] [--clean]

set -e  # 遇到错误立即退出

echo "🚀 图像识别系统构建脚本 - Linux/macOS"
echo "================================================"

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装"
    exit 1
fi

echo "🐍 Python 版本: $(python3 --version)"

# 检查虚拟环境
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  建议在虚拟环境中运行"
    read -p "是否继续? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 安装/更新依赖
echo "📦 检查并安装依赖..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

# 检查 PyInstaller
if ! command -v pyinstaller &> /dev/null; then
    echo "📦 安装 PyInstaller..."
    pip3 install pyinstaller
fi

# 设置权限
chmod +x build.py

# 解析参数
ARGS=""
for arg in "$@"; do
    case $arg in
        --onefile)
            ARGS="$ARGS --onefile"
            echo "📦 单文件模式"
            ;;
        --clean)
            ARGS="$ARGS --clean"
            echo "🧹 清理模式"
            ;;
        *)
            echo "❌ 未知参数: $arg"
            echo "用法: $0 [--onefile] [--clean]"
            exit 1
            ;;
    esac
done

# 执行构建
echo "🔨 开始构建..."
python3 build.py $ARGS

# 构建完成
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 构建成功完成!"
    echo "📁 输出目录: ./dist/"
    
    # 显示文件大小
    if [ -d "./dist/ImageRecognitionSystem" ]; then
        echo "📊 应用大小: $(du -sh ./dist/ImageRecognitionSystem | cut -f1)"
    fi
    
    # macOS 特定处理
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo ""
        echo "🍎 macOS 特定提示:"
        echo "  - 首次运行可能需要在系统偏好设置中允许"
        echo "  - 可以使用 codesign 对应用进行签名"
    fi
    
    echo ""
    echo "🚀 运行应用:"
    echo "  cd ./dist/ImageRecognitionSystem"
    echo "  ./启动_ImageRecognitionSystem.sh"
    
else
    echo ""
    echo "❌ 构建失败!"
    echo "请检查上面的错误信息"
    exit 1
fi
