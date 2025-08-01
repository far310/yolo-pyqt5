#!/bin/bash

# 设置主入口文件
ENTRY_FILE="main.py"

# 设置打包后的程序名称
APP_NAME="objectDetection"

# 是否为单文件打包（true/false）
ONEFILE=true

# 资源文件列表，格式为 "源路径:目标路径"
DATA_FILES=(
  "img:img"
  "model:model"
  "out:out"
)

# 清理旧构建
rm -rf build dist "${APP_NAME}.spec"

# 构建命令
CMD="pyinstaller"

# 单文件模式
if $ONEFILE; then
  CMD+=" --onefile"
fi

# 应用名称
CMD+=" --name $APP_NAME"

# 添加每个资源文件
for DATA in "${DATA_FILES[@]}"; do
  CMD+=" --add-data \"$DATA\""
done

# 禁用确认
CMD+=" --noconfirm"

# 添加主程序文件
CMD+=" $ENTRY_FILE"

# 执行命令
eval $CMD