@echo off
REM Windows 构建脚本

echo 🚀 开始构建图像识别系统...

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python 未安装或未添加到PATH
    pause
    exit /b 1
)

REM 检查pip
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip 未安装
    pause
    exit /b 1
)

REM 创建虚拟环境 (可选)
if "%1"=="--venv" (
    echo 🔧 创建虚拟环境...
    python -m venv build_env
    call build_env\Scripts\activate.bat
    echo ✅ 虚拟环境已激活
)

REM 升级pip
echo 📦 升级pip...
python -m pip install --upgrade pip

REM 安装PyInstaller
echo 📦 安装PyInstaller...
pip install pyinstaller

REM 安装项目依赖
if exist requirements.txt (
    echo 📦 安装项目依赖...
    pip install -r requirements.txt
) else (
    echo ⚠️  requirements.txt 不存在，手动安装依赖...
    pip install PyQt5 opencv-python numpy tensorflow scikit-learn scikit-image psutil
)

REM 运行构建脚本
echo 🔨 运行PyInstaller构建...
python build.py

if errorlevel 1 (
    echo ❌ 构建失败
    pause
    exit /b 1
)

echo 🎉 构建完成！
echo 📁 输出目录: %cd%\dist

REM 清理虚拟环境
if "%1"=="--venv" (
    call deactivate
    echo 🧹 虚拟环境已停用
)

pause
