@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM 图像识别系统构建脚本 - Windows
REM 使用方法: build.bat [--onefile] [--clean]

echo 🚀 图像识别系统构建脚本 - Windows
echo ================================================

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python 未安装或不在 PATH 中
    pause
    exit /b 1
)

echo 🐍 Python 版本:
python --version

REM 检查虚拟环境
if "%VIRTUAL_ENV%"=="" (
    echo ⚠️  建议在虚拟环境中运行
    set /p choice="是否继续? (y/N): "
    if /i not "!choice!"=="y" (
        exit /b 1
    )
)

REM 安装/更新依赖
echo 📦 检查并安装依赖...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

REM 检查 PyInstaller
pyinstaller --version >nul 2>&1
if errorlevel 1 (
    echo 📦 安装 PyInstaller...
    python -m pip install pyinstaller
)

REM 解析参数
set ARGS=
set ONEFILE=0
set CLEAN=0

:parse_args
if "%~1"=="" goto end_parse
if "%~1"=="--onefile" (
    set ARGS=%ARGS% --onefile
    set ONEFILE=1
    echo 📦 单文件模式
) else if "%~1"=="--clean" (
    set ARGS=%ARGS% --clean
    set CLEAN=1
    echo 🧹 清理模式
) else (
    echo ❌ 未知参数: %~1
    echo 用法: %0 [--onefile] [--clean]
    pause
    exit /b 1
)
shift
goto parse_args
:end_parse

REM 执行构建
echo 🔨 开始构建...
python build.py %ARGS%

REM 检查构建结果
if errorlevel 1 (
    echo.
    echo ❌ 构建失败!
    echo 请检查上面的错误信息
    pause
    exit /b 1
)

REM 构建成功
echo.
echo 🎉 构建成功完成!
echo 📁 输出目录: .\dist\

REM 显示文件大小
if exist ".\dist\ImageRecognitionSystem" (
    for /f %%i in ('dir ".\dist\ImageRecognitionSystem" /s /-c ^| find "个文件"') do (
        echo 📊 文件数量: %%i
    )
)

echo.
echo 🚀 运行应用:
echo   cd .\dist\ImageRecognitionSystem
echo   启动_ImageRecognitionSystem.bat

REM Windows 特定提示
echo.
echo 🪟 Windows 特定提示:
echo   - 首次运行可能被 Windows Defender 拦截
echo   - 可以添加到防病毒软件白名单
echo   - 建议在目标机器上测试兼容性

if %CLEAN%==0 (
    echo.
    echo 按任意键退出...
    pause >nul
)
