#!/usr/bin/env python3
"""
PyInstaller 构建脚本
用于将图像识别系统打包成可执行文件
"""

import os
import sys
import shutil
import subprocess
import platform
import json
from pathlib import Path

# 构建配置
BUILD_CONFIG = {
    'app_name': 'ImageRecognitionSystem',
    'version': '3.0.0',
    'author': 'Image Recognition Team',
    'description': '智能图像识别系统',
    'main_script': 'scripts/python_backend_example.py',
    'icon_file': 'assets/app_icon.ico' if platform.system() == 'Windows' else 'assets/app_icon.icns',
    'output_dir': 'dist',
    'build_dir': 'build',
    'spec_file': 'ImageRecognitionSystem.spec'
}

def check_dependencies():
    """检查构建依赖"""
    print("🔍 检查构建依赖...")
    
    required_packages = [
        'pyinstaller',
        'PyQt5',
        'opencv-python',
        'numpy',
        'tensorflow',
        'scikit-learn',
        'scikit-image',
        'psutil'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}")
    
    if missing_packages:
        print(f"\n❌ 缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖检查通过")
    return True

def clean_build():
    """清理构建目录"""
    print("🧹 清理构建目录...")
    
    dirs_to_clean = [BUILD_CONFIG['output_dir'], BUILD_CONFIG['build_dir']]
    files_to_clean = [BUILD_CONFIG['spec_file']]
    
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"  🗑️  删除目录: {dir_path}")
    
    for file_path in files_to_clean:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"  🗑️  删除文件: {file_path}")

def create_assets():
    """创建必要的资源文件"""
    print("🎨 创建资源文件...")
    
    # 创建资源目录
    assets_dir = Path('assets')
    assets_dir.mkdir(exist_ok=True)
    
    # 创建配置目录
    config_dir = Path('config')
    config_dir.mkdir(exist_ok=True)
    
    # 创建默认配置文件
    default_config = {
        "app": {
            "name": BUILD_CONFIG["app_name"],
            "version": BUILD_CONFIG["version"],
            "debug": False
        },
        "camera": {
            "default_index": 0,
            "resolution": [1920, 1080],
            "fps": 30
        },
        "detection": {
            "confidence_threshold": 0.6,
            "nms_threshold": 0.45,
            "max_detections": 100
        }
    }
    
    config_file = config_dir / 'default.json'
    if not config_file.exists():
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        print(f"  ✅ 默认配置文件已创建: {config_file}")

def build_with_pyinstaller():
    """使用PyInstaller构建可执行文件"""
    print("🔨 使用PyInstaller构建可执行文件...")
    
    # PyInstaller 命令参数
    pyinstaller_args = [
        'pyinstaller',
        '--name', BUILD_CONFIG['app_name'],
        '--onedir',  # 创建目录分发
        '--windowed',  # Windows下不显示控制台
        '--clean',
        '--noconfirm',
        '--distpath', BUILD_CONFIG['output_dir'],
        '--workpath', BUILD_CONFIG['build_dir'],
        '--specpath', '.',
    ]
    
    # 添加数据文件
    data_files = [
        ('public', 'public'),
        ('assets', 'assets'),
        ('config', 'config'),
        ('requirements.txt', '.'),
        ('README.md', '.'),
    ]
    
    for src, dst in data_files:
        if os.path.exists(src):
            pyinstaller_args.extend(['--add-data', f'{src}{os.pathsep}{dst}'])
            print(f"  📁 添加数据文件: {src} -> {dst}")
    
    # 添加隐藏导入
    hidden_imports = [
        'PyQt5.QtCore',
        'PyQt5.QtWidgets', 
        'PyQt5.QtWebEngineWidgets',
        'PyQt5.QtWebChannel',
        'cv2',
        'numpy',
        'tensorflow',
        'sklearn',
        'skimage',
        'psutil',
        'queue',
        'threading',
        'concurrent.futures',
        'json',
        'time',
        'pathlib',
        'typing',
    ]
    
    for import_name in hidden_imports:
        pyinstaller_args.extend(['--hidden-import', import_name])
    
    # 排除模块
    excludes = [
        'tkinter',
        'matplotlib',
        'IPython',
        'jupyter',
        'notebook',
        'pandas',
        'scipy',
        'sympy',
    ]
    
    for exclude in excludes:
        pyinstaller_args.extend(['--exclude-module', exclude])
    
    # 添加图标
    if os.path.exists(BUILD_CONFIG['icon_file']):
        pyinstaller_args.extend(['--icon', BUILD_CONFIG['icon_file']])
        print(f"  🎨 添加图标: {BUILD_CONFIG['icon_file']}")
    
    # 添加主脚本
    pyinstaller_args.append(BUILD_CONFIG['main_script'])
    
    print(f"执行命令: {' '.join(pyinstaller_args)}")
    
    try:
        result = subprocess.run(pyinstaller_args, check=True, capture_output=True, text=True)
        print("✅ PyInstaller构建成功!")
        
        # 显示构建输出
        if result.stdout:
            print("📋 构建输出:")
            for line in result.stdout.split('\n')[-10:]:  # 显示最后10行
                if line.strip():
                    print(f"  {line}")
                    
    except subprocess.CalledProcessError as e:
        print("❌ PyInstaller构建失败!")
        print(f"错误代码: {e.returncode}")
        if e.stdout:
            print("标准输出:")
            print(e.stdout)
        if e.stderr:
            print("错误输出:")
            print(e.stderr)
        return False
    
    return True

def create_portable_version():
    """创建便携版本 (单文件)"""
    print("📦 创建便携版本...")
    
    pyinstaller_args = [
        'pyinstaller',
        '--name', f'{BUILD_CONFIG["app_name"]}_Portable',
        '--onefile',  # 单文件分发
        '--windowed',
        '--clean',
        '--noconfirm',
        '--distpath', f'{BUILD_CONFIG["output_dir"]}/portable',
        '--workpath', f'{BUILD_CONFIG["build_dir"]}/portable',
    ]
    
    # 添加必要的数据文件 (嵌入到可执行文件中)
    essential_data = [
        ('config/default.json', 'config'),
        ('assets', 'assets'),
    ]
    
    for src, dst in essential_data:
        if os.path.exists(src):
            pyinstaller_args.extend(['--add-data', f'{src}{os.pathsep}{dst}'])
    
    # 添加隐藏导入
    hidden_imports = [
        'PyQt5.QtCore',
        'PyQt5.QtWidgets', 
        'PyQt5.QtWebEngineWidgets',
        'cv2',
        'numpy',
        'tensorflow.lite.python.interpreter',
    ]
    
    for import_name in hidden_imports:
        pyinstaller_args.extend(['--hidden-import', import_name])
    
    # 添加图标
    if os.path.exists(BUILD_CONFIG['icon_file']):
        pyinstaller_args.extend(['--icon', BUILD_CONFIG['icon_file']])
    
    # 添加主脚本
    pyinstaller_args.append(BUILD_CONFIG['main_script'])
    
    try:
        result = subprocess.run(pyinstaller_args, check=True, capture_output=True, text=True)
        print("✅ 便携版本创建成功!")
        return True
    except subprocess.CalledProcessError as e:
        print("❌ 便携版本创建失败!")
        print(f"错误: {e.stderr}")
        return False

def post_build_tasks():
    """构建后任务"""
    print("🔧 执行构建后任务...")
    
    dist_dir = Path(BUILD_CONFIG['output_dir'])
    app_dir = dist_dir / BUILD_CONFIG['app_name']
    
    if not app_dir.exists():
        print(f"❌ 构建输出目录不存在: {app_dir}")
        return False
    
    # 复制额外文件到输出目录
    extra_files = [
        'README.md',
        'LICENSE',
        'requirements.txt'
    ]
    
    for file_name in extra_files:
        src_file = Path(file_name)
        if src_file.exists():
            dst_file = app_dir / file_name
            shutil.copy2(src_file, dst_file)
            print(f"  📄 复制文件: {file_name}")
    
    # 创建启动脚本
    if platform.system() != 'Windows':
        create_launch_script(app_dir)
    
    # 创建批处理文件 (Windows)
    if platform.system() == 'Windows':
        create_windows_launcher(app_dir)
    
    print("✅ 构建后任务完成")
    return True

def create_launch_script(app_dir):
    """创建Linux/macOS启动脚本"""
    script_content = f'''#!/bin/bash
# {BUILD_CONFIG["app_name"]} 启动脚本

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" &> /dev/null && pwd )"

# 设置环境变量
export QT_QPA_PLATFORM_PLUGIN_PATH="$SCRIPT_DIR"
export LD_LIBRARY_PATH="$SCRIPT_DIR:$LD_LIBRARY_PATH"

# 启动应用
cd "$SCRIPT_DIR"
./{BUILD_CONFIG["app_name"]} "$@"
'''
    
    script_file = app_dir / f'{BUILD_CONFIG["app_name"]}.sh'
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # 设置执行权限
    os.chmod(script_file, 0o755)
    print(f"  🚀 启动脚本已创建: {script_file}")

def create_windows_launcher(app_dir):
    """创建Windows启动批处理文件"""
    bat_content = f'''@echo off
REM {BUILD_CONFIG["app_name"]} 启动脚本

cd /d "%~dp0"
start "" "{BUILD_CONFIG["app_name"]}.exe" %*
'''
    
    bat_file = app_dir / f'{BUILD_CONFIG["app_name"]}.bat'
    with open(bat_file, 'w', encoding='utf-8') as f:
        f.write(bat_content)
    
    print(f"  🚀 启动批处理已创建: {bat_file}")

def show_build_summary():
    """显示构建摘要"""
    print("\n" + "="*60)
    print("🎉 PyInstaller构建完成!")
    print("="*60)
    
    dist_dir = Path(BUILD_CONFIG['output_dir'])
    app_dir = dist_dir / BUILD_CONFIG['app_name']
    
    if platform.system() == 'Windows':
        exe_file = app_dir / f"{BUILD_CONFIG['app_name']}.exe"
        if exe_file.exists():
            size_mb = exe_file.stat().st_size / (1024 * 1024)
            print(f"📁 可执行文件: {exe_file}")
            print(f"📏 文件大小: {size_mb:.1f} MB")
    else:
        exe_file = app_dir / BUILD_CONFIG['app_name']
        if exe_file.exists():
            size_mb = exe_file.stat().st_size / (1024 * 1024)
            print(f"📁 可执行文件: {exe_file}")
            print(f"📏 文件大小: {size_mb:.1f} MB")
    
    # 检查便携版本
    portable_dir = dist_dir / 'portable'
    if portable_dir.exists():
        portable_exe = portable_dir / f"{BUILD_CONFIG['app_name']}_Portable.exe"
        if not portable_exe.exists():
            portable_exe = portable_dir / f"{BUILD_CONFIG['app_name']}_Portable"
        
        if portable_exe.exists():
            size_mb = portable_exe.stat().st_size / (1024 * 1024)
            print(f"📦 便携版本: {portable_exe}")
            print(f"📏 便携版大小: {size_mb:.1f} MB")
    
    print(f"📂 输出目录: {dist_dir.absolute()}")
    print(f"🏷️  版本: {BUILD_CONFIG['version']}")
    print(f"💻 平台: {platform.system()} {platform.machine()}")
    
    print("\n📋 使用说明:")
    print("1. 确保目标机器已安装必要的系统依赖")
    print("2. 首次运行可能需要管理员权限")
    print("3. 摄像头权限需要用户授权")
    
    print("\n🔗 生成的文件:")
    if app_dir.exists():
        print(f"- 完整版本: {app_dir}")
        for file in app_dir.iterdir():
            if file.is_file() and file.suffix in ['.exe', '.sh', '.bat', '']:
                print(f"  • {file.name}")
    
    if (dist_dir / 'portable').exists():
        print(f"- 便携版本: {dist_dir / 'portable'}")

def main():
    """主函数"""
    print(f"🚀 {BUILD_CONFIG['app_name']} PyInstaller构建工具")
    print(f"版本: {BUILD_CONFIG['version']}")
    print(f"平台: {platform.system()} {platform.machine()}")
    print("-" * 60)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 清理构建目录
    clean_build()
    
    # 创建资源文件
    create_assets()
    
    # 使用PyInstaller构建
    if not build_with_pyinstaller():
        sys.exit(1)
    
    # 创建便携版本
    create_portable_version()
    
    # 构建后任务
    if not post_build_tasks():
        sys.exit(1)
    
    # 显示构建摘要
    show_build_summary()
    
    print("\n✅ PyInstaller构建流程完成!")

if __name__ == '__main__':
    main()
