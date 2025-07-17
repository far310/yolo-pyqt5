#!/usr/bin/env python3
"""
PyInstaller 打包脚本 - 图像识别系统
支持 Windows、macOS、Linux 跨平台打包
"""

import os
import sys
import shutil
import subprocess
import platform
import argparse
from pathlib import Path
import json

class ImageRecognitionBuilder:
    def __init__(self):
        self.system = platform.system().lower()
        self.project_root = Path(__file__).parent
        self.dist_dir = self.project_root / "dist"
        self.build_dir = self.project_root / "build"
        self.spec_file = self.project_root / "image_recognition.spec"
        
        # 主程序文件
        self.main_script = self.project_root / "scripts" / "python_backend_example.py"
        
        # 应用信息
        self.app_name = "ImageRecognitionSystem"
        self.app_version = "1.0.0"
        self.app_description = "AI图像识别系统"
        self.app_author = "Your Company"
        
        # 资源文件和数据文件
        self.data_files = [
            ("scripts/utils.py", "scripts"),
            ("scripts/tfLitePool.py", "scripts"),
            ("public/qwebchannel.js", "public"),
            ("requirements.txt", "."),
        ]
        
        # 隐藏导入的模块
        self.hidden_imports = [
            'cv2',
            'numpy',
            'tensorflow',
            'tflite_runtime',
            'PyQt5.QtCore',
            'PyQt5.QtWidgets',
            'PyQt5.QtWebEngineWidgets',
            'PyQt5.QtWebChannel',
            'sklearn.metrics.pairwise',
            'skimage.metrics',
            'psutil',
            'queue',
            'concurrent.futures',
            'threading',
            'json',
            'time',
            'base64',
            "scikit-learn",
            "scikit-image",
            "tflite_runtime.interpreter",
            "tensorflow.lite"
        ]
        
        # 排除的模块（减少打包大小）
        self.excludes = [
            'matplotlib',
            'scipy',
            'pandas',
            'jupyter',
            'notebook',
            'IPython',
            'tkinter',
            'unittest',
            'test',
            'tests',
            'pytest',
            "PIL"
        ]

    def check_dependencies(self):
        """检查构建依赖"""
        print("🔍 检查构建依赖...")
        
        required_packages = [
            'pyinstaller',
            'opencv-python',
            'PyQt5',
            'numpy',
            'psutil',
            "scikit-learn",
            "scikit-image"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"✅ {package}")
            except ImportError:
                missing_packages.append(package)
                print(f"❌ {package}")
        
        if missing_packages:
            print(f"\n⚠️  缺少依赖包: {', '.join(missing_packages)}")
            print("请运行: pip install " + " ".join(missing_packages))
            return False
        
        # 检查 PyInstaller
        try:
            result = subprocess.run(["pyinstaller", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ PyInstaller {result.stdout.strip()}")
            else:
                print("❌ PyInstaller 不可用")
                return False
        except FileNotFoundError:
            print("❌ PyInstaller 未安装")
            return False
        
        # 检查主程序文件
        if not self.main_script.exists():
            print(f"❌ 主程序文件不存在: {self.main_script}")
            return False
        
        print("✅ 所有依赖检查通过")
        return True

    def clean_build_dirs(self):
        """清理构建目录"""
        print("🧹 清理构建目录...")
        
        dirs_to_clean = [self.dist_dir, self.build_dir]
        for dir_path in dirs_to_clean:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                print(f"🗑️  已删除: {dir_path}")
        
        if self.spec_file.exists():
            self.spec_file.unlink()
            print(f"🗑️  已删除: {self.spec_file}")

    def create_pyinstaller_spec(self, onefile=False):
        """创建 PyInstaller spec 文件"""
        print("📝 创建 PyInstaller spec 文件...")
        
        # 构建数据文件列表
        datas_list = []
        for src, dst in self.data_files:
            src_path = self.project_root / src
            if src_path.exists():
                if src_path.is_file():
                    datas_list.append(f"('{src_path}', '{dst}')")
                else:
                    datas_list.append(f"('{src_path}', '{dst}')")
        
        # 构建隐藏导入列表
        hiddenimports_str = "[\n    " + ",\n    ".join([f"'{imp}'" for imp in self.hidden_imports]) + "\n]"
        
        # 构建排除列表
        excludes_str = "[\n    " + ",\n    ".join([f"'{exc}'" for exc in self.excludes]) + "\n]"
        
        # 构建数据文件字符串
        datas_str = "[\n    " + ",\n    ".join(datas_list) + "\n]" if datas_list else "[]"
        
        # 平台特定设置
        console_setting = "True" if self.system != "windows" else "False"
        icon_path = ""
        if self.system == "windows":
            icon_path = "icon='assets/icon.ico'," if (self.project_root / "assets/icon.ico").exists() else ""
        elif self.system == "darwin":
            icon_path = "icon='assets/icon.icns'," if (self.project_root / "assets/icon.icns").exists() else ""
        
        spec_content = f"""# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['{self.main_script}'],
    pathex=['{self.project_root}'],
    binaries=[],
    datas={datas_str},
    hiddenimports={hiddenimports_str},
    hookspath=['pyinstaller_hooks'],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes={excludes_str},
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

{'exe = EXE(' if onefile else 'exe = EXE('}
    pyz,
    a.scripts,
    {'a.binaries,' if onefile else ''}
    {'a.zipfiles,' if onefile else ''}
    {'a.datas,' if onefile else ''}
    [],
    name='{self.app_name}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    {'upx_exclude=[],' if onefile else ''}
    {'runtime_tmpdir=None,' if onefile else ''}
    console={console_setting},
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    {icon_path}
)

{'# 单文件模式，不需要 COLLECT' if onefile else f'''
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='{self.app_name}',
)
'''}

{'# macOS 应用包' if self.system == 'darwin' and not onefile else ''}
{'app = BUNDLE(' if self.system == 'darwin' and not onefile else ''}
{'    coll,' if self.system == 'darwin' and not onefile else ''}
{'    name="{}.app",'.format(self.app_name) if self.system == 'darwin' and not onefile else ''}
{'    icon="{icon}",'.format(icon=icon_path) if self.system == 'darwin' and not onefile and icon_path else ''}
{'    bundle_identifier="com.yourcompany.imagerecognition"' if self.system == 'darwin' and not onefile else ''}
{')' if self.system == 'darwin' and not onefile else ''}
"""
        
        with open(self.spec_file, 'w', encoding='utf-8') as f:
            f.write(spec_content)
        
        print(f"✅ Spec 文件已创建: {self.spec_file}")

    def run_pyinstaller(self, onefile=False):
        """运行 PyInstaller"""
        print("🚀 开始 PyInstaller 打包...")
        
        # 创建 spec 文件
        self.create_pyinstaller_spec(onefile)
        
        # 构建 PyInstaller 命令
        cmd = [
            sys.executable, '-m', 'PyInstaller',
            '--clean',
            '--noconfirm',
            str(self.spec_file)
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        
        # 运行 PyInstaller
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ PyInstaller 打包成功")
            if result.stdout:
                print("输出:", result.stdout)
        except subprocess.CalledProcessError as e:
            print("❌ PyInstaller 打包失败")
            print("错误:", e.stderr)
            return False
        
        return True

    def post_build_tasks(self):
        """构建后任务"""
        print("🔧 执行构建后任务...")
        
        # 检查构建结果
        if self.system == "darwin":
            app_path = self.dist_dir / "ImageRecognitionSystem.app"
            exe_path = self.dist_dir / "ImageRecognitionSystem" / "ImageRecognitionSystem"
        else:
            exe_path = self.dist_dir / "ImageRecognitionSystem" / ("ImageRecognitionSystem.exe" if self.system == "windows" else "ImageRecognitionSystem")
        
        if not exe_path.exists() and not (self.system == "darwin" and app_path.exists()):
            print("❌ 构建失败：找不到可执行文件")
            return False
        
        # 复制额外的配置文件
        config_files = [
            "README.md",
            "requirements.txt",
            ".env.example"
        ]
        
        target_dir = self.dist_dir / "ImageRecognitionSystem"
        for config_file in config_files:
            src_file = self.project_root / config_file
            if src_file.exists():
                dst_file = target_dir / config_file
                shutil.copy2(src_file, dst_file)
                print(f"✅ 已复制: {config_file}")
        
        # 创建启动脚本
        self.create_launcher_scripts()
        
        print("✅ 构建后任务完成")
        return True

    def create_launcher_scripts(self):
        """创建启动脚本"""
        print("📝 创建启动脚本...")
        
        target_dir = self.dist_dir / "ImageRecognitionSystem"
        
        if self.system == "windows":
            # Windows 批处理文件
            bat_content = f"""@echo off
echo 启动图像识别系统...
cd /d "%~dp0"
"{self.app_name}.exe"
pause
"""
            bat_file = target_dir / f"启动_{self.app_name}.bat"
            with open(bat_file, 'w', encoding='gbk') as f:
                f.write(bat_content)
            print("✅ 已创建 Windows 启动脚本")
        
        elif self.system in ["linux", "darwin"]:
            # Unix shell 脚本
            sh_content = f"""#!/bin/bash
echo "启动图像识别系统..."
cd "$(dirname "$0")"
./{self.app_name}
"""
            sh_file = target_dir / f"启动_{self.app_name}.sh"
            with open(sh_file, 'w', encoding='utf-8') as f:
                f.write(sh_content)
            sh_file.chmod(0o755)
            print("✅ 已创建 Unix 启动脚本")
    
    def show_build_results(self):
        """显示构建结果"""
        print("\n🎉 构建完成!")
        print("=" * 50)
        
        app_dir = self.dist_dir / self.app_name
        if app_dir.exists():
            print(f"📁 应用目录: {app_dir}")
            
            # 计算大小
            total_size = sum(f.stat().st_size for f in app_dir.rglob('*') if f.is_file())
            size_mb = total_size / (1024 * 1024)
            print(f"📊 总大小: {size_mb:.1f} MB")
            
            # 列出主要文件
            print("\n📋 主要文件:")
            main_files = [
                f"{self.app_name}.exe" if self.system == "windows" else self.app_name,
                f"启动_{self.app_name}.bat" if self.system == "windows" else f"启动_{self.app_name}.sh",
                "README.md",
                "requirements.txt"
            ]
            
            for file_name in main_files:
                file_path = app_dir / file_name
                if file_path.exists():
                    print(f"✅ {file_name}")
                else:
                    print(f"❌ {file_name} (缺失)")
        
        print("\n🚀 使用方法:")
        if self.system == "windows":
            print(f"  双击运行: {app_dir}/启动_{self.app_name}.bat")
        else:
            print(f"  终端运行: cd {app_dir} && ./启动_{self.app_name}.sh")

    def build(self, onefile=False):
        """主构建流程"""
        print(f"🚀 开始构建 {self.app_name} v{self.app_version}")
        print(f"🖥️  平台: {platform.system()} {platform.machine()}")
        print(f"🐍 Python: {sys.version}")
        print("=" * 50)
        
        # 检查依赖
        if not self.check_dependencies():
            return False
        
        # 清理构建目录
        self.clean_build_dirs()
        
        # 运行 PyInstaller
        if not self.run_pyinstaller(onefile):
            return False
        
        # 构建后任务
        self.post_build_tasks()
        
        return True


def main():
    parser = argparse.ArgumentParser(description="图像识别系统构建工具")
    parser.add_argument("--onefile", action="store_true", 
                       help="打包为单个可执行文件")
    parser.add_argument("--clean", action="store_true",
                       help="仅清理构建目录")
    
    args = parser.parse_args()
    
    builder = ImageRecognitionBuilder()
    
    if args.clean:
        builder.clean_build_dirs()
        print("✅ 清理完成")
        return
    
    success = builder.build(onefile=args.onefile)
    
    if success:
        print("\n🎉 构建成功完成!")
        sys.exit(0)
    else:
        print("\n❌ 构建失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()
