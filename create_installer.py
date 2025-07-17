#!/usr/bin/env python3
"""
安装包创建脚本
支持 Windows (NSIS), macOS (DMG), Linux (AppImage)
"""

import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path
import json

class InstallerCreator:
    """安装包创建器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.dist_dir = self.project_root / "dist"
        self.installer_dir = self.project_root / "installer"
        self.platform = platform.system().lower()
        
        # 应用信息
        self.app_name = "ImageRecognitionSystem"
        self.app_version = "1.0.0"
        self.app_description = "AI图像识别系统"
        self.app_author = "Your Company"
        self.app_url = "https://yourcompany.com"
        
        # 确保安装包目录存在
        self.installer_dir.mkdir(exist_ok=True)
    
    def create_windows_installer(self):
        """创建 Windows NSIS 安装包"""
        print("🪟 创建 Windows 安装包...")
        
        app_dir = self.dist_dir / self.app_name
        if not app_dir.exists():
            print("❌ 应用目录不存在，请先构建应用")
            return False
        
        # NSIS 脚本内容
        nsis_script = f'''
; 图像识别系统安装脚本
!define APPNAME "{self.app_name}"
!define APPVERSION "{self.app_version}"
!define APPDESCRIPTION "{self.app_description}"
!define APPAUTHOR "{self.app_author}"
!define APPURL "{self.app_url}"

; 安装程序属性
Name "${{APPNAME}}"
OutFile "{self.installer_dir}\\${{APPNAME}}_Setup_v${{APPVERSION}}.exe"
InstallDir "$PROGRAMFILES\\${{APPNAME}}"
InstallDirRegKey HKLM "Software\\${{APPNAME}}" "InstallDir"

; 请求管理员权限
RequestExecutionLevel admin

; 现代界面
!include "MUI2.nsh"

; 界面设置
!define MUI_ABORTWARNING
!define MUI_ICON "${{NSISDIR}}\\Contrib\\Graphics\\Icons\\modern-install.ico"
!define MUI_UNICON "${{NSISDIR}}\\Contrib\\Graphics\\Icons\\modern-uninstall.ico"

; 安装页面
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "LICENSE.txt"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

; 卸载页面
!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

; 语言
!insertmacro MUI_LANGUAGE "SimpChinese"

; 版本信息
VIProductVersion "${{APPVERSION}}.0"
VIAddVersionKey "ProductName" "${{APPNAME}}"
VIAddVersionKey "ProductVersion" "${{APPVERSION}}"
VIAddVersionKey "CompanyName" "${{APPAUTHOR}}"
VIAddVersionKey "FileDescription" "${{APPDESCRIPTION}}"
VIAddVersionKey "FileVersion" "${{APPVERSION}}"

; 安装部分
Section "主程序" SecMain
    SetOutPath "$INSTDIR"
    
    ; 复制所有文件
    File /r "{app_dir}\\*"
    
    ; 创建桌面快捷方式
    CreateShortCut "$DESKTOP\\${{APPNAME}}.lnk" "$INSTDIR\\${{APPNAME}}.exe"
    
    ; 创建开始菜单快捷方式
    CreateDirectory "$SMPROGRAMS\\${{APPNAME}}"
    CreateShortCut "$SMPROGRAMS\\${{APPNAME}}\\${{APPNAME}}.lnk" "$INSTDIR\\${{APPNAME}}.exe"
    CreateShortCut "$SMPROGRAMS\\${{APPNAME}}\\卸载.lnk" "$INSTDIR\\Uninstall.exe"
    
    ; 写入注册表
    WriteRegStr HKLM "Software\\${{APPNAME}}" "InstallDir" "$INSTDIR"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}" "DisplayName" "${{APPNAME}}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}" "UninstallString" "$INSTDIR\\Uninstall.exe"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}" "DisplayVersion" "${{APPVERSION}}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}" "Publisher" "${{APPAUTHOR}}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}" "URLInfoAbout" "${{APPURL}}"
    
    ; 创建卸载程序
    WriteUninstaller "$INSTDIR\\Uninstall.exe"
SectionEnd

; 卸载部分
Section "Uninstall"
    ; 删除文件
    RMDir /r "$INSTDIR"
    
    ; 删除快捷方式
    Delete "$DESKTOP\\${{APPNAME}}.lnk"
    RMDir /r "$SMPROGRAMS\\${{APPNAME}}"
    
    ; 删除注册表项
    DeleteRegKey HKLM "Software\\${{APPNAME}}"
    DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}"
SectionEnd
'''
        
        # 写入 NSIS 脚本
        nsis_file = self.installer_dir / f"{self.app_name}.nsi"
        with open(nsis_file, 'w', encoding='utf-8') as f:
            f.write(nsis_script)
        
        # 创建许可证文件
        license_file = self.installer_dir / "LICENSE.txt"
        if not license_file.exists():
            with open(license_file, 'w', encoding='utf-8') as f:
                f.write(f"""
{self.app_name} 软件许可协议

版权所有 (c) 2024 {self.app_author}

本软件按"原样"提供，不提供任何明示或暗示的保证。
使用本软件的风险由用户自行承担。

详细许可条款请访问: {self.app_url}
""")
        
        # 检查 NSIS 是否安装
        try:
            result = subprocess.run(["makensis", "/VERSION"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print("❌ NSIS 未安装，请从 https://nsis.sourceforge.io/ 下载安装")
                return False
        except FileNotFoundError:
            print("❌ NSIS 未安装，请从 https://nsis.sourceforge.io/ 下载安装")
            return False
        
        # 编译安装包
        try:
            cmd = ["makensis", str(nsis_file)]
            result = subprocess.run(cmd, cwd=self.installer_dir, 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("  ✅ Windows 安装包创建成功!")
                installer_file = self.installer_dir / f"{self.app_name}_Setup_v{self.app_version}.exe"
                if installer_file.exists():
                    size_mb = installer_file.stat().st_size / (1024 * 1024)
                    print(f"  📦 安装包: {installer_file} ({size_mb:.1f} MB)")
                return True
            else:
                print("  ❌ 安装包创建失败!")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"  ❌ 安装包创建异常: {e}")
            return False
    
    def create_macos_installer(self):
        """创建 macOS DMG 安装包"""
        print("🍎 创建 macOS 安装包...")
        
        app_bundle = self.dist_dir / f"{self.app_name}.app"
        if not app_bundle.exists():
            print("❌ 应用包不存在，请先构建应用")
            return False
        
        dmg_name = f"{self.app_name}_v{self.app_version}.dmg"
        dmg_path = self.installer_dir / dmg_name
        
        # 创建临时 DMG 目录
        temp_dmg_dir = self.installer_dir / "temp_dmg"
        if temp_dmg_dir.exists():
            shutil.rmtree(temp_dmg_dir)
        temp_dmg_dir.mkdir()
        
        # 复制应用到临时目录
        shutil.copytree(app_bundle, temp_dmg_dir / f"{self.app_name}.app")
        
        # 创建 Applications 链接
        applications_link = temp_dmg_dir / "Applications"
        try:
            applications_link.symlink_to("/Applications")
        except OSError:
            pass
        
        # 创建 DMG
        try:
            cmd = [
                "hdiutil", "create",
                "-volname", self.app_name,
                "-srcfolder", str(temp_dmg_dir),
                "-ov", "-format", "UDZO",
                str(dmg_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("  ✅ macOS 安装包创建成功!")
                if dmg_path.exists():
                    size_mb = dmg_path.stat().st_size / (1024 * 1024)
                    print(f"  📦 安装包: {dmg_path} ({size_mb:.1f} MB)")
                
                # 清理临时目录
                shutil.rmtree(temp_dmg_dir)
                return True
            else:
                print("  ❌ 安装包创建失败!")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"  ❌ 安装包创建异常: {e}")
            return False
    
    def create_linux_installer(self):
        """创建 Linux AppImage"""
        print("🐧 创建 Linux 安装包...")
        
        app_dir = self.dist_dir / self.app_name
        if not app_dir.exists():
            print("❌ 应用目录不存在，请先构建应用")
            return False
        
        # 创建 AppDir 结构
        appdir = self.installer_dir / f"{self.app_name}.AppDir"
        if appdir.exists():
            shutil.rmtree(appdir)
        appdir.mkdir()
        
        # 复制应用文件
        shutil.copytree(app_dir, appdir / "usr" / "bin", dirs_exist_ok=True)
        
        # 创建 desktop 文件
        desktop_content = f"""[Desktop Entry]
Type=Application
Name={self.app_name}
Comment={self.app_description}
Exec={self.app_name}
Icon={self.app_name.lower()}
Categories=Graphics;Photography;
Terminal=false
"""
        
        desktop_file = appdir / f"{self.app_name}.desktop"
        with open(desktop_file, 'w') as f:
            f.write(desktop_content)
        
        # 创建 AppRun 脚本
        apprun_content = f"""#!/bin/bash
SELF=$(readlink -f "$0")
HERE=${{SELF%/*}}
export PATH="${{HERE}}/usr/bin:${{PATH}}"
export LD_LIBRARY_PATH="${{HERE}}/usr/lib:${{LD_LIBRARY_PATH}}"
exec "${{HERE}}/usr/bin/{self.app_name}" "$@"
"""
        
        apprun_file = appdir / "AppRun"
        with open(apprun_file, 'w') as f:
            f.write(apprun_content)
        apprun_file.chmod(0o755)
        
        # 下载 appimagetool
        appimagetool_url = "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
        appimagetool_path = self.installer_dir / "appimagetool"
        
        if not appimagetool_path.exists():
            print("  📥 下载 appimagetool...")
            try:
                import urllib.request
                urllib.request.urlretrieve(appimagetool_url, appimagetool_path)
                appimagetool_path.chmod(0o755)
            except Exception as e:
                print(f"  ❌ 下载 appimagetool 失败: {e}")
                return False
        
        # 创建 AppImage
        appimage_name = f"{self.app_name}_v{self.app_version}.AppImage"
        appimage_path = self.installer_dir / appimage_name
        
        try:
            cmd = [str(appimagetool_path), str(appdir), str(appimage_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("  ✅ Linux 安装包创建成功!")
                if appimage_path.exists():
                    size_mb = appimage_path.stat().st_size / (1024 * 1024)
                    print(f"  📦 安装包: {appimage_path} ({size_mb:.1f} MB)")
                    appimage_path.chmod(0o755)
                
                # 清理临时目录
                shutil.rmtree(appdir)
                return True
            else:
                print("  ❌ 安装包创建失败!")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"  ❌ 安装包创建异常: {e}")
            return False
    
    def create_installer(self):
        """根据平台创建安装包"""
        print(f"📦 为 {self.platform} 平台创建安装包...")
        
        if self.platform == "windows":
            return self.create_windows_installer()
        elif self.platform == "darwin":
            return self.create_macos_installer()
        elif self.platform == "linux":
            return self.create_linux_installer()
        else:
            print(f"❌ 不支持的平台: {self.platform}")
            return False

def main():
    """主函数"""
    print("📦 图像识别系统安装包创建工具")
    print("=" * 40)
    
    creator = InstallerCreator()
    success = creator.create_installer()
    
    if success:
        print("\n🎉 安装包创建成功!")
        print(f"📁 输出目录: {creator.installer_dir}")
    else:
        print("\n❌ 安装包创建失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()
