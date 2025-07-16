"""
创建安装包脚本
支持不同平台的安装包生成
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path

def create_windows_installer():
    """创建Windows NSIS安装包"""
    print("📦 创建Windows安装包...")
    
    nsis_script = '''
; 图像识别系统安装脚本
!define APP_NAME "ImageRecognitionSystem"
!define APP_VERSION "3.0.0"
!define APP_PUBLISHER "Image Recognition Team"
!define APP_DESCRIPTION "智能图像识别系统"

Name "${APP_NAME}"
OutFile "${APP_NAME}_v${APP_VERSION}_Setup.exe"
InstallDir "$PROGRAMFILES\\${APP_NAME}"
RequestExecutionLevel admin

; 安装页面
Page license
Page directory
Page instfiles

; 许可证文件
LicenseData "LICENSE"

Section "MainSection" SEC01
    SetOutPath "$INSTDIR"
    File /r "dist\\ImageRecognitionSystem\\*"
    
    ; 创建桌面快捷方式
    CreateShortCut "$DESKTOP\\${APP_NAME}.lnk" "$INSTDIR\\ImageRecognitionSystem.exe"
    
    ; 创建开始菜单快捷方式
    CreateDirectory "$SMPROGRAMS\\${APP_NAME}"
    CreateShortCut "$SMPROGRAMS\\${APP_NAME}\\${APP_NAME}.lnk" "$INSTDIR\\ImageRecognitionSystem.exe"
    CreateShortCut "$SMPROGRAMS\\${APP_NAME}\\卸载.lnk" "$INSTDIR\\uninstall.exe"
    
    ; 写入注册表
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}" "DisplayName" "${APP_NAME}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}" "UninstallString" "$INSTDIR\\uninstall.exe"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}" "DisplayVersion" "${APP_VERSION}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}" "Publisher" "${APP_PUBLISHER}"
    
    ; 创建卸载程序
    WriteUninstaller "$INSTDIR\\uninstall.exe"
SectionEnd

Section "Uninstall"
    Delete "$DESKTOP\\${APP_NAME}.lnk"
    RMDir /r "$SMPROGRAMS\\${APP_NAME}"
    RMDir /r "$INSTDIR"
    DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}"
SectionEnd
'''
    
    with open('installer.nsi', 'w', encoding='utf-8') as f:
        f.write(nsis_script)
    
    print("✅ NSIS脚本已创建: installer.nsi")
    print("💡 请使用 makensis installer.nsi 编译安装包")

def create_macos_dmg():
    """创建macOS DMG安装包"""
    print("📦 创建macOS DMG安装包...")
    
    app_name = "ImageRecognitionSystem"
    dmg_name = f"{app_name}_v3.0.0.dmg"
    
    if not Path(f"dist/{app_name}").exists():
        print("❌ 应用目录不存在")
        return False
    
    # 创建临时DMG目录
    dmg_dir = Path("dmg_temp")
    dmg_dir.mkdir(exist_ok=True)
    
    # 复制应用到DMG目录
    shutil.copytree(f"dist/{app_name}", dmg_dir / app_name)
    
    # 创建应用程序链接
    os.symlink("/Applications", dmg_dir / "Applications")
    
    # 创建DMG
    cmd = [
        'hdiutil', 'create',
        '-volname', app_name,
        '-srcfolder', str(dmg_dir),
        '-ov', '-format', 'UDZO',
        dmg_name
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ DMG安装包已创建: {dmg_name}")
        
        # 清理临时目录
        shutil.rmtree(dmg_dir)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ DMG创建失败: {e}")
        return False

def create_linux_appimage():
    """创建Linux AppImage"""
    print("📦 创建Linux AppImage...")
    
    app_name = "ImageRecognitionSystem"
    appdir = Path(f"{app_name}.AppDir")
    
    # 创建AppDir结构
    appdir.mkdir(exist_ok=True)
    (appdir / "usr" / "bin").mkdir(parents=True, exist_ok=True)
    (appdir / "usr" / "share" / "applications").mkdir(parents=True, exist_ok=True)
    (appdir / "usr" / "share" / "icons" / "hicolor" / "256x256" / "apps").mkdir(parents=True, exist_ok=True)
    
    # 复制应用文件
    if Path(f"dist/{app_name}").exists():
        shutil.copytree(f"dist/{app_name}", appdir / "usr" / "bin" / app_name)
    
    # 创建desktop文件
    desktop_content = f'''[Desktop Entry]
Type=Application
Name={app_name}
Comment=智能图像识别系统
Exec={app_name}
Icon={app_name}
Categories=Graphics;Photography;
Terminal=false
'''
    
    with open(appdir / f"{app_name}.desktop", 'w') as f:
        f.write(desktop_content)
    
    # 复制desktop文件到标准位置
    shutil.copy2(appdir / f"{app_name}.desktop", 
                 appdir / "usr" / "share" / "applications" / f"{app_name}.desktop")
    
    # 创建AppRun脚本
    apprun_content = f'''#!/bin/bash
SELF=$(readlink -f "$0")
HERE=${{SELF%/*}}
export PATH="${{HERE}}/usr/bin:${{PATH}}"
export LD_LIBRARY_PATH="${{HERE}}/usr/lib:${{LD_LIBRARY_PATH}}"
cd "${{HERE}}/usr/bin/{app_name}"
exec "./{app_name}" "$@"
'''
    
    with open(appdir / "AppRun", 'w') as f:
        f.write(apprun_content)
    
    os.chmod(appdir / "AppRun", 0o755)
    
    print(f"✅ AppDir结构已创建: {appdir}")
    print("💡 请使用 appimagetool 创建 AppImage:")
    print(f"   appimagetool {appdir} {app_name}.AppImage")

def main():
    """主函数"""
    system = platform.system()
    
    print(f"🚀 为 {system} 平台创建安装包...")
    
    if system == 'Windows':
        create_windows_installer()
    elif system == 'Darwin':
        create_macos_dmg()
    elif system == 'Linux':
        create_linux_appimage()
    else:
        print(f"❌ 不支持的平台: {system}")
        sys.exit(1)
    
    print("✅ 安装包创建完成!")

if __name__ == '__main__':
    main()
