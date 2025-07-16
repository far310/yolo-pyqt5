"""
PyInstaller hook for OpenCV
解决 OpenCV 打包问题
"""

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs
import cv2
import os

# 收集 OpenCV 数据文件
datas = collect_data_files('cv2')

# 收集 OpenCV 动态库
binaries = collect_dynamic_libs('cv2')

# 隐藏导入
hiddenimports = [
    'cv2',
    'cv2.cv2',
    'numpy',
]

# 添加 OpenCV 配置文件
cv2_config_path = os.path.join(os.path.dirname(cv2.__file__), 'config.py')
if os.path.exists(cv2_config_path):
    datas.append((cv2_config_path, 'cv2'))

# 添加 OpenCV DLL (Windows)
import platform
if platform.system() == 'Windows':
    cv2_dir = os.path.dirname(cv2.__file__)
    for file in os.listdir(cv2_dir):
        if file.endswith('.dll'):
            dll_path = os.path.join(cv2_dir, file)
            binaries.append((dll_path, '.'))
