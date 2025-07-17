"""
PyInstaller hook for OpenCV (cv2)
解决 OpenCV 打包时的依赖问题
"""

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs
import os
import sys

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

# 平台特定处理
if sys.platform.startswith('win'):
    # Windows 特定的 OpenCV 库
    hiddenimports.extend([
        'cv2.opencv_world',
        'cv2.opencv_imgproc',
        'cv2.opencv_imgcodecs',
        'cv2.opencv_videoio',
    ])
elif sys.platform.startswith('linux'):
    # Linux 特定处理
    hiddenimports.extend([
        'cv2.opencv_core',
        'cv2.opencv_imgproc',
        'cv2.opencv_imgcodecs',
        'cv2.opencv_videoio',
    ])
elif sys.platform.startswith('darwin'):
    # macOS 特定处理
    hiddenimports.extend([
        'cv2.opencv_core',
        'cv2.opencv_imgproc',
        'cv2.opencv_imgcodecs',
        'cv2.opencv_videoio',
    ])

# 尝试收集 OpenCV 配置文件
try:
    import cv2
    cv2_path = os.path.dirname(cv2.__file__)

    # 添加可能的配置文件
    config_files = [
        'opencv_ffmpeg*.dll',  # Windows FFmpeg 支持
        'opencv_videoio_ffmpeg*.dll',
        '*.xml',  # Haar cascades 等
        '*.yml',  # 配置文件
    ]

    for pattern in config_files:
        config_path = os.path.join(cv2_path, pattern)
        if os.path.exists(config_path):
            datas.append((config_path, 'cv2'))

except ImportError:
    pass

# 排除不需要的模块
excludedimports = [
    'cv2.gapi',  # G-API (通常不需要)
    'cv2.samples',  # 示例代码
]
