"""
PyInstaller hook for OpenCV (cv2)
解决OpenCV打包时的依赖问题
"""

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# 收集OpenCV数据文件
datas = collect_data_files('cv2')

# 收集OpenCV动态库
binaries = collect_dynamic_libs('cv2')

# 隐藏导入
hiddenimports = [
    'cv2.cv2',
    'numpy',
]

# 排除不需要的模块
excludedimports = [
    'cv2.gapi',
]
