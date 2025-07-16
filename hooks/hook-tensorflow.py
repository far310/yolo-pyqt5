"""
PyInstaller hook for TensorFlow Lite
解决 TensorFlow Lite 打包问题
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import tensorflow as tf
import os

# 收集 TensorFlow Lite 数据文件
datas = collect_data_files('tensorflow.lite')

# 收集子模块
hiddenimports = collect_submodules('tensorflow.lite')

# 添加特定的隐藏导入
hiddenimports.extend([
    'tensorflow',
    'tensorflow.lite',
    'tensorflow.lite.python',
    'tensorflow.lite.python.interpreter',
    'tensorflow.lite.python.interpreter_wrapper',
])

# 添加 TensorFlow Lite 运行时库
try:
    import tflite_runtime
    datas.extend(collect_data_files('tflite_runtime'))
    hiddenimports.extend(collect_submodules('tflite_runtime'))
except ImportError:
    pass
