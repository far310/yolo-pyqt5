"""
PyInstaller hook for TensorFlow Lite
解决TensorFlow Lite打包时的依赖问题
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# 收集TensorFlow Lite数据文件
datas = collect_data_files('tensorflow.lite')

# 收集所有子模块
hiddenimports = collect_submodules('tensorflow.lite')

# 添加特定的隐藏导入
hiddenimports.extend([
    'tensorflow.lite.python.interpreter',
    'tensorflow.lite.python.interpreter_wrapper',
    'tensorflow.lite.python.metrics',
])
