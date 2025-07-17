"""
PyInstaller hook for TensorFlow Lite
解决 TensorFlow Lite 打包时的依赖问题
"""

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules
import os
import sys

# 收集 TensorFlow Lite 数据文件
datas = []
binaries = []
hiddenimports = []

# 尝试 tflite_runtime
try:
    import tflite_runtime

    # 收集 tflite_runtime 数据文件和库
    datas.extend(collect_data_files('tflite_runtime'))
    binaries.extend(collect_dynamic_libs('tflite_runtime'))

    hiddenimports.extend([
        'tflite_runtime',
        'tflite_runtime.interpreter',
    ])

    # 收集所有子模块
    hiddenimports.extend(collect_submodules('tflite_runtime'))

except ImportError:
    pass

# 尝试完整的 TensorFlow
try:
    import tensorflow as tf

    # 只收集 TensorFlow Lite 相关部分
    hiddenimports.extend([
        'tensorflow',
        'tensorflow.lite',
        'tensorflow.lite.python',
        'tensorflow.lite.python.interpreter',
    ])

    # 收集 TensorFlow Lite 库文件
    tf_path = os.path.dirname(tf.__file__)
    lite_path = os.path.join(tf_path, 'lite')

    if os.path.exists(lite_path):
        # 添加 TensorFlow Lite 相关文件
        for root, dirs, files in os.walk(lite_path):
            for file in files:
                if file.endswith(('.so', '.dll', '.dylib', '.pyd')):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, tf_path)
                    binaries.append((full_path, os.path.dirname(rel_path)))

except ImportError:
    pass

# 平台特定处理
if sys.platform.startswith('win'):
    # Windows 特定的库文件
    hiddenimports.extend([
        '_pywrap_tensorflow_internal',
        'tensorflow.python._pywrap_tensorflow_internal',
    ])
elif sys.platform.startswith('linux'):
    # Linux 特定处理
    hiddenimports.extend([
        'tensorflow.python._pywrap_tensorflow_internal',
    ])
elif sys.platform.startswith('darwin'):
    # macOS 特定处理
    hiddenimports.extend([
        'tensorflow.python._pywrap_tensorflow_internal',
    ])

# 添加常用的数值计算库
hiddenimports.extend([
    'numpy',
    'numpy.core',
    'numpy.core._multiarray_umath',
    'numpy.random',
    'numpy.random._pickle',
])

# 排除不需要的 TensorFlow 模块
excludedimports = [
    'tensorflow.python.eager',
    'tensorflow.python.training',
    'tensorflow.python.saved_model',
    'tensorflow.python.tools',
    'tensorflow.contrib',
    'tensorboard',
    'tensorflow.examples',
]
