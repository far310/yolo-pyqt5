"""
构建配置文件
包含不同平台和构建类型的配置
"""

import platform
from pathlib import Path

# 基础配置
BASE_CONFIG = {
    'app_name': 'ImageRecognitionSystem',
    'version': '3.0.0',
    'author': 'Image Recognition Team',
    'description': '智能图像识别系统',
    'main_script': 'scripts/python_backend_example.py',
}

# 平台特定配置
PLATFORM_CONFIG = {
    'Windows': {
        'icon_file': 'assets/app_icon.ico',
        'console': False,
        'upx': True,
        'additional_hooks': ['hooks/hook-cv2.py'],
        'exclude_modules': [
            'tkinter',
            'matplotlib',
            'IPython',
            'jupyter',
        ],
        'dll_excludes': [
            'MSVCP140.dll',
            'VCRUNTIME140.dll',
        ]
    },
    'Darwin': {
        'icon_file': 'assets/app_icon.icns',
        'console': False,
        'upx': False,  # UPX 在 macOS 上可能有问题
        'bundle_identifier': 'com.imagerecognition.app',
        'codesign_identity': None,  # 设置为开发者证书ID进行代码签名
        'entitlements': {
            'com.apple.security.camera': True,
            'com.apple.security.device.camera': True,
        }
    },
    'Linux': {
        'icon_file': 'assets/app_icon.png',
        'console': False,
        'upx': True,
        'additional_libs': [
            '/usr/lib/x86_64-linux-gnu/libGL.so.1',
            '/usr/lib/x86_64-linux-gnu/libgthread-2.0.so.0',
        ]
    }
}

# 构建类型配置
BUILD_TYPES = {
    'debug': {
        'debug': True,
        'console': True,
        'optimize': False,
        'strip': False,
        'upx': False,
    },
    'release': {
        'debug': False,
        'console': False,
        'optimize': True,
        'strip': True,
        'upx': True,
    },
    'portable': {
        'onefile': True,
        'debug': False,
        'console': False,
        'optimize': True,
        'strip': True,
        'upx': True,
    }
}

# PyInstaller 钩子配置
HOOKS_CONFIG = {
    'cv2': {
        'hidden_imports': [
            'cv2',
            'cv2.cv2',
        ],
        'collect_data': True,
        'collect_binaries': True,
    },
    'tensorflow': {
        'hidden_imports': [
            'tensorflow',
            'tensorflow.lite',
            'tensorflow.lite.python',
            'tensorflow.lite.python.interpreter',
        ],
        'collect_data': True,
    },
    'sklearn': {
        'hidden_imports': [
            'sklearn',
            'sklearn.metrics',
            'sklearn.metrics.pairwise',
        ],
    },
    'PyQt5': {
        'hidden_imports': [
            'PyQt5.QtCore',
            'PyQt5.QtWidgets',
            'PyQt5.QtWebEngineWidgets',
            'PyQt5.QtWebChannel',
            'PyQt5.sip',
        ],
        'collect_data': True,
        'collect_binaries': True,
    }
}

def get_config(build_type='release'):
    """获取构建配置"""
    config = BASE_CONFIG.copy()
    
    # 添加平台配置
    system = platform.system()
    if system in PLATFORM_CONFIG:
        config.update(PLATFORM_CONFIG[system])
    
    # 添加构建类型配置
    if build_type in BUILD_TYPES:
        config.update(BUILD_TYPES[build_type])
    
    return config

def get_hidden_imports():
    """获取隐藏导入列表"""
    imports = []
    for hook_config in HOOKS_CONFIG.values():
        imports.extend(hook_config.get('hidden_imports', []))
    
    # 添加其他必要的导入
    additional_imports = [
        'queue',
        'threading',
        'concurrent.futures',
        'json',
        'time',
        'pathlib',
        'typing',
        'psutil',
        'numpy',
        'PIL',
        'PIL.Image',
    ]
    
    imports.extend(additional_imports)
    return list(set(imports))  # 去重

def get_data_files():
    """获取数据文件列表"""
    data_files = [
        ('public', 'public'),
        ('assets', 'assets'),
        ('config', 'config'),
        ('requirements.txt', '.'),
        ('README.md', '.'),
    ]
    
    # 检查模型目录
    model_dir = Path('model')
    if model_dir.exists():
        data_files.append(('model', 'model'))
    
    # 检查许可证文件
    license_file = Path('LICENSE')
    if license_file.exists():
        data_files.append(('LICENSE', '.'))
    
    return data_files

def get_exclude_modules():
    """获取排除模块列表"""
    base_excludes = [
        'tkinter',
        'matplotlib',
        'IPython',
        'jupyter',
        'notebook',
        'pandas',
        'scipy',
        'sympy',
        'pytest',
        'setuptools',
        'pip',
    ]
    
    system = platform.system()
    if system in PLATFORM_CONFIG:
        platform_excludes = PLATFORM_CONFIG[system].get('exclude_modules', [])
        base_excludes.extend(platform_excludes)
    
    return list(set(base_excludes))
