
"""
运行时环境配置
统一处理源码运行与打包运行两种形态下的路径差异
"""
import sys
import os
import logging
from pathlib import Path
def is_bundled_runtime():
    """检测是否运行在打包环境中"""
    return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')
def get_application_path():
    """获取应用程序路径"""
    if is_bundled_runtime():
        return Path(sys.executable).parent
    else:
        return Path(__file__).parent.parent
def setup_runtime_environment():
    """设置运行时环境"""
    if not is_bundled_runtime():
        return
    else:
        try:
            app_path = get_application_path()
            os.chdir(app_path)
            os.environ['ROPE_APP_PATH'] = str(app_path)
            os.environ['ROPE_PRODUCTION_MODE'] = '1'
            config_dir = app_path / 'config'
            config_dir.mkdir(exist_ok=True)
            logs_dir = app_path / 'logs'
            logs_dir.mkdir(exist_ok=True)
            setup_pytorch_optimizations()
            suppress_verbose_outputs()
        except Exception as e:
            return None
def setup_pytorch_optimizations():
    """设置PyTorch相关优化"""
    optimizations = {'PYTORCH_ALLOC_CONF': 'max_split_size_mb:512', 'CUDA_LAUNCH_BLOCKING': '0', 'TORCH_USE_CUDA_DSA': '0', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'TORCH_CUDA_ARCH_LIST': '', 'CUDNN_API': '80000', 'OMP_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'}
    for key, value in optimizations.items():
        if key not in os.environ:
            os.environ[key] = value
def suppress_verbose_outputs():
    """抑制冗余输出"""
    import warnings
    warnings.filterwarnings('ignore')
    suppressions = {'PYTHONWARNINGS': 'ignore', 'TF_CPP_MIN_LOG_LEVEL': '3', 'PYTORCH_WARNINGS': '0', 'TORCH_SHOW_CPP_STACKTRACES': '0'}
    for key, value in suppressions.items():
        if key not in os.environ:
            os.environ[key] = value
def get_runtime_info():
    """获取运行时信息"""
    return {'is_bundled': is_bundled_runtime(), 'app_path': str(get_application_path()), 'python_version': sys.version, 'platform': sys.platform, 'frozen': getattr(sys, 'frozen', False)}
if __name__ != '__main__':
    setup_runtime_environment()
if __name__ == '__main__':
    print('=== 运行时环境检测 ===')
    info = get_runtime_info()
    for key, value in info.items():
        print(f'{key}: {value}')