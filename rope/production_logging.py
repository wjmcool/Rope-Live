
"""\n生产环境日志配置\n设置适当的日志级别，减少冗余输出\n"""
import logging
import sys
import os
def setup_production_logging():
    """设置生产环境日志配置"""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    try:
        import time
        from logging.handlers import RotatingFileHandler
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'app-%s.log' % time.strftime('%Y%m%d'))
        file_handler = RotatingFileHandler(log_path, maxBytes=10485760, backupCount=5, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-7s %(name)s: %(message)s', '%Y-%m-%d %H:%M:%S'))
        root_logger.addHandler(file_handler)
    except Exception:
        pass
    module_levels = {'rope.RTX50Compatibility': logging.CRITICAL + 1, 'rope.GPUManager': logging.CRITICAL + 1, 'rope.Models': logging.CRITICAL + 1, 'rope.Coordinator': logging.CRITICAL + 1, 'onnxruntime': logging.CRITICAL + 1, 'torch': logging.CRITICAL + 1, 'urllib3': logging.CRITICAL + 1, 'requests': logging.CRITICAL + 1, 'PIL': logging.CRITICAL + 1, 'cv2': logging.CRITICAL + 1}
    for module_name, level in module_levels.items():
        logger = logging.getLogger(module_name)
        logger.setLevel(level)
        logger.propagate = False
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='torch')
    warnings.filterwarnings('ignore', message='.*CUDA.*')
    warnings.filterwarnings('ignore', message='.*sm_120.*')
    warnings.filterwarnings('ignore', message='.*compute capability.*')
    os.environ['PYTHONWARNINGS'] = 'ignore'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def setup_debug_logging():
    """设置调试环境日志配置（显示详细信息）"""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    print('[注意] 调试环境日志配置已应用 - 显示详细信息')
def get_logging_mode():
    """检测当前应该使用的日志模式"""
    debug_mode = os.getenv('DEBUG', '').lower() in ['1', 'true', 'yes', 'on']
    dev_mode = os.getenv('DEVELOPMENT', '').lower() in ['1', 'true', 'yes', 'on']
    production_mode = os.getenv('ROPE_PRODUCTION_MODE', '').lower() in ['1', 'true', 'yes', 'on']
    is_frozen_build = getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')
    if '--debug' in sys.argv or '--verbose' in sys.argv:
        debug_mode = True
    if is_frozen_build or production_mode:
        return False
    else:
        return debug_mode or dev_mode
def configure_logging():
    """自动配置日志"""
    if get_logging_mode():
        setup_debug_logging()
    else:
        setup_production_logging()
if __name__ != '__main__':
    configure_logging()
if __name__ == '__main__':
    print('=== 日志配置测试 ===')
    print('\n1. 生产环境模式:')
    setup_production_logging()
    logging.debug('这是DEBUG信息（不应该显示）')
    logging.info('这是INFO信息（不应该显示）')
    logging.warning('这是WARNING信息（应该显示）')
    logging.error('这是ERROR信息（应该显示）')
    print('\n2. 调试环境模式:')
    setup_debug_logging()
    logging.debug('这是DEBUG信息（应该显示）')
    logging.info('这是INFO信息（应该显示）')
    logging.warning('这是WARNING信息（应该显示）')
    logging.error('这是ERROR信息（应该显示）')