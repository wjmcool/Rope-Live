
"""\n日志配置模块\n统一管理项目的日志配置\n"""
import logging
import logging.handlers
import os
import sys
from datetime import datetime
def setup_logger(name='LiveLog', level=logging.INFO, log_to_file=True, log_to_console=True):
    """\n    设置日志配置\n    \n    Args:\n        name: 日志器名称\n        level: 日志级别\n        log_to_file: 是否记录到文件\n        log_to_console: 是否输出到控制台\n    \n    Returns:\n        logger: 配置好的日志器\n    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    else:
        logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        if log_to_file:
            log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
            os.makedirs(log_dir, exist_ok=True)
            log_filename = f"{datetime.now().strftime('%Y%m%d')}.log"
            log_filepath = os.path.join(log_dir, log_filename)
            file_handler = logging.handlers.RotatingFileHandler(log_filepath, maxBytes=10485760, backupCount=5, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        return logger
def get_logger(name='LiveLog'):
    """\n    获取日志器实例\n    \n    Args:\n        name: 日志器名称\n    \n    Returns:\n        logger: 日志器实例\n    """
    return logging.getLogger(name)
default_logger = setup_logger()
def log_exception(logger=None, exc_info=True):
    """\n    记录异常信息的装饰器\n    \n    Args:\n        logger: 日志器实例\n        exc_info: 是否包含异常堆栈信息\n    """
    if logger is None:
        logger = default_logger
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f'函数 {func.__name__} 执行异常: {e}', exc_info=exc_info)
                raise
        return wrapper
    return decorator
def cleanup_old_logs(log_dir=None, days_to_keep=7):
    """\n    清理旧的日志文件\n    \n    Args:\n        log_dir: 日志目录路径\n        days_to_keep: 保留天数\n    """
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    if not os.path.exists(log_dir):
        return
    else:
        import time
        current_time = time.time()
        cutoff_time = current_time - days_to_keep * 24 * 60 * 60
        try:
            for filename in os.listdir(log_dir):
                if filename.startswith('rope_live_') and filename.endswith('.log'):
                        filepath = os.path.join(log_dir, filename)
                        if os.path.getmtime(filepath) < cutoff_time:
                            os.remove(filepath)
                            default_logger.info(f'已删除旧日志文件: {filename}')
        except Exception as e:
            default_logger.error(f'清理日志文件失败: {e}')
cleanup_old_logs()