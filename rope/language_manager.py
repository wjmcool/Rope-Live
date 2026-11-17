"""语言字典加载与用户配置文件的定位/自愈。

设计要点（解决"换台机器/首次运行就报 加载语言配置失败: 'config\\data.json'"）：

* 路径一律相对**项目根目录**解析，不依赖当前工作目录；
* `config/data.json` 不存在时自动写入一份默认配置，而不是抛错；
* 文件损坏（JSON 错误）时把它改名成 `data.json.broken` 备份并重建默认值；
* 以上情况都不向控制台打印错误——首次运行属于正常路径。
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)

current_dict = None

#: 与 GUI 的 json_dict 默认值保持一致
DEFAULT_CONFIG = {
    'source videos': None,
    'source faces': None,
    'saved videos': None,
    'dock_win_geom': [1400, 800, 176, 113],
    'language': 'Chinese',
}


def config_dir():
    """项目根目录下的 config 目录（不存在则创建）。"""
    path = os.path.join(PROJECT_ROOT, 'config')
    os.makedirs(path, exist_ok=True)
    return path


def data_json_path():
    """用户配置文件 config/data.json 的绝对路径。"""
    return os.path.join(config_dir(), 'data.json')


def _write_default(path):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as handle:
        json.dump(DEFAULT_CONFIG, handle, indent=4, ensure_ascii=False)
    os.replace(tmp, path)


def ensure_data_json():
    """确保配置文件存在，返回其绝对路径。

    缺失 -> 写入默认值；无法解析 -> 备份为 .broken 后重建。
    """
    path = data_json_path()
    if not os.path.exists(path):
        try:
            _write_default(path)
            logger.debug('已创建默认配置: %s', path)
        except OSError as exc:
            logger.debug('创建默认配置失败: %s', exc)
        return path
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        try:
            os.replace(path, path + '.broken')
            _write_default(path)
            logger.debug('配置损坏已重建，原文件备份为 data.json.broken')
        except OSError as exc:
            logger.debug('重建配置失败: %s', exc)
    return path


def get_current_dict():
    """获取当前语言的字典"""
    return current_dict


def set_current_dict(dict_data):
    """设置当前语言字典"""
    global current_dict
    current_dict = dict_data


def load_language_dict():
    """根据配置加载对应语言的字典。

    配置文件缺失或损坏时静默回退到中文，不打印错误（首次运行即属正常路径）。
    """
    language = 'Chinese'
    try:
        with open(ensure_data_json(), 'r', encoding='utf-8') as handle:
            language = json.load(handle).get('language', 'Chinese')
    except (OSError, ValueError, AttributeError) as exc:
        logger.debug('读取语言配置失败，回退中文: %s', exc)

    if language == 'English':
        from . import Dicts_EN as Dicts
    else:
        from . import Dicts_CN as Dicts

    set_current_dict(Dicts.DEFAULT_DATA)
    return Dicts
