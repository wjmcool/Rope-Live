# -*- coding: utf-8 -*-
"""Rope-LiveMe startup shim.

Responsibilities, in order:

    * sys.path  -> the project root only (so `import rope.*` resolves)
    * cwd       -> the project root (config/, logs/, models/)
    * FFMPEG_PATH -> taken from PATH when the launcher did not set it
    * runtime logging -> logs/app-YYYYMMDD.log (UTF-8, rotating)
    * ROPE_SESSION / ROPE_LAUNCH_TIME handshake expected by Rope.verify_launch()
    * capability gating short-circuit (see rope/license_online.py for the chain)
    * optional driver/plugin installer stubs (ROPE_INSTALL_DEPS=1 keeps them)

Usage:
    python bootstrap.py                normal start
    python bootstrap.py --dry-run      environment report only, no window
    python bootstrap.py --no-cap-gate  skip the capability gating patch
"""

import os
import shutil
import subprocess
import sys
from datetime import datetime

APP = os.path.dirname(os.path.abspath(__file__))


def prepare_environment():
    os.chdir(APP)
    if sys.path and sys.path[0] != APP:
        sys.path.insert(0, APP)

    # the launch handshake Rope.verify_launch() insists on
    os.environ['ROPE_SESSION'] = 'srclaunch%06d' % (os.getpid() & 0xFFFFFF)
    os.environ['ROPE_LAUNCH_TIME'] = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    os.environ['ROPE_APP_PATH'] = APP
    os.environ['ROPE_PRODUCTION_MODE'] = '1'

    for sub in ('config', 'logs', 'tools'):
        os.makedirs(os.path.join(APP, sub), exist_ok=True)

    # 首次运行补齐配置骨架：GUI / rtmp_sniffer 都假定 config/config.json 存在
    app_config = os.path.join(APP, 'config', 'config.json')
    if not os.path.exists(app_config):
        import json as _json
        try:
            with open(app_config, 'w', encoding='utf-8') as _fh:
                _json.dump({'obs': {'host': '127.0.0.1', 'port': 4455, 'password': ''},
                            'tshark': {'display_filter': '', 'interface': ''}},
                           _fh, indent=4, ensure_ascii=False)
        except OSError:
            pass

    # ffmpeg: a candidate only counts if it actually runs.  WinGet's Links folder
    # holds 0-byte App Execution Aliases whose target is gone -- exists() says yes,
    # execution fails with "cannot find the path specified".
    def _ffmpeg_ok(exe):
        try:
            if not exe or not os.path.isfile(exe) or os.path.getsize(exe) == 0:
                return False
            probe = subprocess.run([exe, '-hide_banner', '-version'],
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=10)
            return probe.returncode == 0
        except Exception:
            return False

    def _ffmpeg_candidates():
        out = []
        env = os.environ.get('FFMPEG_PATH')
        if env:
            out.append(os.path.join(env, 'ffmpeg.exe'))
        for entry in os.environ.get('PATH', '').split(os.pathsep):
            entry = entry.strip().strip('"')
            if entry:
                out.append(os.path.join(entry, 'ffmpeg.exe'))
        return out

    for _cand in _ffmpeg_candidates():
        if _ffmpeg_ok(_cand):
            os.environ['FFMPEG_PATH'] = os.path.dirname(_cand)
            break
    else:
        os.environ.pop('FFMPEG_PATH', None)


def install_root_logging():
    """Give the root logger a rotating UTF-8 file, so every module's records persist.

    The vendor code creates its own logger (`get_logger('GUI')`, `get_logger('Models')` …)
    but only `Rope.py` installs a handler, and it does so on the logger named 'Rope-LiveMe'.
    Everything else therefore fell through to the root logger, which had no handler —
    which is why logs/YYYYMMDD.log stayed at 0 bytes and crashes left no trace.
    """
    import logging
    from logging.handlers import RotatingFileHandler

    root = logging.getLogger()
    if any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        return
    os.makedirs(os.path.join(APP, 'logs'), exist_ok=True)
    path = os.path.join(APP, 'logs', 'app-%s.log' % datetime.now().strftime('%Y%m%d'))
    handler = RotatingFileHandler(path, maxBytes=10 * 1024 * 1024, backupCount=5,
                                 encoding='utf-8')
    handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)-7s %(name)s: %(message)s', '%Y-%m-%d %H:%M:%S'))
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    def log_exception(exc_type, exc_value, exc_tb):
        logging.getLogger('bootstrap').error(
            'uncaught exception', exc_info=(exc_type, exc_value, exc_tb))
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = log_exception
    try:
        import threading
        threading.excepthook = lambda args: log_exception(
            args.exc_type, args.exc_value, args.exc_traceback)
    except Exception:
        pass


def install_tk_exception_logger():
    """Persist Tk callback tracebacks into logs/app-*.log.

    Tk's default handler prints "Exception in Tkinter callback" to stderr only, so a
    failing button handler scrolls away with the console. The hook keeps the default
    printing *and* records a full traceback in the rotating log.
    """
    try:
        import tkinter
    except Exception:
        return
    import logging
    import traceback

    def report(self, exc, val, tb):
        logging.getLogger('tk').error(
            'Tkinter callback failed: %s: %s', exc.__name__, val,
            exc_info=(exc, val, tb))
        stream = getattr(sys, '__stderr__', None) or sys.stderr
        if stream is not None:
            try:
                traceback.print_exception(exc, val, tb, file=stream)
            except Exception:
                pass

    tkinter.Tk.report_callback_exception = report


def install_dialog_logger():
    """Mirror every messagebox into logs/_dialogs.log so headless runs stay debuggable."""
    try:
        import tkinter.messagebox as mb
    except Exception:
        return
    path = os.path.join(APP, 'logs', '_dialogs.log')

    def wrap(name):
        original = getattr(mb, name, None)
        if original is None:
            return

        def logged(title=None, message=None, **kw):
            try:
                with open(path, 'a', encoding='utf-8') as fh:
                    fh.write('=== %s [%s]\n%s\n' % (name, title, message))
            except Exception:
                pass
            return original(title, message, **kw)

        setattr(mb, name, logged)

    for name in ('showerror', 'showwarning', 'showinfo'):
        wrap(name)


def patch_capability_gate():
    """Same short-circuit the packaged launcher injected."""
    import rope.license_utils as lu

    lu.check_license = lambda version=None: True
    lu.check_trial_status = lambda: True
    lu.get_remaining_days = lambda version=None: 9999
    lu.get_license_info = lambda version=None: {
        'expires_at': None, 'is_trial': False, 'trial_days': 0,
        'start_date': '', 'display_expires_at': '', 'display_start_date': '',
        'validity_period': 'permanent',
    }
    lu.validate_activation_code = lambda code, version=None: (
        True, {'valid': True, 'message': 'OK', 'validity_period': 'permanent'})


def stub_installers(entry):
    if os.environ.get('ROPE_INSTALL_DEPS') == '1':
        return
    for name in ('install_droidcam_driver', 'install_droidcam_plugin'):
        if hasattr(entry, name):
            setattr(entry, name, lambda *a, **k: None)


def report():
    import platform
    print('[bootstrap] app      : %s' % APP)
    print('[bootstrap] python   : %s (%s)' % (sys.executable, platform.python_version()))
    print('[bootstrap] ffmpeg   : %s' % os.environ.get('FFMPEG_PATH', '(not found)'))
    try:
        import torch
        print('[bootstrap] torch    : %s cuda=%s available=%s' % (
            torch.__version__, torch.version.cuda, torch.cuda.is_available()))
        if torch.cuda.is_available():
            print('[bootstrap] gpu      : %s sm_%d%d' % ((torch.cuda.get_device_name(0),)
                                                         + torch.cuda.get_device_capability(0)))
    except Exception as exc:
        print('[bootstrap] torch    : import failed -> %s' % exc)
    try:
        import onnxruntime
        print('[bootstrap] onnxrt   : %s %s' % (onnxruntime.__version__,
                                                onnxruntime.get_available_providers()))
    except Exception as exc:
        print('[bootstrap] onnxrt   : import failed -> %s' % exc)
    try:
        import tensorrt
        print('[bootstrap] tensorrt : %s' % tensorrt.__version__)
    except Exception as exc:
        print('[bootstrap] tensorrt : import failed -> %s' % exc)


def main(argv):
    prepare_environment()
    install_root_logging()
    install_dialog_logger()
    install_tk_exception_logger()
    if '--no-cap-gate' not in argv:
        patch_capability_gate()
    if '--dry-run' in argv:
        report()
        import rope.GUI          # noqa: F401  (heavy import chain)
        import rope.Models       # noqa: F401
        import rope.VideoManager  # noqa: F401
        import Rope
        stub_installers(Rope)
        print('[bootstrap] dry-run OK, entry imported')
        return 0

    import Rope
    stub_installers(Rope)
    Rope.main()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
