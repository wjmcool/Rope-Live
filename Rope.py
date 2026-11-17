
try:
    from rope.runtime_env import setup_runtime_environment
    setup_runtime_environment()
except ImportError:
    pass
try:
    from rope.production_logging import configure_logging
    configure_logging()
except ImportError:
    import logging
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import atexit
import tkinter as tk
from tkinter import ttk, messagebox
import sys
from rope import Coordinator
from rope.license_utils import validate_activation_code, get_machine_code, check_license
from datetime import datetime
import os
from pathlib import Path
from rope.language_manager import load_language_dict
import json
import random
import subprocess
import win32api
import win32con
import cv2
from rope.config import SOFTWARE_VERSION
from rope.logger_config import setup_logger, get_logger
logger = setup_logger('Rope-LiveMe', log_to_console=False)
Dicts = load_language_dict()
DEFAULT_DATA = Dicts.DEFAULT_DATA
def verify_launch():
    """验证是否通过启动器启动"""
    session_key = os.environ.get('ROPE_SESSION')
    launch_time = os.environ.get('ROPE_LAUNCH_TIME')
    if not session_key or not launch_time:
        logger.error('启动验证失败：缺少会话密钥或启动时间')
        messagebox.showerror(DEFAULT_DATA['LaunchError'], DEFAULT_DATA['LaunchErrorMsg'])
        sys.exit(1)
    time_formats = ['%Y/%m/%d %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%m/%d/%Y %H:%M:%S']
    launch_datetime = None
    for time_format in time_formats:
        try:
            launch_datetime = datetime.strptime(launch_time, time_format)
        except ValueError:
            continue
    if not launch_datetime:
        raise ValueError(f"{DEFAULT_DATA['Error']}: {launch_time}")
    try:
        time_diff = (datetime.now() - launch_datetime).total_seconds()
        if time_diff > 600:
            logger.error(f'启动超时：时间差为 {time_diff} 秒')
            messagebox.showerror(DEFAULT_DATA['Error'], DEFAULT_DATA['LaunchTimeout'])
            sys.exit(1)
    except Exception as e:
        logger.error(f'启动验证异常: {e}')
        messagebox.showerror(DEFAULT_DATA['LaunchError'], str(e))
        sys.exit(1)


def cleanup():
    """清理资源，避免创建额外的Tk窗口"""
    try:
        import tkinter as tk
        if tk._default_root is not None:
            tk._default_root.quit()
            tk._default_root.destroy()
            tk._default_root = None
    except Exception:
        return None
atexit.register(cleanup)
def initialize_license_system():
    BASE_DIR = Path(__file__).resolve().parent
    TOOLS_DIR = BASE_DIR / 'tools'
    LICENSE_ENC_PATH = TOOLS_DIR / 'license.enc'
    os.makedirs(TOOLS_DIR, exist_ok=True)
    try:
        license_valid = check_license(SOFTWARE_VERSION)
        return license_valid
    except Exception as e:
        return False
def show_activation_dialog(root):
    try:
        dialog = tk.Toplevel(root)
        dialog.title(DEFAULT_DATA['ActivationTitle'])
        dialog.geometry('300x220')
        dialog.resizable(False, False)
        dialog.configure(bg='#F0F0E0')
        dialog.protocol('WM_DELETE_WINDOW', lambda: sys.exit())
        try:
            style = ttk.Style()
            style.theme_use('clam')
            style.configure('TButton', font=('Arial', 10), borderwidth=1, relief='raised', background='#f0f0f0')
            style.configure('TLabel', font=('Arial', 10), background='#f0f0f0')
            style.configure('TEntry', font=('Arial', 10), fieldbackground='#ffffff')
            style.configure('TFrame', background='#f0f0f0')
        except Exception as e:
            messagebox.showwarning('Style Warning', f'Error configuring style: {e}')
        main_frame = ttk.Frame(dialog, padding='20 20 20 20', style='TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True)
        dialog.image_refs = []
        def activate():
            try:
                activation_code = activation_code_entry.get().strip()
                success, response = validate_activation_code(activation_code, SOFTWARE_VERSION)
                if success and isinstance(response, dict):
                    validity_period = response.get('validity_period')
                    period_display = {'permanent': DEFAULT_DATA['Permanent'], '1day': DEFAULT_DATA['1day'], '3days': DEFAULT_DATA['3days'], '1week': DEFAULT_DATA['1week'], '1month': DEFAULT_DATA['1month'], '3months': DEFAULT_DATA['3months'], '6months': DEFAULT_DATA['6months'], '1year': DEFAULT_DATA['1year']}.get(validity_period, validity_period)
                    messagebox.showinfo(DEFAULT_DATA['ActivationSuccess'], DEFAULT_DATA['ActivationSuccessMsg'].format(period_display))
                    dialog.destroy()
                    timestamp = datetime.now().strftime('%H%M%S')
                    session_key = f'{random.randint(1000, 9999)}{random.randint(1000, 9999)}{timestamp}'
                    launch_time = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                    os.environ['ROPE_SESSION'] = session_key
                    os.environ['ROPE_LAUNCH_TIME'] = launch_time
                else:
                    error_message = response if isinstance(response, str) else DEFAULT_DATA['LicenseActivationError']
                    messagebox.showerror(DEFAULT_DATA['ActivationError'], error_message)
            except Exception as e:
                messagebox.showerror(DEFAULT_DATA['Error'], DEFAULT_DATA['InvalidActivationCode'])
        ttk.Label(main_frame, text=DEFAULT_DATA['ActivationCodeLabel']).pack(anchor=tk.W, pady=(0, 5))
        activation_code_entry = ttk.Entry(main_frame, width=40)
        activation_code_entry.pack(fill=tk.X, pady=(0, 10))
        activate_button = ttk.Button(main_frame, text=DEFAULT_DATA['ActivateButton'], command=activate)
        activate_button.pack(fill=tk.X, pady=(0, 10))
        machine_code_label = ttk.Label(main_frame, text=DEFAULT_DATA['BuyActivationCodeLabel'], wraplength=260, foreground='red')
        machine_code_label.pack(fill=tk.X, pady=(5, 0), side=tk.BOTTOM)
        def buy_activation_code():
            import webbrowser
            if SOFTWARE_VERSION == 'rope-plus':
                webbrowser.open(DEFAULT_DATA['TelegramUrl'])
            else:
                if SOFTWARE_VERSION == 'rope':
                    webbrowser.open(DEFAULT_DATA['BuyActivationCodeLink'])
        buy_activation_code_button = ttk.Button(main_frame, text=DEFAULT_DATA['BuyActivationCode'], command=buy_activation_code)
        buy_activation_code_button.pack(fill=tk.X)
        dialog.update_idletasks()
        dialog.geometry('+{}+{}'.format((dialog.winfo_screenwidth() - dialog.winfo_width()) // 2, (dialog.winfo_screenheight() - dialog.winfo_height()) // 2))
        dialog.grab_set()
        dialog.wait_window()
    except Exception as e:
        messagebox.showerror(DEFAULT_DATA['Error'], DEFAULT_DATA['CreateActivationWindowError'].format(str(e)))
        sys.exit(1)
    if not check_license(SOFTWARE_VERSION):
        sys.exit()
def print_startup_info():
    """打印启动信息"""
    return
def update_obs_config():
    obs_config_path = os.path.join(os.getenv('APPDATA'), 'obs-studio', 'plugin_config', 'obs-websocket', 'config.json')
    if not os.path.exists(obs_config_path):
        return False
    obs_config = {}
    with open(obs_config_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    if content:
        obs_config = json.loads(content)
    try:
        subprocess.run(['taskkill', '/F', '/IM', 'obs64.exe'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        obs_config['server_enabled'] = True
        with open(obs_config_path, 'w', encoding='utf-8') as f:
            json.dump(obs_config, f, indent=4)
        server_password = obs_config.get('server_password', '')
        server_port = obs_config.get('server_port', 4455)
        if getattr(sys, 'frozen', False):
            exe_path = os.path.abspath(sys.executable)
            base_path = os.path.dirname(exe_path)
            config_path = os.path.join(base_path, 'config', 'config.json')
        else:
            config_path = Path(__file__).resolve().parent / 'config' / 'config.json'
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        config_data = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    c_text = f.read().strip()
                if c_text:
                    config_data = json.loads(c_text)
            except Exception:
                config_data = {}
        config_data['obs'] = config_data.get('obs', {})
        config_data['obs'].update({'host': '127.0.0.1', 'port': server_port, 'password': server_password})
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f'更新配置文件时出错: {e}')
        return False


def uninstall_droidcam():
    try:
        uninstall_path = 'C:\\Program Files\\DroidCam\\Drivers\\uninstall.exe'
        if os.path.exists(uninstall_path):
            subprocess.Popen([uninstall_path, '/S'], shell=False, creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS)
    except Exception as e:
        return None
atexit.register(uninstall_droidcam)
def delete_droidcam_plugin():
    """删除DroidCam插件"""
    try:
        obs_path = ''
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 'SOFTWARE\\WOW6432Node\\OBS Studio') as key:
                obs_path, _ = winreg.QueryValueEx(key, '')
        except Exception:
            pass
        if not obs_path:
            for drive in 'CDEFGH':
                path = f'{drive}:\\Program Files\\obs-studio'
                if os.path.exists(path):
                    obs_path = path
                    break
        if obs_path:
            subprocess.run(['taskkill', '/F', '/IM', 'obs64.exe'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            plugin_dll = os.path.join(obs_path, 'obs-plugins', '64bit', 'droidcam-virtual-output.dll')
            if os.path.exists(plugin_dll):
                os.remove(plugin_dll)
            plugin_data = os.path.join(obs_path, 'data', 'obs-plugins', 'droidcam-virtual-output')
            if os.path.exists(plugin_data):
                import shutil
                shutil.rmtree(plugin_data)
            return True
        else:
            return False
    except Exception:
        return False
def perform_cleanup():
    """执行所有清理操作"""
    try:
        subprocess.run(['taskkill', '/F', '/IM', 'obs64.exe'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
    except Exception as e:
        pass
    try:
        obs_path = ''
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 'SOFTWARE\\WOW6432Node\\OBS Studio') as key:
                obs_path, _ = winreg.QueryValueEx(key, '')
        except Exception:
            pass
        if not obs_path:
            for drive in 'CDEFGH':
                path = f'{drive}:\\Program Files\\obs-studio'
                if os.path.exists(path):
                    obs_path = path
                    break
        if obs_path:
            plugin_dll = os.path.join(obs_path, 'obs-plugins', '64bit', 'droidcam-virtual-output.dll')
            if os.path.exists(plugin_dll):
                os.remove(plugin_dll)
            plugin_data = os.path.join(obs_path, 'data', 'obs-plugins', 'droidcam-virtual-output')
            if os.path.exists(plugin_data):
                import shutil
                shutil.rmtree(plugin_data)
        else:
            pass
    except Exception as e:
        pass
    try:
        uninstall_path = 'C:\\Program Files\\DroidCam\\Drivers\\uninstall.exe'
        if os.path.exists(uninstall_path):
            subprocess.Popen([uninstall_path, '/S'], shell=True, creationflags=subprocess.CREATE_NO_WINDOW)
    except Exception as e:
        return None
def handle_console_exit(ctrl_type):
    """处理控制台关闭事件"""
    if ctrl_type in [win32con.CTRL_C_EVENT, win32con.CTRL_BREAK_EVENT, win32con.CTRL_CLOSE_EVENT]:
        perform_cleanup()
    return False
win32api.SetConsoleCtrlHandler(handle_console_exit, True)
import atexit
atexit.register(perform_cleanup)
def install_droidcam_driver():
    """安装DroidCam驱动"""
    if getattr(sys, 'frozen', False):
        exe_path = os.path.abspath(sys.executable)
        base_path = os.path.dirname(exe_path)
        local_installer_path = os.path.join(base_path, 'ext_dependencies', 'wireshark', 'DroidCam.Drivers.7.1.1.exe')
    else:
        local_installer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ext_dependencies', 'wireshark', 'DroidCam.Drivers.7.1.1.exe')
    if os.path.exists(local_installer_path):
        try:
            subprocess.run([local_installer_path, '/S'], check=True)
        except Exception as e:
            messagebox.showerror('驱动安装错误', f'驱动安装失败: {str(e)}')
    else:
        messagebox.showerror('驱动安装错误', f'驱动安装程序不存在: {local_installer_path}')
def install_droidcam_plugin():
    """安装DroidCam插件"""
    if getattr(sys, 'frozen', False):
        exe_path = os.path.abspath(sys.executable)
        base_path = os.path.dirname(exe_path)
        local_installer_path = os.path.join(base_path, 'ext_dependencies', 'wireshark', 'OBS.Plugin.0.2.1.exe')
    else:
        local_installer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ext_dependencies', 'wireshark', 'OBS.Plugin.0.2.1.exe')
    if os.path.exists(local_installer_path):
        try:
            subprocess.run([local_installer_path, '/S'], check=True)
        except Exception as e:
            messagebox.showerror('插件安装错误', f'插件安装失败: {str(e)}')
    else:
        messagebox.showerror('插件安装错误', f'插件安装程序不存在: {local_installer_path}')
def limit_camera_resolution():
    """限制摄像头分辨率，优化性能"""
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            camera_name = cap.getBackendName()
            if 'C1000e' in camera_name or 'Logitech' in camera_name:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                cap.set(cv2.CAP_PROP_FPS, 30)
            cap.release()
    except Exception as e:
        return None
def main():
    verify_launch()
    license_valid = initialize_license_system()
    update_obs_config()
    limit_camera_resolution()
    if not license_valid:
        root = tk.Tk()
        root.withdraw()
        show_activation_dialog(root)
        if check_license(SOFTWARE_VERSION):
            try:
                root.destroy()
                Coordinator.run()
            except Exception as e:
                messagebox.showerror('错误', DEFAULT_DATA['ErrorRunningCoordinator'].format(str(e)))
                raise
        else:
            sys.exit()
    else:
        install_droidcam_driver()
        install_droidcam_plugin()
        try:
            Coordinator.run()
        except tk.TclError as e:
            messagebox.showerror('Tkinter错误', DEFAULT_DATA['TkinterError'].format(str(e)))
            if 'pyimage' in str(e):
                messagebox.showerror('资源错误', DEFAULT_DATA['ImageResourceLoadFailed'])
            raise
        except Exception as e:
            messagebox.showerror('错误', DEFAULT_DATA['ErrorRunningCoordinator'].format(str(e)))
            raise


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        messagebox.showerror('错误', DEFAULT_DATA['ErrorRunningCoordinator'].format(str(e)))
        try:
            for widget in tk.Tk().winfo_children():
                widget.destroy()
        except Exception:
            pass
        sys.exit(1)
    perform_cleanup()
