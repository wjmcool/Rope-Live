
import os
import subprocess
import psutil
import json
import time
import sys
from pathlib import Path
from tkinter import messagebox
from rope.language_manager import load_language_dict
from obsws_python import ReqClient
Dicts = load_language_dict()
DEFAULT_DATA = Dicts.DEFAULT_DATA
obs_config_path = os.path.expandvars('%AppData%\\\\obs-studio\\\\global.ini')
def find_obs_path_from_process():
    # irreducible cflow, using cdg fallback
    for proc in psutil.process_iter(['pid', 'name', 'exe']):
        try:
            if proc.info['name'] == 'obs64.exe':
                obs_path = proc.info['exe']
                if obs_path:
                    proc.kill()
                    proc.wait()
                    return obs_path
        except (psutil.AccessDenied, psutil.NoSuchProcess) as e:
            print(DEFAULT_DATA['Error'], DEFAULT_DATA['AccessDenied'].format(str(e)))
def find_obs_path():
    """通过注册表和常见路径查找 OBS 安装位置"""
    obs_path = find_obs_path_from_process()
    if obs_path:
        return obs_path
    import winreg
    import subprocess
    try:
        result = subprocess.run(['reg', 'query', 'HKLM\\SOFTWARE\\WOW6432Node\\OBS Studio', '/ve'], capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            if '(默认)' in line or 'REG_SZ' in line:
                parts = line.strip().split('    ')
                if len(parts) >= 3:
                    install_path = parts[(-1)]
                    obs_path = os.path.join(install_path, 'bin', '64bit', 'obs64.exe')
                    if Path(obs_path).is_file():
                        return obs_path
    except subprocess.CalledProcessError:
        pass
    for key_path in ['SOFTWARE\\OBS Studio', 'SOFTWARE\\WOW6432Node\\OBS Studio']:
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                install_path, _ = winreg.QueryValueEx(key, '')
                obs_path = os.path.join(install_path, 'bin', '64bit', 'obs64.exe')
                if Path(obs_path).is_file():
                    return obs_path
        except FileNotFoundError:
            pass
        except Exception as e:
            print(DEFAULT_DATA['Error'], f'注册表查询失败: {str(e)}')
    common_paths = ['C:\\Program Files\\obs-studio\\bin\\64bit\\obs64.exe', 'C:\\Program Files (x86)\\obs-studio\\bin\\64bit\\obs64.exe', 'D:\\Program Files\\obs-studio\\bin\\64bit\\obs64.exe', 'D:\\Program Files (x86)\\obs-studio\\bin\\64bit\\obs64.exe', 'E:\\Program Files\\obs-studio\\bin\\64bit\\obs64.exe', 'E:\\Program Files (x86)\\obs-studio\\bin\\64bit\\obs64.exe']
    for path in common_paths:
        if Path(path).is_file():
            return path
    print(DEFAULT_DATA['Error'], DEFAULT_DATA['OBSPathNotFound'])
def update_obs_config():
    try:
        if not os.path.exists(obs_config_path):
            print(DEFAULT_DATA['Error'], DEFAULT_DATA['OBSConfigFileNotFound'])
            return False
        else:
            with open(obs_config_path, 'r', encoding='utf-8-sig') as file:
                lines = file.readlines()
        in_section = False
        section_found = False
        autostart_found = False
        section_index = 0
        for i, line in enumerate(lines):
            if line.strip() == '[DroidCamVirtualOutput]':
                in_section = True
                section_found = True
                section_index = i
            else:
                if in_section and line.startswith('AutoStart='):
                        lines[i] = 'AutoStart=true\n'
                        autostart_found = True
                        break
                if in_section and line.startswith('['):
                        break
        if not section_found:
            lines.extend(['\n[DroidCamVirtualOutput]\n', 'AutoStart=true\n'])
        else:
            if not autostart_found:
                lines.insert(section_index + 1, 'AutoStart=true\n')
        with open(obs_config_path, 'w', encoding='utf-8-sig') as file:
            file.writelines(lines)
    except Exception as e:
        print(DEFAULT_DATA['Error'], DEFAULT_DATA['UpdateOBSConfigError'].format(str(e)))
        return False
def terminate_process(process_name):
    try:
        for process in psutil.process_iter(['pid', 'name']):
            if process.info['name'] == process_name:
                psutil.Process(process.info['pid']).terminate()
                return True
        return True
    except Exception as e:
        print(DEFAULT_DATA['Error'], DEFAULT_DATA['TerminateProcessError'].format(process_name, str(e)))
        return False
def start_application(app_path, is_obs=False):
    if app_path and Path(app_path).is_file():
        try:
            app_dir = os.path.dirname(app_path)
            current_dir = os.getcwd()
            os.chdir(app_dir)
            if is_obs:
                subprocess.Popen([app_path, '--safe-mode=false', '--minimize-to-tray', '--disable-shutdown-check', '--disable-preview'])
            else:
                subprocess.Popen(app_path)
            os.chdir(current_dir)
            return True
        except Exception as e:
            print(DEFAULT_DATA['Error'], DEFAULT_DATA['StartApplicationError'].format(app_path, str(e)))
            return False
    else:
        print(DEFAULT_DATA['Error'], DEFAULT_DATA['InvalidAppPath'].format(app_path))
        return False
def Camera_Extension_Setup():
    try:
        cfg_path = Path(sys.executable).parent / 'config' / 'config.json' if getattr(sys, 'frozen', False) else Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / 'config' / 'config.json'
        cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
        client = ReqClient(host=cfg['obs']['host'], port=cfg['obs']['port'], password=cfg['obs']['password'])
        scenes = {s['sceneName'] for s in client.get_scene_list().scenes}
        if '换脸' not in scenes:
            client.create_scene('换脸')
        client.set_current_program_scene('换脸')
        src_name = '换脸视频源'
        src_kind = 'dshow_input'
        src_settings = {'video_device_id': 'OBS Virtual Camera:', 'device': 'OBS Virtual Camera', 'resolution': '1920x1080', 'fps': 30, 'buffering': True, 'active': True, 'path': 'video=OBS Virtual Camera:', 'last_video_device_id': 'OBS Virtual Camera:'}
        try:
            client.get_input_list()
        except Exception:
            pass
        if not any((i['inputName'] == src_name for i in client.get_input_list().inputs)):
            client.create_input('换脸', src_name, src_kind, src_settings, True)
        video = client.get_video_settings()
        cw, ch = (video.base_width, video.base_height)
        items = client.get_scene_item_list('换脸').scene_items
        item_id = next((i['sceneItemId'] for i in items if i['sourceName'] == src_name), None)
        if item_id is not None:
            client.set_scene_item_transform('换脸', item_id, {'boundsType': 'OBS_BOUNDS_SCALE_INNER', 'boundsWidth': cw, 'boundsHeight': ch, 'boundsAlignment': 5, 'positionX': 0, 'positionY': 0})
        print(DEFAULT_DATA['ExtendCamerasSuccess'])
        return True
    except Exception as e:
        print(DEFAULT_DATA['ExtendCamerasError'], e)
        return False
def main():
    print('\n')
    print(DEFAULT_DATA['ExtendCameras'])
    success = True
    obs_path = find_obs_path()
    if not obs_path:
        return
    else:
        if not update_obs_config():
            success = False
        time.sleep(2)
        if not start_application(obs_path, is_obs=True):
            success = False
        if success and (not Camera_Extension_Setup()):
                success = False
        if success:
            messagebox.showinfo(DEFAULT_DATA['Success'], DEFAULT_DATA['ExtendCamerasSuccess'])
        else:
            messagebox.showerror(DEFAULT_DATA['Error'], DEFAULT_DATA['ExtendCamerasError'])
if __name__ == '__main__':
    main()