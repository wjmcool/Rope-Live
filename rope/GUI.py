
global current_dict
import os
import traceback
import threading
import cv2
import tkinter as tk
from tkinter import filedialog, font
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk
import json
import time
import copy
import bisect
import torch
import torchvision
import customtkinter as ctk
import random as rd
from datetime import datetime
import sys
from .logger_config import setup_logger, get_logger
from .config import rope_live_version
torchvision.disable_beta_transforms_warning()
import mimetypes
import webbrowser
import rope.GUIElements as GE
import rope.Styles as style
from skimage import transform as trans
from torchvision.transforms import v2
from tkinter import messagebox
from . import language_manager
from os import listdir
from os.path import isfile, join
from platform import system
from rope.Dicts import CAMERA_BACKENDS, get_media_path, cv2_imread_unicode
from rope.FaceLandmarks import FaceLandmarks
from rope.FaceEditor import FaceEditor
from rope.rtmp_sniffer import get_base_dir
import rope.extend_cameras as extend_cameras
import sys
import gc
from obswebsocket import obsws, requests
from rope.license_utils import check_license, get_remaining_days, get_license_info, get_aware_datetime
from .license_online import LicenseClient, validate_activation_code as online_validate_activation_code
from rope.rtmp_sniffer import get_rtmp_info
import simpleobsws
import asyncio
import subprocess
from datetime import datetime
class GUI(tk.Tk):
    def __init__(self, models, media_path=None, software_version=None):
        super().__init__()
        self.logger = get_logger('GUI')
        self.software_version = software_version or 'rope'
        self.current_language = 'Chinese'
        self.language_var = tk.StringVar(value=self.current_language)
        self.widget = {}
        check_mark_path = os.path.join(os.path.dirname(__file__), 'media', 'check.png')
        check_mark_image = Image.open(check_mark_path)
        check_mark_image = check_mark_image.resize((16, 16), Image.Resampling.LANCZOS)
        self.widget['CheckMark'] = ImageTk.PhotoImage(check_mark_image)
        Dicts = language_manager.load_language_dict()
        self.DEFAULT_DATA = Dicts.DEFAULT_DATA
        self.PARAM_VARS = Dicts.PARAM_VARS
        self.PARAMS = Dicts.PARAMS
        self.CAMERA_BACKENDS = Dicts.CAMERA_BACKENDS
        self.models = models
        if media_path is None:
            media_path = os.path.join(os.path.dirname(__file__), 'media')
        self.media_path = media_path
        self.image_cache = {}
        if self.software_version == 'rope-plus':
            self.title(self.DEFAULT_DATA['PlusWindowTitle'])
        else:
            self.title(self.DEFAULT_DATA['WindowTitle'])
        self.target_media = []
        self.target_video_file = []
        self.action_q = []
        self.video_image = []
        self.video_loaded = False
        self.image_loaded = False
        self.image_file_name = []
        self.stop_marker = []
        self.stop_image = []
        self.stop_marker_icon = []
        self.window_last_change = []
        self.blank = tk.PhotoImage()
        self.output_folder = []
        self.output_videos_text = []
        self.target_media_buttons = []
        self.input_videos_button = []
        self.input_videos_text = []
        self.target_media_canvas = []
        self.source_faces_buttons = []
        self.input_videos_button = []
        self.input_faces_text = []
        self.shift_i_len = 0
        self.source_faces_canvas = []
        self.video = []
        self.video_slider = []
        self.found_faces_canvas = []
        self.merged_embedding_name = []
        self.merged_embeddings_text = []
        self.me_name = []
        self.merged_faces_canvas = []
        self.parameters = {}
        self.parameters_face_editor = {}
        self.control = {}
        self.widget = {}
        self.static_widget = {}
        self.layer = {}
        self.temp_emb = []
        self.arcface_dst = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
        self.json_dict = {'source videos': None, 'source faces': None, 'saved videos': None, 'dock_win_geom': [1400, 800, self.winfo_screenwidth() / 2 - 400, self.winfo_screenheight() / 2 - 510]}
        self.marker = {'框架': '0', '参数': '', 'icon_ref': ''}
        self.markers = []
        self.target_face = {'TKButton': [], '按钮状态': 'off', '图像': [], 'Embedding': [], 'SourceFaceAssignments': [], 'EmbeddingNumber': 0, 'AssignedEmbedding': [], 'DFLModel': False}
        self.target_faces = []
        self.source_face = {'TKButton': [], '按钮状态': 'off', 'Image': [], 'Embedding': [], 'DFLModel': False}
        self.source_faces = []
        config_path = os.path.join(get_base_dir(), 'config', 'config.json')
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = json.load(file)
            if not isinstance(config, dict):
                config = {}
        except (OSError, ValueError):
            # 首次运行或文件损坏：用内置默认值，不阻塞启动
            config = {}
        obs_config = config.get('obs') or {}
        self.obs_host = obs_config.get('host', '')
        self.obs_port = obs_config.get('port', 4455)
        self.obs_password = obs_config.get('password', '')
        self.loop = asyncio.get_event_loop()
        script_dir = os.path.dirname(__file__)
        icon_path = os.path.join(script_dir, 'media', 'rope.ico')
        if system() != 'Linux' and os.path.exists(icon_path):
                self.iconbitmap(icon_path)
        self.default_params_visibility = {}
        self.default_params_face_editor_visibility = {}
        def load_shortcuts_from_json():
            try:
                with open('shortcuts.json', 'r', encoding='utf-8') as json_file:
                    return json.load(json_file)
            except FileNotFoundError:
                return {'时间轴开始': 'z', '向左轻移30帧': 'a', '右移30帧': 'd', '记录': 'r', 'Play': 'space', '保存图像': 'ctrl+s', '添加标记': 'f', '删除标记': 'alt+f', '上一个标记': 'q', '下一个标记': 'w', '切换恢复': '1', '切换恢复2': 'h', '切换方向': '2', '切换强度': '3', '切换差异': '4', '切换遮挡': '5', '切换人脸解析': '6', '切换基于文本的蒙版': '7', '切换颜色调整': '8', '切换面部调整': '9', 'Swap Faces（换脸）': 's', '向左轻移1帧': 'c', '右移1帧': 'v', '显示蒙版': 'x'}
        shortcuts = load_shortcuts_from_json()
        text_vars = {}
        for shortcut_name, default_value in shortcuts.items():
            text_vars[shortcut_name] = tk.StringVar(value=default_value)
        self.key_actions ={
            shortcuts['时间轴开始']: lambda: self.preview_control('q'),
            shortcuts['向左轻移30帧']: lambda: self.preview_control('a'),
            shortcuts['右移30帧']: lambda: self.preview_control('d'),
            shortcuts['记录']: lambda: self.toggle_rec_video(),
            shortcuts['Play']: lambda: self.toggle_play_video(),
            shortcuts['保存图像']: lambda: self.save_image(),
            shortcuts['添加标记']: lambda: self.update_marker('add'),
            shortcuts['删除标记']: lambda: self.update_marker('delete'),
            shortcuts['上一个标记']: lambda: self.update_marker('prev'),
            shortcuts['下一个标记']: lambda: self.update_marker('next'),
            shortcuts['切换恢复']: lambda: self.toggle_and_update('Restorer', 'Restorer'),
            shortcuts['切换方向']: lambda: self.toggle_and_update('Orient', 'Orientation'),
            shortcuts['切换强度']: lambda: self.toggle_and_update('Strength', 'Strength'),
            shortcuts['切换差异']: lambda: self.toggle_and_update('Diff', 'Differencing'),
            shortcuts['切换遮挡']: lambda: self.toggle_and_update('Occluder', 'Occluder'),
            shortcuts['切换人脸解析']: lambda: self.toggle_and_update('FaceParser', 'Face Parser'),
            shortcuts['切换基于文本的蒙版']: lambda: self.toggle_and_update('CLIP', 'Text-Based Masking'),
            shortcuts['切换颜色调整']: lambda: self.toggle_and_update('Color', 'Color Adjustments'),
            shortcuts['切换面部调整']: lambda: self.toggle_and_update('FaceAdj', 'Input Face Adjustments'),
            shortcuts['Swap Faces（换脸）']: lambda: self.toggle_swapper(),
            shortcuts['向左轻移1帧']: lambda: self.back_one_frame(),
            shortcuts['右移1帧']: lambda: self.forward_one_frame(),
            shortcuts['显示蒙版']: lambda: self.toggle_maskview(),
        }
        self.bind('<Key>', self.handle_key_press)
        self.bind('<Return>', lambda event: self.focus_set())
    def load_language_setting(self):
        """加载保存的语言设置"""
        try:
            if os.path.exists(self.language_file):
                with open(self.language_file, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            else:
                return 'Chinese'
        except Exception as e:
            return 'Chinese'
    def save_language_setting(self, language):
        """保存语言设置到现有的data.json"""
        try:
            with open(language_manager.data_json_path(), 'r', encoding='utf-8') as f:
                config = json.load(f)
            if not isinstance(config, dict):
                config = {}
            config['language'] = language
            with open(language_manager.data_json_path(), 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            self.current_language = language
        except Exception as e:
            messagebox.showerror(self.DEFAULT_DATA['Error'], self.DEFAULT_DATA['LanguageChangeError'] + str(e))
    def handle_key_press(self, event):
        if isinstance(self.focus_get(), tk.Entry):
            return
        else:
            f_keys = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12']
            modifiers = [mod for mod, mask in [('shift', 1), ('ctrl', 4), ('alt', 131072)] if event.state & mask]
            key_combination = event.keysym if event.keysym in f_keys else '+'.join(filter(None, ['+'.join(modifiers), event.keysym.lower()]))
            action = self.key_actions.get(key_combination)
            action and action()
    def toggle_and_update(self, switch_name, parameter_name):
        self.widget[f'{switch_name}Switch'].set(not self.widget[f'{switch_name}Switch'].get())
    def update_license_status(self):
        if check_license():
            license_info = get_license_info()
            remaining_days = get_remaining_days()
            if remaining_days == 9999 or (license_info and license_info.get('validity_period') == 'permanent'):
                status_text = self.DEFAULT_DATA['LicenseStatusText'].format(self.DEFAULT_DATA['Permanent'])
            else:
                expires_at = license_info.get('expires_at') if license_info else None
                if not expires_at:
                    status_text = self.DEFAULT_DATA['LicenseStatusText'].format('永久有效')
                else:
                    expires_date = get_aware_datetime(expires_at)
                    formatted_date = expires_date.strftime('%Y-%m-%d')
                    if remaining_days == 0:
                        status_text = self.DEFAULT_DATA['LicenseStatusText'].format(self.DEFAULT_DATA['ExpiringToday'].format(formatted_date))
                    else:
                        status_text = self.DEFAULT_DATA['LicenseStatusText'].format(self.DEFAULT_DATA['ExpiresOn'].format(formatted_date))
        else:
            status_text = self.DEFAULT_DATA['LicenseStatusText'].format('未激活')
        # 状态栏文字已隐藏：控件保留，只清空内容
        status_text = ''
        self.license_status_label.config(text=status_text)
    def on_topright_resize(self, event):
        """处理topright层大小变化，动态调整设置按钮位置"""
        if hasattr(self, 'settings_button') and self.settings_button.winfo_exists():
                topright_width = event.width
                button_width = 60
                new_x = max(20, topright_width - button_width - 20)
                self.settings_button.place(x=new_x, y=9)
    def show_settings_menu(self):
        if self.settings_menu is None:
            self.settings_menu = tk.Menu(self, tearoff=0)
            self.settings_menu.add_command(label=self.DEFAULT_DATA['GetStreamKeyTitle'], command=lambda: self.on_settings_select(self.DEFAULT_DATA['GetStreamKeyTitle']))
            self.settings_menu.add_command(label=self.DEFAULT_DATA['CameraExtensionTitle'], command=lambda: self.on_settings_select(self.DEFAULT_DATA['CameraExtensionTitle']))
            self.language_menu = tk.Menu(self.settings_menu, tearoff=0)
            self.settings_menu.add_cascade(label=self.DEFAULT_DATA['LanguageSettingsTitle'], menu=self.language_menu)
            languages = {'中文': 'Chinese', 'English': 'English'}
            self.settings_menu.add_command(label=self.DEFAULT_DATA['RegisterSoftware'], command=lambda: self.on_settings_select(self.DEFAULT_DATA['RegisterSoftware']))
            current_display_name = next((name for name, code in languages.items() if code == self.current_language), '中文')
            self.language_var.set(current_display_name)
            for display_name, lang_code in languages.items():
                self.language_menu.add_radiobutton(label=display_name, value=display_name, variable=self.language_var, command=lambda l=lang_code: self.on_settings_select(l))
        self.settings_menu.post(self.settings_button.winfo_rootx(), self.settings_button.winfo_rooty() + self.settings_button.winfo_height())
    def check_npcap(self):
        try:
            subprocess.run(['sc', 'query', 'npcap'], check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False
    def on_settings_select(self, choice):
        if choice in ['Chinese', 'English']:
            self.change_language(choice)
            return
        else:
            if choice == self.DEFAULT_DATA['GetStreamKeyTitle']:
                if not self.check_npcap():
                    messagebox.showerror(self.DEFAULT_DATA['NpcapErrorTitle'], self.DEFAULT_DATA['NpcapErrorMessage'])
                    webbrowser.open('https://npcap.com/#download')
                    return
                else:
                    self.get_stream_key()
            else:
                if choice == self.DEFAULT_DATA['CameraExtensionTitle']:
                    extend_cameras.main()
                else:
                    if choice == self.DEFAULT_DATA['RegisterSoftware']:
                        self.open_activation_window()
                    else:
                        if choice == 'English':
                            self.change_language('English')
                        else:
                            if choice == '中文':
                                self.change_language('Chinese')
    def open_activation_window(self):
        """打开激活窗口（当前版本授权已关闭，保留入口备用）。"""
        activation_window = ctk.CTkToplevel(self)
        activation_window.title(self.DEFAULT_DATA.get('RegisterSoftware', '注册软件'))
        window_width, window_height = (400, 250)
        x = (activation_window.winfo_screenwidth() - window_width) // 2
        y = (activation_window.winfo_screenheight() - window_height) // 2
        activation_window.geometry(f'{window_width}x{window_height}+{x}+{y}')
        activation_window.resizable(False, False)
        activation_window.lift()
        activation_window.focus_force()
        activation_window.grab_set()

        main_frame = ctk.CTkFrame(activation_window)
        main_frame.pack(padx=20, pady=20, fill='both', expand=True)

        machine_code = ''
        try:
            from .license_utils import get_machine_code
            machine_code = get_machine_code()
        except Exception:
            machine_code = ''
        ctk.CTkLabel(main_frame, text=str(machine_code)).pack(pady=(20, 5))

        ctk.CTkLabel(main_frame,
                     text=self.DEFAULT_DATA.get('ActivationCodeLabel', '激活码：')).pack(pady=(0, 5))
        code_entry = ctk.CTkEntry(main_frame, width=300)
        code_entry.pack(pady=(0, 20))

        def finish(result):
            success, message = result if isinstance(result, tuple) else (False, result)
            activate_button.configure(state='normal',
                                      text=self.DEFAULT_DATA.get('ActivateButton', '激活'))
            if success:
                messagebox.showinfo(self.DEFAULT_DATA.get('ActivationSuccess', '激活成功'),
                                    self.DEFAULT_DATA.get('RestartText', '激活成功，重启软件'))
                self.restart_application()
            else:
                if isinstance(message, dict):
                    message = message.get('message') or str(message)
                messagebox.showerror(self.DEFAULT_DATA.get('Error', '错误'),
                                     message or '激活失败')

        def worker(code):
            try:
                outcome = online_validate_activation_code(code)
            except Exception:
                outcome = (False, '激活失败')
            try:
                self.after(0, lambda: finish(outcome))
            except Exception:
                pass

        def validate_and_activate():
            code = code_entry.get().strip()
            if not code:
                messagebox.showerror(self.DEFAULT_DATA.get('Error', '错误'),
                                     self.DEFAULT_DATA.get('EmptyActivationCode', '请输入激活码'))
                return
            activate_button.configure(state='disabled',
                                      text=self.DEFAULT_DATA.get('ActivatingText', '激活中…'))
            import threading
            threading.Thread(target=worker, args=(code,), daemon=True).start()

        activate_button = ctk.CTkButton(main_frame,
                                        text=self.DEFAULT_DATA.get('ActivateButton', '激活'),
                                        command=validate_and_activate)
        activate_button.pack(pady=(0, 10))
        activate_button = ctk.CTkButton(main_frame, text=self.DEFAULT_DATA.get('ActivateButton', '激活'), command=validate_and_activate)
        activate_button.pack(pady=(10, 30))
    def get_stream_key(self):
        try:
            server, code = get_rtmp_info()
            if server and code:
                    full_rtmp = f'{server}/{code}'
        except Exception as e:
            messagebox.showerror(self.DEFAULT_DATA['Error'], self.DEFAULT_DATA['GetStreamKeyError'] + str(e))
    async def configure_obs_for_douyin(self, rtmp_url):
        # irreducible cflow, using cdg fallback
        ws = simpleobsws.WebSocketClient(url=f'ws://{self.obs_host}:{self.obs_port}', password=self.obs_password)
        try:
            await ws.connect()
            await ws.wait_until_identified()
            get_current_settings_request = simpleobsws.Request('GetStreamServiceSettings')
            get_current_settings_response = await ws.call(get_current_settings_request)
            if get_current_settings_response.ok():
                self.logger.info(f"{self.DEFAULT_DATA['CurrentStreamSettings']}: {json.dumps(get_current_settings_response.responseData, indent=2)}")
            else:
                messagebox.showerror(self.DEFAULT_DATA['Error'], self.DEFAULT_DATA['GetStreamKeyError'] + str(get_current_settings_response.responseData))
            server, stream_key = rtmp_url.split('/third/', 1)
            server = f'{server}/third'
            new_settings = {'server': server, 'key': stream_key}
            request = simpleobsws.Request('SetStreamServiceSettings', {'streamServiceType': 'rtmp_custom', 'streamServiceSettings': new_settings})
            response = await ws.call(request)
            if response.ok():
                pass
            else:
                messagebox.showerror(self.DEFAULT_DATA['Error'], self.DEFAULT_DATA['GetStreamKeyError'] + str(response.responseData))
            get_request = simpleobsws.Request('GetStreamServiceSettings')
            get_response = await ws.call(get_request)
            if get_response.ok():
                updated_settings = get_response.responseData
                if updated_settings['streamServiceSettings']['server'] == server and updated_settings['streamServiceSettings']['key'] == stream_key:
                    self.after(0, lambda: messagebox.showinfo(self.DEFAULT_DATA['Warning'], self.DEFAULT_DATA['OBSStreamSuccess']))
                else:
                    self.after(0, lambda: messagebox.showwarning(self.DEFAULT_DATA['Warning'], self.DEFAULT_DATA['OBSStreamWarning']))
            else:
                messagebox.showerror(self.DEFAULT_DATA['Error'], self.DEFAULT_DATA['GetStreamKeyError'] + str(get_response.responseData))
        except Exception as e:
            error_message = self.DEFAULT_DATA['Error'] + str(e)
            self.after(0, lambda: messagebox.showerror(self.DEFAULT_DATA['Error'], error_message))
        finally:
            await ws.disconnect()
    def start_stream(self):
        try:
            server, code = get_rtmp_info()
            if server and code:
                full_rtmp = f'{server}/{code}'
                self.loop.run_until_complete(self.configure_obs_for_douyin(full_rtmp))
            else:
                messagebox.showinfo(self.DEFAULT_DATA['Error'], self.DEFAULT_DATA['NoStreamInfoError'])
        except Exception as e:
            messagebox.showerror(self.DEFAULT_DATA['Error'], self.DEFAULT_DATA['StreamSettingsError'] + str(e))
    def on_closing(self):
        """处理窗口关闭事件"""
        try:
            uninstall_path = 'C:\\Program Files\\DroidCam\\Drivers\\uninstall.exe'
            if os.path.exists(uninstall_path):
                subprocess.Popen([uninstall_path, '/S'], shell=False, creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS)
        except Exception as e:
            pass
        self.destroy()
        sys.exit(0)
    def run(self):
        self.mainloop()
    def create_gui(self):
        """创建GUI元素"""
        global pv_frame
        global tmp
        global pv_frame2
        global ks_frame
        global v_f_frame
        try:
            current_row = 0
            self.widget = {name: widget for name, widget in self.widget.items() if hasattr(widget, '__class__') and widget.self.__module__ == 'rope.GUIElements'}
            top_frame = tk.Frame(self, style.canvas_frame_label_1)
            top_frame.grid(row=0, column=0, sticky='NEWS', padx=1, pady=1)
            top_frame.grid_columnconfigure(0, weight=1)
            top_frame.grid_columnconfigure(1, weight=0)
            self.layer['topleft'] = tk.Frame(top_frame, style.canvas_frame_label_1, height=42)
            self.layer['topleft'].grid(row=0, column=0, sticky='NEWS', pady=0)
            self.widget['StartButton'] = GE.Button(self.layer['topleft'], 'StartRope', 1, self.load_all, None, 'control', 10, 9, width=200)
            self.widget['OutputFolderButton'] = GE.Button(self.layer['topleft'], 'OutputFolder', 1, self.select_save_video_path, None, 'control', x=240, y=1, width=190)
            self.output_videos_text = GE.Text(self.layer['topleft'], '', 1, 240, 20, 190, 20)
            self.configure(bg=style.bg)
            ctk.set_appearance_mode('dark')
            middle_frame = tk.Frame(self, style.canvas_frame_label_1)
            middle_frame.grid(row=1, column=0, sticky='NEWS', padx=1, pady=1)
            middle_frame.grid_columnconfigure(0, weight=0)
            middle_frame.grid_columnconfigure(1, weight=1)
            middle_frame.grid_columnconfigure(2, weight=0)
            middle_frame.grid_columnconfigure(3, weight=0)
            self.layer['parameters_frame'] = tk.Frame(middle_frame, style.canvas_frame_label_3)
            self.layer['parameters_frame'].grid(row=0, column=2, sticky='NEWS', pady=0, padx=1)
            widget_items = list(self.widget.items())
            for widget_name, widget_instance in widget_items:
                try:
                    if widget_instance.parent == self.layer.get('parameters_frame'):
                        widget_instance.grid(row=current_row, column=0, padx=2, pady=2)
                        current_row += 1
                        if hasattr(widget_instance, 'related_widgets'):
                            for related_widget in widget_instance.related_widgets:
                                if related_widget:
                                    related_widget.grid(row=current_row - 1, column=1, padx=2, pady=2)
                except Exception as e:
                    import logging
                    logging.error(f'处理widget {widget_name}时出错: {e}')
        except Exception as e:
            import logging
            logging.error(f'创建GUI时出错: {e}')
            logging.error(traceback.format_exc())
        self.configure(bg=style.bg)
        ctk.set_appearance_mode('dark')
        tmp = ctk.CTkFrame(self, border_width=0, fg_color=style.main, bg_color=style.bg)
        tmp.grid(row=1, column=0, sticky='NEWS', padx=0, pady=0)
        tmp.grid_forget()
        def vidupdate():
            self.resize_image()
        def input_panel_checkbox():
            current_state = self.checkbox.get()
            if current_state:
                self.layer['InputVideoFrame'].grid(row=0, column=0, sticky='NEWS', padx=0, pady=(0, 0))
                ks_frame.grid_forget()
                pv_frame_container.grid_forget()
                self.collapse_keyboardshortcuts.deselect()
                self.collapse_parametersvisibility.deselect()
            else:
                self.layer['InputVideoFrame'].grid_forget()
                v_f_frame.grid_forget()
                self.after(10, vidupdate)
        def collapse_faces_panel():
            current_state = self.collapse_bottom.get()
            if current_state:
                ff_frame.grid(row=5, column=0, sticky='NEWS', padx=0, pady=(1, 0))
                mf_frame.grid(row=6, column=0, sticky='NEWS', padx=0, pady=1)
                self.after(10, vidupdate)
            else:
                ff_frame.grid_forget()
                mf_frame.grid_forget()
                self.after(10, vidupdate)
        def collapse_params_panel():
            current_state = self.collapse_params.get()
            if current_state:
                self.layer['parameter_frame'].grid(row=0, column=2, sticky='NEWS', pady=0, padx=0)
                self.after(10, vidupdate)
            else:
                self.layer['parameter_frame'].grid_forget()
                self.after(10, vidupdate)
        def keyboard_shortcuts():
            current_state = self.collapse_keyboardshortcuts.get()
            if current_state:
                ks_frame.grid(row=0, column=0, sticky='NEWS', padx=0, pady=(0, 0))
                self.after(10, vidupdate)
                self.layer['InputVideoFrame'].grid_forget()
                pv_frame_container.grid_forget()
                self.after(10, vidupdate)
                self.checkbox.deselect()
                self.collapse_parametersvisibility.deselect()
            else:
                ks_frame.grid_forget()
                pv_frame_container.grid_forget()
                v_f_frame.grid_forget()
                self.after(10, vidupdate)
                self.layer['InputVideoFrame'].grid(row=0, column=0, sticky='NEWS', padx=0, pady=(0, 0))
                self.after(10, vidupdate)
                self.checkbox.select()
        def parameters_visibility():
            current_state = self.collapse_parametersvisibility.get()
            if current_state:
                pv_frame_container.grid(row=0, column=0, sticky='NEWS', padx=0, pady=(0, 0))
                self.after(10, vidupdate)
                self.layer['InputVideoFrame'].grid_forget()
                ks_frame.grid_forget()
                self.after(10, vidupdate)
                self.checkbox.deselect()
                self.collapse_keyboardshortcuts.deselect()
            else:
                pv_frame_container.grid_forget()
                ks_frame.grid_forget()
                v_f_frame.grid_forget()
                self.after(10, vidupdate)
                self.layer['InputVideoFrame'].grid(row=0, column=0, sticky='NEWS', padx=0, pady=(0, 0))
                self.after(10, vidupdate)
                self.checkbox.select()
                self.collapse_keyboardshortcuts.deselect()
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.configure(style.frame_style_bg)
        top_frame = tk.Frame(self, style.canvas_frame_label_1)
        top_frame.grid(row=0, column=0, sticky='NEWS', padx=1, pady=1)
        top_frame.grid_columnconfigure(0, weight=1)
        top_frame.grid_columnconfigure(1, weight=0)
        middle_frame = tk.Frame(self, style.frame_style_bg)
        middle_frame.grid(row=1, column=0, sticky='NEWS', padx=0, pady=0)
        middle_frame.grid_rowconfigure(0, weight=1)
        middle_frame.grid_columnconfigure(0, weight=0)
        middle_frame.grid_columnconfigure(1, weight=1)
        middle_frame.grid_columnconfigure(2, weight=0)
        middle_frame.grid_columnconfigure(3, weight=0)
        v_f_frame = ctk.CTkFrame(middle_frame, height=42, border_width=0, fg_color=style.main)
        v_f_frame.grid(row=0, column=0, sticky='NEWS', padx=0, pady=(0, 0))
        y = 0
        x = 10
        ks_frame = ctk.CTkFrame(middle_frame, height=42, width=250, border_width=0, fg_color=style.main, background_corner_colors=(style.main, style.main, style.main, style.main))
        ks_frame.grid(row=0, column=0, sticky='NEWS', padx=0, pady=(0, 0))
        ks_frame.grid_forget()
        def load_shortcuts_from_json():
            try:
                with open('shortcuts.json', 'r', encoding='utf-8') as json_file:
                    return json.load(json_file)
            except FileNotFoundError:
                return {'时间轴开始': 'z', '左移30帧': 'a', '右移30帧': 'd', '记录': 'r', '播放': 'space', '保存图像': 'ctrl+s', '添加标记': 'f', '删除标记': 'alt+f', '上一个标记': 'q', '下一个标记': 'w', '切换恢复': '1', '切换恢复2': 'h', '切换方向': '2', '切换强度': '3', '切换差异': '4', '切换遮挡': '5', '切换人脸解析': '6', '切换基于文本的蒙版': '7', '切换颜色调整': '8', '切换面部调整': '9', '清除显存': 'F1', '换脸': 's', '左移1帧': 'c', '右移1帧': 'v', '显示蒙版': 'x'}
        shortcuts = load_shortcuts_from_json()
        def save_shortcuts_to_json(shortcuts):
            with open('shortcuts.json', 'w', encoding='utf-8') as json_file:
                json.dump(shortcuts, json_file)
        def update_key_actions():
            self.key_actions ={
                shortcuts['时间轴开始']: lambda: self.preview_control('q'),
                shortcuts['向左轻移30帧']: lambda: self.preview_control('a'),
                shortcuts['记录']: lambda: self.toggle_rec_video(),
                shortcuts['Play']: lambda: self.toggle_play_video(),
                shortcuts['右移30帧']: lambda: self.preview_control('d'),
                shortcuts['保存图像']: lambda: self.save_image(),
                shortcuts['添加标记']: lambda: self.update_marker('add'),
                shortcuts['删除标记']: lambda: self.update_marker('delete'),
                shortcuts['上一个标记']: lambda: self.update_marker('prev'),
                shortcuts['下一个标记']: lambda: self.update_marker('next'),
                shortcuts['切换恢复']: lambda: self.toggle_and_update('Restorer', 'Restorer'),
                shortcuts['切换方向']: lambda: self.toggle_and_update('Orient', 'Orientation'),
                shortcuts['切换强度']: lambda: self.toggle_and_update('Strength', 'Strength'),
                shortcuts['切换差异']: lambda: self.toggle_and_update('Diff', 'Differencing'),
                shortcuts['切换遮挡']: lambda: self.toggle_and_update('Occluder', 'Occluder'),
                shortcuts['切换人脸解析']: lambda: self.toggle_and_update('FaceParser', 'Face Parser'),
                shortcuts['切换基于文本的蒙版']: lambda: self.toggle_and_update('CLIP', 'Text-Based Masking'),
                shortcuts['切换颜色调整']: lambda: self.toggle_and_update('Color', 'Color Adjustments'),
                shortcuts['切换面部调整']: lambda: self.toggle_and_update('FaceAdj', 'Input Face Adjustments'),
                shortcuts['Swap Faces（换脸）']: lambda: self.toggle_swapper(),
                shortcuts['向左轻移1帧']: lambda: self.back_one_frame(),
                shortcuts['右移1帧']: lambda: self.forward_one_frame(),
                shortcuts['显示蒙版']: lambda: self.toggle_maskview(),
            }
        shortcuts = load_shortcuts_from_json()
        text_vars = {}
        for shortcut_name, default_value in shortcuts.items():
            text_vars[shortcut_name] = tk.StringVar(value=default_value)
        def save_shortcuts():
            for shortcut_name, text_var in text_vars.items():
                shortcuts[shortcut_name] = text_var.get()
            save_shortcuts_to_json(shortcuts)
            update_key_actions()
        save_ks_button = ctk.CTkButton(ks_frame, text='保存快捷键', command=save_shortcuts, width=150, height=15, corner_radius=3, fg_color=style.main2, hover_color=style.main3)
        save_ks_button.place(x=40, y=20)
        y = 60
        x = 10
        for shortcut_name, default_value in shortcuts.items():
            ctk.CTkLabel(ks_frame, text=shortcut_name).place(x=x, y=y)
            ctk.CTkEntry(ks_frame, textvariable=text_vars[shortcut_name], width=50, height=15, border_width=0).place(x=180, y=y)
            y += 20
        pv_frame_container = tk.Frame(middle_frame, style.frame_style_bg)
        pv_frame_container.grid(row=0, column=0, sticky='NEWS', padx=0, pady=0)
        pv_frame_container.grid_rowconfigure(0, weight=0)
        pv_frame_container.grid_rowconfigure(1, weight=0)
        pv_frame_container.grid_rowconfigure(2, weight=1)
        empty_row = ctk.CTkLabel(pv_frame_container, text='', fg_color=style.main2, height=15)
        empty_row.grid(row=0, column=0, sticky='NS', padx=0, pady=0)
        tabview_main_visibility = ctk.CTkTabview(pv_frame_container, width=350, height=100, corner_radius=6, border_width=1, fg_color=style.main, border_color=style.main3, segmented_button_selected_hover_color='#b1b1b2', segmented_button_unselected_hover_color=style.main, segmented_button_selected_color='#7562ee', segmented_button_unselected_color=style.main, text_color='#F1E5AC', text_color_disabled=style.main2)
        tabview_main_visibility.grid(row=2, column=0, sticky='nsew')
        tab_face_swapper_visibility = tabview_main_visibility.add(self.DEFAULT_DATA['FaceSwapperParametersTitle'])
        tab_face_editor_visibility = tabview_main_visibility.add(self.DEFAULT_DATA['FaceEditorTitle'])
        pv_frame = GE.CTkScrollableFrame(tab_face_swapper_visibility, allow_drag_and_drop=True, allowed_widget_type=GE.ParamSwitch, border_width=0, fg_color=style.main, background_corner_colors=(style.main, style.main, style.main, style.main))
        pv_frame.grid(row=0, column=0, sticky='nsew', padx=0, pady=(0, 0))
        tab_face_swapper_visibility.grid_rowconfigure(0, weight=1)
        tab_face_swapper_visibility.grid_columnconfigure(0, weight=1)
        pv_frame2 = GE.CTkScrollableFrame(tab_face_editor_visibility, allow_drag_and_drop=True, allowed_widget_type=GE.ParamSwitch, border_width=0, fg_color=style.main, background_corner_colors=(style.main, style.main, style.main, style.main))
        pv_frame2.grid(row=0, column=0, sticky='nsew', padx=0, pady=(0, 0))
        tab_face_editor_visibility.grid_rowconfigure(0, weight=1)
        tab_face_editor_visibility.grid_columnconfigure(0, weight=1)
        def load_params_visibility_from_json(task='startup', initial_dir='.'):
            try:
                if task == 'startup':
                    with open('startup_parameters_visibility.json', 'r', encoding='utf-8') as json_file:
                        config_data = json.load(json_file)
                        file_name = json_file.name
                else:
                    with filedialog.askopenfile(mode='r', encoding='utf-8', initialdir=initial_dir, filetypes=[('JSON files', '*.json'), ('All files', '*.*')]) as json_file:
                        config_data = json.load(json_file)
                        file_name = json_file.name
                if config_data.get('config_type') != 'parameters_visibility':
                    import logging
                    logging.error(f'错误: {file_name} 配置类型无效!')
                    return (None, None)
                else:
                    return (config_data.get('parameters', {}), config_data.get('parameters_face_editor', {}))
            except FileNotFoundError:
                return ({}, {})
            except json.JSONDecodeError:
                import logging
                logging.error(f'JSON文件解码错误: {file_name}')
                return (None, None)
        def get_params_visibility_from_frame(param_type='all'):
            params_visibility = {}
            params_face_editor_visibility = {}
            if param_type == 'all' or param_type == 'parameters':
                max_row = pv_frame.scrollable_frame.grid_size()[1]
                for row in range(max_row):
                    widgets_in_row = pv_frame.scrollable_frame.grid_slaves(row=row)
                    for widget in widgets_in_row:
                        if hasattr(widget, 'draggable_object_instance'):
                            params_visibility[widget.draggable_object_instance.name] = widget.draggable_object_instance.get()
            if param_type == 'all' or param_type == 'parameters_face_editor':
                max_row = pv_frame2.scrollable_frame.grid_size()[1]
                for row in range(max_row):
                    widgets_in_row = pv_frame2.scrollable_frame.grid_slaves(row=row)
                    for widget in widgets_in_row:
                        if hasattr(widget, 'draggable_object_instance'):
                            params_face_editor_visibility[widget.draggable_object_instance.name] = widget.draggable_object_instance.get()
            return (params_visibility, params_face_editor_visibility)
        def save_params_visibility_to_json(params_visibility, params_face_editor_visibility, initial_dir='.', default_filename='startup_parameters_visibility.json'):
            config_data = {'config_type': 'parameters_visibility', 'version': '1.0', 'parameters': params_visibility, 'parameters_faceeditor': params_face_editor_visibility}
            save_file = filedialog.asksaveasfile(mode='w', encoding='utf-8', initialdir=initial_dir, defaultextension='.json', filetypes=[('JSON files', '*.json'), ('All files', '*.*')], initialfile=default_filename)
            if save_file:
                with save_file:
                    json.dump(config_data, save_file, indent=4)
        def save_params_visibility():
            params_visibility, params_face_editor_visibility = get_params_visibility_from_frame(param_type='all')
            save_params_visibility_to_json(params_visibility, params_face_editor_visibility)
        def remove_param_switch_widgets(param_type='all'):
            if param_type == 'all' or param_type == 'parameters':
                max_row = pv_frame.scrollable_frame.grid_size()[1]
                for row in range(max_row):
                    widgets_in_row = pv_frame.scrollable_frame.grid_slaves(row=row)
                    for widget in widgets_in_row:
                        if hasattr(widget, 'draggable_object_instance'):
                            widget.destroy()
            if param_type == 'all' or param_type == 'parameters_face_editor':
                max_row = pv_frame2.scrollable_frame.grid_size()[1]
                for row in range(max_row):
                    widgets_in_row = pv_frame2.scrollable_frame.grid_slaves(row=row)
                    for widget in widgets_in_row:
                        if hasattr(widget, 'draggable_object_instance'):
                            widget.destroy()
        def load_params_visibility_configuration():
            json_conf, json_conf2 = load_params_visibility_from_json(task='manual', initial_dir='.')
            if json_conf:
                remove_param_switch_widgets(param_type='parameters')
                for widget_name, widget_instance in self.default_params_visibility.items():
                    if widget_name not in json_conf:
                        json_conf[widget_name] = True
                apply_params_visibility_configuration(json_conf, None, param_type='parameters', reload=True)
            if json_conf2:
                remove_param_switch_widgets(param_type='parameters_face_editor')
                for widget_name, widget_instance in self.default_params_face_editor_visibility.items():
                    if widget_name not in json_conf2:
                        json_conf2[widget_name] = True
                apply_params_visibility_configuration(None, json_conf2, param_type='parameters_face_editor', reload=True)
        def default_params_visibility_configuration():
            remove_param_switch_widgets(param_type='all')
            apply_params_visibility_configuration(self.default_params_visibility, self.default_params_face_editor_visibility, param_type='all', reload=True)
        def apply_params_visibility_configuration(params_visibility=None, params_face_editor_visibility=None, param_type='all', reload=False):
            if param_type == 'all' or param_type == 'parameters':
                if params_visibility == None:
                    params_visibility, _ = get_params_visibility_from_frame(param_type='parameters')
                padx = 1
                pady = 0
                pv_row = 0
                row = 1
                column = 0
                for widget_name, widget_value in params_visibility.items():
                    if widget_name in self.widget:
                        pv_row += 1
                        row += 1
                        if reload:
                            GE.ParamSwitch(pv_frame.scrollable_frame, widget_name, self.widget[widget_name].display_text, 3, self.update_param_visibility, widget_value, 398, 20, pv_row, 0, padx, pady, allow_drag_and_drop=True)
                        if hasattr(self.widget[widget_name], 'frame'):
                            self.widget[widget_name].frame.grid(row=row, column=column, padx=padx, pady=pady)
                        if not widget_value:
                            self.widget[widget_name].hide()
                        else:
                            if widget_value:
                                self.widget[widget_name].unhide()
            if param_type == 'all' or param_type == 'parameters_face_editor':
                if params_face_editor_visibility == None:
                    _, params_face_editor_visibility = get_params_visibility_from_frame(param_type='parameters_face_editor')
                padx = 1
                pady = 0
                pv_row = 0
                row = 1
                column = 0
                for widget_name, widget_value in params_face_editor_visibility.items():
                    if widget_name in self.widget:
                        pv_row += 1
                        row += 1
                        if reload:
                            GE.ParamSwitch(pv_frame2.scrollable_frame, widget_name, self.widget[widget_name].display_text, 3, self.update_param_visibility, widget_value, 398, 20, pv_row, 0, padx, pady, allow_drag_and_drop=True)
                        if hasattr(self.widget[widget_name], 'frame'):
                            self.widget[widget_name].frame.grid(row=row, column=column, padx=padx, pady=pady)
                        if not widget_value:
                            self.widget[widget_name].hide()
                        else:
                            if widget_value:
                                self.widget[widget_name].unhide()
            self.static_widget['parameters_scrollbar'].resize_scrollbar(None)
        button_frame = tk.Frame(pv_frame_container, style.frame_style_bg)
        button_frame.grid(row=1, column=0, sticky='EW', padx=0, pady=0)
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        button_frame.grid_columnconfigure(2, weight=1)
        button_frame.grid_columnconfigure(3, weight=1)
        save_pv_button = ctk.CTkButton(button_frame, text=self.DEFAULT_DATA['Save'], command=save_params_visibility, width=80, height=15, corner_radius=3, fg_color=style.main, hover_color=style.main3, text_color='#FFFFE0', anchor='center')
        save_pv_button.grid(row=0, column=0, padx=0, pady=0)
        apply_pv_button = ctk.CTkButton(button_frame, text=self.DEFAULT_DATA['Apply'], command=lambda: apply_params_visibility_configuration(params_visibility=None, params_face_editor_visibility=None, param_type='all', reload=False), width=80, height=15, corner_radius=3, fg_color=style.main, hover_color=style.main3, text_color='#FFFFE0', anchor='center')
        apply_pv_button.grid(row=0, column=1, padx=0, pady=0)
        load_pv_button = ctk.CTkButton(button_frame, text=self.DEFAULT_DATA['Load'], command=load_params_visibility_configuration, width=80, height=15, corner_radius=3, fg_color=style.main, hover_color=style.main3, text_color='#FFFFE0', anchor='center')
        load_pv_button.grid(row=0, column=2, padx=0, pady=0)
        default_pv_button = ctk.CTkButton(button_frame, text=self.DEFAULT_DATA['Default'], command=default_params_visibility_configuration, width=80, height=15, corner_radius=3, fg_color=style.main, hover_color=style.main3, text_color='#FFFFE0', anchor='center')
        default_pv_button.grid(row=0, column=3, padx=0, pady=0)
        pv_frame_container.grid_forget()
        bottom_frame = tk.Frame(self, style.canvas_frame_label_1)
        bottom_frame.grid(row=2, column=0, sticky='NEWS', padx=1, pady=1)
        bottom_frame.grid_columnconfigure(0, minsize=100)
        bottom_frame.grid_columnconfigure(1, weight=1)
        bottom_frame.grid_columnconfigure(2, minsize=100)
        self.layer['topleft'] = self.layer['topleft'].grid(row=0, column=0, sticky='NEWS', pady=0)
        self.widget['StartButton'] = GE.Button(self.layer['topleft'], 'StartRope', 1, self.load_all, None, 'control', 10, 9, width=200)
        self.widget['OutputFolderButton'] = GE.Button(self.layer['topleft'], 'OutputFolder', 1, self.select_save_video_path, None, 'control', x=240, y=1, width=190)
        self.output_videos_text = GE.Text(self.layer['topleft'], '', 1, 240, 20, 190, 20)
        self.layer['topright'] = tk.Frame(top_frame, style.canvas_frame_label_1, height=42, width=413)
        self.layer['topright'].grid(row=0, column=1, sticky='NEWS', pady=0)
        self.settings_menu = None
        self.static_widget['vram_indicator'] = GE.VRAM_Indicator(self.layer['topright'], 1, 190, 20, 20, 11)
        self.settings_button = tk.Button(self.layer['topright'], text=self.DEFAULT_DATA['SettingsTitle'], command=self.show_settings_menu, width=6, height=1, bg='#151515', fg='#b0b0b0')
        self.settings_button.place(x=20, y=9)
        self.layer['topright'].bind('<Configure>', self.on_topright_resize)
        self.checkbox = ctk.CTkCheckBox(self.layer['topleft'], text=self.DEFAULT_DATA['InputPanelTitle'], text_color='#B0B0B0', command=input_panel_checkbox, onvalue=True, offvalue=False, checkbox_width=18, checkbox_height=18, border_width=0, hover_color='#303030', fg_color=style.main)
        self.checkbox.place(x=500, y=10)
        self.checkbox.select()
        self.collapse_bottom = ctk.CTkCheckBox(self.layer['topleft'], text=self.DEFAULT_DATA['FacesPanelTitle'], text_color='#B0B0B0', command=collapse_faces_panel, onvalue=True, offvalue=False, checkbox_width=18, checkbox_height=18, border_width=0, hover_color='#303030', fg_color=style.main)
        self.collapse_bottom.place(x=600, y=10)
        self.collapse_bottom.select()
        self.collapse_params = ctk.CTkCheckBox(self.layer['topleft'], text=self.DEFAULT_DATA['ParamsPanelTitle'], text_color='#B0B0B0', command=collapse_params_panel, onvalue=True, offvalue=False, checkbox_width=18, checkbox_height=18, border_width=0, hover_color='#303030', fg_color=style.main)
        self.collapse_params.place(x=705, y=10)
        self.collapse_params.select()
        self.collapse_keyboardshortcuts = ctk.CTkCheckBox(self.layer['topleft'], text=self.DEFAULT_DATA['KeyboardShortcutsTitle'], text_color='#B0B0B0', command=keyboard_shortcuts, onvalue=True, offvalue=False, checkbox_width=18, checkbox_height=18, border_width=0, hover_color='#303030', fg_color=style.main)
        self.collapse_keyboardshortcuts.place(x=840, y=10)
        self.collapse_parametersvisibility = ctk.CTkCheckBox(self.layer['topleft'], text=self.DEFAULT_DATA['ParametersVisibilityTitle'], text_color='#B0B0B0', command=parameters_visibility, onvalue=True, offvalue=False, checkbox_width=18, checkbox_height=18, border_width=0, hover_color='#303030', fg_color=style.main)
        self.collapse_parametersvisibility.place(x=985, y=10)
        self.layer['InputVideoFrame'] = tk.Frame(middle_frame, style.canvas_frame_label_3)
        self.layer['InputVideoFrame'].grid(row=0, column=0, sticky='NEWS', padx=1, pady=0)
        self.layer['InputVideoFrame'].grid_rowconfigure(0, weight=0)
        self.layer['InputVideoFrame'].grid_rowconfigure(1, weight=1)
        self.layer['InputVideoFrame'].grid_columnconfigure(0, weight=0)
        self.layer['InputVideoFrame'].grid_columnconfigure(1, weight=0)
        self.layer['InputVideoFrame'].grid_columnconfigure(0, weight=0)
        self.layer['InputVideoFrame'].grid_columnconfigure(1, weight=0)
        frame = tk.Frame(self.layer['InputVideoFrame'], style.canvas_frame_label_2, height=42)
        frame.grid(row=0, column=0, columnspan=2, sticky='NEWS', padx=0, pady=0)
        self.widget['VideoFolderButton'] = GE.Button(frame, 'LoadTVideos', 2, self.select_video_path, None, 'control', 10, 1, width=195)
        self.input_videos_text = GE.Text(frame, '', 2, 10, 20, 190, 20)
        self.target_media_canvas = tk.Canvas(self.layer['InputVideoFrame'], style.canvas_frame_label_3, height=100, width=195)
        self.target_media_canvas.grid(row=1, column=0, sticky='NEWS', padx=10, pady=10)
        self.target_media_canvas.bind('<MouseWheel>', self.target_videos_mouse_wheel)
        self.target_media_canvas.create_text(8, 20, anchor='w', fill='grey25', font=('Microsoft YaHei', 16), text=f" {self.DEFAULT_DATA['TargetVideoTitle']}")
        scroll_canvas = tk.Canvas(self.layer['InputVideoFrame'], style.canvas_frame_label_3, bd=0)
        scroll_canvas.grid(row=1, column=1, sticky='NEWS', padx=0, pady=0)
        scroll_canvas.grid_rowconfigure(0, weight=1)
        scroll_canvas.grid_columnconfigure(0, weight=1)
        self.static_widget['input_videos_scrollbar'] = GE.Scrollbar_y(scroll_canvas, self.target_media_canvas)
        frame = tk.Frame(self.layer['InputVideoFrame'], style.canvas_frame_label_2, height=42)
        frame.grid(row=0, column=2, columnspan=2, sticky='NEWS', padx=0, pady=0)
        self.widget['FacesFolderButton'] = GE.Button(frame, 'LoadSFaces', 2, self.select_faces_path, None, 'control', 10, 1, width=195)
        self.input_faces_text = GE.Text(frame, '', 2, 10, 20, 190, 20)
        self.source_faces_canvas = tk.Canvas(self.layer['InputVideoFrame'], style.canvas_frame_label_3, height=100, width=195)
        self.source_faces_canvas.grid(row=1, column=2, sticky='NEWS', padx=10, pady=10)
        self.source_faces_canvas.bind('<MouseWheel>', self.source_faces_mouse_wheel)
        self.source_faces_canvas.create_text(8, 20, anchor='w', fill='grey25', font=('Microsoft YaHei', 16), text=self.DEFAULT_DATA['DesiredAvatarTitle'])
        scroll_canvas = tk.Canvas(self.layer['InputVideoFrame'], style.canvas_frame_label_3, bd=0)
        scroll_canvas.grid(row=1, column=3, sticky='NEWS', padx=0, pady=0)
        scroll_canvas.grid_rowconfigure(0, weight=1)
        scroll_canvas.grid_columnconfigure(0, weight=1)
        self.static_widget['input_faces_scrollbar'] = GE.Scrollbar_y(scroll_canvas, self.source_faces_canvas)
        GE.Separator_y(self.layer['InputVideoFrame'], 229, 0)
        GE.Separator_x(self.layer['InputVideoFrame'], 0, 41)
        self.layer['preview_column'] = tk.Frame(middle_frame, style.canvas_bg)
        self.layer['preview_column'].grid(row=0, column=1, sticky='NEWS', pady=0)
        self.layer['preview_column'].grid_columnconfigure(0, weight=1)
        self.layer['preview_column'].grid_rowconfigure(0, weight=0)
        self.layer['preview_column'].grid_rowconfigure(1, weight=1)
        self.layer['preview_column'].grid_rowconfigure(2, weight=0)
        self.layer['preview_column'].grid_rowconfigure(3, weight=0)
        self.layer['preview_column'].grid_rowconfigure(4, weight=0)
        self.layer['preview_column'].grid_rowconfigure(5, weight=0)
        self.layer['preview_column'].grid_rowconfigure(6, weight=0)
        preview_data = tk.Frame(self.layer['preview_column'], style.canvas_frame_label_2, height=24)
        preview_data.grid(row=0, column=0, sticky='NEWS', pady=0)
        preview_data.grid_columnconfigure(0, weight=1)
        preview_data.grid_columnconfigure(1, weight=1)
        preview_data.grid_columnconfigure(2, weight=1)
        preview_data.grid_columnconfigure(3, weight=1)
        preview_data.grid_columnconfigure(4, weight=1)
        preview_data.grid_rowconfigure(0, weight=0)
        frame = tk.Frame(preview_data, style.canvas_frame_label_2, height=24, width=100)
        frame.grid(row=0, column=0)
        self.widget['AudioButton'] = GE.Button(frame, 'Audio', 2, self.toggle_audio, None, 'control', x=0, y=0, width=100)
        frame = tk.Frame(preview_data, style.canvas_frame_label_2, height=24, width=100)
        frame.grid(row=0, column=1)
        self.widget['MaskViewButton'] = GE.Button(frame, 'MaskView', 2, self.toggle_maskview, None, 'control', x=0, y=0, width=100)
        frame = tk.Frame(preview_data, style.canvas_frame_label_2, height=24, width=100)
        frame.grid(row=0, column=2)
        self.widget['CompareViewButton'] = GE.Button(frame, 'CompareView', 2, self.toggle_compareview, None, 'control', x=0, y=0, width=100)
        frame = tk.Frame(preview_data, style.canvas_frame_label_2, height=24, width=100)
        frame.grid(row=0, column=3)
        self.widget['FloatPreviewButton'] = GE.Button(frame, 'FloatPreview', 2, self.toggle_float_preview, None, 'control', x=0, y=0, width=100)
        frame = tk.Frame(preview_data, style.canvas_frame_label_2, height=24, width=200)
        frame.grid(row=0, column=4)
        self.widget['PreviewModeTextSel'] = GE.TextSelection(frame, 'PreviewModeTextSel', '', 2, self.set_view, True, 'control', width=200, height=20, row=0, column=0, padx=1, pady=0, text_percent=1)
        self.video = tk.Label(self.layer['preview_column'], bg='black')
        self.video.grid(row=1, column=0, sticky='NEWS', padx=0, pady=0)
        self.video.bind('<MouseWheel>', self.iterate_through_merged_embeddings)
        self.video.bind('<ButtonRelease-1>', lambda event: self.toggle_play_video())
        self.video.bind('<Button-3>', self.show_preview_context_menu)
        self.preview_rotation_angle = 0
        self.float_preview_window = None
        self.float_preview_label = None
        self.layer['slider_frame'] = tk.Frame(self.layer['preview_column'], style.canvas_frame_label_2, height=50)
        self.layer['slider_frame'].grid(row=2, column=0, sticky='NEWS', pady=0)
        self.video_slider = GE.Timeline(self.layer['slider_frame'], self.widget, self.temp_toggle_swapper, self.temp_toggle_enhancer, self.temp_toggle_faces_editor, self.add_action)
        self.layer['markers_canvas'] = tk.Canvas(self.layer['preview_column'], style.canvas_frame_label_2, height=20)
        self.layer['markers_canvas'].grid(row=3, column=0, sticky='NEWS')
        self.layer['markers_canvas'].bind('<Configure>', lambda e: self.update_marker(e.width))
        self.layer['preview_frame'] = tk.Frame(self.layer['preview_column'], style.canvas_bg, height=40)
        self.layer['preview_frame'].grid(row=4, column=0, sticky='NEWS')
        self.layer['preview_frame'].grid_columnconfigure(0, weight=0)
        self.layer['preview_frame'].grid_columnconfigure(1, weight=1)
        self.layer['preview_frame'].grid_columnconfigure(2, weight=0)
        self.layer['preview_frame'].grid_rowconfigure(0, weight=0)
        self.layer['preview_frame'].grid_rowconfigure(1, weight=0)
        self.layer['play_controls_left'] = tk.Frame(self.layer['preview_frame'], style.canvas_frame_label_2, height=30, width=100)
        self.layer['play_controls_left'].grid(row=0, column=0, sticky='NEWS', pady=0)
        self.widget['SaveImageButton'] = GE.Button(self.layer['play_controls_left'], 'SaveImageButton', 2, self.save_image, None, 'control', x=0, y=5, width=100)
        cente_frame = tk.Frame(self.layer['preview_frame'], style.canvas_frame_label_2, height=30)
        cente_frame.grid(row=0, column=1, sticky='NEWS', pady=0)
        cente_frame.grid_columnconfigure(0, weight=0)
        cente_frame.grid_rowconfigure(0, weight=0)
        play_control_frame = tk.Frame(cente_frame, style.canvas_frame_label_2, height=30, width=320)
        play_control_frame.place(anchor='c', relx=0.5, rely=0.5)
        column = 0
        col_delta = 50
        self.widget['TLBegButton'] = GE.Button(play_control_frame, 'TLBeginning', 2, self.preview_control, 'q', 'control', x=column, y=2, width=20)
        column += col_delta
        self.widget['TLLeftButton'] = GE.Button(play_control_frame, 'TLLeft', 2, self.preview_control, 'a', 'control', x=column, y=2, width=20)
        column += col_delta
        self.widget['TLRecButton'] = GE.Button(play_control_frame, 'Record', 2, self.toggle_rec_video, None, 'control', x=column, y=2, width=20)
        column += col_delta
        self.widget['TLPlayButton'] = GE.Button(play_control_frame, 'Play', 2, self.toggle_play_video, None, 'control', x=column, y=2, width=20)
        column += col_delta
        self.widget['TLLoopButton'] = GE.Button(play_control_frame, 'Loop', 2, self.toggle_loop_video, None, 'control', x=column, y=2, width=20)
        column += col_delta
        self.widget['TLRightButton'] = GE.Button(play_control_frame, 'TLRight', 2, self.preview_control, 'd', 'control', x=column, y=2, width=20)
        right_playframe = tk.Frame(self.layer['preview_frame'], style.canvas_frame_label_2, height=30, width=120)
        right_playframe.grid(row=0, column=2, sticky='NEWS', pady=0)
        self.widget['AddMarkerButton'] = GE.Button(right_playframe, 'AddMarkerButton', 2, self.update_marker, 'add', 'control', x=0, y=5, width=20)
        self.widget['DelMarkerButton'] = GE.Button(right_playframe, 'DelMarkerButton', 2, self.update_marker, 'delete', 'control', x=25, y=5, width=20)
        self.widget['PrevMarkerButton'] = GE.Button(right_playframe, 'PrevMarkerButton', 2, self.update_marker, 'prev', 'control', x=50, y=5, width=20)
        self.widget['NextMarkerButton'] = GE.Button(right_playframe, 'NextMarkerButton', 2, self.update_marker, 'next', 'control', x=75, y=5, width=20)
        self.widget['SaveMarkerButton'] = GE.Button(right_playframe, 'SaveMarkerButton', 2, self.save_markers_json, None, 'control', x=95, y=5, width=20)
        self.layer['image_controls'] = tk.Frame(self.layer['preview_column'], style.canvas_frame_label_2, height=80)
        self.layer['image_controls'].grid(row=2, column=0, rowspan=2, sticky='NEWS', pady=0)
        self.widget['SaveImageButton'] = GE.Button(self.layer['image_controls'], 'SaveImageButton', 2, self.save_image, None, 'control', x=0, y=5, width=100)
        self.widget['AutoSwapButton'] = GE.Button(self.layer['image_controls'], 'AutoSwapButton', 2, self.toggle_auto_swap, None, 'control', x=150, y=5, width=100)
        self.layer['image_controls'].grid_forget()
        self.layer['FaceLab_controls'] = tk.Frame(self.layer['preview_column'], style.canvas_frame_label_2, height=80)
        self.layer['FaceLab_controls'].grid(row=2, column=0, rowspan=2, sticky='NEWS', pady=0)
        self.layer['FaceLab_controls'].grid_forget()
        ff_frame = tk.Frame(self.layer['preview_column'], style.canvas_frame_label_1)
        ff_frame.grid(row=5, column=0, sticky='NEWS', pady=1)
        ff_frame.grid_columnconfigure(0, weight=0)
        ff_frame.grid_columnconfigure(1, weight=1)
        ff_frame.grid_rowconfigure(0, weight=0)
        button_frame = tk.Frame(ff_frame, style.canvas_frame_label_2, height=99, width=224)
        button_frame.grid(row=0, column=0)
        self.widget['FindFacesButton'] = GE.Button(button_frame, 'FindFaces', 2, self.find_faces, None, 'control', x=112, y=0, width=112, height=33)
        self.widget['ClearFacesButton'] = GE.Button(button_frame, 'ClearFaces', 2, self.clear_faces, None, 'control', x=112, y=33, width=112, height=33)
        self.widget['SwapFacesButton'] = GE.Button(button_frame, 'SwapFaces', 2, self.toggle_swapper, None, 'control', x=0, y=0, width=112, height=33)
        self.widget['EditFacesButton'] = GE.Button(button_frame, 'EditFaces', 2, self.toggle_faces_editor, None, 'control', x=0, y=33, width=112, height=33)
        self.widget['EnhanceFrameButton'] = GE.Button(button_frame, 'EnhanceFrame', 2, self.toggle_enhancer, None, 'control', x=0, y=66, width=112, height=33)
        self.found_faces_canvas = tk.Canvas(ff_frame, style.canvas_frame_label_3, height=100)
        self.found_faces_canvas.grid(row=0, column=1, sticky='NEWS')
        self.found_faces_canvas.bind('<MouseWheel>', self.target_faces_mouse_wheel)
        self.found_faces_canvas.create_text(8, 45, anchor='w', fill='grey25', font=('Microsoft YaHei', 16), text=self.DEFAULT_DATA['FoundFacesTitle'])
        self.static_widget['23'] = GE.Separator_y(ff_frame, 111, 0)
        mf_frame = tk.Frame(self.layer['preview_column'], style.canvas_frame_label_1)
        mf_frame.grid(row=6, column=0, sticky='NEWS', pady=0)
        mf_frame.grid_columnconfigure(0, minsize=10)
        mf_frame.grid_columnconfigure(1, weight=1)
        mf_frame.grid_rowconfigure(0, weight=0)
        button_frame = tk.Frame(mf_frame, style.canvas_frame_label_2, height=100, width=112)
        button_frame.grid(row=0, column=0)
        self.widget['DelEmbedButton'] = GE.Button(button_frame, 'DelEmbed', 2, self.delete_merged_embedding, None, 'control', x=0, y=30, width=112, height=33)
        self.merged_embedding_name = tk.StringVar()
        self.merged_embeddings_text = tk.Entry(button_frame, style.entry_2, textvariable=self.merged_embedding_name)
        self.merged_embeddings_text.place(x=8, y=8, width=96, height=20)
        self.merged_embeddings_text.bind('<Return>', lambda event: self.save_selected_source_faces(self.merged_embedding_name))
        self.me_name = self.nametowidget(self.merged_embeddings_text)
        self.merged_faces_canvas = tk.Canvas(mf_frame, style.canvas_frame_label_3, height=100)
        self.merged_faces_canvas.grid(row=0, column=1, sticky='NEWS')
        self.merged_faces_canvas.grid_rowconfigure(0, weight=1)
        self.merged_faces_canvas.bind('<MouseWheel>', lambda event: self.merged_faces_canvas.xview_scroll(-int(event.delta / 120.0), 'units'))
        self.merged_faces_canvas.create_text(8, 45, anchor='w', fill='grey25', font=('Microsoft YaHei', 16), text=self.DEFAULT_DATA['DFMModelTitle'])
        self.static_widget['24'] = GE.Separator_y(mf_frame, 111, 0)
        width = 398
        self.layer['parameter_frame'] = tk.Frame(middle_frame, style.canvas_frame_label_3, bd=0, width=width)
        self.layer['parameter_frame'].grid(row=0, column=2, sticky='NEWS', pady=0, padx=1)
        self.layer['parameter_frame'].grid_rowconfigure(0, weight=0)
        self.layer['parameter_frame'].grid_rowconfigure(1, weight=1)
        self.layer['parameter_frame'].grid_rowconfigure(2, weight=0)
        self.layer['parameter_frame'].grid_columnconfigure(0, weight=0)
        self.layer['parameter_frame'].grid_columnconfigure(1, weight=0)
        parameters_control_frame = tk.Frame(self.layer['parameter_frame'], style.canvas_frame_label_2, bd=0, width=width, height=42)
        parameters_control_frame.grid(row=0, column=0, columnspan=2, sticky='NEWS', pady=0, padx=0)
        parameters_control_frame.grid_columnconfigure(0, weight=1)
        parameters_control_frame.grid_columnconfigure(1, weight=1)
        parameters_control_frame.grid_columnconfigure(2, weight=1)
        parameters_control_frame.grid_rowconfigure(0, weight=0)
        frame = tk.Frame(parameters_control_frame, style.canvas_frame_label_2, height=42, width=100)
        frame.grid(row=0, column=0)
        self.widget['SaveParamsButton'] = GE.Button(frame, 'SaveParamsButton', 2, self.parameter_io, 'save', 'control', x=0, y=8, width=100)
        frame = tk.Frame(parameters_control_frame, style.canvas_frame_label_2, height=42, width=100)
        frame.grid(row=0, column=1)
        self.widget['LoadParamsButton'] = GE.Button(frame, 'LoadParamsButton', 2, self.parameter_io, 'load', 'control', x=0, y=8, width=100)
        frame = tk.Frame(parameters_control_frame, style.canvas_frame_label_2, height=42, width=100)
        frame.grid(row=0, column=2)
        self.widget['DefaultParamsButton'] = GE.Button(frame, 'DefaultParamsButton', 2, self.parameter_io, 'default', 'control', x=0, y=8, width=100)
        self.layer['parameters_canvas'] = tk.Canvas(self.layer['parameter_frame'], style.canvas_frame_label_3, bd=0, width=width)
        self.layer['parameters_canvas'].grid(row=1, column=0, sticky='NEWS', pady=0, padx=0)
        tabview_main = ctk.CTkTabview(self.layer['parameters_canvas'], width=398, height=2050, corner_radius=6, border_width=1, fg_color=style.main, border_color=style.main3, segmented_button_selected_hover_color='#b1b1b2', segmented_button_unselected_hover_color=style.main, segmented_button_selected_color='#7562ee', segmented_button_unselected_color=style.main, text_color='#F1E5AC', text_color_disabled=style.main2)
        tabview_main.pack(fill='both', expand=True)
        self.layer['parameters_canvas'].create_window(0, 0, window=tabview_main, anchor='nw')
        tab_face_swapper = tabview_main.add(self.DEFAULT_DATA['FaceSwapperTitle'])
        tab_live_portrait = tabview_main.add(self.DEFAULT_DATA['LivePortraitTitle'])
        self.layer['parameters_frame'] = tk.Frame(tab_face_swapper, style.canvas_frame_label_3, bd=0, width=width, height=2050)
        self.layer['parameters_frame'].grid(row=0, column=0, sticky='NEWS', pady=0, padx=0)
        self.layer['parameters_face_editor_frame'] = tk.Frame(tab_live_portrait, style.canvas_frame_label_3, bd=0, width=width, height=2050)
        self.layer['parameters_face_editor_frame'].grid(row=0, column=0, sticky='NEWS', pady=0, padx=0)
        self.layer['parameter_scroll_canvas'] = tk.Canvas(self.layer['parameter_frame'], style.canvas_frame_label_3, bd=0)
        self.layer['parameter_scroll_canvas'].grid(row=1, column=1, sticky='NEWS', pady=0)
        self.layer['parameter_scroll_canvas'].grid_rowconfigure(0, weight=1)
        self.layer['parameter_scroll_canvas'].grid_columnconfigure(0, weight=1)
        self.static_widget['parameters_scrollbar'] = GE.Scrollbar_y(self.layer['parameter_scroll_canvas'], self.layer['parameters_canvas'])
        self.static_widget['30'] = GE.Separator_x(parameters_control_frame, 0, 41)
        row = 1
        column = 0
        padx = 1
        pady = 0
        row = row + 1
        self.widget['ProvidersPriorityTextSel'] = GE.TextSelection(self.layer['parameters_frame'], 'ProvidersPriorityTextSel', self.DEFAULT_DATA['ProcessingCoreTitle'], 3, self.update_data, 'parameter', 'parameter', 398, 20, row, 0, padx, pady, 0.72)
        row = row + 1
        self.widget['ThreadsSlider'] = GE.Slider2(self.layer['parameters_frame'], 'ThreadsSlider', self.DEFAULT_DATA['ThreadsTitle'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.72)
        row = row + 1
        self.widget['FaceSwapperModelTextSel'] = GE.TextSelectionComboBox(self.layer['parameters_frame'], 'FaceSwapperModelTextSel', self.DEFAULT_DATA['FaceSwapperModelTitle'], 3, self.update_data, 'parameter', 'parameter', 398, 20, row, 0, padx, pady, 0.72, 150)
        row = row + 1
        self.widget['SwapperTypeTextSel'] = GE.TextSelection(self.layer['parameters_frame'], 'SwapperTypeTextSel', self.DEFAULT_DATA['SwapperResolution'], 3, self.update_data, 'parameter', 'parameter', 398, 20, row, 0, padx, pady, 0.72)
        row = row + 1
        self.widget['WebCamBackendSel'] = GE.TextSelectionComboBox(self.layer['parameters_frame'], 'WebCamBackendSel', self.DEFAULT_DATA['WebcamBackendTitle'], 3, self.update_data, 'parameter', 'parameter', 398, 20, row, 0, padx, pady, 0.72, 150)
        row = row + 1
        self.widget['WebCamMaxResolSel'] = GE.TextSelectionComboBox(self.layer['parameters_frame'], 'WebCamMaxResolSel', self.DEFAULT_DATA['WebcamResolution'], 3, self.update_data, 'parameter', 'parameter', 398, 20, row, 0, padx, pady, 0.72, 150)
        row = row + 1
        self.widget['WebCamOrientationSel'] = GE.TextSelection(self.layer['parameters_frame'], 'WebCamOrientationSel', self.DEFAULT_DATA['WebcamOrientation'], 3, self.update_webcam_orientation, 'parameter', 'parameter', 398, 20, row, 0, padx, pady, 0.72)
        row = row + 1
        self.widget['WebCamMaxFPSSel'] = GE.TextSelection(self.layer['parameters_frame'], 'WebCamMaxFPSSel', self.DEFAULT_DATA['WebcamFPS'], 3, self.update_data, 'parameter', 'parameter', 398, 20, row, 0, padx, pady, 0.72)
        row = row + 1
        self.widget['WebCamMaxNoSlider'] = GE.Slider2(self.layer['parameters_frame'], 'WebCamMaxNoSlider', self.DEFAULT_DATA['WebcamCount'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.72)
        row = row + 1
        self.widget['VirtualCameraSwitch'] = GE.Switch2(self.layer['parameters_frame'], 'VirtualCameraSwitch', self.DEFAULT_DATA['SendFramesToVirtualCamera'], 3, self.toggle_virtualcam, 'control', 398, 20, row, 0, padx, pady)
        row = row + 1
        self.widget['RestorerSwitch'] = GE.Switch2(self.layer['parameters_frame'], 'RestorerSwitch', self.DEFAULT_DATA['RestorerTitle'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady)
        row = row + 1
        self.widget['RestorerTypeTextSel'] = GE.TextSelectionComboBox(self.layer['parameters_frame'], 'RestorerTypeTextSel', self.DEFAULT_DATA['RestorerModelTitle'], 3, self.update_data, 'parameter', 'parameter', 398, 20, row, 0, padx, pady, 0.72, 150)
        row = row + 1
        self.widget['RestorerDetTypeTextSel'] = GE.TextSelection(self.layer['parameters_frame'], 'RestorerDetTypeTextSel', self.DEFAULT_DATA['DetectionAlignment'], 3, self.update_data, 'parameter', 'parameter', 398, 20, row, 0, padx, pady, 0.72)
        row = row + 1
        self.widget['VQFRFidelitySlider'] = GE.Slider2(self.layer['parameters_frame'], 'VQFRFidelitySlider', self.DEFAULT_DATA['RestorerFidelity'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.72)
        row = row + 1
        self.widget['RestorerSlider'] = GE.Slider2(self.layer['parameters_frame'], 'RestorerSlider', self.DEFAULT_DATA['RestorerBlend'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.72)
        row = row + 1
        self.widget['FrameEnhancerTypeTextSel'] = GE.TextSelectionComboBox(self.layer['parameters_frame'], 'FrameEnhancerTypeTextSel', self.DEFAULT_DATA['EnhancerType'], 3, self.update_data, 'parameter', 'parameter', 398, 20, row, 0, padx, pady, 0.72, 150)
        row = row + 1
        self.widget['EnhancerSlider'] = GE.Slider2(self.layer['parameters_frame'], 'EnhancerSlider', self.DEFAULT_DATA['FrameEnhancerBlend'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.72)
        row = row + 1
        self.widget['OccluderSwitch'] = GE.Switch2(self.layer['parameters_frame'], 'OccluderSwitch', self.DEFAULT_DATA['Occluder'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady)
        row = row + 1
        self.widget['OccluderSlider'] = GE.Slider2(self.layer['parameters_frame'], 'OccluderSlider', self.DEFAULT_DATA['OccluderSize'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['BeautySwitch'] = GE.Switch2(self.layer['parameters_frame'], 'BeautySwitch', self.DEFAULT_DATA['BeautyFilterTitle'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady)
        row = row + 1
        self.widget['BeautyWhitenSlider'] = GE.Slider2(self.layer['parameters_frame'], 'BeautyWhitenSlider', self.DEFAULT_DATA['BeautyWhiten'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.72)
        row = row + 1
        self.widget['BeautyFaceSlimSlider'] = GE.Slider2(self.layer['parameters_frame'], 'BeautyFaceSlimSlider', self.DEFAULT_DATA['BeautyFaceSlim'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.72)
        row = row + 1
        self.widget['BeautyFilterTypeTextSel'] = GE.TextSelection(self.layer['parameters_frame'], 'BeautyFilterTypeTextSel', self.DEFAULT_DATA['BeautyFilterType'], 3, self.update_data, 'parameter', 'parameter', 398, 20, row, 0, padx, pady, 0.72)
        row = row + 1
        self.widget['BeautyFilterSlider'] = GE.Slider2(self.layer['parameters_frame'], 'BeautyFilterSlider', self.DEFAULT_DATA['BeautyFilterStrength'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.72)
        row = row + 1
        self.widget['FaceParserSwitch'] = GE.Switch2(self.layer['parameters_frame'], 'FaceParserSwitch', self.DEFAULT_DATA['FaceParser'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady)
        row = row + 1
        self.widget['BGParserBlurSlider'] = GE.Slider2(self.layer['parameters_frame'], 'BGParserBlurSlider', self.DEFAULT_DATA['BGParserBlur'], 3, self.update_data, 'parameter', 300, 20, row, 0, padx, pady, 0.62, 40)
        row = row + 1
        self.widget['ParserBlurSlider'] = GE.Slider2(self.layer['parameters_frame'], 'ParserBlurSlider', self.DEFAULT_DATA['ParserBlur'], 3, self.update_data, 'parameter', 300, 20, row, 0, padx, pady, 0.62, 40)
        row = row + 1
        self.widget['FaceParserSlider'] = GE.Slider2(self.layer['parameters_frame'], 'FaceParserSlider', self.DEFAULT_DATA['FaceParserGB'], 3, self.update_data, 'parameter', 300, 20, row, 0, padx, pady, 0.62, 40)
        row = row + 1
        self.widget['NeckParserSlider'] = GE.Slider2(self.layer['parameters_frame'], 'NeckParserSlider', self.DEFAULT_DATA['NeckParser'], 3, self.update_data, 'parameter', 300, 20, row, 0, padx, pady, 0.62, 40)
        row = row + 1
        self.widget['LeftEyeBrowParserSlider'] = GE.Slider2(self.layer['parameters_frame'], 'LeftEyeBrowParserSlider', self.DEFAULT_DATA['LeftEyeBrowParser'], 3, self.update_data, 'parameter', 300, 20, row, 0, padx, pady, 0.62, 40)
        row = row + 1
        self.widget['RightEyeBrowParserSlider'] = GE.Slider2(self.layer['parameters_frame'], 'RightEyeBrowParserSlider', self.DEFAULT_DATA['RightEyeBrowParser'], 3, self.update_data, 'parameter', 300, 20, row, 0, padx, pady, 0.62, 40)
        row = row + 1
        self.widget['LeftEyeParserSlider'] = GE.Slider2(self.layer['parameters_frame'], 'LeftEyeParserSlider', self.DEFAULT_DATA['LeftEyeParser'], 3, self.update_data, 'parameter', 300, 20, row, 0, padx, pady, 0.62, 40)
        row = row + 1
        self.widget['RightEyeParserSlider'] = GE.Slider2(self.layer['parameters_frame'], 'RightEyeParserSlider', self.DEFAULT_DATA['RightEyeParser'], 3, self.update_data, 'parameter', 300, 20, row, 0, padx, pady, 0.62, 40)
        row = row + 1
        self.widget['NoseParserSlider'] = GE.Slider2(self.layer['parameters_frame'], 'NoseParserSlider', self.DEFAULT_DATA['NoseParser'], 3, self.update_data, 'parameter', 300, 20, row, 0, padx, pady, 0.62, 40)
        row = row + 1
        self.widget['MouthParserSlider'] = GE.Slider2(self.layer['parameters_frame'], 'MouthParserSlider', self.DEFAULT_DATA['MouthParser'], 3, self.update_data, 'parameter', 300, 20, row, 0, padx, pady, 0.62, 40)
        row = row + 1
        self.widget['UpperLipParserSlider'] = GE.Slider2(self.layer['parameters_frame'], 'UpperLipParserSlider', self.DEFAULT_DATA['UpperLipParser'], 3, self.update_data, 'parameter', 300, 20, row, 0, padx, pady, 0.62, 40)
        row = row + 1
        self.widget['LowerLipParserSlider'] = GE.Slider2(self.layer['parameters_frame'], 'LowerLipParserSlider', self.DEFAULT_DATA['LowerLipParser'], 3, self.update_data, 'parameter', 300, 20, row, 0, padx, pady, 0.62, 40)
        row = row + 1
        self.widget['OrientSwitch'] = GE.Switch2(self.layer['parameters_frame'], 'OrientSwitch', self.DEFAULT_DATA['Orientation'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady)
        row = row + 1
        self.widget['OrientSlider'] = GE.Slider2(self.layer['parameters_frame'], 'OrientSlider', self.DEFAULT_DATA['OrientationAngle'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['StrengthSwitch'] = GE.Switch2(self.layer['parameters_frame'], 'StrengthSwitch', self.DEFAULT_DATA['Strength'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady)
        row = row + 1
        self.widget['StrengthSlider'] = GE.Slider2(self.layer['parameters_frame'], 'StrengthSlider', self.DEFAULT_DATA['StrengthValue'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['BorderTopSlider'] = GE.Slider2(self.layer['parameters_frame'], 'BorderTopSlider', self.DEFAULT_DATA['BorderTop'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['BorderLeftSlider'] = GE.Slider2(self.layer['parameters_frame'], 'BorderLeftSlider', self.DEFAULT_DATA['BorderLeft'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['BorderRightSlider'] = GE.Slider2(self.layer['parameters_frame'], 'BorderRightSlider', self.DEFAULT_DATA['BorderRight'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['BorderBottomSlider'] = GE.Slider2(self.layer['parameters_frame'], 'BorderBottomSlider', self.DEFAULT_DATA['BorderBottom'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['BorderBlurSlider'] = GE.Slider2(self.layer['parameters_frame'], 'BorderBlurSlider', self.DEFAULT_DATA['BorderBlur'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['DiffSwitch'] = GE.Switch2(self.layer['parameters_frame'], 'DiffSwitch', self.DEFAULT_DATA['Diff'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady)
        row = row + 1
        self.widget['DiffSlider'] = GE.Slider2(self.layer['parameters_frame'], 'DiffSlider', self.DEFAULT_DATA['DiffValue'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['DiffingBlurSlider'] = GE.Slider2(self.layer['parameters_frame'], 'DiffingBlurSlider', self.DEFAULT_DATA['DiffingBlur'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['DFLXSegSwitch'] = GE.Switch2(self.layer['parameters_frame'], 'DFLXSegSwitch', self.DEFAULT_DATA['DFLXSeg'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady)
        row = row + 1
        self.widget['DFLXSegSlider'] = GE.Slider2(self.layer['parameters_frame'], 'DFLXSegSlider', self.DEFAULT_DATA['DFLXSegSliderSize'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['OccluderBlurSlider'] = GE.Slider2(self.layer['parameters_frame'], 'OccluderBlurSlider', self.DEFAULT_DATA['OccluderBlur'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['FinalBlurSwitch'] = GE.Switch2(self.layer['parameters_frame'], 'FinalBlurSwitch', self.DEFAULT_DATA['FinalBlurSwitch'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady)
        row = row + 1
        self.widget['FinalBlurSlider'] = GE.Slider2(self.layer['parameters_frame'], 'FinalBlurSlider', self.DEFAULT_DATA['FinalBlurValue'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['BlendSlider'] = GE.Slider2(self.layer['parameters_frame'], 'BlendSlider', self.DEFAULT_DATA['OverallMaskBlend'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['DFLRCTColorSwitch'] = GE.Switch2(self.layer['parameters_frame'], 'DFLRCTColorSwitch', self.DEFAULT_DATA['DFLRCTColorTransfer'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady + 5)
        row = row + 1
        self.widget['DFLLoadOnlyOneSwitch'] = GE.Switch2(self.layer['parameters_frame'], 'DFLLoadOnlyOneSwitch', self.DEFAULT_DATA['DFLLoadOnlyOne'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady + 5)
        row = row + 1
        self.widget['DFLAmpMorphSlider'] = GE.Slider2(self.layer['parameters_frame'], 'DFLAmpMorphSlider', self.DEFAULT_DATA['DFLAmpMorphFactor'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady + 5, 0.62)
        row = row + 1
        self.widget['CLIPSwitch'] = GE.Switch2(self.layer['parameters_frame'], 'CLIPSwitch', self.DEFAULT_DATA['CLIPMask'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady)
        row = row + 1
        self.widget['CLIPTextEntry'] = GE.Text_Entry(self.layer['parameters_frame'], 'CLIPTextEntry', self.DEFAULT_DATA['CLIPText'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['CLIPSlider'] = GE.Slider2(self.layer['parameters_frame'], 'CLIPSlider', self.DEFAULT_DATA['CLIPValue'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['RestoreEyesSwitch'] = GE.Switch2(self.layer['parameters_frame'], 'RestoreEyesSwitch', self.DEFAULT_DATA['RestoreEyes'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady)
        row = row + 1
        self.widget['RestoreEyesSlider'] = GE.Slider2(self.layer['parameters_frame'], 'RestoreEyesSlider', self.DEFAULT_DATA['RestoreEyesBlend'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['Eyes_Mouth_BlurSlider'] = GE.Slider2(self.layer['parameters_frame'], 'Eyes_Mouth_BlurSlider', self.DEFAULT_DATA['EyesMouthBlur'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['RestoreEyesFeatherSlider'] = GE.Slider2(self.layer['parameters_frame'], 'RestoreEyesFeatherSlider', self.DEFAULT_DATA['RestoreEyesFeather'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['RestoreEyesSizeSlider'] = GE.Slider2(self.layer['parameters_frame'], 'RestoreEyesSizeSlider', self.DEFAULT_DATA['RestoreEyesSize'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['RestoreEyesRadiusFactorXSlider'] = GE.Slider2(self.layer['parameters_frame'], 'RestoreEyesRadiusFactorXSlider', self.DEFAULT_DATA['RestoreEyesRadiusFactorX'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['RestoreEyesRadiusFactorYSlider'] = GE.Slider2(self.layer['parameters_frame'], 'RestoreEyesRadiusFactorYSlider', self.DEFAULT_DATA['RestoreEyesRadiusFactorY'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['RestoreEyesSpacingOffsetSlider'] = GE.Slider2(self.layer['parameters_frame'], 'RestoreEyesSpacingOffsetSlider', self.DEFAULT_DATA['RestoreEyesSpacingOffset'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['RestoreEyesXoffsetSlider'] = GE.Slider2(self.layer['parameters_frame'], 'RestoreEyesXoffsetSlider', self.DEFAULT_DATA['RestoreEyesXoffset'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['RestoreEyesYoffsetSlider'] = GE.Slider2(self.layer['parameters_frame'], 'RestoreEyesYoffsetSlider', self.DEFAULT_DATA['RestoreEyesYoffset'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['RestoreMouthSwitch'] = GE.Switch2(self.layer['parameters_frame'], 'RestoreMouthSwitch', self.DEFAULT_DATA['RestoreMouth'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady)
        row = row + 1
        self.widget['RestoreMouthSlider'] = GE.Slider2(self.layer['parameters_frame'], 'RestoreMouthSlider', self.DEFAULT_DATA['RestoreMouthBlend'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['RestoreMouthFeatherSlider'] = GE.Slider2(self.layer['parameters_frame'], 'RestoreMouthFeatherSlider', self.DEFAULT_DATA['RestoreMouthFeather'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['RestoreMouthSizeSlider'] = GE.Slider2(self.layer['parameters_frame'], 'RestoreMouthSizeSlider', self.DEFAULT_DATA['RestoreMouthSize'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['RestoreMouthRadiusFactorXSlider'] = GE.Slider2(self.layer['parameters_frame'], 'RestoreMouthRadiusFactorXSlider', self.DEFAULT_DATA['RestoreMouthRadiusFactorX'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['RestoreMouthRadiusFactorYSlider'] = GE.Slider2(self.layer['parameters_frame'], 'RestoreMouthRadiusFactorYSlider', self.DEFAULT_DATA['RestoreMouthRadiusFactorY'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['RestoreMouthXoffsetSlider'] = GE.Slider2(self.layer['parameters_frame'], 'RestoreMouthXoffsetSlider', self.DEFAULT_DATA['RestoreMouthXoffset'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['RestoreMouthYoffsetSlider'] = GE.Slider2(self.layer['parameters_frame'], 'RestoreMouthYoffsetSlider', self.DEFAULT_DATA['RestoreMouthYoffset'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['AutoColorSwitch'] = GE.Switch2(self.layer['parameters_frame'], 'AutoColorSwitch', self.DEFAULT_DATA['AutoColor'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady)
        row = row + 1
        self.widget['AutoColorTypeTextSel'] = GE.TextSelection(self.layer['parameters_frame'], 'AutoColorTypeTextSel', self.DEFAULT_DATA['SimilarityType'], 3, self.update_data, 'parameter', 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['AutoColorSlider'] = GE.Slider2(self.layer['parameters_frame'], 'AutoColorSlider', self.DEFAULT_DATA['AutoColorBlend'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['JpegCompressionSwitch'] = GE.Switch2(self.layer['parameters_frame'], 'JpegCompressionSwitch', self.DEFAULT_DATA['JpegCompression'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady)
        row = row + 1
        self.widget['JpegCompressionSlider'] = GE.Slider2(self.layer['parameters_frame'], 'JpegCompressionSlider', self.DEFAULT_DATA['JpegCompressionValue'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['ColorSwitch'] = GE.Switch2(self.layer['parameters_frame'], 'ColorSwitch', self.DEFAULT_DATA['ColorAdjustment'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady)
        row = row + 1
        self.widget['ColorRedSlider'] = GE.Slider2(self.layer['parameters_frame'], 'ColorRedSlider', self.DEFAULT_DATA['ColorRed'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['ColorGreenSlider'] = GE.Slider2(self.layer['parameters_frame'], 'ColorGreenSlider', self.DEFAULT_DATA['ColorGreen'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['ColorBlueSlider'] = GE.Slider2(self.layer['parameters_frame'], 'ColorBlueSlider', self.DEFAULT_DATA['ColorBlue'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['ColorBrightSlider'] = GE.Slider2(self.layer['parameters_frame'], 'ColorBrightSlider', self.DEFAULT_DATA['ColorBright'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['ColorContrastSlider'] = GE.Slider2(self.layer['parameters_frame'], 'ColorContrastSlider', self.DEFAULT_DATA['ColorContrast'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['ColorSaturationSlider'] = GE.Slider2(self.layer['parameters_frame'], 'ColorSaturationSlider', self.DEFAULT_DATA['ColorSaturation'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['ColorSharpnessSlider'] = GE.Slider2(self.layer['parameters_frame'], 'ColorSharpnessSlider', self.DEFAULT_DATA['ColorSharpness'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['ColorHueSlider'] = GE.Slider2(self.layer['parameters_frame'], 'ColorHueSlider', self.DEFAULT_DATA['ColorHue'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['ColorGammaSlider'] = GE.Slider2(self.layer['parameters_frame'], 'ColorGammaSlider', self.DEFAULT_DATA['ColorGamma'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['NoiseSlider'] = GE.Slider2(self.layer['parameters_frame'], 'NoiseSlider', self.DEFAULT_DATA['Noise'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['FaceAdjSwitch'] = GE.Switch2(self.layer['parameters_frame'], 'FaceAdjSwitch', self.DEFAULT_DATA['FaceAdj'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady)
        row = row + 1
        self.widget['KPSXSlider'] = GE.Slider2(self.layer['parameters_frame'], 'KPSXSlider', self.DEFAULT_DATA['KPSX'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['KPSYSlider'] = GE.Slider2(self.layer['parameters_frame'], 'KPSYSlider', self.DEFAULT_DATA['KPSY'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['KPSScaleSlider'] = GE.Slider2(self.layer['parameters_frame'], 'KPSScaleSlider', self.DEFAULT_DATA['KPSScale'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['FaceScaleSlider'] = GE.Slider2(self.layer['parameters_frame'], 'FaceScaleSlider', self.DEFAULT_DATA['FaceScale'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['FaceLikenessSwitch'] = GE.Switch2(self.layer['parameters_frame'], 'FaceLikenessSwitch', self.DEFAULT_DATA['FaceLikeness'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady)
        row = row + 1
        self.widget['FaceLikenessFactorSlider'] = GE.Slider2(self.layer['parameters_frame'], 'FaceLikenessFactorSlider', self.DEFAULT_DATA['FaceLikenessFactor'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['ThresholdSlider'] = GE.Slider2(self.layer['parameters_frame'], 'ThresholdSlider', self.DEFAULT_DATA['Threshold'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['DetectTypeTextSel'] = GE.TextSelectionComboBox(self.layer['parameters_frame'], 'DetectTypeTextSel', self.DEFAULT_DATA['DetectType'], 3, self.update_data, 'parameter', 'parameter', 398, 20, row, 0, padx, pady, 0.62, 150)
        row = row + 1
        self.widget['DetectScoreSlider'] = GE.Slider2(self.layer['parameters_frame'], 'DetectScoreSlider', self.DEFAULT_DATA['DetectScore'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['SimilarityTypeTextSel'] = GE.TextSelection(self.layer['parameters_frame'], 'SimilarityTypeTextSel', self.DEFAULT_DATA['SimilarityType'], 3, self.update_data, 'parameter', 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['AutoRotationSwitch'] = GE.Switch2(self.layer['parameters_frame'], 'AutoRotationSwitch', self.DEFAULT_DATA['AutoRotation'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady)
        row = row + 1
        self.widget['LandmarksDetectionAdjSwitch'] = GE.Switch2(self.layer['parameters_frame'], 'LandmarksDetectionAdjSwitch', self.DEFAULT_DATA['LandmarksDetectionAdj'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady)
        row = row + 1
        self.widget['LandmarksAlignModeFromPointsSwitch'] = GE.Switch2(self.layer['parameters_frame'], 'LandmarksAlignModeFromPointsSwitch', self.DEFAULT_DATA['LandmarksAlignModeFromPoints'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 30, 40)
        row = row + 1
        self.widget['LandmarksDetectTypeTextSel'] = GE.TextSelectionComboBox(self.layer['parameters_frame'], 'LandmarksDetectTypeTextSel', self.DEFAULT_DATA['LandmarksDetectType'], 3, self.update_data, 'parameter', 'parameter', 398, 20, row, 0, padx, pady, 0.62, 150)
        row = row + 1
        self.widget['LandmarksDetectScoreSlider'] = GE.Slider2(self.layer['parameters_frame'], 'LandmarksDetectScoreSlider', self.DEFAULT_DATA['LandmarksDetectScore'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['ShowLandmarksSwitch'] = GE.Switch2(self.layer['parameters_frame'], 'ShowLandmarksSwitch', self.DEFAULT_DATA['ShowLandmarks'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady)
        row = row + 1
        self.widget['LandmarksPositionAdjSwitch'] = GE.Switch2(self.layer['parameters_frame'], 'LandmarksPositionAdjSwitch', self.DEFAULT_DATA['LandmarksPositionAdj'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady)
        row = row + 1
        self.widget['FaceIDSlider'] = GE.Slider2(self.layer['parameters_frame'], 'FaceIDSlider', 'Face ID: ', 3, self.update_face_landmarks_data, 'parameter', 300, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['EyeLeftXSlider'] = GE.Slider2(self.layer['parameters_frame'], 'EyeLeftXSlider', self.DEFAULT_DATA['EyeLeftX'], 3, self.update_face_landmarks_data, 'parameter', 300, 20, row, 0, padx, pady, 0.62, 40)
        row = row + 1
        self.widget['EyeLeftYSlider'] = GE.Slider2(self.layer['parameters_frame'], 'EyeLeftYSlider', self.DEFAULT_DATA['EyeLeftY'], 3, self.update_face_landmarks_data, 'parameter', 300, 20, row, 0, padx, pady, 0.62, 40)
        row = row + 1
        self.widget['EyeRightXSlider'] = GE.Slider2(self.layer['parameters_frame'], 'EyeRightXSlider', self.DEFAULT_DATA['EyeRightX'], 3, self.update_face_landmarks_data, 'parameter', 300, 20, row, 0, padx, pady, 0.62, 40)
        row = row + 1
        self.widget['EyeRightYSlider'] = GE.Slider2(self.layer['parameters_frame'], 'EyeRightYSlider', self.DEFAULT_DATA['EyeRightY'], 3, self.update_face_landmarks_data, 'parameter', 300, 20, row, 0, padx, pady, 0.62, 40)
        row = row + 1
        self.widget['NoseXSlider'] = GE.Slider2(self.layer['parameters_frame'], 'NoseXSlider', self.DEFAULT_DATA['NoseX'], 3, self.update_face_landmarks_data, 'parameter', 300, 20, row, 0, padx, pady, 0.62, 40)
        row = row + 1
        self.widget['NoseYSlider'] = GE.Slider2(self.layer['parameters_frame'], 'NoseYSlider', self.DEFAULT_DATA['NoseY'], 3, self.update_face_landmarks_data, 'parameter', 300, 20, row, 0, padx, pady, 0.62, 40)
        row = row + 1
        self.widget['MouthLeftXSlider'] = GE.Slider2(self.layer['parameters_frame'], 'MouthLeftXSlider', self.DEFAULT_DATA['MouthLeftX'], 3, self.update_face_landmarks_data, 'parameter', 300, 20, row, 0, padx, pady, 0.62, 40)
        row = row + 1
        self.widget['MouthLeftYSlider'] = GE.Slider2(self.layer['parameters_frame'], 'MouthLeftYSlider', self.DEFAULT_DATA['MouthLeftY'], 3, self.update_face_landmarks_data, 'parameter', 300, 20, row, 0, padx, pady, 0.62, 40)
        row = row + 1
        self.widget['MouthRightXSlider'] = GE.Slider2(self.layer['parameters_frame'], 'MouthRightXSlider', self.DEFAULT_DATA['MouthRightX'], 3, self.update_face_landmarks_data, 'parameter', 300, 20, row, 0, padx, pady, 0.62, 40)
        row = row + 1
        self.widget['MouthRightYSlider'] = GE.Slider2(self.layer['parameters_frame'], 'MouthRightYSlider', self.DEFAULT_DATA['MouthRightY'], 3, self.update_face_landmarks_data, 'parameter', 300, 20, row, 0, padx, pady, 0.62, 40)
        row = row + 1
        self.widget['RecordTypeTextSel'] = GE.TextSelection(self.layer['parameters_frame'], 'RecordTypeTextSel', self.DEFAULT_DATA['RecordType'], 3, self.update_data, 'parameter', 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['VideoQualSlider'] = GE.Slider2(self.layer['parameters_frame'], 'VideoQualSlider', self.DEFAULT_DATA['VideoQual'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['AudioSpeedSlider'] = GE.Slider2(self.layer['parameters_frame'], 'AudioSpeedSlider', self.DEFAULT_DATA['AudioSpeed'], 3, self.update_data, 'parameter', 398, 20, row, 0, padx, pady, 0.62)
        row = row + 1
        self.widget['MergeTextSel'] = GE.TextSelection(self.layer['parameters_frame'], 'MergeTextSel', self.DEFAULT_DATA['Merge'], 3, self.select_input_faces, 'merge', '', 398, 20, row, 0, padx, pady, 0.62)
        row = 1
        column = 0
        padx = 1
        pady = 0
        row = row + 1
        self.widget['FaceEditorTypeTextSel'] = GE.TextSelection(self.layer['parameters_face_editor_frame'], 'FaceEditorTypeTextSel', self.DEFAULT_DATA['FaceEditorType'], 3, self.update_face_editor_data, 'parameter_face_editor', 'parameter_face_editor', 398, 20, row, 0, padx, pady, 0.6)
        row = row + 1
        self.widget['FaceEditorIDSlider'] = GE.Slider2(self.layer['parameters_face_editor_frame'], 'FaceEditorIDSlider', self.DEFAULT_DATA['FaceEditorID'], 3, self.update_face_editor_data, 'parameter_face_editor', 398, 20, row, 0, padx, pady, 0.6)
        row = row + 1
        self.widget['CropScaleSlider'] = GE.Slider2(self.layer['parameters_face_editor_frame'], 'CropScaleSlider', self.DEFAULT_DATA['CropScale'], 3, self.update_face_editor_data, 'parameter_face_editor', 398, 20, row, 0, padx, pady, 0.6, 40)
        row = row + 1
        self.widget['EyesOpenRatioSlider'] = GE.Slider2(self.layer['parameters_face_editor_frame'], 'EyesOpenRatioSlider', self.DEFAULT_DATA['EyesOpenRatio'], 3, self.update_face_editor_data, 'parameter_face_editor', 398, 20, row, 0, padx, pady, 0.6, 40)
        row = row + 1
        self.widget['LipsOpenRatioSlider'] = GE.Slider2(self.layer['parameters_face_editor_frame'], 'LipsOpenRatioSlider', self.DEFAULT_DATA['LipsOpenRatio'], 3, self.update_face_editor_data, 'parameter_face_editor', 398, 20, row, 0, padx, pady, 0.6, 40)
        row = row + 1
        self.widget['HeadPitchSlider'] = GE.Slider2(self.layer['parameters_face_editor_frame'], 'HeadPitchSlider', self.DEFAULT_DATA['HeadPitch'], 3, self.update_face_editor_data, 'parameter_face_editor', 398, 20, row, 0, padx, pady, 0.6, 40)
        row = row + 1
        self.widget['HeadYawSlider'] = GE.Slider2(self.layer['parameters_face_editor_frame'], 'HeadYawSlider', self.DEFAULT_DATA['HeadYaw'], 3, self.update_face_editor_data, 'parameter_face_editor', 398, 20, row, 0, padx, pady, 0.6, 40)
        row = row + 1
        self.widget['HeadRollSlider'] = GE.Slider2(self.layer['parameters_face_editor_frame'], 'HeadRollSlider', self.DEFAULT_DATA['HeadRoll'], 3, self.update_face_editor_data, 'parameter_face_editor', 398, 20, row, 0, padx, pady, 0.6, 40)
        row = row + 1
        self.widget['XAxisMovementSlider'] = GE.Slider2(self.layer['parameters_face_editor_frame'], 'XAxisMovementSlider', self.DEFAULT_DATA['XAxisMovement'], 3, self.update_face_editor_data, 'parameter_face_editor', 398, 20, row, 0, padx, pady, 0.6, 40)
        row = row + 1
        self.widget['YAxisMovementSlider'] = GE.Slider2(self.layer['parameters_face_editor_frame'], 'YAxisMovementSlider', self.DEFAULT_DATA['YAxisMovement'], 3, self.update_face_editor_data, 'parameter_face_editor', 398, 20, row, 0, padx, pady, 0.6, 40)
        row = row + 1
        self.widget['ZAxisMovementSlider'] = GE.Slider2(self.layer['parameters_face_editor_frame'], 'ZAxisMovementSlider', self.DEFAULT_DATA['ZAxisMovement'], 3, self.update_face_editor_data, 'parameter_face_editor', 398, 20, row, 0, padx, pady, 0.6, 40)
        row = row + 1
        self.widget['MouthPoutingSlider'] = GE.Slider2(self.layer['parameters_face_editor_frame'], 'MouthPoutingSlider', self.DEFAULT_DATA['MouthPouting'], 3, self.update_face_editor_data, 'parameter_face_editor', 398, 20, row, 0, padx, pady, 0.6, 40)
        row = row + 1
        self.widget['MouthPursingSlider'] = GE.Slider2(self.layer['parameters_face_editor_frame'], 'MouthPursingSlider', self.DEFAULT_DATA['MouthPursing'], 3, self.update_face_editor_data, 'parameter_face_editor', 398, 20, row, 0, padx, pady, 0.6, 40)
        row = row + 1
        self.widget['MouthGrinSlider'] = GE.Slider2(self.layer['parameters_face_editor_frame'], 'MouthGrinSlider', self.DEFAULT_DATA['MouthGrin'], 3, self.update_face_editor_data, 'parameter_face_editor', 398, 20, row, 0, padx, pady, 0.6, 40)
        row = row + 1
        self.widget['LipsCloseOpenSlider'] = GE.Slider2(self.layer['parameters_face_editor_frame'], 'LipsCloseOpenSlider', self.DEFAULT_DATA['LipsCloseOpen'], 3, self.update_face_editor_data, 'parameter_face_editor', 398, 20, row, 0, padx, pady, 0.6, 40)
        row = row + 1
        self.widget['MouthSmileSlider'] = GE.Slider2(self.layer['parameters_face_editor_frame'], 'MouthSmileSlider', self.DEFAULT_DATA['MouthSmile'], 3, self.update_face_editor_data, 'parameter_face_editor', 398, 20, row, 0, padx, pady, 0.6, 40)
        row = row + 1
        self.widget['EyeWinkSlider'] = GE.Slider2(self.layer['parameters_face_editor_frame'], 'EyeWinkSlider', self.DEFAULT_DATA['EyeWink'], 3, self.update_face_editor_data, 'parameter_face_editor', 398, 20, row, 0, padx, pady, 0.6, 40)
        row = row + 1
        self.widget['EyeBrowsDirectionSlider'] = GE.Slider2(self.layer['parameters_face_editor_frame'], 'EyeBrowsDirectionSlider', self.DEFAULT_DATA['EyeBrowsDirection'], 3, self.update_face_editor_data, 'parameter_face_editor', 398, 20, row, 0, padx, pady, 0.6, 40)
        row = row + 1
        self.widget['EyeGazeHorizontalSlider'] = GE.Slider2(self.layer['parameters_face_editor_frame'], 'EyeGazeHorizontalSlider', self.DEFAULT_DATA['EyeGazeHorizontal'], 3, self.update_face_editor_data, 'parameter_face_editor', 398, 20, row, 0, padx, pady, 0.6, 40)
        row = row + 1
        self.widget['EyeGazeVerticalSlider'] = GE.Slider2(self.layer['parameters_face_editor_frame'], 'EyeGazeVerticalSlider', self.DEFAULT_DATA['EyeGazeVertical'], 3, self.update_face_editor_data, 'parameter_face_editor', 398, 20, row, 0, padx, pady, 0.6, 40)
        params_visibility, params_face_editor_visibility = load_params_visibility_from_json()
        if params_visibility == None:
            params_visibility = {}
        for widget_name, widget_instance in self.widget.items():
            if widget_instance.parent == self.layer['parameters_frame']:
                self.default_params_visibility[widget_name] = True
                if widget_name not in params_visibility:
                    params_visibility[widget_name] = True
        if params_face_editor_visibility == None:
            params_face_editor_visibility = {}
        for widget_name, widget_instance in self.widget.items():
            if widget_instance.parent == self.layer['parameters_face_editor_frame']:
                self.default_params_face_editor_visibility[widget_name] = True
                if widget_name not in params_face_editor_visibility:
                    params_face_editor_visibility[widget_name] = True
        apply_params_visibility_configuration(params_visibility, params_face_editor_visibility, param_type='all', reload=True)
        self.layer['tooltip_frame'] = tk.Frame(self.layer['parameter_frame'], style.canvas_frame_label_3, height=80)
        self.layer['tooltip_frame'].grid(row=2, column=0, columnspan=2, sticky='NEWS', padx=0, pady=0)
        self.layer['tooltip_label'] = tk.Label(self.layer['tooltip_frame'], style.info_label, wraplength=width - 10, image=self.blank, compound='left', height=80, width=width - 10)
        self.layer['tooltip_label'].place(x=5, y=5)
        self.static_widget['22'] = GE.Separator_x(self.layer['tooltip_frame'], 0, 0)
        self.layer['facelab_canvas'] = tk.Canvas(self.layer['parameter_frame'], style.canvas_frame_label_3, bd=0, width=width)
        self.layer['facelab_canvas'].grid(row=1, column=0, sticky='NEWS', pady=0, padx=0)
        self.layer['facelab_frame'] = tk.Frame(self.layer['facelab_canvas'], style.canvas_frame_label_3, bd=0, width=width, height=11000)
        self.layer['facelab_frame'].grid(row=0, column=0, sticky='NEWS', pady=0, padx=0)
        self.layer['facelab_canvas'].create_window(0, 0, window=self.layer['facelab_frame'], anchor='nw')
        self.layer['facelab_scroll_canvas'] = tk.Canvas(self.layer['parameter_frame'], style.canvas_frame_label_3, bd=0)
        self.layer['facelab_scroll_canvas'].grid(row=1, column=1, sticky='NEWS', pady=0)
        self.layer['facelab_scroll_canvas'].grid_rowconfigure(0, weight=1)
        self.layer['facelab_scroll_canvas'].grid_columnconfigure(0, weight=1)
        self.static_widget['facelab_scrollbar'] = GE.Scrollbar_y(self.layer['facelab_scroll_canvas'], self.layer['facelab_canvas'])
        # 状态栏保留，但其中的文字/链接一律不显示
        left_text = ''
        left_url = ''
        right_url = ''
        self.status_left_label = tk.Label(bottom_frame, style.donate_1, cursor='hand2', text=left_text)
        self.status_left_label.grid(row=0, column=0, sticky='NEWS')
        # if left_url:
        #     self.status_left_label.bind('<Button-1>', lambda e: self.callback(left_url))
        self.announcement_url = ''
        self.announcements = ['']
        self.announcement_urls = ['']
        self.current_announcement_index = 0
        self.status_label_frame = tk.Frame(bottom_frame, bg=style.main)
        self.status_label_frame.grid(row=0, column=1, sticky='NEWS')
        self.status_label_frame.grid_columnconfigure(0, weight=1)
        self.status_canvas = tk.Canvas(self.status_label_frame, bg=style.main, bd=0, highlightthickness=0, height=20)
        self.status_canvas.grid(row=0, column=0, sticky='NEWS')
        self.announcement_text = self.status_canvas.create_text(0, 10, text='', fill='#7562ee', font=('微软雅黑 Semibold', 10), anchor='center')
        # 状态栏文字与点击已关闭（控件本身保留）
        # if right_url:
        #     self.status_canvas.bind('<Button-1>', lambda e: self.callback(right_url))
        #     self.status_canvas.bind('<Enter>', lambda e: self.status_canvas.config(cursor='hand2'))
        #     self.status_canvas.bind('<Leave>', lambda e: self.status_canvas.config(cursor=''))
        self.text_pos = 0
        self.canvas_width = 0
        self.text_width = 0
        self.scroll_delay = 30
        self.scroll_step = 1
        self.update_announcements()
        self.animate_scroll()
        self.license_status_label = tk.Label(bottom_frame, style.donate_1, text='', anchor='e')
        self.license_status_label.grid(row=0, column=2, sticky='NEWS')
        self.update_license_status()
        from .FaceLandmarks import FaceLandmarks
        self.face_landmarks = FaceLandmarks(self.widget, self.parameters, self.add_action)
        self.add_action('face_landmarks', self.face_landmarks)
        from .FaceEditor import FaceEditor
        self.face_editor = FaceEditor(self.widget, self.parameters_face_editor, self.add_action)
        self.add_action('face_editor', self.face_editor)
    def create_language_menu(self):
        """创建语言菜单"""
        language_menu = tk.Menu(self.menubar, tearoff=0)
        self.widget['LanguageMenu'] = language_menu
        def change_lang(lang):
            return lambda: self.change_language(lang)
        language_menu.add_command(label='中文', command=change_lang('Chinese'), image=self.widget['CheckMark'] if self.current_language == 'Chinese' else '', compound='left')
        language_menu.add_command(label='English', command=change_lang('English'), image=self.widget['CheckMark'] if self.current_language == 'English' else '', compound='left')
        return language_menu
    def update_face_landmarks_data(self, mode, name, use_markers=False):
        # irreducible cflow, using cdg fallback
        if mode == 'parameter':
            frame_number = self.video_slider.get()
            face_id = self.widget['FaceIDSlider'].get()
            parameter_value = self.widget[name].get()
            landmarks = self.face_landmarks.get_landmarks(frame_number, face_id)
            if landmarks is None:
                landmarks = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
                self.face_landmarks.add_landmarks(frame_number, face_id, landmarks)
            match name:
                case 'EyeLeftXSlider':
                    landmarks[0] = tuple((parameter_value, landmarks[0][1]))
                case 'EyeLeftYSlider':
                    landmarks[0] = tuple((landmarks[0][0], parameter_value))
                case 'EyeRightXSlider':
                    landmarks[1] = tuple((parameter_value, landmarks[1][1]))
                case 'EyeRightYSlider':
                    landmarks[1] = tuple((landmarks[1][0], parameter_value))
                case 'NoseXSlider':
                    landmarks[2] = tuple((parameter_value, landmarks[2][1]))
                case 'NoseYSlider':
                    landmarks[2] = tuple((landmarks[2][0], parameter_value))
                case 'MouthLeftXSlider':
                    landmarks[3] = tuple((parameter_value, landmarks[3][1]))
                case 'MouthLeftYSlider':
                    landmarks[3] = tuple((landmarks[3][0], parameter_value))
                case 'MouthRightXSlider':
                    landmarks[4] = tuple((parameter_value, landmarks[4][1]))
                case 'MouthRightYSlider':
                    landmarks[4] = tuple((landmarks[4][0], parameter_value))
            self.add_action('face_landmarks', self.face_landmarks)
            self.face_landmarks.apply_changes_to_widget_and_parameters(frame_number, face_id)
            if use_markers:
                pass
        self.add_action('get_requested_video_frame', frame_number)
        self.add_action('get_requested_video_frame_without_markers', frame_number)
        from .FaceEditor import FaceEditor
        self.face_editor = FaceEditor(self.widget, self.parameters_face_editor, self.add_action)
        self.add_action('face_editor', self.face_editor)
    def update_face_editor_data(self, mode, name, use_markers=False):
        if mode == 'parameter_face_editor':
            frame_number = self.video_slider.get()
            face_id = self.widget['FaceEditorIDSlider'].get()
            parameter_value = self.widget[name].get()
            parameters = self.face_editor.get_parameters(frame_number, face_id)
            if parameters is None:
                parameters = ['Human-Face', 2.5, 0.0, 0, 0, 0, 0.0, 0.0, 1.0, 0.0, 0, 0.0, 0.0]
                self.face_editor.add_parameters(frame_number, face_id, parameters)
            if name in self.face_editor.parameter_map:
                index = self.face_editor.parameter_map[name]
                parameters[index] = parameter_value
            self.add_action('face_editor', self.face_editor)
            self.face_editor.apply_changes_to_widget_and_parameters(frame_number, face_id)
            if use_markers:
                self.add_action('get_requested_video_frame', frame_number)
            else:
                self.add_action('get_requested_video_frame_without_markers', frame_number)
    def update_data(self, mode, name, use_markers=False):
        if mode == 'parameter':
            self.parameters[name] = self.widget[name].get()
            self.add_action('parameters', self.parameters)
            if name == 'SimilarityTypeTextSel' or name == 'FaceSwapperModelTextSel':
                if self.video_loaded or self.image_loaded:
                    for face in self.target_faces:
                        if face['ButtonState']:
                            face['SourceFaceAssignments'] = []
                    self.clear_faces()
                    self.load_input_faces()
            else:
                if name == 'ProvidersPriorityTextSel':
                    provider_value = self.models.switch_providers_priority(self.parameters[name])
                    if provider_value != self.parameters[name]:
                        self.parameters[name] = provider_value
                        self.widget[name].set(provider_value, request_frame=False)
                    else:
                        self.models.delete_models()
                        torch.cuda.empty_cache()
                else:
                    if name == 'WebCamMaxResolSel' or name == 'WebCamMaxFPSSel':
                        self.add_action('change_webcam_resolution_and_fps')
                    else:
                        if name == 'WebCamBackendSel':
                            self.add_action('change_webcam_resolution_and_fps')
                            self.populate_target_videos()
                        else:
                            if name == 'ThreadsSlider':
                                self.models.set_number_of_threads(self.parameters[name])
        else:
            if mode == 'control':
                self.control[name] = self.widget[name].get()
                self.add_action('control', self.control)
        if use_markers:
            self.add_action('get_requested_video_frame', self.video_slider.get())
        else:
            self.add_action('get_requested_video_frame_without_markers', self.video_slider.get())
    def update_webcam_orientation(self, mode, name, use_markers=False):
        """处理摄像头方向变化,自动交换分辨率的宽高"""
        if mode == 'parameter':
            orientation = self.widget[name].get()
            self.parameters[name] = orientation
            current_resolution = self.parameters.get('WebCamMaxResolSel', '1920x1080')
            try:
                original_resolutions = self.DEFAULT_DATA['WebCamMaxResolSelModes']
                if orientation == '竖屏':
                    new_resolutions = []
                    for res in original_resolutions:
                        w, h = res.split('x')
                        if int(w) > int(h):
                            new_resolutions.append(f'{h}x{w}')
                        else:
                            new_resolutions.append(res)
                    self.rotate_preview(90)
                else:
                    new_resolutions = original_resolutions.copy()
                    self.rotate_preview(0)
                self.widget['WebCamMaxResolSel'].combo_box.configure(values=new_resolutions)
                width, height = current_resolution.split('x')
                if orientation == '竖屏':
                    if int(width) > int(height):
                        new_resolution = f'{height}x{width}'
                    else:
                        new_resolution = current_resolution
                else:
                    if int(height) > int(width):
                        new_resolution = f'{height}x{width}'
                    else:
                        new_resolution = current_resolution
                self.widget['WebCamMaxResolSel'].combo_box.set(new_resolution)
                self.parameters['WebCamMaxResolSel'] = new_resolution
                if new_resolution != current_resolution:
                    self.add_action('change_webcam_resolution_and_fps')
            except Exception as e:
                import logging
                logging.error(f'更新摄像头方向时出错: {e}')
            self.add_action('parameters', self.parameters)
            if use_markers:
                self.add_action('get_requested_video_frame', self.video_slider.get())
            else:
                self.add_action('get_requested_video_frame_without_markers', self.video_slider.get())
    def update_param_visibility(self, name, visible):
        if name in self.widget:
            if visible:
                self.widget[name].unhide()
            else:
                self.widget[name].hide()
        self.static_widget['parameters_scrollbar'].resize_scrollbar(None)
    def callback(self, url):
        webbrowser.open_new_tab(url)
    def target_faces_mouse_wheel(self, event):
        self.found_faces_canvas.xview_scroll(1 * int(event.delta / 120.0), 'units')
    def source_faces_mouse_wheel(self, event):
        self.source_faces_canvas.yview_scroll(-int(event.delta / 120.0), 'units')
        center = (self.source_faces_canvas.yview()[1] - self.source_faces_canvas.yview()[0]) / 2
        center = center + self.source_faces_canvas.yview()[0]
        self.static_widget['input_faces_scrollbar'].set(center)
    def target_videos_mouse_wheel(self, event):
        self.target_media_canvas.yview_scroll(-int(event.delta / 120.0), 'units')
        center = (self.target_media_canvas.yview()[1] - self.target_media_canvas.yview()[0]) / 2
        center = center + self.target_media_canvas.yview()[0]
        self.static_widget['input_videos_scrollbar'].set(center)
    def create_language_menu(self):
        """创建语言选择菜单"""
        self.language_menu = tk.Menu(self.root, tearoff=0)
        for lang in self.supported_languages:
            self.language_menu.add_command(label=lang, command=lambda l=lang: self.set_language(l))
    def update_announcements(self):
        """状态栏公告已关闭：保留方法以兼容调用方"""
        return
    def _update_announcement_ui(self, announcements, announcement_urls):
        """在主线程中更新公告UI"""
        self.announcements = announcements
        self.announcement_urls = announcement_urls
        if self.announcements:
            self.current_announcement_index = 0
            current_text = self.announcements[self.current_announcement_index]
            if not hasattr(self, 'status_canvas') or hasattr(self, 'announcement_text'):
                    self.status_canvas.itemconfig(self.announcement_text, text=current_text)
                    current_url = self.announcement_urls[self.current_announcement_index] if self.announcement_urls else ''
                    if current_url:
                        self.status_canvas.bind('<Button-1>', lambda e: self.callback(current_url))
                        self.status_canvas.bind('<Enter>', lambda e: self.status_canvas.config(cursor='hand2'))
                        self.status_canvas.bind('<Leave>', lambda e: self.status_canvas.config(cursor=''))
                    else:
                        try:
                            self.status_canvas.unbind('<Button-1>')
                            self.status_canvas.unbind('<Enter>')
                            self.status_canvas.unbind('<Leave>')
                            self.status_canvas.config(cursor='')
                        except:
                            return
    def animate_scroll(self):
        """状态栏文字已关闭：不再启动滚动画布。"""
        return
    def forward_one_frame(self):
        frame = self.video_slider.get()
        video_length = self.video_slider.get_length()
        frame += 1
        if frame > video_length:
            frame = video_length
        self.video_slider.set(frame)
        self.add_action('get_requested_video_frame', frame)
    def back_one_frame(self):
        """向左移动 1 帧。"""
        if self.widget['PreviewModeTextSel'].get() != 'Video' or not self.video_loaded:
            return
        frame = self.video_slider.get() - 1
        if frame < 0:
            frame = 0
        self.video_slider.set(frame)
        self.add_action('get_requested_video_frame', frame)

    def preview_control(self, event):
        """时间轴快捷控制：q 回到起点 / a 左移 30 帧 / d 右移 30 帧 / w 右移 1 帧 / s 左移 1 帧 / 空格 播放暂停。"""
        if self.widget['PreviewModeTextSel'].get() != 'Video' or not self.video_loaded:
            return
        frame = self.video_slider.get()
        video_length = self.video_slider.get_length()
        if event == ' ':
            self.toggle_play_video()
            return
        if event == 'q':
            frame = 0
        elif event == 'a':
            frame -= 30
        elif event == 'd':
            frame += 30
        elif event == 'w':
            frame += 1
        elif event == 's':
            frame -= 1
        frame = max(0, min(frame, video_length))
        self.video_slider.set(frame)
        self.add_action('get_requested_video_frame', frame)

    def parameters_mouse_wheel(self, event):
        self.canvas.yview_scroll(1 * int(event.delta / 120.0), 'units')
    def initialize_gui(self):
        json_object = {}
        try:
            data_json_file = open(language_manager.data_json_path(), 'r', encoding='utf-8')
        except:
            self.json_dict['language'] = self.current_language
            with open(language_manager.data_json_path(), 'w', encoding='utf-8') as outfile:
                json.dump(self.json_dict, outfile, ensure_ascii=False)
        else:
            json_object = json.load(data_json_file)
            data_json_file.close()
        self.geometry('%dx%d+%d+%d' % (self.json_dict['dock_win_geom'][0], self.json_dict['dock_win_geom'][1], self.json_dict['dock_win_geom'][2], self.json_dict['dock_win_geom'][3]))
        self.window_last_change = self.winfo_geometry()
        self.resizable(width=True, height=True)
        self.create_gui()
        splash_path = get_media_path('splash_next.png')
        splash_img = cv2_imread_unicode(splash_path)
        if splash_img is not None:
            self.video_image = cv2.cvtColor(splash_img, cv2.COLOR_BGR2RGB)
        else:
            self.video_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.resize_image()
        for key, value in self.widget.items():
            self.widget[key].add_info_frame(self.layer['tooltip_label'])
            if self.widget[key].get_data_type() == 'parameter':
                self.parameters[key] = self.widget[key].get()
            else:
                if self.widget[key].get_data_type() == 'control':
                    self.control[key] = self.widget[key].get()
        for key, value in self.widget.items():
            if self.widget[key].get_data_type() == 'parameter_face_editor':
                self.parameters_face_editor[key] = self.widget[key].get()
        try:
            self.json_dict['source videos'] = json_object['source videos']
        except KeyError:
            self.widget['VideoFolderButton'].error_button()
        else:
            if self.json_dict['source videos'] == None:
                self.widget['VideoFolderButton'].error_button()
            else:
                path = self.create_path_string(self.json_dict['source videos'], 28)
                self.input_videos_text.configure(text=path)
        try:
            self.json_dict['source faces'] = json_object['source faces']
        except KeyError:
            self.widget['FacesFolderButton'].error_button()
        else:
            if self.json_dict['source faces'] == None:
                self.widget['FacesFolderButton'].error_button()
            else:
                path = self.create_path_string(self.json_dict['source faces'], 28)
                self.input_faces_text.configure(text=path)
        try:
            self.json_dict['saved videos'] = json_object['saved videos']
        except KeyError:
            self.widget['OutputFolderButton'].error_button()
        else:
            if self.json_dict['saved videos'] == None:
                self.widget['OutputFolderButton'].error_button()
            else:
                path = self.create_path_string(self.json_dict['saved videos'], 28)
                self.output_videos_text.configure(text=path)
                self.add_action('saved_video_path', self.json_dict['saved videos'])
        try:
            parameters_json_file = open('startup_parameters.json', 'r', encoding='utf-8')
        except:
            pass
        else:
            temp = json.load(parameters_json_file)
            parameters_json_file.close()
            if temp.get('config_type') == 'parameters':
                temp = temp.get('parameters', {})
                for key, value in self.parameters.items():
                    try:
                        if key in self.parameters:
                            self.parameters[key] = temp[key]
                            if key == 'ProvidersPriorityTextSel':
                                provider_value = self.models.switch_providers_priority(temp[key])
                                if provider_value != temp[key]:
                                    self.parameters[key] = provider_value
                            else:
                                if key == 'ThreadsSlider':
                                    self.models.set_number_of_threads(value)
                    except KeyError:
                        continue
                for key, value in self.parameters.items():
                    self.widget[key].set(value, request_frame=False)
                temp = temp.get('parameters_face_editor', {})
                for key, value in self.parameters_face_editor.items():
                    try:
                        if key in self.parameters_face_editor:
                            self.parameters_face_editor[key] = temp[key]
                    except KeyError:
                        continue
                for key, value in self.parameters_face_editor.items():
                    self.widget[key].set(value, request_frame=False)
            else:
                import logging
                logging.error('错误: startup_parameters.json 配置类型无效!')
        self.add_action('parameters', self.parameters)
        self.add_action('control', self.control)
        self.widget['StartButton'].error_button()
        self.set_view(False, '')
    def create_path_string(self, path, text_len):
        if len(path) > text_len:
            last_folder = os.path.basename(os.path.normpath(path))
            last_folder_len = len(last_folder)
            if last_folder_len > text_len:
                path = path[:3] + '...' + path[-last_folder_len + 6:]
                return path
            else:
                path = path[:text_len - last_folder_len] + '.../' + path[-last_folder_len:]
        return path
    def load_all(self):
        if not self.json_dict['source videos'] or not self.json_dict['source faces']:
            messagebox.showinfo('Set Faces folder', 'Please set faces and videos folders first!')
            return
        else:
            self.populate_target_videos()
            self.load_input_faces()
            self.widget['StartButton'].enable_button()
    def select_video_path(self):
        temp = self.json_dict['source videos']
        self.json_dict['source videos'] = filedialog.askdirectory(title='Select Target Videos Folder', initialdir=temp)
        self.json_dict['language'] = self.current_language
        path = self.create_path_string(self.json_dict['source videos'], 28)
        self.input_videos_text.configure(text=path)
        with open(language_manager.data_json_path(), 'w', encoding='utf-8') as outfile:
            json.dump(self.json_dict, outfile, ensure_ascii=False, indent=4)
            outfile.close()
        self.widget['VideoFolderButton'].set(False, request_frame=False)
        self.populate_target_videos()
    def select_save_video_path(self):
        temp = self.json_dict['saved videos']
        self.json_dict['saved videos'] = filedialog.askdirectory(title='Select Save Video Folder', initialdir=temp)
        self.json_dict['language'] = self.current_language
        path = self.create_path_string(self.json_dict['saved videos'], 28)
        self.output_videos_text.configure(text=path)
        with open(language_manager.data_json_path(), 'w', encoding='utf-8') as outfile:
            json.dump(self.json_dict, outfile, ensure_ascii=False, indent=4)
            outfile.close()
        self.widget['OutputFolderButton'].set(False, request_frame=False)
        self.add_action('saved_video_path', self.json_dict['saved videos'])
    def select_faces_path(self):
        temp = self.json_dict['source faces']
        self.json_dict['source faces'] = filedialog.askdirectory(title='Select Source Faces Folder', initialdir=temp)
        self.json_dict['language'] = self.current_language
        path = self.create_path_string(self.json_dict['source faces'], 28)
        self.input_faces_text.configure(text=path)
        with open(language_manager.data_json_path(), 'w', encoding='utf-8') as outfile:
            json.dump(self.json_dict, outfile, ensure_ascii=False, indent=4)
            outfile.close()
        self.widget['FacesFolderButton'].set(False, request_frame=False)
        self.load_input_faces()
    def load_dfl_input_models(self):
        text_font = font.Font(family='Microsoft YaHei', size=10)
        dfl_models_dir = 'dfl_models'
        j = len(self.source_faces)
        for model_file in listdir(dfl_models_dir):
            if model_file == '.gitkeep':
                continue
            else:
                new_source_face = self.source_face.copy()
                new_source_face['ButtonState'] = False
                new_source_face['Embedding'] = False
                new_source_face['DFLModel'] = model_file
                new_source_face['DFLModelPath'] = f'{dfl_models_dir}/{model_file}'
                button_text = f"(DFM) {model_file.split('.')[0]}"
                text_width = text_font.measure('ABCDEFGHIJKLMNO')
                new_source_face['TKButton'] = tk.Button(self.merged_faces_canvas, style.media_button_off_3, image=self.blank, text=button_text, height=14, width=text_width, compound='left', anchor='w')
                new_source_face['TKButton'].bind('<ButtonRelease-1>', lambda event, arg=j: self.select_input_faces(event, arg))
                new_source_face['TKButton'].bind('<MouseWheel>', lambda event: self.merged_faces_canvas.xview_scroll(-int(event.delta / 120.0), 'units'))
                new_source_face['TextWidth'] = text_width
                x_width = 20
                if len(self.source_faces) > 0:
                    x_width += self.get_adjacent_element_width(j)
                new_source_face['XCoord'] = x_width
                self.merged_faces_canvas.create_window(x_width, 8 + 22 * (j % 4), window=new_source_face['TKButton'], anchor='nw')
                self.source_faces.append(new_source_face)
                j += 1
    def get_adjacent_element_width(self, cur_index=0):
        x_width = 0
        if len(self.source_faces) >= 4 and cur_index >= 4:
                adjacent_elem_index = cur_index - 4
                x_width = self.source_faces[adjacent_elem_index].get('XCoord', 0) + self.source_faces[adjacent_elem_index].get('TextWidth', 0)
        return x_width
    def load_input_faces(self):
        self.source_faces = []
        self.merged_faces_canvas.delete('all')
        self.source_faces_canvas.delete('all')
        text_font = font.Font(family='Microsoft YaHei', size=10)
        try:
            temp0 = []
            try:
                with open('merged_embeddings.txt', 'r', encoding='utf-8') as embedfile:
                    temp = embedfile.read().splitlines()
                    for i in range(0, len(temp), 513):
                        to = [temp[i][6:], np.array(temp[i + 1:i + 513], dtype='float32')]
                        temp0.append(to)
            except:
                pass
            for j in range(len(temp0)):
                new_source_face = self.source_face.copy()
                new_source_face['ButtonState'] = False
                new_source_face['Embedding'] = temp0[j][1]
                text_width = text_font.measure('ABCDEFGHIJKLMNO')
                new_source_face['TKButton'] = tk.Button(self.merged_faces_canvas, style.media_button_off_3, image=self.blank, text=temp0[j][0], height=14, width=text_width, compound='left', anchor='w')
                new_source_face['TKButton'].bind('<ButtonRelease-1>', lambda event, arg=j: self.select_input_faces(event, arg))
                new_source_face['TKButton'].bind('<MouseWheel>', lambda event: self.merged_faces_canvas.xview_scroll(-int(event.delta / 120.0), 'units'))
                new_source_face['TextWidth'] = text_width
                x_width = 20
                if len(self.source_faces) > 0:
                    x_width += self.get_adjacent_element_width(j)
                new_source_face['XCoord'] = x_width
                self.merged_faces_canvas.create_window(x_width, 8 + 22 * (j % 4), window=new_source_face['TKButton'], anchor='nw')
                self.source_faces.append(new_source_face)
            self.load_dfl_input_models()
            self.merged_faces_canvas.configure(scrollregion=self.merged_faces_canvas.bbox('all'))
            self.merged_faces_canvas.xview_moveto(0)
        except Exception as e:
            pass
        self.shift_i_len = len(self.source_faces)
        directory = self.json_dict['source faces']
        filenames = [os.path.join(dirpath, f) for dirpath, dirnames, filenames in os.walk(directory) for f in filenames]
        i = 0
        for file in filenames:
            try:
                file_type = mimetypes.guess_type(file)[0][:5]
            except Exception as e:
                import logging
                logging.warning(f'无法识别的文件类型: {file}, 错误: {e}')
            else:
                if file_type == 'image':
                    img = cv2_imread_unicode(file)
                    if img is not None:
                        img = torch.from_numpy(img.astype('uint8')).to(self.models.device)
                        pad_scale = 0.2
                        padded_width = int(img.size()[1] * (1.0 + pad_scale))
                        padded_height = int(img.size()[0] * (1.0 + pad_scale))
                        padding = torch.zeros((padded_height, padded_width, 3), dtype=torch.uint8, device=self.models.device)
                        width_start = int(img.size()[1] * pad_scale / 2)
                        width_end = width_start + int(img.size()[1])
                        height_start = int(img.size()[0] * pad_scale / 2)
                        height_end = height_start + int(img.size()[0])
                        padding[height_start:height_end, width_start:width_end, :] = img
                        img = padding
                        img = img.permute(2, 0, 1)
                        try:
                            if self.parameters['AutoRotationSwitch']:
                                rotation_angles = [0, 90, 180, 270]
                            else:
                                rotation_angles = [0]
                            bboxes, kpss_5, _ = self.models.run_detect(img, detect_mode=self.parameters['DetectTypeTextSel'], max_num=1, score=0.5, use_landmark_detection=self.parameters['LandmarksDetectionAdjSwitch'], landmark_detect_mode=self.parameters['LandmarksDetectTypeTextSel'], landmark_score=0.5, from_points=self.parameters['LandmarksAlignModeFromPointsSwitch'], rotation_angles=rotation_angles)
                            kpss_5 = kpss_5[0]
                        except IndexError:
                            import logging
                            logging.warning(f'图像裁剪过于接近: {file}')
                            continue
                        face_emb, cropped_image = self.models.run_recognize(img, kpss_5, self.parameters['SimilarityTypeTextSel'], self.parameters['FaceSwapperModelTextSel'])
                        crop = cv2.cvtColor(cropped_image.cpu().numpy(), cv2.COLOR_BGR2RGB)
                        crop = cv2.resize(crop, (85, 85))
                        new_source_face = self.source_face.copy()
                        self.source_faces.append(new_source_face)
                        self.source_faces[(-1)]['Image'] = ImageTk.PhotoImage(image=Image.fromarray(crop))
                        self.source_faces[(-1)]['Embedding'] = face_emb
                        self.source_faces[(-1)]['TKButton'] = tk.Button(self.source_faces_canvas, style.media_button_off_3, image=self.source_faces[(-1)]['Image'], height=90, width=90)
                        self.source_faces[(-1)]['ButtonState'] = False
                        self.source_faces[(-1)]['file'] = file
                        self.source_faces[(-1)]['TKButton'].bind('<ButtonRelease-1>', lambda event, arg=len(self.source_faces) - 1: self.select_input_faces(event, arg))
                        self.source_faces[(-1)]['TKButton'].bind('<MouseWheel>', self.source_faces_mouse_wheel)
                        self.source_faces_canvas.create_window(i % 2 * 100, i // 2 * 100, window=self.source_faces[(-1)]['TKButton'], anchor='nw')
                        self.static_widget['input_faces_scrollbar'].resize_scrollbar(None)
                        i = i + 1
                    else:
                        import logging
                        logging.warning(f'无效文件: {file}')
        torch.cuda.empty_cache()
    def find_faces(self):
        try:
            img = torch.from_numpy(self.video_image).to(self.models.device)
            img = img.permute(2, 0, 1)
            if self.parameters['AutoRotationSwitch']:
                rotation_angles = [0, 90, 180, 270]
            else:
                rotation_angles = [0]
            bboxes, kpss_5, _ = self.models.run_detect(img, detect_mode=self.parameters['DetectTypeTextSel'], max_num=50, score=self.parameters['DetectScoreSlider'] / 100.0, use_landmark_detection=self.parameters['LandmarksDetectionAdjSwitch'], landmark_detect_mode=self.parameters['LandmarksDetectTypeTextSel'], landmark_score=self.parameters['LandmarksDetectScoreSlider'] / 100.0, from_points=self.parameters['LandmarksAlignModeFromPointsSwitch'], rotation_angles=rotation_angles)
            ret = []
            for face_kps in kpss_5:
                face_emb, cropped_img = self.models.run_recognize(img, face_kps, self.parameters['SimilarityTypeTextSel'], self.parameters['FaceSwapperModelTextSel'])
                ret.append([face_kps, face_emb, cropped_img])
        except Exception as e:
            messagebox.showinfo('No Media', 'No media selected')
            import logging
            logging.warning('未选择媒体文件')
            return None
        if ret:
            threshhold = self.parameters['ThresholdSlider']
            for face in ret:
                found = False
                for emb in self.target_faces:
                    if self.findCosineDistance(emb['Embedding'], face[1]) >= threshhold:
                        found = True
                        break
                if not found:
                    crop = cv2.resize(face[2].cpu().numpy(), (82, 82))
                    new_target_face = self.target_face.copy()
                    self.target_faces.append(new_target_face)
                    last_index = len(self.target_faces) - 1
                    self.target_faces[last_index]['TKButton'] = tk.Button(self.found_faces_canvas, style.media_button_off_3, height=86, width=86)
                    self.target_faces[last_index]['TKButton'].bind('<MouseWheel>', self.target_faces_mouse_wheel)
                    self.target_faces[last_index]['ButtonState'] = False
                    self.target_faces[last_index]['Image'] = ImageTk.PhotoImage(image=Image.fromarray(crop))
                    self.target_faces[last_index]['Embedding'] = face[1]
                    self.target_faces[last_index]['EmbeddingNumber'] = 1
                    self.target_faces[(-1)]['TKButton'].config(pady=10, image=self.target_faces[last_index]['Image'], command=lambda k=last_index: self.toggle_found_faces_buttons_state(k))
                    self.found_faces_canvas.create_window(last_index * 92, 8, window=self.target_faces[last_index]['TKButton'], anchor='nw')
                    self.found_faces_canvas.configure(scrollregion=self.found_faces_canvas.bbox('all'))
    def clear_faces(self):
        self.target_faces = []
        self.found_faces_canvas.delete('all')
    def toggle_found_faces_buttons_state(self, button):
        for i in range(len(self.target_faces)):
            self.target_faces[i]['ButtonState'] = False
            self.target_faces[i]['TKButton'].config(style.media_button_off_3)
        self.target_faces[button]['ButtonState'] = True
        self.target_faces[button]['TKButton'].config(style.media_button_on_3)
        for i in range(len(self.source_faces)):
            self.source_faces[i]['ButtonState'] = False
            self.source_faces[i]['TKButton'].config(style.media_button_off_3)
        for i in range(len(self.target_faces[button]['SourceFaceAssignments'])):
            self.source_faces[self.target_faces[button]['SourceFaceAssignments'][i]]['ButtonState'] = True
            self.source_faces[self.target_faces[button]['SourceFaceAssignments'][i]]['TKButton'].config(style.media_button_on_3)
    def select_input_faces(self, event, button):
        try:
            if event.state & 4 != 0:
                modifier = 'ctrl'
            else:
                if event.state & 1 != 0:
                    modifier = 'shift'
                else:
                    modifier = 'none'
        except:
            modifier = event
        if modifier != 'auto':
            for face in self.source_faces:
                face['TKButton'].config(style.media_button_off_3)
                if modifier == 'none':
                    face['ButtonState'] = False
            if modifier != 'merge':
                self.source_faces[button]['ButtonState'] = not self.source_faces[button]['ButtonState']
            if modifier == 'shift':
                if self.source_faces[button]['DFLModel']:
                    for i in range(len(self.source_faces)):
                        if i == button:
                            continue
                        else:
                            if self.source_faces[i]['ButtonState'] and self.source_faces[i]['DFLModel']:
                                    self.source_faces[button]['ButtonState'] = False
                                    messagebox.showinfo('You cannot combine DFL Models!', 'You cannot combine DFL Models!')
                                    for face in self.source_faces:
                                        face['ButtonState'] = False
                                    break
                for i in range(button - 1, self.shift_i_len - 1, (-1)):
                    if self.source_faces[i]['ButtonState']:
                        for j in range(i, button, 1):
                            self.source_faces[j]['ButtonState'] = True
                        break
                for i in range(button + 1, len(self.source_faces), 1):
                    if self.source_faces[i]['ButtonState']:
                        for j in range(button, i, 1):
                            self.source_faces[j]['ButtonState'] = True
                        break
            for face in self.source_faces:
                if face['ButtonState']:
                    face['TKButton'].config(style.media_button_on_3)
                    if self.widget['PreviewModeTextSel'].get() == 'FaceLab':
                        self.add_action('load_target_image', face['file'])
                        self.image_loaded = True
            if self.source_faces[button]['DFLModel']:
                if self.models.dfl_models:
                    if self.parameters['DFLLoadOnlyOneSwitch']:
                        for model in list(self.models.dfl_models):
                            if model != self.source_faces[button]['DFLModel']:
                                del self.models.dfl_models[model]._sess
                                del self.models.dfl_models[model]
                        gc.collect()
        for tface in self.target_faces:
            if tface['ButtonState']:
                tface['SourceFaceAssignments'] = []
                tface['DFLModel'] = False
                temp_holder = []
                for j in range(len(self.source_faces)):
                    if self.source_faces[j]['ButtonState']:
                        tface['SourceFaceAssignments'].append(j)
                        if not self.source_faces[j]['DFLModel']:
                            temp_holder.append(self.source_faces[j]['Embedding'])
                        if self.source_faces[j]['DFLModel']:
                            if self.models.dfl_models and self.parameters['DFLLoadOnlyOneSwitch']:
                                    for model in list(self.models.dfl_models):
                                        del self.models.dfl_models[model]._sess
                                        del self.models.dfl_models[model]
                                    gc.collect()
                            tface['DFLModel'] = self.source_faces[j]['DFLModel']
                if temp_holder:
                    if self.widget['MergeTextSel'].get() == 'Median':
                        tface['AssignedEmbedding'] = np.median(temp_holder, 0)
                    else:
                        if self.widget['MergeTextSel'].get() == 'Mean':
                            tface['AssignedEmbedding'] = np.mean(temp_holder, 0)
                    self.temp_emb = tface['AssignedEmbedding']
                else:
                    tface['AssignedEmbedding'] = []
                break
        self.add_action('target_faces', self.target_faces)
        self.add_action('get_requested_video_frame', self.video_slider.get())
    def populate_target_videos(self):
        # irreducible cflow, using cdg fallback
        videos = []
        camera_backend = CAMERA_BACKENDS[self.parameters['WebCamBackendSel']]
        for i in range(self.parameters['WebCamMaxNoSlider']):
            try:
                camera = cv2.VideoCapture(i, camera_backend)
                if not camera.isOpened():
                    continue
                success, webcam_frame = camera.read()
                if not success:
                    continue
                ratio = float(webcam_frame.shape[0]) / webcam_frame.shape[1]
                new_height = 50
                new_width = int(new_height / ratio)
                webcam_frame = cv2.resize(webcam_frame, (new_width, new_height))
                webcam_frame = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB)
                webcam_frame[:new_height, :new_width, :] = webcam_frame
                videos.append([webcam_frame, f'Webcam {i}'])
                camera.release()
            except Exception as e:
                import logging
                logging.error(f'摄像头访问错误: {e}')
        directory = self.json_dict['source videos']
        filenames = [os.path.join(dirpath, f) for dirpath, dirnames, filenames in os.walk(directory) for f in filenames]
        images = []
        self.target_media = []
        self.target_media_buttons = []
        self.target_media_canvas.delete('all')
        for file in filenames:
            try:
                file_type = mimetypes.guess_type(file)[0][:5]
            except Exception as e:
                import logging
                logging.warning(f'无法识别的文件类型: {file}, 错误: {e}')
            else:
                if file_type == 'image':
                    try:
                        image = cv2_imread_unicode(file)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    except Exception as e:
                        import logging
                        logging.error(f'读取文件失败: {file}, 错误: {e}')
                    else:
                        ratio = float(image.shape[0]) / image.shape[1]
                        new_height = 100
                        new_width = int(new_height / ratio)
                        image = cv2.resize(image, (new_width, new_height))
                        image[:new_height, :new_width, :] = image
                        images.append([image, file])
                else:
                    if file_type == 'video':
                        try:
                            video = cv2.VideoCapture(file)
                        except Exception as e:
                            import logging
                            logging.error(f'读取视频文件失败: {file}, 错误: {e}')
                        else:
                            if video.isOpened():
                                video.set(cv2.CAP_PROP_POS_FRAMES, int(video.get(cv2.CAP_PROP_FRAME_COUNT) / 2))
                                success, video_frame = video.read()
                                if success:
                                    video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
                                    ratio = float(video_frame.shape[0]) / video_frame.shape[1]
                                    new_height = 100
                                    new_width = int(new_height / ratio)
                                    video_frame = cv2.resize(video_frame, (new_width, new_height))
                                    video_frame[:new_height, :new_width, :] = video_frame
                                    videos.append([video_frame, file])
                                    video.release()
                                else:
                                    import logging
                                    logging.error(f'读取视频文件失败: {file}')
                            else:
                                import logging
                                logging.error(f'打开视频文件失败: {file}')
        delx, dely = (100, 120)
        if self.widget['PreviewModeTextSel'].get() == 'Image':
            for i in range(len(images)):
                self.target_media_buttons.append(tk.Button(self.target_media_canvas, style.media_button_off_3, height=115, width=190))
                rgb_video = Image.fromarray(images[i][0])
                self.target_media.append(ImageTk.PhotoImage(image=rgb_video))
                self.target_media_buttons[i].config(image=self.target_media[i], command=lambda i=i: self.load_target(i, images[i][1], self.widget['PreviewModeTextSel'].get()))
                self.target_media_buttons[i].bind('<MouseWheel>', self.target_videos_mouse_wheel)
                self.target_media_canvas.create_window(0, i * dely, window=self.target_media_buttons[i], anchor='nw')
            self.static_widget['input_videos_scrollbar'].resize_scrollbar(None)
        else:
            if self.widget['PreviewModeTextSel'].get() == 'Video':
                for i in range(len(videos)):
                    self.target_media_buttons.append(tk.Button(self.target_media_canvas, style.media_button_off_3, height=115, width=190))
                    self.target_media.append(ImageTk.PhotoImage(image=Image.fromarray(videos[i][0])))
                    filename = os.path.basename(videos[i][1])
                    if len(filename) > 32:
                        filename = filename[:29] + '...'
                    self.target_media_buttons[i].bind('<MouseWheel>', self.target_videos_mouse_wheel)
                    self.target_media_buttons[i].config(image=self.target_media[i], text=filename, compound='top', anchor='n', command=lambda i=i: self.load_target(i, videos[i][1], self.widget['PreviewModeTextSel'].get()))
                    self.target_media_canvas.create_window(0, i * dely, window=self.target_media_buttons[i], anchor='nw')
                self.static_widget['input_videos_scrollbar'].resize_scrollbar(None)
    def auto_swap(self):
        try:
            self.find_faces()
            self.target_faces[0]['ButtonState'] = True
            self.target_faces[0]['TKButton'].config(style.media_button_on_3)
            self.select_input_faces('auto', '')
            self.toggle_swapper(True)
        except:
            return None
    def toggle_auto_swap(self):
        self.widget['AutoSwapButton'].toggle_button()
    def load_target(self, button, media_file, media_type):
        self.toggle_play_video('stop')
        self.image_loaded = False
        self.video_loaded = False
        self.clear_faces()
        if media_type == 'Video':
            self.video_slider.set(0)
            self.add_action('load_target_video', media_file)
            self.media_file_name = os.path.splitext(os.path.basename(media_file))
            self.video_loaded = True
        else:
            if media_type == 'Image':
                self.add_action('load_target_image', media_file)
                self.media_file_name = os.path.splitext(os.path.basename(media_file))
                self.image_loaded = True
                if self.widget['AutoSwapButton'].get():
                    self.add_action('function', 'gui.auto_swap()')
        for i in range(len(self.target_media_buttons)):
            self.target_media_buttons[i].config(style.media_button_off_3)
        self.target_media_buttons[button].config(style.media_button_on_3)
        self.layer['markers_canvas'].delete('all')
        self.markers = []
        self.stop_marker = []
        self.load_markers_json()
        self.add_action('update_markers_canvas', self.markers)
        self.add_action('markers', self.markers)
    def set_image(self, image, requested):
        self.video_image = image[0]
        frame = image[1]
        if not requested:
            self.video_slider.set(frame)
            self.parameter_update_from_marker(frame)
        self.resize_image()
    def resize_image(self):
        image = self.video_image
        if len(image) != 0:
            if hasattr(self, 'preview_rotation_angle') and self.preview_rotation_angle != 0:
                    if self.preview_rotation_angle == 90:
                        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    else:
                        if self.preview_rotation_angle == 180:
                            image = cv2.rotate(image, cv2.ROTATE_180)
                        else:
                            if self.preview_rotation_angle == 270:
                                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x1 = float(self.video.winfo_width())
            y1 = float(self.video.winfo_height())
            x2 = float(image.shape[1])
            y2 = float(image.shape[0])
            m1 = x1 / y1
            m2 = x2 / y2
            if m2 > m1:
                x2 = x1
                y2 = x1 / m2
                image = cv2.resize(image, (int(x2), int(y2)))
                padding = int((y1 - y2) / 2.0)
                image = cv2.copyMakeBorder(image, padding, padding, 0, 0, cv2.BORDER_CONSTANT)
            else:
                y2 = y1
                x2 = y2 * m2
                image = cv2.resize(image, (int(x2), int(y2)))
                padding = int((x1 - x2) / 2.0)
                image = cv2.copyMakeBorder(image, 0, 0, padding, padding, cv2.BORDER_CONSTANT)
            pil_image = Image.fromarray(image)
            image = ImageTk.PhotoImage(pil_image)
            self.video.image = image
            self.video.configure(image=self.video.image)
            if self.float_preview_window and self.float_preview_window.winfo_exists():
                    self.update_float_preview_image(pil_image)
    def check_for_video_resize(self):
        win_geom = '%dx%d+%d+%d' % (self.json_dict['dock_win_geom'][0], self.json_dict['dock_win_geom'][1], self.json_dict['dock_win_geom'][2], self.json_dict['dock_win_geom'][3])
        if self.winfo_geometry() != win_geom:
            self.resize_image()
            for k, v in self.widget.items():
                v.is_resizing = True
                v.hide()
                v.is_resizing = False
            for k, v in self.static_widget.items():
                v.is_resizing = True
                v.hide()
                v.is_resizing = False
            if self.winfo_geometry() != self.window_last_change:
                self.window_last_change = self.winfo_geometry()
                return
            else:
                for k, v in self.widget.items():
                    v.is_resizing = True
                    v.unhide()
                    v.is_resizing = False
                for k, v in self.static_widget.items():
                    v.is_resizing = True
                    v.unhide()
                    v.is_resizing = False
                str1 = self.winfo_geometry().split('x')
                str2 = str1[1].split('+')
                win_geom = [str1[0], str2[0], str2[1], str2[2]]
                win_geom = [int(strings) for strings in win_geom]
                self.json_dict['dock_win_geom'] = win_geom
                self.json_dict['language'] = self.current_language
                with open(language_manager.data_json_path(), 'w', encoding='utf-8') as outfile:
                    json.dump(self.json_dict, outfile, ensure_ascii=False, indent=4)
    def get_action(self):
        action = self.action_q[0]
        self.action_q.pop(0)
        return action
    def get_action_length(self):
        return len(self.action_q)
    def set_video_slider_length(self, video_length):
        self.video_slider.set_length(video_length)
    def set_video_slider_fps(self, fps):
        self.video_slider.set_fps(fps)
    def findCosineDistance(self, vector1, vector2):
        vector1 = vector1.ravel()
        vector2 = vector2.ravel()
        cos_dist = 1 - np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        return 100 - cos_dist * 50
    def toggle_play_video(self, set_value='toggle'):
        if self.video_loaded:
            if set_value == 'toggle':
                self.widget['TLPlayButton'].toggle_button()
            if set_value == 'stop':
                self.widget['TLPlayButton'].disable_button()
            if set_value == 'play':
                self.widget['TLPlayButton'].enable_button()
            if self.widget['TLPlayButton'].get():
                if not self.video_loaded:
                    import logging
                    logging.warning('请先选择视频!')
                    return
                else:
                    loop_enabled = self.widget['TLLoopButton'].get() if 'TLLoopButton' in self.widget else False
                    self.add_action('set_loop_mode', loop_enabled)
                    if self.widget['TLRecButton'].get():
                        if not self.json_dict['saved videos']:
                            messagebox.showinfo('Set saved videos folder', 'PLease set a folder to save videos before starting to record ')
                            import logging
                            logging.warning('请先设置视频保存文件夹!')
                            self.add_action('play_video', 'stop_from_gui')
                        else:
                            self.add_action('play_video', 'record')
                    else:
                        self.add_action('play_video', 'play')
            else:
                self.add_action('play_video', 'stop_from_gui')
    def toggle_loop_video(self):
        """切换循环播放模式"""
        if 'TLLoopButton' in self.widget:
            self.widget['TLLoopButton'].toggle_button()
            loop_enabled = self.widget['TLLoopButton'].get()
            self.add_action('set_loop_mode', loop_enabled)
            status = '已启用' if loop_enabled else '已禁用'
    def set_player_buttons_to_inactive(self):
        self.widget['TLRecButton'].disable_button()
        self.widget['TLPlayButton'].disable_button()
    def set_virtual_cam_toggle_disable(self):
        self.widget['VirtualCameraSwitch'].toggle_switch(False)
    def toggle_swapper(self, toggle_value=(-1)):
        if toggle_value == (-1):
            self.widget['SwapFacesButton'].toggle_button()
        else:
            if toggle_value:
                self.widget['SwapFacesButton'].enable_button()
            else:
                self.widget['SwapFacesButton'].disable_button()
        if self.widget['PreviewModeTextSel'].get() == 'Video' or self.widget['PreviewModeTextSel'].get() == 'Theater':
            self.update_data('control', 'SwapFacesButton', use_markers=True)
        else:
            if self.widget['PreviewModeTextSel'].get() == 'Image':
                self.update_data('control', 'SwapFacesButton', use_markers=False)
            else:
                if self.widget['PreviewModeTextSel'].get() == 'FaceLab':
                    self.update_data('control', 'SwapFacesButton', use_markers=False)
    def temp_toggle_swapper(self, state):
        if state == 'off':
            self.widget['SwapFacesButton'].temp_disable_button()
        else:
            if state == 'on':
                self.widget['SwapFacesButton'].temp_enable_button()
        self.update_data('control', 'SwapFacesButton', use_markers=True)
    def toggle_enhancer(self, toggle_value=(-1)):
        if toggle_value == (-1):
            self.widget['EnhanceFrameButton'].toggle_button()
        else:
            if toggle_value:
                self.widget['EnhanceFrameButton'].enable_button()
            else:
                self.widget['EnhanceFrameButton'].disable_button()
        if self.widget['PreviewModeTextSel'].get() == 'Video' or self.widget['PreviewModeTextSel'].get() == 'Theater':
            self.update_data('control', 'EnhanceFrameButton', use_markers=True)
        else:
            if self.widget['PreviewModeTextSel'].get() == 'Image':
                self.update_data('control', 'EnhanceFrameButton', use_markers=False)
            else:
                if self.widget['PreviewModeTextSel'].get() == 'FaceLab':
                    self.update_data('control', 'EnhanceFrameButton', use_markers=False)
    def temp_toggle_enhancer(self, state):
        if state == 'off':
            self.widget['EnhanceFrameButton'].temp_disable_button()
        else:
            if state == 'on':
                self.widget['EnhanceFrameButton'].temp_enable_button()
        self.update_data('control', 'EnhanceFrameButton', use_markers=True)
    def toggle_faces_editor(self, toggle_value=(-1)):
        if toggle_value == (-1):
            self.widget['EditFacesButton'].toggle_button()
        else:
            if toggle_value:
                self.widget['EditFacesButton'].enable_button()
            else:
                self.widget['EditFacesButton'].disable_button()
        if self.widget['PreviewModeTextSel'].get() == 'Video' or self.widget['PreviewModeTextSel'].get() == 'Theater':
            self.update_data('control', 'EditFacesButton', use_markers=True)
        else:
            if self.widget['PreviewModeTextSel'].get() == 'Image':
                self.update_data('control', 'EditFacesButton', use_markers=False)
            else:
                if self.widget['PreviewModeTextSel'].get() == 'FaceLab':
                    self.update_data('control', 'EditFacesButton', use_markers=False)
    def temp_toggle_faces_editor(self, state):
        if state == 'off':
            self.widget['EditFacesButton'].temp_disable_button()
        else:
            if state == 'on':
                self.widget['EditFacesButton'].temp_enable_button()
        self.update_data('control', 'EditFacesButton', use_markers=True)
    def toggle_rec_video(self):
        self.save_markers_json()
        if not self.widget['TLPlayButton'].get():
            self.widget['TLRecButton'].toggle_button()
            if self.widget['TLRecButton'].get():
                self.widget['TLRecButton'].enable_button()
            else:
                self.widget['TLRecButton'].disable_button()
    def add_action(self, action, parameter=None):
        if action != 'get_requested_video_frame' and action != 'get_requested_video_frame_without_markers':
            self.action_q.append([action, parameter])
        else:
            if self.video_loaded and (not self.widget['TLPlayButton'].get()):
                self.action_q.append([action, parameter])
            else:
                if self.image_loaded:
                    self.action_q.append([action, parameter])
    def update_vram_indicator(self):
        try:
            used, total = self.models.get_gpu_memory()
        except:
            return None
        self.static_widget['vram_indicator'].set(used, total)
    def save_selected_source_faces(self, text):
        text = text.get()
        temp_holder = []
        for button in self.source_faces:
            if button['ButtonState']:
                temp_holder.append(button['Embedding'])
        if temp_holder:
            if self.widget['MergeTextSel'].get() == 'Median':
                ave_embedding = np.median(temp_holder, 0)
            else:
                if self.widget['MergeTextSel'].get() == 'Mean':
                    ave_embedding = np.mean(temp_holder, 0)
            for tface in self.target_faces:
                if tface['ButtonState']:
                    ave_embedding = tface['AssignedEmbedding']
            if text != '':
                with open('merged_embeddings.txt', 'a', encoding='utf-8') as embedfile:
                    identifier = 'Name: ' + text
                    embedfile.write('%s\n' % identifier)
                    for number in ave_embedding:
                        embedfile.write('%s\n' % number)
            else:
                import logging
                logging.warning('未指定嵌入名称')
        else:
            import logging
            logging.warning('未选择源图像')
        self.focus()
        self.load_input_faces()
    def delete_merged_embedding(self):
        sel = []
        for j in range(len(self.source_faces)):
            if self.source_faces[j]['ButtonState']:
                sel = j
                break
        temp0 = []
        if os.path.exists('merged_embeddings.txt'):
            with open('merged_embeddings.txt', 'r', encoding='utf-8') as embedfile:
                temp = embedfile.read().splitlines()
                for i in range(0, len(temp), 513):
                    to = [temp[i], np.array(temp[i + 1:i + 513], dtype='float32')]
                    temp0.append(to)
        if j < len(temp0):
            temp0.pop(j)
            with open('merged_embeddings.txt', 'w', encoding='utf-8') as embedfile:
                for line in temp0:
                    embedfile.write('%s\n' % line[0])
                    for i in range(512):
                        embedfile.write('%s\n' % line[1][i])
        self.load_input_faces()
    def iterate_through_merged_embeddings(self, event):
        if event.delta > 0:
            for i in range(len(self.source_faces)):
                if self.source_faces[i]['ButtonState'] and i < len(self.source_faces) - 1:
                        self.select_input_faces('none', i + 1)
                        break
        else:
            if event.delta < 0:
                for i in range(len(self.source_faces)):
                    if self.source_faces[i]['ButtonState'] and i > 0:
                            self.select_input_faces('none', i - 1)
                            break
    def set_view(self, load_target_videos, b):
        if load_target_videos and self.widget['PreviewModeTextSel'].get() != 'Theater':
                self.populate_target_videos()
        self.layer['slider_frame'].grid_forget()
        self.layer['preview_frame'].grid_forget()
        self.layer['markers_canvas'].grid_forget()
        self.layer['image_controls'].grid_forget()
        self.layer['FaceLab_controls'].grid_forget()
        self.layer['InputVideoFrame'].grid_forget()
        self.layer['parameter_frame'].grid_forget()
        self.layer['parameters_canvas'].grid_forget()
        self.layer['parameter_scroll_canvas'].grid_forget()
        self.layer['facelab_canvas'].grid_forget()
        self.layer['facelab_scroll_canvas'].grid_forget()
        if self.widget['PreviewModeTextSel'].get() == 'Video':
            self.image_loaded = False
            self.layer['slider_frame'].grid(row=2, column=0, sticky='NEWS', pady=0)
            self.layer['preview_frame'].grid(row=4, column=0, sticky='NEWS')
            self.layer['markers_canvas'].grid(row=3, column=0, sticky='NEWS')
            self.layer['parameter_frame'].grid(row=0, column=2, sticky='NEWS', pady=0, padx=1)
            self.layer['parameters_canvas'].grid(row=1, column=0, sticky='NEWS', pady=0, padx=0)
            self.layer['parameter_scroll_canvas'].grid(row=1, column=1, sticky='NEWS', pady=0)
            self.layer['InputVideoFrame'].grid(row=0, column=0, sticky='NEWS', padx=1, pady=0)
        else:
            if self.widget['PreviewModeTextSel'].get() == 'Image':
                self.video_loaded = False
                self.layer['image_controls'].grid(row=2, column=0, rowspan=2, sticky='NEWS', pady=0)
                self.layer['parameters_canvas'].grid(row=1, column=0, sticky='NEWS', pady=0, padx=0)
                self.layer['parameter_scroll_canvas'].grid(row=1, column=1, sticky='NEWS', pady=0)
                self.layer['InputVideoFrame'].grid(row=0, column=0, sticky='NEWS', padx=1, pady=0)
                self.layer['parameter_frame'].grid(row=0, column=2, sticky='NEWS', pady=0, padx=1)
            else:
                if self.widget['PreviewModeTextSel'].get() == 'FaceLab':
                    self.video_loaded = False
                    self.layer['FaceLab_controls'].grid(row=2, column=0, rowspan=2, sticky='NEWS', pady=0)
                    self.layer['facelab_canvas'].grid(row=1, column=0, sticky='NEWS', pady=0, padx=0)
                    self.layer['facelab_scroll_canvas'].grid(row=1, column=1, sticky='NEWS', pady=0)
                    self.layer['InputVideoFrame'].grid(row=0, column=0, sticky='NEWS', padx=1, pady=0)
                    self.layer['parameter_frame'].grid(row=0, column=2, sticky='NEWS', pady=0, padx=1)
                else:
                    if self.widget['PreviewModeTextSel'].get() == 'Theater':
                        self.image_loaded = False
                        self.layer['slider_frame'].grid(row=2, column=0, sticky='NEWS', pady=0)
                        self.layer['preview_frame'].grid(row=4, column=0, sticky='NEWS')
                        self.layer['markers_canvas'].grid(row=3, column=0, sticky='NEWS')
    def update_marker(self, action):
        if action == 'add':
            for i in range(len(self.markers)):
                if self.markers[i]['frame'] == self.video_slider.get():
                    self.layer['markers_canvas'].delete(self.markers[i]['icon_ref'])
                    self.markers.pop(i)
                    break
            width = self.layer['markers_canvas'].winfo_width() - 20 - 40 - 20
            position = 20 + int(width * self.video_slider.get() / self.video_slider.get_length())
            temp_param = copy.deepcopy(self.parameters)
            temp = {'frame': self.video_slider.get(), 'parameters': temp_param, 'icon_ref': self.layer['markers_canvas'].create_line(position, 0, position, 15, fill='light goldenrod')}
            self.markers.append(temp)
            def sort(e):
                return e['frame']
            self.markers.sort(key=sort)
            self.add_action('markers', self.markers)
        else:
            if action == 'delete':
                for i in range(len(self.markers)):
                    if self.markers[i]['frame'] == self.video_slider.get():
                        self.layer['markers_canvas'].delete(self.markers[i]['icon_ref'])
                        self.markers.pop(i)
                        break
            else:
                if action == 'prev':
                    temp = []
                    for i in range(len(self.markers)):
                        temp.append(self.markers[i]['frame'])
                    idx = bisect.bisect_left(temp, self.video_slider.get())
                    if idx > 0:
                        self.video_slider.set(self.markers[idx - 1]['frame'])
                        self.add_action('get_requested_video_frame', self.markers[idx - 1]['frame'])
                        self.parameter_update_from_marker(self.markers[idx - 1]['frame'])
                else:
                    if action == 'next':
                        temp = []
                        for i in range(len(self.markers)):
                            temp.append(self.markers[i]['frame'])
                        idx = bisect.bisect(temp, self.video_slider.get())
                        if idx < len(self.markers):
                            self.video_slider.set(self.markers[idx]['frame'])
                            self.add_action('get_requested_video_frame', self.markers[idx]['frame'])
                            self.parameter_update_from_marker(self.markers[idx]['frame'])
                    else:
                        self.layer['markers_canvas'].delete('all')
                        width = self.layer['markers_canvas'].winfo_width() - 20 - 40 - 20
                        for marker in self.markers:
                            position = 20 + int(width * marker['frame'] / self.video_slider.get_length())
                            marker['icon_ref'] = self.layer['markers_canvas'].create_line(position, 0, position, 15, fill='light goldenrod')
    def save_markers_json(self):
        if len(self.markers) == 0 or len(self.media_file_name) == 0:
            return None
        else:
            json_file_path = os.path.join(self.json_dict['source videos'], self.media_file_name[0] + '_markers.json')
            with open(json_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(self.markers, json_file)
                import logging
                logging.warning('标记已保存')
    def load_markers_json(self):
        if len(self.media_file_name) == 0:
            return
        else:
            json_file_path = os.path.join(self.json_dict['source videos'], self.media_file_name[0] + '_markers.json')
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r', encoding='utf-8') as json_file:
                    loaded_markers = json.load(json_file)
                update_rules = {'RestorerTypeTextSel': lambda value, loaded_marker: {'RestorerTypeTextSel': {'GFPGAN': 'GFPGAN-v1.4', 'CF': 'CodeFormer', 'GPEN256': 'GPEN-256', 'GPEN512': 'GPEN-512'}.get(value, value)}, 'Restorer2TypeTextSel': lambda value, loaded_marker: {'MouthParserSlider': max(5, value), 'UpperLipParserSlider': value // 2 if value > 0 else 8, 'LowerLipParserSlider': value // 2 if value > 0 else 8}, 'MouthParserSlider': lambda value, loaded_marker: {'MouthParserSlider': value}}
                updated_markers = []
                for loaded_marker in loaded_markers:
                    updated_parameters = {}
                    rules_updated_keys = set()
                    for key in self.parameters:
                        value = loaded_marker['parameters'].get(key, self.parameters[key])
                        if key in update_rules:
                            updates = update_rules[key](value, loaded_marker['parameters'])
                            for update_key, update_value in updates.items():
                                updated_parameters[update_key] = update_value
                                rules_updated_keys.add(update_key)
                        else:
                            if key not in rules_updated_keys:
                                updated_parameters[key] = value
                    updated_marker = {'frame': loaded_marker['frame'], 'parameters': updated_parameters, 'icon_ref': loaded_marker.get('icon_ref')}
                    updated_markers.append(updated_marker)
                self.markers = updated_markers
                self.add_action('update_markers_canvas', self.markers)
    def update_markers_canvas(self):
        self.layer['markers_canvas'].delete('all')
        width = self.layer['markers_canvas'].winfo_width() - 20 - 40 - 20
        for marker in self.markers:
            position = 20 + int(width * marker['frame'] / self.video_slider.get_length())
            marker['icon_ref'] = self.layer['markers_canvas'].create_line(position, 0, position, 15, fill='light goldenrod')
    def toggle_stop(self):
        if self.stop_marker == self.video_slider.self.timeline_position:
            self.stop_marker = []
            self.add_action('set_stop', (-1))
            self.video_slider_canvas.delete(self.stop_image)
        else:
            self.video_slider_canvas.delete(self.stop_image)
            self.stop_marker = self.video_slider.self.timeline_position
            self.add_action('set_stop', self.stop_marker)
            width = self.video_slider_canvas.winfo_width() - 30
            position = 15 + int(width * self.video_slider.self.timeline_position / self.video_slider.configure('to')[4])
            self.stop_image = self.video_slider_canvas.create_image(position, 30, image=self.stop_marker_icon)
    def save_image(self):
        filename = self.media_file_name[0] + '_' + str(time.time())[:10]
        filename = os.path.join(self.json_dict['saved videos'], filename)
        cv2.imwrite(filename + '.png', cv2.cvtColor(self.video_image, cv2.COLOR_BGR2RGB))
        import logging
        logging.warning(f'图像已保存为: {filename}.png')
    def clear_mem(self):
        """\n        手动释放显存功能 - 已被智能内存管理替代\n        该函数保留以防兼容性问题，但推荐使用智能内存管理\n        """
        start_time = time.time()
        self.widget['RestorerSwitch'].set(False)
        self.widget['OccluderSwitch'].set(False)
        self.widget['FaceParserSwitch'].set(False)
        self.widget['CLIPSwitch'].set(False)
        self.toggle_swapper(False)
        self.toggle_enhancer(False)
        self.toggle_faces_editor(False)
        self.models.delete_models()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        cleanup_time = time.time() - start_time
        try:
            from rope.OptimizationUtils import memory_optimizer
        except ImportError:
            return None
    def parameter_update_from_marker(self, frame):
        temp = []
        for i in range(len(self.markers)):
            temp.append(self.markers[i]['frame'])
        idx = bisect.bisect(temp, frame)
        if idx > 0:
            self.parameters = copy.deepcopy(self.markers[idx - 1]['parameters'])
            for key, value in self.parameters.items():
                self.widget[key].set(self.parameters[key], request_frame=False)
    def toggle_audio(self):
        self.add_action('play_video', 'stop_from_gui')
        self.widget['AudioButton'].toggle_button()
        self.control['AudioButton'] = self.widget['AudioButton'].get()
        self.add_action('control', self.control)
        if self.widget['TLPlayButton'].get():
            self.add_action('play_video', 'play')
    def toggle_maskview(self):
        self.widget['MaskViewButton'].toggle_button()
        self.control['MaskViewButton'] = self.widget['MaskViewButton'].get()
        self.add_action('control', self.control)
        self.add_action('get_requested_video_frame', self.video_slider.get())
    def toggle_compareview(self):
        self.widget['CompareViewButton'].toggle_button()
        self.control['CompareViewButton'] = self.widget['CompareViewButton'].get()
        self.add_action('control', self.control)
        self.add_action('get_requested_video_frame', self.video_slider.get())
    def toggle_float_preview(self):
        """切换悬浮预览窗口的显示状态"""
        if self.float_preview_window is None or not self.float_preview_window.winfo_exists():
            self.create_float_preview_window()
            self.widget['FloatPreviewButton'].enable_button()
            return
        else:
            self.close_float_preview_window()
            self.widget['FloatPreviewButton'].disable_button()
    def create_float_preview_window(self):
        """创建悬浮预览窗口"""
        import tkinter as tk
        self.float_preview_window = tk.Toplevel(self)
        self.float_preview_window.title('悬浮预览')
        self.float_window_has_border = True
        self.float_preview_window.overrideredirect(False)
        self.float_preview_window.title('悬浮预览')
        self.float_preview_window.attributes('-topmost', True)
        self.float_preview_window.configure(bg='black')
        window_width, window_height = self.get_float_window_size()
        x = self.winfo_x() + self.winfo_width() - window_width - 20
        y = self.winfo_y() + 50
        self.float_preview_window.geometry(f'{window_width}x{window_height}+{x}+{y}')
        self.float_preview_label = tk.Label(self.float_preview_window, bg='black')
        self.float_preview_label.pack(fill='both', expand=True)
        self.float_preview_label.bind('<Button-1>', self.start_drag_float_window)
        self.float_preview_label.bind('<B1-Motion>', self.drag_float_window)
        self.float_preview_label.bind('<Double-Button-1>', lambda e: self.toggle_float_preview())
        self.float_preview_label.bind('<Button-3>', self.show_float_context_menu)
        self.float_preview_label.focus_set()
        def delayed_image_update():
            if not hasattr(self, 'video_image') or len(self.video_image) > 0:
                    try:
                        from PIL import Image
                        pil_image = Image.fromarray(self.video_image)
                        self.update_float_preview_image(pil_image)
                    except Exception as e:
                        print(f'初始化悬浮窗口图像时出错: {e}')
        self.float_preview_window.after(100, delayed_image_update)
    def get_float_window_size(self):
        # irreducible cflow, using cdg fallback
        """根据视频画面比例计算悬浮窗口尺寸"""
        try:
            if hasattr(self, 'models') and hasattr(self.models, 'vm') and self.models.vm.capture and (not isinstance(self.models.vm.capture, list)):
                original_width = int(self.models.vm.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                original_height = int(self.models.vm.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if original_width > 0 and original_height > 0:
                    aspect_ratio = original_width / original_height
                    base_height = 300
                    window_width = int(base_height * aspect_ratio)
                    window_height = base_height
                    max_width = 800
                    min_width = 200
                    max_height = 600
                    min_height = 150
                    if window_width > max_width:
                        window_width = max_width
                        window_height = int(window_width / aspect_ratio)
                    else:
                        if window_width < min_width:
                            window_width = min_width
                            window_height = int(window_width / aspect_ratio)
                    if window_height > max_height:
                        window_height = max_height
                        window_width = int(window_height * aspect_ratio)
                    else:
                        if window_height < min_height:
                            window_height = min_height
                            window_width = int(window_height * aspect_ratio)
                    return (window_width, window_height)
                if hasattr(self, 'video_image'):
                    if len(self.video_image) > 0:
                        img_height, img_width = self.video_image.shape[:2]
                        if img_width > 0 and img_height > 0:
                                aspect_ratio = img_width / img_height
                                base_height = 300
                                window_width = int(base_height * aspect_ratio)
                                window_height = base_height
                                window_width = max(200, min(800, window_width))
                                window_height = max(150, min(600, window_height))
                                return (window_width, window_height)
                    else:
                        return (400, 225)
                else:
                    return (400, 225)
        except Exception as e:
                print(f'计算悬浮窗口尺寸时出错: {e}')
                return (400, 225)
    def close_float_preview_window(self):
        """关闭悬浮预览窗口"""
        if self.float_preview_window and self.float_preview_window.winfo_exists():
                self.float_preview_window.destroy()
        self.float_preview_window = None
        self.float_preview_label = None
    def start_drag_float_window(self, event):
        """开始拖拽悬浮窗口"""
        if self.float_preview_window and self.float_preview_window.winfo_exists():
                self.drag_start_x = event.x_root
                self.drag_start_y = event.y_root
    def drag_float_window(self, event):
        """拖拽悬浮窗口"""
        if self.float_preview_window and self.float_preview_window.winfo_exists() and hasattr(self, 'drag_start_x') and hasattr(self, 'drag_start_y'):
                        try:
                            x = self.float_preview_window.winfo_x() + event.x_root - self.drag_start_x
                            y = self.float_preview_window.winfo_y() + event.y_root - self.drag_start_y
                            self.float_preview_window.geometry(f'+{x}+{y}')
                            self.drag_start_x = event.x_root
                            self.drag_start_y = event.y_root
                        except Exception as e:
                            print(f'拖拽窗口时出错: {e}')
    def show_float_context_menu(self, event):
        """显示悬浮窗口的右键菜单"""
        try:
            import tkinter as tk
            context_menu = tk.Menu(self.float_preview_window, tearoff=0)
            context_menu.add_command(label='关闭悬浮窗口', command=self.toggle_float_preview)
            context_menu.add_separator()
            if hasattr(self, 'float_window_has_border') and self.float_window_has_border:
                context_menu.add_command(label='隐藏边框', command=self.toggle_float_window_border)
            else:
                context_menu.add_command(label='显示边框', command=self.toggle_float_window_border)
            context_menu.add_separator()
            context_menu.add_command(label='放大 (+10%)', command=lambda: self.scale_float_window(1.1))
            context_menu.add_command(label='缩小 (-10%)', command=lambda: self.scale_float_window(0.9))
            context_menu.add_separator()
            context_menu.add_command(label='调整为视频比例', command=self.resize_float_window_to_video_ratio)
            context_menu.add_separator()
            context_menu.add_command(label='小尺寸 (高度200)', command=lambda: self.resize_float_window_by_ratio(200))
            context_menu.add_command(label='中尺寸 (高度300)', command=lambda: self.resize_float_window_by_ratio(300))
            context_menu.add_command(label='大尺寸 (高度400)', command=lambda: self.resize_float_window_by_ratio(400))
            context_menu.tk_popup(event.x_root, event.y_root)
        except Exception as e:
            print(f'显示右键菜单时出错: {e}')
    def create_float_window_context_menu(self):
        """创建悬浮窗口的右键菜单（已废弃，使用show_float_context_menu）"""
        return
    def show_preview_context_menu(self, event):
        """显示主预览窗口的右键菜单"""
        try:
            import tkinter as tk
            context_menu = tk.Menu(self.video, tearoff=0)
            context_menu.add_command(label='旋转 90°', command=lambda: self.rotate_preview(90))
            context_menu.add_command(label='旋转 180°', command=lambda: self.rotate_preview(180))
            context_menu.add_command(label='旋转 270°', command=lambda: self.rotate_preview(270))
            context_menu.add_separator()
            context_menu.add_command(label='重置旋转', command=lambda: self.rotate_preview(0))
            context_menu.tk_popup(event.x_root, event.y_root)
        except Exception as e:
            print(f'显示预览窗口右键菜单时出错: {e}')
    def rotate_preview(self, angle):
        """旋转预览窗口画面"""
        try:
            old_angle = getattr(self, 'preview_rotation_angle', 0)
            self.preview_rotation_angle = angle
            if hasattr(self, 'models') and hasattr(self.models, 'vm'):
                    vm = self.models.vm
                    old_is_rotated = old_angle == 90 or old_angle == 270
                    new_is_rotated = angle == 90 or angle == 270
                    if old_is_rotated != new_is_rotated and vm.control.get('VirtualCameraSwitch', False):
                            vm.enable_virtualcam()
            self.resize_image()
        except Exception as e:
            print(f'旋转预览窗口时出错: {e}')
    def resize_float_window(self, width, height):
        """调整悬浮窗口大小"""
        if self.float_preview_window and self.float_preview_window.winfo_exists():
                x = self.float_preview_window.winfo_x()
                y = self.float_preview_window.winfo_y()
                self.float_preview_window.geometry(f'{width}x{height}+{x}+{y}')
    def resize_float_window_to_video_ratio(self):
        """调整悬浮窗口为视频原始比例"""
        if self.float_preview_window and self.float_preview_window.winfo_exists():
                try:
                    current_height = self.float_preview_window.winfo_height()
                    if current_height <= 1:
                        current_height = 300
                    new_width, new_height = self.get_float_window_size_by_height(current_height)
                    self.resize_float_window(new_width, new_height)
                except Exception as e:
                    print(f'调整悬浮窗口为视频比例时出错: {e}')
    def resize_float_window_by_ratio(self, target_height):
        """根据指定高度和视频比例调整悬浮窗口大小"""
        if self.float_preview_window and self.float_preview_window.winfo_exists():
                try:
                    new_width, new_height = self.get_float_window_size_by_height(target_height)
                    self.resize_float_window(new_width, new_height)
                except Exception as e:
                    print(f'按比例调整悬浮窗口大小时出错: {e}')
    def get_float_window_size_by_height(self, target_height):
        """根据目标高度和视频比例计算窗口尺寸"""
        try:
            aspect_ratio = None
            if hasattr(self, 'models') and hasattr(self.models, 'vm') and self.models.vm.capture and (not isinstance(self.models.vm.capture, list)):
                            original_width = int(self.models.vm.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                            original_height = int(self.models.vm.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            if original_width > 0 and original_height > 0:
                                    aspect_ratio = original_width / original_height
            if aspect_ratio is None and hasattr(self, 'video_image') and (len(self.video_image) > 0):
                        img_height, img_width = self.video_image.shape[:2]
                        if img_width > 0 and img_height > 0:
                                aspect_ratio = img_width / img_height
            if aspect_ratio is None:
                aspect_ratio = 1.7777777777777777
            target_width = int(target_height * aspect_ratio)
            target_width = max(200, min(800, target_width))
            target_height = max(150, min(600, target_height))
            return (target_width, target_height)
        except Exception as e:
            print(f'计算窗口尺寸时出错: {e}')
            return (int(target_height * 16 / 9), target_height)
    def toggle_float_window_border(self):
        """切换悬浮窗口边框显示/隐藏"""
        if self.float_preview_window and self.float_preview_window.winfo_exists():
                try:
                    self.float_window_has_border = not self.float_window_has_border
                    x = self.float_preview_window.winfo_x()
                    y = self.float_preview_window.winfo_y()
                    width = self.float_preview_window.winfo_width()
                    height = self.float_preview_window.winfo_height()
                    if self.float_window_has_border:
                        self.float_preview_window.overrideredirect(False)
                        self.float_preview_window.attributes('-topmost', True)
                        self.float_preview_window.title('悬浮预览')
                    else:
                        self.float_preview_window.overrideredirect(True)
                        self.float_preview_window.attributes('-topmost', True)
                    self.float_preview_window.geometry(f'{width}x{height}+{x}+{y}')
                except Exception as e:
                    print(f'切换窗口边框时出错: {e}')
    def scale_float_window(self, scale_factor):
        """按比例缩放悬浮窗口"""
        if self.float_preview_window and self.float_preview_window.winfo_exists():
                try:
                    current_width = self.float_preview_window.winfo_width()
                    current_height = self.float_preview_window.winfo_height()
                    new_width = int(current_width * scale_factor)
                    new_height = int(current_height * scale_factor)
                    min_width, min_height = (100, 75)
                    max_width, max_height = (1920, 1080)
                    new_width = max(min_width, min(new_width, max_width))
                    new_height = max(min_height, min(new_height, max_height))
                    x = self.float_preview_window.winfo_x()
                    y = self.float_preview_window.winfo_y()
                    self.float_preview_window.geometry(f'{new_width}x{new_height}+{x}+{y}')
                except Exception as e:
                    print(f'缩放悬浮窗口时出错: {e}')
    def update_float_preview_image(self, image):
        """更新悬浮窗口中的预览图像"""
        if self.float_preview_window and self.float_preview_window.winfo_exists() and self.float_preview_label:
                    try:
                        window_width = self.float_preview_window.winfo_width()
                        window_height = self.float_preview_window.winfo_height()
                        if window_width > 1 and window_height > 1:
                                from PIL import Image, ImageTk
                                if hasattr(image, 'resize'):
                                    img_width, img_height = image.size
                                    ratio = max(window_width / img_width, window_height / img_height)
                                    scaled_width = int(img_width * ratio)
                                    scaled_height = int(img_height * ratio)
                                    scaled_image = image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
                                    left = (scaled_width - window_width) // 2
                                    top = (scaled_height - window_height) // 2
                                    right = left + window_width
                                    bottom = top + window_height
                                    cropped_image = scaled_image.crop((left, top, right, bottom))
                                    photo = ImageTk.PhotoImage(cropped_image)
                                    self.float_preview_label.configure(image=photo)
                                    self.float_preview_label.image = photo
                    except Exception as e:
                        import logging
                        logging.error(f'更新悬浮预览窗口图像时出错: {e}')
    def parameter_io(self, task, initial_dir='.'):
        if task == 'save':
            save_file = filedialog.asksaveasfile(mode='w', encoding='utf-8', initialdir=initial_dir, initialfile='startup_parameters.json', defaultextension='.json', filetypes=[('JSON files', '*.json'), ('All files', '*.*')])
            if save_file:
                config_data = {'config_type': 'parameters', 'version': '1.0', 'parameters': self.parameters, 'parameters_face_editor': self.parameters_face_editor}
                json.dump(config_data, save_file, indent=4)
                save_file.close()
        else:
            if task == 'load':
                try:
                    load_file = filedialog.askopenfile(mode='r', encoding='utf-8', initialdir=initial_dir, filetypes=[('JSON files', '*.json'), ('All files', '*.*')])
                    if load_file:
                        config_data = json.load(load_file)
                        file_name = load_file.name
                        load_file.close()
                        if config_data.get('config_type') != 'parameters':
                            import logging
                            logging.error(f'错误: {file_name} 配置类型无效!')
                            return
                        else:
                            temp = config_data.get('parameters', {})
                            for key, value in temp.items():
                                if key in self.parameters:
                                    self.parameters[key] = value
                            temp = config_data.get('parameters_face_editor', {})
                            for key, value in temp.items():
                                if key in self.parameters_face_editor:
                                    self.parameters_face_editor[key] = value
                            self.update_ui_with_parameters()
                            self.add_action('parameters', self.parameters)
                            self.add_action('parameters_face_editor', self.parameters_face_editor)
                            self.add_action('control', self.control)
                            self.add_action('get_requested_video_frame', self.video_slider.get())
                except FileNotFoundError:
                    import logging
                    logging.warning('尚未创建保存文件!')
                except json.JSONDecodeError:
                    import logging
                    logging.error('JSON文件解码错误，请检查文件格式。')
            else:
                if task == 'default':
                    self.load_default_parameters()
                    self.add_action('parameters', self.parameters)
                    self.add_action('parameters_face_editor', self.parameters_face_editor)
                    self.add_action('control', self.control)
                    self.add_action('get_requested_video_frame', self.video_slider.get())
    def update_ui_with_parameters(self):
        for key, value in self.parameters.items():
            self.widget[key].set(value, request_frame=False)
            if key == 'ProvidersPriorityTextSel':
                provider_value = self.models.switch_providers_priority(value)
                if provider_value != value:
                    self.widget[key].set(provider_value, request_frame=False)
                else:
                    self.models.delete_models()
                    torch.cuda.empty_cache()
            else:
                if key == 'ThreadsSlider':
                    self.models.set_number_of_threads(value)
        for key, value in self.parameters_face_editor.items():
            self.widget[key].set(value, request_frame=False)
    def load_default_parameters(self):
        for key, value in self.parameters.items():
            self.widget[key].load_default()
        for key, value in self.parameters_face_editor.items():
            self.widget[key].load_default()
    def findCosineDistance2(self, vector1, vector2):
        cos_dist = 1.0 - np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        return cos_dist
    def toggle_virtualcam(self, mode, name, use_markers=False):
        self.control[name] = self.widget[name].get()
        self.add_action('control', self.control)
        if self.control[name]:
            self.add_action('enable_virtualcam')
        else:
            self.add_action('disable_virtualcam')
    def disable_record_button(self):
        self.widget['TLRecButton'].disable_button()
    def change_language(self, language):
        """切换界面语言"""
        try:
            original_language = self.current_language
            if messagebox.askokcancel(self.DEFAULT_DATA['Warning'], self.DEFAULT_DATA['LanguageSwitched']):
                with open(language_manager.data_json_path(), 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                self.current_language = original_language
                self.language_var.set(original_language)
                languages = {'中文': 'Chinese', 'English': 'English'}
                current_display_name = next((name for name, code in languages.items() if code == original_language), '中文')
                self.language_var.set(current_display_name)
            config['language'] = language
            with open(language_manager.data_json_path(), 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            self.current_language = language
            self.language_var.set(language)
            self.restart_application()
            return
        except Exception as e:
            messagebox.showerror(self.DEFAULT_DATA['Error'], self.DEFAULT_DATA['LanguageChangeError'] + str(e))
    def restart_application(self):
        """重启应用"""
        try:
            timestamp = datetime.now().strftime('%H%M%S')
            rand1 = str(rd.randint(1000, 9999))
            rand2 = str(rd.randint(1000, 9999))
            session_key = f'{rand1}{rand2}{timestamp}'
            launch_time = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
            os.environ['ROPE_SESSION'] = session_key
            os.environ['ROPE_LAUNCH_TIME'] = launch_time
            if getattr(sys, 'frozen', False):
                base_path = os.path.dirname(sys.executable)
                launcher_exe = os.path.join(base_path, 'Launcher.exe')
                launcher_bat = os.path.join(base_path, '启动.bat')
                if os.path.exists(launcher_exe):
                    subprocess.Popen([launcher_exe], shell=True, env=os.environ)
                else:
                    if os.path.exists(launcher_bat):
                        subprocess.Popen([launcher_bat], shell=True, env=os.environ)
                    else:
                        subprocess.Popen([sys.executable], shell=True, env=os.environ)
            else:
                base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                launcher_bat = os.path.join(base_path, '启动.bat')
                if os.path.exists(launcher_bat):
                    subprocess.Popen([launcher_bat], shell=True, env=os.environ)
                else:
                    python = sys.executable
                    os.execl(python, python, *sys.argv)
            self.quit()
            sys.exit(0)
        except Exception as e:
            import logging
            logging.error(f'重启失败: {e}')
            if getattr(sys, 'frozen', False):
                subprocess.Popen([sys.executable], shell=True, env=os.environ)
            else:
                python = sys.executable
                os.execl(python, python, *sys.argv)
            self.quit()
            sys.exit(0)
    def load_language_dict(self):
        """根据配置加载对应语言的字典"""
        try:
            with open(language_manager.data_json_path(), 'r', encoding='utf-8') as f:
                config = json.load(f)
                language = config.get('language', 'Chinese')
            if language == 'English':
                from . import Dicts_EN as Dicts
            else:
                if language == 'Chinese':
                    from . import Dicts_CN as Dicts
                else:
                    from . import Dicts_CN as Dicts
            self.DEFAULT_DATA = Dicts.DEFAULT_DATA
            self.PARAM_VARS = Dicts.PARAM_VARS
            self.PARAMS = Dicts.PARAMS
            self.CAMERA_BACKENDS = Dicts.CAMERA_BACKENDS
            current_dict = Dicts.DEFAULT_DATA
        except Exception as e:
            from . import Dicts_CN as Dicts
            self.DEFAULT_DATA = Dicts.DEFAULT_DATA
            self.PARAM_VARS = Dicts.PARAM_VARS
            self.PARAMS = Dicts.PARAMS
            self.CAMERA_BACKENDS = Dicts.CAMERA_BACKENDS
            current_dict = Dicts.DEFAULT_DATA
current_dict = None
def get_current_dict():
    """获取当前语言的字典"""
    return current_dict
if __name__ == '__main__':
    models = {}
    app = GUI(models)
    app.run()