
"""\nGUI公告集成模块 - 简化版本，直接替换GUI.py中的公告代码\n保持原有的显示位置和结构，只是将代码模块化\n"""
import threading
import time
import webbrowser
from .logger_config import get_logger
from .announcement_fetcher import fetch_announcement, parse_announcements
logger = get_logger('GUIAnnouncement')
class GUIAnnouncementHelper:
    """\n    GUI公告助手 - 直接集成到现有GUI结构中\n    """
    def __init__(self, gui_instance, announcement_url):
        """\n        初始化公告助手\n        \n        Args:\n            gui_instance: GUI实例\n            announcement_url: 公告URL\n        """
        self.gui = gui_instance
        self.announcement_url = announcement_url
        self.software_version = self.gui.software_version
        if self.software_version == 'rope-plus':
            self.announcements = ['电报: @ttt87877']
            self.announcement_urls = ['https://example.com/community']
        else:
            self.announcements = ['公告: https://example.com/announcement']
            self.announcement_urls = ['https://example.com/announcement']
        self.current_announcement_index = 0
        self._announcement_updating = False
        self._announcement_timer_set = False
        self.scroll_step = 1
        self.scroll_delay = 50
        self.switch_interval = 10000
        self.update_interval = 300000
    def update_announcements(self):
        """\n        更新公告内容（异步执行避免界面卡死）\n        """
        if self._announcement_updating:
            return
        else:
            self._announcement_updating = True
            thread = threading.Thread(target=self._fetch_announcements_async, daemon=True)
            thread.start()
    def _fetch_announcements_async(self):
        # irreducible cflow, using cdg fallback
        """\n        异步获取公告内容\n        """
        try:
            success, content = fetch_announcement(self.announcement_url)
            if success:
                announcements, announcement_urls = parse_announcements(content)
                self._safe_update_ui(announcements, announcement_urls)
            else:
                error_announcements = [content]
                self._safe_update_ui(error_announcements, [''])
        except Exception as e:
            self._safe_update_ui(['公告功能暂时不可用'], [''])
            self._announcement_updating = False
    def _safe_update_ui(self, announcements, announcement_urls):
        """\n        线程安全的UI更新 - 直接调用GUI的方法\n        """
        try:
            if hasattr(self.gui, 'winfo_exists') and self.gui.winfo_exists():
                self.gui.after_idle(lambda: self._update_announcement_ui(announcements, announcement_urls))
            else:
                logger.warning('GUI已销毁，跳过UI更新')
        except Exception as e:
            return None
    def _update_announcement_ui(self, announcements, announcement_urls):
        """\n        在主线程中更新公告UI\n        """
        try:
            self.announcements = announcements
            self.announcement_urls = announcement_urls
            self.current_announcement_index = 0
            self._update_current_announcement_display()
            self.start_announcement_switching()
            if not self._announcement_timer_set:
                self._announcement_timer_set = True
                self.gui.after(self.update_interval, self._schedule_next_update)
        except Exception as e:
            return None
    def _update_current_announcement_display(self):
        """\n        更新当前显示的公告\n        """
        try:
            if not hasattr(self.gui, 'status_canvas') or not hasattr(self.gui, 'announcement_text'):
                return None
            else:
                current_text = self.announcements[self.current_announcement_index]
                self.gui.status_canvas.itemconfig(self.gui.announcement_text, text=current_text)
                self.gui.text_pos = self.gui.status_canvas.winfo_width() if hasattr(self.gui, 'status_canvas') else 0
                current_url = self.announcement_urls[self.current_announcement_index] if self.announcement_urls else ''
                self.gui.status_canvas.unbind('<Button-1>')
                self.gui.status_canvas.unbind('<Enter>')
                self.gui.status_canvas.unbind('<Leave>')
                if current_url:
                    self.gui.status_canvas.bind('<Button-1>', lambda e: self._open_url(current_url))
                    self.gui.status_canvas.bind('<Enter>', lambda e: self.gui.status_canvas.config(cursor='hand2'))
                    self.gui.status_canvas.bind('<Leave>', lambda e: self.gui.status_canvas.config(cursor=''))
                else:
                    self.gui.status_canvas.config(cursor='')
        except Exception as e:
            return None
    def _open_url(self, url):
        """\n        打开URL链接\n        """
        try:
            webbrowser.open(url)
        except Exception as e:
            return None
    def animate_scroll(self):
        """\n        动画滚动文本\n        """
        try:
            if not hasattr(self.gui, 'status_canvas') or not hasattr(self.gui, 'announcement_text'):
                self.gui.after(self.scroll_delay, self.animate_scroll)
                return
            else:
                canvas_width = self.gui.status_canvas.winfo_width()
                bbox = self.gui.status_canvas.bbox(self.gui.announcement_text)
                text_width = bbox[2] - bbox[0] if bbox else 0
                if text_width > canvas_width:
                    if not hasattr(self.gui, 'text_pos'):
                        self.gui.text_pos = canvas_width
                    self.gui.text_pos -= self.scroll_step
                    if self.gui.text_pos < -text_width:
                        self.gui.text_pos = canvas_width
                    self.gui.status_canvas.coords(self.gui.announcement_text, self.gui.text_pos, 10)
                else:
                    center_x = canvas_width // 2
                    self.gui.status_canvas.coords(self.gui.announcement_text, center_x, 10)
                    self.gui.status_canvas.itemconfig(self.gui.announcement_text, anchor='center')
                self.gui.after(self.scroll_delay, self.animate_scroll)
        except Exception as e:
            self.gui.after(self.scroll_delay, self.animate_scroll)
    def start_announcement_switching(self):
        """\n        启动公告切换系统\n        """
        if len(self.announcements) > 1:
            self.gui.after(self.switch_interval, self._switch_to_next_announcement)
    def _switch_to_next_announcement(self):
        """\n        切换到下一条公告\n        """
        try:
            if len(self.announcements) > 1:
                self.current_announcement_index = (self.current_announcement_index + 1) % len(self.announcements)
                self._update_current_announcement_display()
            self.gui.after(self.switch_interval, self._switch_to_next_announcement)
        except Exception as e:
            self.gui.after(self.switch_interval, self._switch_to_next_announcement)
    def _schedule_next_update(self):
        """\n        调度下一次远程更新\n        """
        try:
            self._announcement_updating = False
            self.update_announcements()
            self.gui.after(self.update_interval, self._schedule_next_update)
        except Exception as e:
            return None
def create_announcement_helper(gui_instance, software_version='rope'):
    """\n    创建公告助手实例\n    \n    Args:\n        gui_instance: GUI实例\n        software_version: 软件版本\n        \n    Returns:\n        GUIAnnouncementHelper实例\n    """
    if software_version == 'rope-plus':
        announcement_url = 'https://example.com/rope-announcement.txt'
    else:
        announcement_url = 'https://example.com/announcement.txt'
    return GUIAnnouncementHelper(gui_instance, announcement_url)