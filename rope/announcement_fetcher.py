# Rebuilt from announcement_fetcher.pyc
"""公告获取模块 - 专门处理网络请求问题"""
import ssl
import time
import urllib.error
import urllib.request


def _is_plus_tier():
    """The vendor build referenced a global `self` here; keep it safe."""
    obj = globals().get('self')
    return getattr(obj, 'software_version', None) == 'rope-plus'


def fetch_announcement(url, max_retries=2):
    """获取公告内容，多种方法确保成功"""
    try:
        import requests
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/plain,text/html,*/*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Connection': 'close',
            'Cache-Control': 'no-cache',
        }
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=10, verify=False,
                                        proxies={'http': None, 'https': None})
                if response.status_code == 200:
                    return (True, response.text)
                raise Exception(f'HTTP {response.status_code}')
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    break
    except ImportError:
        pass
    try:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/plain,text/html,*/*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Connection': 'close',
        }
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(url, headers=headers)
                proxy_handler = urllib.request.ProxyHandler({})
                https_handler = urllib.request.HTTPSHandler(context=ssl_context)
                opener = urllib.request.build_opener(proxy_handler, https_handler)
                response = opener.open(req, timeout=10)
                content = response.read().decode('utf-8')
                response.close()
                return (True, content)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
    except Exception:
        pass
    if _is_plus_tier():
        return (False, '电报: @ttt87877')
    return (False, '公告: https://example.com/announcement')


def parse_announcements(content):
    """解析公告内容"""
    if not content or not content.strip():
        if _is_plus_tier():
            return (['电报: @ttt87877'], ['https://example.com/community'])
        return (['公告: https://example.com/announcement'], ['https://example.com/announcement'])
    lines = content.strip().split('\n')
    announcements = []
    announcement_urls = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if '|' in line:
            parts = line.split('|', 1)
            announcements.append(parts[0].strip())
            announcement_urls.append(parts[1].strip())
            continue
        announcements.append(line)
        announcement_urls.append('')
    if not announcements:
        if _is_plus_tier():
            announcements = ['电报: @ttt87877']
            announcement_urls = ['https://example.com/community']
        else:
            announcements = ['公告: https://example.com/announcement']
            announcement_urls = ['https://example.com/announcement']
    return (announcements, announcement_urls)
