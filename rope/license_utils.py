
import hashlib
from datetime import datetime, timedelta, timezone
import uuid
from pathlib import Path
import json
from rope.Dicts import DEFAULT_DATA
from typing import Tuple, Dict
from .license_online import check_license as online_check_license, validate_activation_code as online_validate_activation_code, get_machine_code as online_get_machine_code, get_remaining_days as online_get_remaining_days, LicenseClient
import requests
import winreg
import time
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import base64
from rope.config import SOFTWARE_VERSION
BASE_DIR = Path(__file__).resolve().parent.parent
LICENSE_SERVER_URL = 'https://license.invalid'
LICENSE_SERVER_FALLBACK_URL = 'https://license.invalid'
LICENSE_SERVER_URLS = [LICENSE_SERVER_URL, LICENSE_SERVER_FALLBACK_URL]
REGISTRY_KEY = 'Software\\Rope-LiveMe'
REGISTRY_VALUE = 'TrialInfo'
ENCRYPTION_KEY = hashlib.sha256(b'Rope-LiveMe2024').digest()[:16]
IV = b'Rope-LiveMeInit2024'[:16]
def encrypt_data(data: str) -> str:
    """加密数据"""
    try:
        cipher = AES.new(ENCRYPTION_KEY, AES.MODE_CBC, IV)
        padded_data = pad(data.encode(), AES.block_size)
        encrypted_data = cipher.encrypt(padded_data)
        return base64.b64encode(encrypted_data).decode('utf-8')
    except Exception:
        return None
def decrypt_data(encrypted_data: str) -> str:
    """解密数据"""
    try:
        encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
        cipher = AES.new(ENCRYPTION_KEY, AES.MODE_CBC, IV)
        decrypted_data = unpad(cipher.decrypt(encrypted_bytes), AES.block_size)
        return decrypted_data.decode('utf-8')
    except Exception:
        return None
def get_local_timezone():
    """获取本地时区"""
    local_offset = -time.timezone
    hours = local_offset // 3600
    minutes = abs(local_offset % 3600 // 60)
    return timezone(timedelta(hours=hours, minutes=minutes))
def get_aware_datetime(dt_str):
    """将时间字符串转换为本地时区的 datetime 对象"""
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(get_local_timezone())
    except Exception:
        return datetime.now(get_local_timezone())
def get_current_time():
    """获取当前本地时间"""
    return datetime.now(get_local_timezone())
def post_to_license_server(path: str, data: Dict, timeout: int):
    """优先请求新许可证服务，失败后回退到旧地址。"""
    headers = {'Content-Type': 'application/json'}
    last_exception = None
    last_response = None
    for base_url in LICENSE_SERVER_URLS:
        url = f"{base_url.rstrip('/')}{path}"
        try:
            try:
                response = requests.post(url, json=data, headers=headers, timeout=timeout, proxies={'http': None, 'https': None})
            except requests.exceptions.RequestException:
                response = requests.post(url, json=data, headers=headers, timeout=timeout)
            if response.status_code in [404, 500, 502, 503, 504]:
                last_response = response
            else:
                return response
        except Exception as e:
            last_exception = e
    if last_response is not None:
        return last_response
    else:
        if last_exception is not None:
            raise last_exception
        else:
            raise RuntimeError('未能连接到任何许可证服务')
def get_registry_trial_info():
    """从注册表获取试用信息"""
    try:
        key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, REGISTRY_KEY)
        encrypted_value, _ = winreg.QueryValueEx(key, REGISTRY_VALUE)
        winreg.CloseKey(key)
        decrypted_data = decrypt_data(encrypted_value)
        if decrypted_data:
            return json.loads(decrypted_data)
    except:
        return None
def save_registry_trial_info(trial_info):
    """保存试用信息到注册表"""
    try:
        json_data = json.dumps(trial_info)
        encrypted_data = encrypt_data(json_data)
        if not encrypted_data:
            return False
        else:
            key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, REGISTRY_KEY)
            winreg.SetValueEx(key, REGISTRY_VALUE, 0, winreg.REG_SZ, encrypted_data)
            winreg.CloseKey(key)
            return True
    except Exception:
        return False
def verify_trial_with_server(machine_id):
    """在线验证试用状态"""
    try:
        data = {'machine_id': machine_id}
        response = post_to_license_server('/api/trial/verify/', data, 15)
        if response.status_code == 200:
            result = response.json()
            if result.get('is_trial'):
                return (True, result)
            else:
                return (False, None)
        if response.status_code == 403:
            try:
                reg_trial_info = get_registry_trial_info()
                if reg_trial_info:
                    reg_trial_info['is_disabled'] = True
                    save_registry_trial_info(reg_trial_info)
            except Exception:
                pass
            return (False, None)
        return (False, None)
    except Exception:
        return (False, None)


def create_trial_with_server(machine_id):
    """在服务器创建试用记录"""
    try:
        current_time = get_current_time()
        start_time = current_time.isoformat()
        trial_days = 1
        data = {
            'machine_id': machine_id,
            'start_date': start_time,
            'trial_days': trial_days,
        }
        response = post_to_license_server('/api/trial/create/', data, 5)
        if response.status_code == 200:
            result = response.json()
            return result
        return None
    except Exception as e:
        return None

def check_trial_status():
    """检查试用状态"""
    try:
        machine_id = get_machine_code()
        reg_trial_info = get_registry_trial_info()
        if reg_trial_info:
            if reg_trial_info.get('is_disabled', False):
                return False
            if reg_trial_info.get('machine_id') != machine_id:
                return False
            start_date = get_aware_datetime(reg_trial_info['start_date'])
            trial_period = timedelta(days=reg_trial_info.get('trial_days', 1))
            current_time = get_current_time()
            start_date = start_date.astimezone(get_local_timezone())
            current_time = current_time.astimezone(get_local_timezone())
            end_date = (start_date + trial_period).astimezone(get_local_timezone())
            if current_time <= end_date:
                try:
                    success, server_trial = verify_trial_with_server(machine_id)
                    if not success:
                        return False
                except Exception as e:
                    return True
                return True
            return False
        try:
            success, server_trial = verify_trial_with_server(machine_id)
            if success:
                trial_info = {'machine_id': machine_id, 'start_date': server_trial['start_date'],
                              'trial_days': server_trial.get('trial_days', 1), 'is_disabled': False}
                save_registry_trial_info(trial_info)
                return True
            trial_info = create_trial_with_server(machine_id)
            if trial_info:
                save_registry_trial_info({'machine_id': machine_id, 'start_date': trial_info['start_date'],
                                          'trial_days': trial_info.get('trial_days', 1), 'is_disabled': False})
                return True
            return False
        except Exception as e:
            return False
    except Exception as e:
        return False



def check_license(version: str=SOFTWARE_VERSION) -> bool:
    """检查许可证是否有效"""
    try:
        client = LicenseClient(LICENSE_SERVER_URL, version=version)
        success, message = client.check_license()
        if success:
            return True
        else:
            print(f'许可证验证失败: {message}')
            if check_trial_status():
                return True
            else:
                return False
    except Exception as e:
        print(f'许可证验证过程发生错误: {str(e)}')
        return False
def validate_activation_code(activation_code: str, version: str=SOFTWARE_VERSION) -> Tuple[bool, Dict]:
    try:
        client = LicenseClient(LICENSE_SERVER_URL, version=version)
        success, message = client.activate_license(activation_code)
        if success:
            return (True, {'valid': True, 'message': message, 'validity_period': client.load_license_info().get('validity_period', '1month')})
        else:
            return (False, message)
    except Exception:
        return (False, '激活失败，请联系客服处理')
def get_machine_code() -> str:
    """获取机器码"""
    client = LicenseClient(LICENSE_SERVER_URL)
    return client.get_machine_id()
def get_remaining_days(version: str=SOFTWARE_VERSION) -> int:
    """获取剩余天数"""
    client = LicenseClient(LICENSE_SERVER_URL, version=version)
    return client.get_remaining_days()
def get_license_info(version: str=SOFTWARE_VERSION):
    """获取许可证信息"""
    try:
        client = LicenseClient(LICENSE_SERVER_URL, version=version)
        license_info = client.load_license_info()
        if license_info and license_info.get('expires_at'):
            return license_info
        else:
            trial_info = get_registry_trial_info()
            if trial_info:
                start_date = get_aware_datetime(trial_info['start_date'])
                trial_period = timedelta(days=trial_info.get('trial_days', 1))
                current_time = get_current_time()
                expiry_date = start_date + trial_period
                start_date = start_date.astimezone(get_local_timezone())
                current_time = current_time.astimezone(get_local_timezone())
                expiry_date = expiry_date.astimezone(get_local_timezone())
                if current_time > expiry_date:
                    return
                else:
                    expiry_date_str = expiry_date.strftime('%Y-%m-%d')
                    start_date_str = start_date.strftime('%Y-%m-%d')
                    result = {'expires_at': expiry_date.isoformat(), 'is_trial': True, 'trial_days': trial_info.get('trial_days', 1), 'start_date': start_date.isoformat(), 'display_expires_at': expiry_date_str, 'display_start_date': start_date_str}
                    return result
    except Exception as e:
        return None
__all__ = ['check_license', 'validate_activation_code', 'get_machine_code', 'get_remaining_days', 'get_license_info', 'check_trial_status']