
import requests
import json
from datetime import datetime
from pathlib import Path
import uuid
import hashlib
from typing import Optional, Tuple, Dict
import os
from cryptography.fernet import Fernet
from rope.language_manager import load_language_dict
import base64
import platform
import sys
import traceback
import logging
from rope.config import SOFTWARE_VERSION
_LOG_DIR = Path(__file__).resolve().parent.parent / 'logs'
try:
    os.makedirs(_LOG_DIR, exist_ok=True)
    _license_logger = logging.getLogger('license_client')
    if not _license_logger.handlers:
        _license_logger.setLevel(logging.DEBUG)
        _handler = logging.FileHandler(_LOG_DIR / 'license_client.log', encoding='utf-8')
        _handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        _license_logger.addHandler(_handler)
except Exception:
    _license_logger = logging.getLogger('license_client_fallback')
    _license_logger.addHandler(logging.NullHandler())
_SERVER_DOMAINS = ('license.invalid',)
def _scrub_log(text) -> str:
    """把日志中的服务器域名替换为占位符"""
    s = str(text)
    for domain in _SERVER_DOMAINS:
        s = s.replace(domain, '[license-server]')
    return s
def _safe_json(response) -> Optional[Dict]:
    """安全解析响应JSON，失败时返回None并记录日志"""
    try:
        return response.json()
    except Exception as e:
        try:
            body_preview = response.text[:500]
        except Exception:
            body_preview = '<无法读取响应体>'
        _license_logger.error(f"响应JSON解析失败: HTTP {getattr(response, 'status_code', '?')} url={_scrub_log(getattr(response, 'url', '?'))} error={_scrub_log(e)} body={_scrub_log(body_preview)}")
Dicts = load_language_dict()
DEFAULT_DATA = Dicts.DEFAULT_DATA
PRIMARY_LICENSE_SERVER_URL = 'https://license.invalid'
FALLBACK_LICENSE_SERVER_URL = 'https://license.invalid'
class LicenseClient:
    def __init__(self, server_url: str=PRIMARY_LICENSE_SERVER_URL, version: str=SOFTWARE_VERSION):
        try:
            if isinstance(server_url, (list, tuple)):
                raw_server_urls = list(server_url)
            else:
                raw_server_urls = [server_url, FALLBACK_LICENSE_SERVER_URL]
            self.server_urls = []
            for url in raw_server_urls:
                normalized_url = str(url).rstrip('/')
                if normalized_url and normalized_url not in self.server_urls:
                        self.server_urls.append(normalized_url)
            if not self.server_urls:
                self.server_urls = [PRIMARY_LICENSE_SERVER_URL, FALLBACK_LICENSE_SERVER_URL]
            self.server_url = self.server_urls[0]
            self.version = version
            self.license_file = Path(__file__).resolve().parent.parent / 'tools' / 'license.enc'
            os.makedirs(self.license_file.parent, exist_ok=True)
            self._key = base64.urlsafe_b64encode(hashlib.sha256(b'1your-secret-key').digest())
            self._fernet = Fernet(self._key)
        except Exception as e:
            raise
    def get_machine_id(self) -> str:
        """获取机器ID"""
        system_info = [platform.machine(), platform.processor()]
        try:
            drive = os.environ['SystemDrive']
            volume_info = os.popen(f'vol {drive}').read()
            if 'Serial Number' in volume_info:
                disk_serial = volume_info.split('Serial Number is ')[1].split('\n')[0]
                system_info.append(disk_serial)
        except:
            mac = ':'.join(['{:02x}'.format(uuid.getnode() >> elements & 255) for elements in range(0, 12, 2)][::(-1)])
            system_info.append(mac)
        machine_id = '+'.join(filter(None, system_info))
        return hashlib.md5(machine_id.encode()).hexdigest()
    def _post_with_fallback(self, path: str, json_data: Dict, headers: Dict, timeout: int, session=None):
        """优先请求新地址，失败后自动回退到旧地址。"""
        last_exception = None
        last_response = None
        for base_url in self.server_urls:
            request_url = f'{base_url}{path}'
            try:
                if session is not None:
                    response = session.post(request_url, json=json_data, headers=headers, timeout=timeout)
                else:
                    response = requests.post(request_url, json=json_data, headers=headers, timeout=timeout)
                if response.status_code in [404, 500, 502, 503, 504]:
                    last_response = response
                else:
                    self.server_url = base_url
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
    def save_license_info(self, license_info: Dict) -> bool:
        """保存许可证信息到文件"""
        try:
            sign_data = f"{license_info['license_code']}{license_info['machine_id']}{license_info['expires_at']}"
            signature = hashlib.sha256(sign_data.encode()).hexdigest()
            license_info['signature'] = signature
            encrypted_data = self._fernet.encrypt(json.dumps(license_info).encode())
            with open(self.license_file, 'wb') as f:
                f.write(encrypted_data)
        except Exception as e:
            return False
    def load_license_info(self, ignore_expiry: bool=False) -> Optional[Dict]:
        """加载许可证信息

        ignore_expiry=True 时即使本地已过期也返回信息，
        用于过期后在线验证服务器是否已续期。
        """
        try:
            if not self.license_file.exists():
                return None
            encrypted_data = self.license_file.read_text(encoding='utf-8')
            try:
                decrypted_data = self._fernet.decrypt(encrypted_data.encode()).decode()
                license_info = json.loads(decrypted_data)
                if not self.verify_license_signature(license_info):
                    return None
                if not ignore_expiry and license_info.get('expires_at'):
                    from datetime import datetime, timezone
                    import sys
                    sys.path.append('..')
                    from rope.license_utils import get_aware_datetime, get_local_timezone, get_current_time
                    expires_at = get_aware_datetime(license_info['expires_at'])
                    current_time = get_current_time()
                    expires_at = expires_at.astimezone(get_local_timezone())
                    current_time = current_time.astimezone(get_local_timezone())
                    if current_time > expires_at and license_info.get('validity_period') != 'permanent':
                        return None
                return license_info
            except Exception as e:
                return None
        except Exception as e:
            return None


    def verify_license_signature(self, license_info: Dict) -> bool:
        """验证许可证签名"""
        try:
            sign_data = f"{license_info['license_code']}{license_info['machine_id']}{license_info['expires_at']}"
            expected_signature = hashlib.sha256(sign_data.encode()).hexdigest()
            return True
        except Exception as e:
            return False
    def _try_online_refresh(self, license_info: Dict) -> Optional[Dict]:
        """本地许可证已过期时，尝试在线验证：若服务器端已续期，则更新本地文件并返回更新后的信息。\n        验证失败（未续期/机器不匹配/网络异常等）返回 None。"""
        try:
            current_machine_id = self.get_machine_id()
            if current_machine_id != license_info.get('machine_id'):
                return
            else:
                headers = {'Content-Type': 'application/json'}
                data = {'license_code': license_info['license_code'], 'machine_id': current_machine_id, 'version': self.version}
                response = self._post_with_fallback('/api/licenses/verify/', data, headers, 5)
                if response.status_code != 200:
                    return
                else:
                    result = _safe_json(response)
                    if not result or not result.get('valid'):
                        return None
                    else:
                        license_version = result.get('version')
                        if license_version and license_version != self.version:
                            return
                        else:
                            old_expires = license_info.get('expires_at')
                            if result.get('expires_at'):
                                license_info['expires_at'] = result['expires_at']
                            if result.get('validity_period'):
                                license_info['validity_period'] = result['validity_period']
                            license_info['last_verified'] = datetime.now().isoformat()
                            if self.save_license_info(license_info):
                                _license_logger.info(f"本地许可证已过期，但服务器验证有效（已续期），本地文件已更新: expires_at {old_expires} -> {license_info.get('expires_at')}")
                                return license_info
        except Exception as e:
            _license_logger.warning(f'过期许可证在线刷新失败: {_scrub_log(e)}')
            return None
    def check_license(self) -> Tuple[bool, str]:
        """检查许可证是否有效"""
        try:
            license_info = self.load_license_info()
            if not license_info:
                raw_info = self.load_license_info(ignore_expiry=True)
                if raw_info and raw_info.get('license_code') and raw_info.get('machine_id') and self._try_online_refresh(raw_info):
                    return (True, DEFAULT_DATA['LicenseValid'])
                return (False, DEFAULT_DATA['NoLicenseInfo'])
            if not license_info.get('license_code') or not license_info.get('machine_id'):
                return (False, DEFAULT_DATA['IncompleteLicenseInfo'])
            current_machine_id = self.get_machine_id()
            if current_machine_id != license_info['machine_id']:
                return (False, DEFAULT_DATA['LicenseMachineIDMismatch'])
            license_version = license_info.get('version')
            if license_version and license_version != self.version:
                return (False, DEFAULT_DATA['LicenseVersionMismatch'])
            if license_info.get('expires_at'):
                expires_at = datetime.fromisoformat(license_info['expires_at'].replace('Z', '+00:00'))
                if datetime.now().astimezone() > expires_at:
                    return (False, DEFAULT_DATA['LicenseExpired'])
            last_verified = license_info.get('last_verified')
            if not last_verified or (datetime.now() - datetime.fromisoformat(last_verified)).days >= 1:
                url = f'{self.server_url}/api/licenses/verify/'
                headers = {'Content-Type': 'application/json'}
                data = {
                    'license_code': license_info['license_code'],
                    'machine_id': current_machine_id,
                    'version': self.version,
                }
                try:
                    response = self._post_with_fallback(
                        '/api/licenses/verify/',
                        data,
                        headers,
                        5,
                    )
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('valid'):
                            license_version = result.get('version')
                            if license_version and license_version != self.version:
                                return (False, DEFAULT_DATA['LicenseVersionMismatch'])
                            license_info['last_verified'] = datetime.now().isoformat()
                            if result.get('expires_at'):
                                if result['expires_at'] != license_info.get('expires_at'):
                                    _license_logger.info(
                                        f"许可证过期时间已更新: {license_info.get('expires_at')} -> {result['expires_at']}")
                                license_info['expires_at'] = result['expires_at']
                            if result.get('validity_period'):
                                license_info['validity_period'] = result['validity_period']
                            self.save_license_info(license_info)
                            return (True, DEFAULT_DATA['LicenseValid'])
                        error_message = result.get('message', DEFAULT_DATA['LicenseInvalid'])
                        return (False, error_message)
                except Exception:
                    pass
            return (True, DEFAULT_DATA['LicenseValid'])
        except Exception as e:
            return (False, str(e))

    def get_remaining_days(self) -> int:
        """获取剩余天数"""
        try:
            license_info = self.load_license_info()
            if license_info and license_info.get('expires_at'):
                if license_info.get('validity_period') == 'permanent' or license_info.get('is_permanent', False):
                    return 9999
                expires_at_str = license_info['expires_at']
                from datetime import datetime, timedelta, timezone
                import sys
                sys.path.append('..')
                from rope.license_utils import get_local_timezone
                try:
                    expires_at = datetime.fromisoformat(expires_at_str.replace('Z', '+00:00'))
                except ValueError:
                    try:
                        expires_at = datetime.strptime(expires_at_str, '%Y-%m-%d')
                        expires_at = expires_at.replace(hour=23, minute=59, second=59)
                        expires_at = expires_at.astimezone(get_local_timezone())
                    except ValueError:
                        return 0
                now = datetime.now().astimezone(get_local_timezone())
                expires_at = expires_at.astimezone(get_local_timezone())
                remaining = expires_at - now
                return max(0, remaining.days)
            else:
                import sys
                sys.path.append('..')
                from rope.license_utils import get_registry_trial_info, get_aware_datetime, get_local_timezone, get_current_time
                from datetime import datetime, timedelta
                trial_info = get_registry_trial_info()
                if not trial_info:
                    return 0
                start_date = get_aware_datetime(trial_info['start_date'])
                trial_period = timedelta(days=trial_info.get('trial_days', 1))
                end_date = start_date + trial_period
                current_time = get_current_time()
                start_date = start_date.astimezone(get_local_timezone())
                current_time = current_time.astimezone(get_local_timezone())
                end_date = end_date.astimezone(get_local_timezone())
                remaining = end_date - current_time
                return max(0, remaining.days)
        except Exception as e:
            return 0


    def is_in_trial_period(self) -> bool:
        """检查是否在试用期内"""
        try:
            from ..license_utils import check_trial_status
            return check_trial_status()
        except Exception as e:
            return False
    def _preflight_network_check(self) -> Optional[str]:
        """激活前网络自检：DNS解析 + TCP连通性。\n        返回 None 表示网络正常；否则返回给用户的友好提示语。"""
        import socket
        dns_fail_count = 0
        for base_url in self.server_urls:
            host = base_url.split('://')[(-1)].split('/')[0]
            try:
                socket.getaddrinfo(host, 443, socket.AF_UNSPEC, socket.SOCK_STREAM)
            except socket.gaierror as e:
                dns_fail_count += 1
                _license_logger.warning(f'网络自检: DNS解析失败 {_scrub_log(host)}: {_scrub_log(e)}')
            else:
                try:
                    with socket.create_connection((host, 443), timeout=5):
                        _license_logger.info(f'网络自检通过: {_scrub_log(host)}')
                        return
                except OSError as e:
                    _license_logger.warning(f'网络自检: TCP连接失败 {_scrub_log(host)}: {_scrub_log(e)}')
        if dns_fail_count == len(self.server_urls):
            return '连接服务器失败，请检查DNS设置或关闭代理/VPN后重试'
        else:
            return '连接服务器失败，请检查网络或关闭代理/VPN后重试'
    def activate_license(self, license_code: str) -> Tuple[bool, Dict]:
        """激活许可证"""
        try:
            import re
            original_code = license_code
            license_code = re.sub('[^0-9a-zA-Z\\-]', '', str(license_code).strip())
            if license_code != original_code.strip():
                _license_logger.warning(
                    f'激活码包含非常规字符，已清洗: 原始长度={len(original_code)}, 清洗后={license_code}')
            _license_logger.info(f'开始激活: version={self.version}, machine_id={self.get_machine_id()}')
            network_error = self._preflight_network_check()
            if network_error:
                _license_logger.warning(f'激活前网络自检未通过: {network_error}')
                return (False, network_error)
            session = requests.Session()
            session.verify = True
            session.trust_env = True
            no_proxy_session = None
            url = f'{self.server_url}/api/licenses/activate/'
            headers = {'Content-Type': 'application/json'}
            current_machine_id = self.get_machine_id()
            verify_url = f'{self.server_url}/api/licenses/verify_status/'
            verify_data = {
                'license_code': license_code.strip(),
                'version': self.version,
            }
            try:
                verify_response = self._post_with_fallback(
                    '/api/licenses/verify_status/',
                    verify_data,
                    headers,
                    10,
                    session=session,
                )
            except Exception as e:
                _license_logger.warning(f'代理会话请求verify_status失败: {_scrub_log(e)}，尝试无代理会话')
                no_proxy_session = requests.Session()
                no_proxy_session.verify = True
                no_proxy_session.trust_env = False
                no_proxy_session.proxies = {'http': '', 'https': ''}
                try:
                    verify_response = self._post_with_fallback(
                        '/api/licenses/verify_status/',
                        verify_data,
                        headers,
                        15,
                        session=no_proxy_session,
                    )
                except Exception as inner_e:
                    _license_logger.error(
                        f'verify_status连接失败(含无代理回退): {_scrub_log(inner_e)}\n{_scrub_log(traceback.format_exc())}')
                    return (False, {'message': '无法连接到许可证服务器，请检查网络后重试', 'error': 'CONNECTION_ERROR'})
            active_session = no_proxy_session if no_proxy_session else session
            if verify_response.status_code == 200:
                verify_result = _safe_json(verify_response)
                if verify_result is None:
                    return (False, '许可证服务器响应异常，请稍后重试或联系客服（日志见 logs/license_client.log）')
                license_version = verify_result.get('version')
                if verify_result.get('is_activated'):
                    if verify_result.get('machine_id') == current_machine_id:
                        if license_version == self.version or not license_version:
                            pass
                        else:
                            _license_logger.warning(
                                f'许可证版本不匹配: 当前版本 {self.version}, 许可证版本 {license_version}')
                            return (False, DEFAULT_DATA['LicenseVersionMismatch'])
                    else:
                        _license_logger.warning(
                            f"许可证已在其他机器上激活: 当前机器ID {current_machine_id}, 许可证机器ID {verify_result.get('machine_id')}")
                        return (False, DEFAULT_DATA['LicenseUsedOnOtherMachine'])
                elif verify_result.get('is_expired'):
                    _license_logger.warning(f"许可证已过期: {verify_result.get('expires_at')}")
                    return (False, DEFAULT_DATA['LicenseExpired'])
                if license_version and license_version != self.version:
                    _license_logger.warning(f'许可证版本不匹配: 当前版本 {self.version}, 许可证版本 {license_version}')
                    return (False, DEFAULT_DATA['LicenseVersionMismatch'])
                data = {
                    'license_code': license_code.strip(),
                    'machine_id': current_machine_id,
                    'version': self.version,
                    'action': 'activate',
                }
                response = self._post_with_fallback(
                    url,
                    data,
                    headers,
                    15,
                    session=active_session,
                )
                if response.status_code == 200:
                    result = _safe_json(response)
                    if result is None:
                        return (False, '许可证服务器响应异常，请稍后重试或联系客服（日志见 logs/license_client.log）')
                    if not result.get('valid'):
                        error_message = result.get('message', DEFAULT_DATA['LicenseActivationError'])
                        _license_logger.warning(f'激活被服务器拒绝: {_scrub_log(error_message)} (响应: {_scrub_log(result)})')
                        return (False, error_message)
                    if result.get('machine_id') != current_machine_id:
                        _license_logger.warning(
                            f"许可证已在其他机器上激活: 当前机器ID {current_machine_id}, 许可证机器ID {result.get('machine_id')}")
                        return (False, DEFAULT_DATA['LicenseUsedOnOtherMachine'])
                    license_version = result.get('version')
                    if license_version and license_version != self.version:
                        _license_logger.warning(f'许可证版本不匹配: 当前版本 {self.version}, 许可证版本 {license_version}')
                        return (False, DEFAULT_DATA['LicenseVersionMismatch'])
                    license_info = {
                        'license_code': license_code,
                        'machine_id': current_machine_id,
                        'expires_at': result.get('expires_at'),
                        'status': result.get('status'),
                        'activated_at': datetime.now().isoformat(),
                        'validity_period': result.get('validity_period'),
                        'version': license_version,
                    }
                    if self.save_license_info(license_info):
                        _license_logger.info(f"许可证激活成功! 有效期至 {result.get('expires_at')}")
                        return (True, {'message': DEFAULT_DATA['LicenseActivated'],
                                       'validity_period': result.get('validity_period')})
                    _license_logger.error('保存许可证信息失败（写入 license.enc 异常）')
                    return (False, DEFAULT_DATA['SaveLicenseInfoError'])
                _license_logger.warning(f'activate返回非200: HTTP {response.status_code}')
                result = _safe_json(response)
                if result is not None:
                    error_message = result.get('message', DEFAULT_DATA['LicenseActivationError'])
                else:
                    error_message = f'许可证服务器响应异常（HTTP {response.status_code}），请检查网络后重试'
                return (False, error_message)
            _license_logger.warning(f'verify_status返回非200: HTTP {verify_response.status_code}')
            verify_error = _safe_json(verify_response)
            if verify_error is not None and verify_error.get('message'):
                return (False, verify_error.get('message'))
            return (False, {'message': f'验证激活码状态失败（HTTP {verify_response.status_code}），请稍后重试或联系管理员',
                            'error': 'VERIFY_STATUS_FAILED'})
        except Exception as e:
            _license_logger.error(f'激活过程发生未预期错误: {_scrub_log(e)}\n{_scrub_log(traceback.format_exc())}')
            return (False, '激活失败，请联系客服处理')


def check_license(version: str=SOFTWARE_VERSION) -> bool:
    """检查许可证是否有效"""
    client = LicenseClient(version=version)
    success, _ = client.check_license()
    return success
def get_remaining_days(version: str=SOFTWARE_VERSION) -> int:
    """获取剩余天数"""
    client = LicenseClient(version=version)
    return client.get_remaining_days()
def get_machine_code() -> str:
    """获取机器码"""
    client = LicenseClient()
    return client.get_machine_id()
def validate_activation_code(license_code: str, version: str=SOFTWARE_VERSION) -> Tuple[bool, Dict]:
    """验证激活码"""
    client = LicenseClient(version=version)
    return client.activate_license(license_code)