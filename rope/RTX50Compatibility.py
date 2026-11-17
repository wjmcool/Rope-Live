
"""\nRTX 50系列兼容性模块\n解决PyTorch不支持sm_120架构的问题\n支持RTX 5090/5080/5070/5060/5050等所有RTX 50系列显卡\n"""
global rtx50_compat
import torch
import logging
import os
import warnings
from typing import Optional, Dict, Any, List
import subprocess
import sys
logger = logging.getLogger(__name__)
class RTX50Compatibility:
    """RTX 50系列兼容性处理器"""
    def __init__(self):
        self.is_rtx_50_series = False
        self.rtx_50_model = None
        self.compatibility_mode = None
        self.fallback_device = None
    def detect_rtx_50_series(self) -> bool:
        # irreducible cflow, using cdg fallback
        """检测是否为RTX 50系列显卡"""
        if not torch.cuda.is_available():
            import subprocess
        else:
            device_name = torch.cuda.get_device_name(0)
            compute_capability = torch.cuda.get_device_capability(0)
            if compute_capability == (12, 0) or 'RTX 50' in device_name:
                self.is_rtx_50_series = True
                if 'RTX 5090' in device_name:
                    self.rtx_50_model = 'RTX 5090'
                else:
                    if 'RTX 5080' in device_name:
                        self.rtx_50_model = 'RTX 5080'
                    else:
                        if 'RTX 5070' in device_name:
                            self.rtx_50_model = 'RTX 5070'
                        else:
                            if 'RTX 5060' in device_name:
                                self.rtx_50_model = 'RTX 5060'
                            else:
                                if 'RTX 5050' in device_name:
                                    self.rtx_50_model = 'RTX 5050'
                                else:
                                    self.rtx_50_model = 'RTX 50系列 (未知型号)'
                logger.debug(f'检测到RTX 50系列显卡: {device_name} ({self.rtx_50_model}), 计算能力: {compute_capability}')
                return True
            return False
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                gpu_name = result.stdout.strip()
                if 'RTX 50' in gpu_name or '5070' in gpu_name or '5080' in gpu_name or ('5090' in gpu_name):
                    self.is_rtx_50_series = True
                    self.rtx_50_model = 'RTX 5070' if '5070' in gpu_name else 'RTX 50系列'
                    logger.debug(f'通过nvidia-smi检测到RTX 50系列显卡: {gpu_name}')
                    return True
        except Exception as e:
            logger.error(f'RTX 50系列检测失败: {e}')
            return False
        logger.debug('CUDA不可用，无法检测RTX 50系列显卡')
        return False
    def setup_compatibility_mode(self) -> bool:
        """设置RTX 50系列兼容模式"""
        if not self.is_rtx_50_series:
            return True
        else:
            try:
                logger.debug(f'设置{self.rtx_50_model}兼容模式...')
                self.compatibility_mode = 'gpu_force'
                self.fallback_device = 'cuda'
                os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
                os.environ['TORCH_USE_CUDA_DSA'] = '0'
                os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                warnings.filterwarnings('ignore', message='.*CUDA capability.*')
                warnings.filterwarnings('ignore', message='.*not compatible.*')
                warnings.filterwarnings('ignore', message='.*sm_120.*')
                logger.debug(f'{self.rtx_50_model}兼容模式设置完成: {self.compatibility_mode}')
                return True
            except Exception as e:
                logger.error(f'{self.rtx_50_model}兼容模式设置失败: {e}')
                return False
    def get_compatible_device(self, preferred_device: str='cuda') -> str:
        """\n        获取兼容的设备\n        \n        Args:\n            preferred_device: 首选设备\n            \n        Returns:\n            str: 兼容的设备名称\n        """
        try:
            cuda_available = torch.cuda.is_available()
        except (RuntimeError, AssertionError) as e:
            if 'torch not compiled with cuda enabled' in str(e).lower():
                logger.debug('PyTorch未编译CUDA支持，强制使用CPU')
                return 'cpu'
            else:
                cuda_available = False
        if not cuda_available:
            logger.debug('由于CUDA不可用，使用CPU模式')
            return 'cpu'
        else:
            if not self.is_rtx_50_series:
                return preferred_device
            else:
                if self.compatibility_mode == 'cpu_fallback':
                    logger.debug(f'{self.rtx_50_model}使用CPU fallback模式')
                    return 'cpu'
                else:
                    if self.compatibility_mode == 'gpu_experimental':
                        logger.debug(f'{self.rtx_50_model}尝试使用GPU实验模式')
                        return preferred_device
                    else:
                        if self.compatibility_mode == 'gpu_force':
                            logger.debug(f'{self.rtx_50_model}强制使用GPU模式，忽略架构警告')
                            return 'cuda'
                        else:
                            return preferred_device
    def create_compatible_tensor(self, *args, device: Optional[str]=None, **kwargs) -> torch.Tensor:
        # irreducible cflow, using cdg fallback
        """\n        创建兼容的张量\n        \n        Args:\n            *args: torch.tensor的位置参数\n            device: 设备名称\n            **kwargs: torch.tensor的关键字参数\n            \n        Returns:\n            torch.Tensor: 兼容的张量\n        """
        if device is None:
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except (RuntimeError, AssertionError):
                device = 'cpu'
        if self.is_rtx_50_series:
            device = self.get_compatible_device(device)
        try:
            return torch.tensor(*args, device=device, **kwargs)
        except (RuntimeError, AssertionError) as e:
            error_msg = str(e).lower()
            error_text = str(e)
            error_exc = e
        if ('no kernel image is available' in error_msg or 'torch not compiled with cuda enabled' in error_msg or 'cuda' in error_msg) and self.is_rtx_50_series:
            logger.warning(f'{self.rtx_50_model} CUDA错误，强制使用CPU: {error_text}')
            kwargs_copy = kwargs.copy()
            kwargs_copy.pop('device', None)
            return torch.tensor(*args, device='cpu', **kwargs_copy)
        else:
            raise error_exc
    def create_compatible_zeros(self, *args, device: Optional[str]=None, **kwargs) -> torch.Tensor:
        # irreducible cflow, using cdg fallback
        """\n        创建兼容的零张量\n        \n        Args:\n            *args: torch.zeros的位置参数\n            device: 设备名称\n            **kwargs: torch.zeros的关键字参数\n            \n        Returns:\n            torch.Tensor: 兼容的零张量\n        """
        if device is None:
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except (RuntimeError, AssertionError):
                device = 'cpu'
        if self.is_rtx_50_series:
            device = self.get_compatible_device(device)
        try:
            return torch.zeros(*args, device=device, **kwargs)
        except (RuntimeError, AssertionError) as e:
            error_msg = str(e).lower()
            error_text = str(e)
            error_exc = e
        if ('no kernel image is available' in error_msg or 'torch not compiled with cuda enabled' in error_msg or 'cuda' in error_msg) and self.is_rtx_50_series:
            logger.debug(f'{self.rtx_50_model} CUDA错误，强制使用CPU: {error_text}')
            kwargs_copy = kwargs.copy()
            kwargs_copy.pop('device', None)
            return torch.zeros(*args, device='cpu', **kwargs_copy)
        else:
            raise error_exc
    def create_compatible_ones(self, *args, device: Optional[str]=None, **kwargs) -> torch.Tensor:
        # irreducible cflow, using cdg fallback
        """\n        创建兼容的一张量\n        \n        Args:\n            *args: torch.ones的位置参数\n            device: 设备名称\n            **kwargs: torch.ones的关键字参数\n            \n        Returns:\n            torch.Tensor: 兼容的一张量\n        """
        if device is None:
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except (RuntimeError, AssertionError):
                device = 'cpu'
        if self.is_rtx_50_series:
            device = self.get_compatible_device(device)
        try:
            return torch.ones(*args, device=device, **kwargs)
        except (RuntimeError, AssertionError) as e:
            error_msg = str(e).lower()
            error_text = str(e)
            error_exc = e
        if ('no kernel image is available' in error_msg or 'torch not compiled with cuda enabled' in error_msg or 'cuda' in error_msg) and self.is_rtx_50_series:
            logger.debug(f'{self.rtx_50_model} CUDA错误，强制使用CPU: {error_text}')
            kwargs_copy = kwargs.copy()
            kwargs_copy.pop('device', None)
            return torch.ones(*args, device='cpu', **kwargs_copy)
        else:
            raise error_exc
    def wrap_cuda_operation(self, operation, *args, **kwargs):
        """\n        包装CUDA操作以提供RTX 50系列兼容性\n        \n        Args:\n            operation: 要执行的CUDA操作\n            *args: 操作的位置参数\n            **kwargs: 操作的关键字参数\n            \n        Returns:\n            操作的结果\n        """
        if not self.is_rtx_50_series:
            return operation(*args, **kwargs)
        else:
            try:
                if self.compatibility_mode == 'gpu_force':
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        return operation(*args, **kwargs)
                else:
                    return operation(*args, **kwargs)
            except RuntimeError as e:
                error_msg = str(e).lower()
                if 'no kernel image is available' in error_msg or 'cuda error' in error_msg:
                    logger.warning(f'{self.rtx_50_model} CUDA操作失败，尝试CPU fallback: {e}')
                    if self.compatibility_mode != 'gpu_force':
                        cpu_args = []
                        for arg in args:
                            if isinstance(arg, torch.Tensor) and arg.is_cuda:
                                cpu_args.append(arg.cpu())
                            else:
                                cpu_args.append(arg)
                        cpu_kwargs = {}
                        for key, value in kwargs.items():
                            if isinstance(value, torch.Tensor) and value.is_cuda:
                                cpu_kwargs[key] = value.cpu()
                            else:
                                if key == 'device' and 'cuda' in str(value):
                                    cpu_kwargs[key] = 'cpu'
                                else:
                                    cpu_kwargs[key] = value
                        return operation(*cpu_args, **cpu_kwargs)
                    else:
                        logger.error(f'{self.rtx_50_model} 强制GPU模式下操作失败: {e}')
                        raise
                else:
                    raise
    def get_status(self) -> Dict[str, Any]:
        """获取兼容性状态"""
        return {'is_rtx_50_series': self.is_rtx_50_series, 'rtx_50_model': self.rtx_50_model, 'compatibility_mode': self.compatibility_mode, 'fallback_device': self.fallback_device, 'cuda_available': torch.cuda.is_available()}
    def install_pytorch_nightly(self) -> bool:
        """\n        尝试安装支持RTX 50系列的PyTorch nightly版本\n        \n        Returns:\n            bool: 安装成功返回True\n        """
        try:
            logger.debug(f"尝试安装支持{self.rtx_50_model or 'RTX 50系列'}的PyTorch nightly版本...")
            result = subprocess.run([sys.executable, '-m', 'pip', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error('pip不可用，无法安装PyTorch nightly')
                return False
            else:
                install_cmd = [sys.executable, '-m', 'pip', 'install', '--upgrade', '--index-url', 'https://download.pytorch.org/whl/nightly/cu124', 'torch', 'torchvision', 'torchaudio']
                logger.debug('执行安装命令...')
                result = subprocess.run(install_cmd, capture_output=True, text=True, timeout=600)
                if result.returncode == 0:
                    logger.debug('PyTorch nightly安装成功，请重启应用程序')
                    return True
                else:
                    logger.error(f'PyTorch nightly安装失败: {result.stderr}')
                    return False
        except Exception as e:
            logger.error(f'安装PyTorch nightly时出错: {e}')
            return False
    def suggest_solutions(self) -> List[str]:
        """提供RTX 50系列兼容性解决方案建议"""
        solutions = []
        if self.is_rtx_50_series:
            model_name = self.rtx_50_model or 'RTX 50系列'
            solutions.extend([f'1. 安装支持{model_name}的PyTorch nightly版本', '2. 使用CPU模式运行（性能较低但稳定）', '3. 等待PyTorch官方发布支持sm_120架构的稳定版本', '4. 使用CUDA 12.4+和最新的NVIDIA驱动程序', '5. 考虑使用TensorRT进行模型推理优化', '6. 支持所有RTX 50系列显卡: RTX 5090/5080/5070/5060/5050'])
        return solutions
rtx50_compat = RTX50Compatibility()
def initialize_rtx50_compatibility():
    """初始化RTX 50系列兼容性"""
    if rtx50_compat.detect_rtx_50_series():
        rtx50_compat.setup_compatibility_mode()
        logger.debug('RTX 50系列兼容性解决方案:')
        for solution in rtx50_compat.suggest_solutions():
            logger.debug(f'  {solution}')
    return rtx50_compat
def _is_cuda_fallback_error(error_msg):
    """判断是否为需要回退到CPU的CUDA相关错误"""
    return 'no kernel image is available' in error_msg or 'torch not compiled with cuda enabled' in error_msg or 'nvidia driver' in error_msg or ('no nvidia gpu' in error_msg) or ('cuda' in error_msg)
def compatible_zeros(*args, **kwargs):
    # irreducible cflow, using cdg fallback
    """RTX 50系列兼容的torch.zeros"""
    try:
        return torch.zeros(*args, **kwargs)
    except (RuntimeError, AssertionError) as e:
        error_msg = str(e).lower()
        error_text = str(e)
        error_exc = e
    if _is_cuda_fallback_error(error_msg):
        kwargs_copy = kwargs.copy()
        kwargs_copy['device'] = 'cpu'
        logger.warning(f'CUDA错误，回退到CPU模式: {error_text}')
        return torch.zeros(*args, **kwargs_copy)
    else:
        raise error_exc
def compatible_ones(*args, **kwargs):
    # irreducible cflow, using cdg fallback
    """RTX 50系列兼容的torch.ones"""
    try:
        return torch.ones(*args, **kwargs)
    except (RuntimeError, AssertionError) as e:
        error_msg = str(e).lower()
        error_text = str(e)
        error_exc = e
    if _is_cuda_fallback_error(error_msg):
        kwargs_copy = kwargs.copy()
        kwargs_copy['device'] = 'cpu'
        logger.warning(f'CUDA错误，回退到CPU模式: {error_text}')
        return torch.ones(*args, **kwargs_copy)
    else:
        raise error_exc
def compatible_tensor(*args, **kwargs):
    # irreducible cflow, using cdg fallback
    """RTX 50系列兼容的torch.tensor"""
    try:
        return torch.tensor(*args, **kwargs)
    except (RuntimeError, AssertionError) as e:
        error_msg = str(e).lower()
        error_text = str(e)
        error_exc = e
    if _is_cuda_fallback_error(error_msg):
        kwargs_copy = kwargs.copy()
        kwargs_copy['device'] = 'cpu'
        logger.warning(f'CUDA错误，回退到CPU模式: {error_text}')
        return torch.tensor(*args, **kwargs_copy)
    else:
        raise error_exc
def compatible_arange(*args, **kwargs):
    # irreducible cflow, using cdg fallback
    """RTX 50系列兼容的torch.arange"""
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    if rtx50_compat.is_rtx_50_series:
        device = rtx50_compat.get_compatible_device(device)
        kwargs['device'] = device
    try:
        return torch.arange(*args, **kwargs)
    except (RuntimeError, AssertionError) as e:
        error_msg = str(e).lower()
        error_text = str(e)
        error_exc = e
    if _is_cuda_fallback_error(error_msg):
        logger.warning(f'CUDA错误，回退到CPU模式: {error_text}')
        kwargs['device'] = 'cpu'
        return torch.arange(*args, **kwargs)
    else:
        raise error_exc
def compatible_meshgrid(*args, **kwargs):
    # irreducible cflow, using cdg fallback
    """RTX 50系列兼容的torch.meshgrid"""
    try:
        return torch.meshgrid(*args, **kwargs)
    except (RuntimeError, AssertionError) as e:
        error_msg = str(e).lower()
        error_text = str(e)
        error_exc = e
    if _is_cuda_fallback_error(error_msg):
        logger.warning(f'CUDA错误，回退到CPU模式: {error_text}')
        cpu_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.is_cuda:
                cpu_args.append(arg.cpu())
            else:
                cpu_args.append(arg)
        return torch.meshgrid(*cpu_args, **kwargs)
    else:
        raise error_exc
def compatible_empty(*args, **kwargs):
    # irreducible cflow, using cdg fallback
    """RTX 50系列兼容的torch.empty"""
    try:
        return torch.empty(*args, **kwargs)
    except (RuntimeError, AssertionError) as e:
        error_msg = str(e).lower()
        error_text = str(e)
        error_exc = e
    if _is_cuda_fallback_error(error_msg):
        kwargs_copy = kwargs.copy()
        kwargs_copy['device'] = 'cpu'
        logger.warning(f'CUDA错误，回退到CPU模式: {error_text}')
        return torch.empty(*args, **kwargs_copy)
    else:
        raise error_exc