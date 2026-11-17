
import torch
import logging
import warnings
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import platform
import subprocess
import json
import os
from pathlib import Path
try:
    from .RTX50Compatibility import initialize_rtx50_compatibility, rtx50_compat
except ImportError:
    class DummyCompatibility:
        def detect_rtx_50_series(self):
            return False
        def setup_compatibility_mode(self):
            return True
        def get_compatible_device(self, device):
            return device
        def get_status(self):
            return {'is_rtx_50_series': False}
    rtx50_compat = DummyCompatibility()
    def initialize_rtx50_compatibility():
        return rtx50_compat
logger = logging.getLogger(__name__)
@dataclass
class GPUInfo:
    """GPU信息数据类"""
    name: str
    compute_capability: Tuple[int, int]
    memory_total: int
    memory_free: int
    architecture: str
    cuda_cores: int
    device_id: int
    driver_version: str
    cuda_version: str
@dataclass
class OptimizationConfig:
    """GPU优化配置数据类"""
    batch_size: int
    memory_fraction: float
    precision_mode: str
    tensor_core_enabled: bool
    async_execution: bool
    max_workspace_size: int
    enable_cudnn_benchmark: bool
    memory_pool_init_size: int
class GPUManager:
    """GPU管理器 - 负责GPU检测、配置和优化"""
    COMPATIBILITY_MATRIX = {'RTX_5090': {'compute_capability': (12, 0), 'min_pytorch_version': '2.5.0', 'min_cuda_version': '12.4', 'architecture': 'Ada Lovelace', 'cuda_cores': 21760, 'recommended_settings': {'batch_size': 12, 'memory_fraction': 0.9, 'precision_mode': 'mixed', 'tensor_core_enabled': True, 'async_execution': True, 'max_workspace_size': 4294967296, 'enable_cudnn_benchmark': True, 'memory_pool_init_size': 2048}}, 'RTX_5080': {'compute_capability': (12, 0), 'min_pytorch_version': '2.5.0', 'min_cuda_version': '12.4', 'architecture': 'Ada Lovelace', 'cuda_cores': 10752, 'recommended_settings': {'batch_size': 10, 'memory_fraction': 0.9, 'precision_mode': 'mixed', 'tensor_core_enabled': True, 'async_execution': True, 'max_workspace_size': 3221225472, 'enable_cudnn_benchmark': True, 'memory_pool_init_size': 1536}}, 'RTX_5070': {'compute_capability': (12, 0), 'min_pytorch_version': '2.5.0', 'min_cuda_version': '12.4', 'architecture': 'Ada Lovelace', 'cuda_cores': 6144, 'recommended_settings': {'batch_size': 8, 'memory_fraction': 0.9, 'precision_mode': 'mixed', 'tensor_core_enabled': True, 'async_execution': True, 'max_workspace_size': 3221225472, 'enable_cudnn_benchmark': True, 'memory_pool_init_size': 1024}}, 'RTX_5060': {'compute_capability': (12, 0), 'min_pytorch_version': '2.5.0', 'min_cuda_version': '12.4', 'architecture': 'Ada Lovelace', 'cuda_cores': 3584, 'recommended_settings': {'batch_size': 6, 'memory_fraction': 0.85, 'precision_mode': 'mixed', 'tensor_core_enabled': True, 'async_execution': True, 'max_workspace_size': 2147483648, 'enable_cudnn_benchmark': True, 'memory_pool_init_size': 768}}, 'RTX_5050': {'compute_capability': (12, 0), 'min_pytorch_version': '2.5.0', 'min_cuda_version': '12.4', 'architecture': 'Ada Lovelace', 'cuda_cores': 2560, 'recommended_settings': {'batch_size': 4, 'memory_fraction': 0.8, 'precision_mode': 'mixed', 'tensor_core_enabled': True, 'async_execution': True, 'max_workspace_size': 1073741824, 'enable_cudnn_benchmark': True, 'memory_pool_init_size': 512}}, 'RTX_4090': {'compute_capability': (8, 9), 'min_pytorch_version': '2.0.0', 'min_cuda_version': '11.8', 'architecture': 'Ada Lovelace', 'cuda_cores': 16384, 'recommended_settings': {'batch_size': 6, 'memory_fraction': 0.85, 'precision_mode': 'fp16', 'tensor_core_enabled': True, 'async_execution': True, 'max_workspace_size': 4294967296, 'enable_cudnn_benchmark': True, 'memory_pool_init_size': 2048}}, 'RTX_4080': {'compute_capability': (8, 9), 'min_pytorch_version': '2.0.0', 'min_cuda_version': '11.8', 'architecture': 'Ada Lovelace', 'cuda_cores': 9728, 'recommended_settings': {'batch_size': 4, 'memory_fraction': 0.85, 'precision_mode': 'fp16', 'tensor_core_enabled': True, 'async_execution': True, 'max_workspace_size': 2147483648, 'enable_cudnn_benchmark': True, 'memory_pool_init_size': 1024}}, 'RTX_3090': {'compute_capability': (8, 6), 'min_pytorch_version': '1.9.0', 'min_cuda_version': '11.1', 'architecture': 'Ampere', 'cuda_cores': 10496, 'recommended_settings': {'batch_size': 4, 'memory_fraction': 0.8, 'precision_mode': 'fp16', 'tensor_core_enabled': True, 'async_execution': True, 'max_workspace_size': 2147483648, 'enable_cudnn_benchmark': True, 'memory_pool_init_size': 1024}}, 'DEFAULT': {'compute_capability': (0, 0), 'min_pytorch_version': '1.8.0', 'min_cuda_version': '11.0', 'architecture': 'Unknown', 'cuda_cores': 0, 'recommended_settings': {'batch_size': 2, 'memory_fraction': 0.7, 'precision_mode': 'fp32', 'tensor_core_enabled': False, 'async_execution': False, 'max_workspace_size': 1073741824, 'enable_cudnn_benchmark': False, 'memory_pool_init_size': 512}}}
    def __init__(self):
        self.gpu_info = None
        self.optimization_config = None
        self._cuda_available = torch.cuda.is_available()
        self.rtx50_compat = initialize_rtx50_compatibility()
    def detect_gpu(self) -> Optional[GPUInfo]:
        """\n        检测GPU型号和架构\n        \n        Returns:\n            GPUInfo: GPU信息对象，如果检测失败返回None\n        """
        try:
            if not self._cuda_available:
                logger.warning('CUDA不可用，无法检测GPU')
                return
            else:
                device_id = 0
                device_name = torch.cuda.get_device_name(device_id)
                compute_capability = torch.cuda.get_device_capability(device_id)
                memory_total = torch.cuda.get_device_properties(device_id).total_memory // 1048576
                memory_free = (torch.cuda.get_device_properties(device_id).total_memory - torch.cuda.memory_allocated(device_id)) // 1048576
                cuda_version = torch.version.cuda if torch.version.cuda else 'Unknown'
                driver_version = self._get_driver_version()
                architecture, cuda_cores = self._determine_architecture_and_cores(device_name, compute_capability)
                self.gpu_info = GPUInfo(name=device_name, compute_capability=compute_capability, memory_total=memory_total, memory_free=memory_free, architecture=architecture, cuda_cores=cuda_cores, device_id=device_id, driver_version=driver_version, cuda_version=cuda_version)
                logger.debug(f'检测到GPU: {device_name}, 计算能力: {compute_capability}, 内存: {memory_total}MB, 架构: {architecture}')
                return self.gpu_info
        except Exception as e:
            logger.error(f'GPU检测失败: {e}')
    def _get_driver_version(self) -> str:
        # irreducible cflow, using cdg fallback
        """获取显卡驱动版本"""
        try:
            if platform.system() == 'Windows':
                result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    return result.stdout.strip()
                return 'Unknown'
        except Exception:
            return 'Unknown'
    def _determine_architecture_and_cores(self, device_name: str, compute_capability: Tuple[int, int]) -> Tuple[str, int]:
        """\n        根据设备名称和计算能力确定架构和CUDA核心数\n        \n        Args:\n            device_name: GPU设备名称\n            compute_capability: 计算能力元组\n            \n        Returns:\n            Tuple[str, int]: (架构名称, CUDA核心数)\n        """
        if compute_capability == (12, 0) or 'RTX 50' in device_name:
            if 'RTX 5090' in device_name:
                return ('Ada Lovelace', 21760)
            else:
                if 'RTX 5080' in device_name:
                    return ('Ada Lovelace', 10752)
                else:
                    if 'RTX 5070' in device_name:
                        return ('Ada Lovelace', 6144)
                    else:
                        if 'RTX 5060' in device_name:
                            return ('Ada Lovelace', 3584)
                        else:
                            if 'RTX 5050' in device_name:
                                return ('Ada Lovelace', 2560)
                            else:
                                return ('Ada Lovelace', 6144)
        else:
            if 'RTX 40' in device_name or compute_capability == (8, 9):
                if '4090' in device_name:
                    return ('Ada Lovelace', 16384)
                else:
                    if '4080' in device_name:
                        return ('Ada Lovelace', 9728)
                    else:
                        if '4070' in device_name:
                            return ('Ada Lovelace', 5888)
                        else:
                            return ('Ada Lovelace', 7680)
            else:
                if 'RTX 30' in device_name or compute_capability == (8, 6):
                    if '3090' in device_name:
                        return ('Ampere', 10496)
                    else:
                        if '3080' in device_name:
                            return ('Ampere', 8704)
                        else:
                            if '3070' in device_name:
                                return ('Ampere', 5888)
                            else:
                                return ('Ampere', 6912)
                else:
                    if 'RTX 20' in device_name or compute_capability == (7, 5):
                        if '2080 Ti' in device_name:
                            return ('Turing', 4352)
                        else:
                            if '2080' in device_name:
                                return ('Turing', 2944)
                            else:
                                if '2070' in device_name:
                                    return ('Turing', 2304)
                                else:
                                    return ('Turing', 2560)
                    else:
                        major, minor = compute_capability
                        if major >= 12:
                            return ('Ada Lovelace (Next-Gen)', 0)
                        else:
                            if major == 8:
                                if minor >= 9:
                                    return ('Ada Lovelace', 0)
                                else:
                                    if minor >= 6:
                                        return ('Ampere', 0)
                                    else:
                                        return ('Turing', 0)
                            else:
                                if major == 7:
                                    return ('Turing', 0)
                                else:
                                    if major == 6:
                                        return ('Pascal', 0)
                                    else:
                                        return ('Unknown', 0)
    def get_optimal_config(self, gpu_info: Optional[GPUInfo]=None) -> OptimizationConfig:
        """\n        获取针对特定GPU的优化配置\n        \n        Args:\n            gpu_info: GPU信息，如果为None则使用已检测的GPU信息\n            \n        Returns:\n            OptimizationConfig: 优化配置对象\n        """
        if gpu_info is None:
            gpu_info = self.gpu_info
        if gpu_info is None:
            logger.warning('无GPU信息，使用默认配置')
            return self._create_config_from_matrix('DEFAULT')
        else:
            gpu_key = self._match_gpu_key(gpu_info)
            config = self._create_config_from_matrix(gpu_key)
            config = self._adjust_config_for_memory(config, gpu_info.memory_free)
            self.optimization_config = config
            logger.debug(f'为GPU {gpu_info.name} 生成优化配置: 批处理大小={config.batch_size}, 精度模式={config.precision_mode}')
            return config
    def _match_gpu_key(self, gpu_info: GPUInfo) -> str:
        """匹配GPU型号到配置键"""
        if gpu_info.compute_capability == (12, 0) or 'RTX 50' in gpu_info.name:
            if 'RTX 5090' in gpu_info.name:
                return 'RTX_5090'
            else:
                if 'RTX 5080' in gpu_info.name:
                    return 'RTX_5080'
                else:
                    if 'RTX 5070' in gpu_info.name:
                        return 'RTX_5070'
                    else:
                        if 'RTX 5060' in gpu_info.name:
                            return 'RTX_5060'
                        else:
                            if 'RTX 5050' in gpu_info.name:
                                return 'RTX_5050'
                            else:
                                return 'RTX_5070'
        else:
            if 'RTX 4090' in gpu_info.name:
                return 'RTX_4090'
            else:
                if 'RTX 4080' in gpu_info.name:
                    return 'RTX_4080'
                else:
                    if 'RTX 3090' in gpu_info.name:
                        return 'RTX_3090'
                    else:
                        return 'DEFAULT'
    def _create_config_from_matrix(self, gpu_key: str) -> OptimizationConfig:
        """从兼容性矩阵创建配置"""
        settings = self.COMPATIBILITY_MATRIX[gpu_key]['recommended_settings']
        return OptimizationConfig(batch_size=settings['batch_size'], memory_fraction=settings['memory_fraction'], precision_mode=settings['precision_mode'], tensor_core_enabled=settings['tensor_core_enabled'], async_execution=settings['async_execution'], max_workspace_size=settings['max_workspace_size'], enable_cudnn_benchmark=settings['enable_cudnn_benchmark'], memory_pool_init_size=settings['memory_pool_init_size'])
    def _adjust_config_for_memory(self, config: OptimizationConfig, available_memory: int) -> OptimizationConfig:
        """根据可用内存调整配置"""
        if available_memory < 4096:
            config.batch_size = max(1, config.batch_size // 2)
            config.memory_fraction = min(0.7, config.memory_fraction)
            config.memory_pool_init_size = min(512, config.memory_pool_init_size)
            logger.info(f'内存不足({available_memory}MB)，调整配置: 批处理大小={config.batch_size}')
        if available_memory < 2048:
            config.batch_size = 1
            config.precision_mode = 'fp32'
            config.tensor_core_enabled = False
            config.memory_fraction = 0.6
            config.memory_pool_init_size = 256
            logger.warning(f'内存严重不足({available_memory}MB)，使用最小配置')
        return config
    def apply_optimizations(self, config: Optional[OptimizationConfig]=None) -> bool:
        """\n        应用GPU优化设置到PyTorch\n        \n        Args:\n            config: 优化配置，如果为None则使用已生成的配置\n            \n        Returns:\n            bool: 应用成功返回True，失败返回False\n        """
        if config is None:
            config = self.optimization_config
        if config is None:
            logger.error('无优化配置可应用')
            return False
        else:
            try:
                if not self._cuda_available:
                    logger.warning('CUDA不可用，跳过GPU优化')
                    return False
                else:
                    if hasattr(torch.cuda, 'set_memory_fraction'):
                        torch.cuda.set_memory_fraction(config.memory_fraction)
                        logger.debug(f'设置GPU内存分配比例: {config.memory_fraction}')
                    if config.enable_cudnn_benchmark:
                        torch.backends.cudnn.benchmark = True
                        logger.debug('启用cuDNN基准测试')
                    if config.async_execution:
                        torch.backends.cuda.enable_async_execution = True
                        logger.debug('启用异步执行')
                    self._apply_precision_settings(config)
                    if config.memory_pool_init_size > 0:
                        try:
                            dummy_tensor = torch.zeros((config.memory_pool_init_size * 1024 * 256,), device='cuda', dtype=torch.float32)
                            del dummy_tensor
                            torch.cuda.empty_cache()
                            logger.debug(f'初始化GPU内存池: {config.memory_pool_init_size}MB')
                        except Exception as e:
                            logger.warning(f'内存池初始化失败: {e}')
                    self._apply_tensorrt_settings(config)
                    self._suppress_cuda_warnings()
                    logger.debug('GPU优化配置应用成功')
                    return True
            except Exception as e:
                logger.error(f'应用GPU优化失败: {e}')
                return False
    def _apply_precision_settings(self, config: OptimizationConfig):
        """应用精度设置"""
        try:
            if config.precision_mode == 'fp16':
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                logger.debug('启用FP16精度模式')
            else:
                if config.precision_mode == 'mixed':
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    if config.tensor_core_enabled:
                        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
                    logger.debug('启用混合精度模式')
                else:
                    if config.precision_mode == 'fp32':
                        torch.backends.cuda.matmul.allow_tf32 = False
                        torch.backends.cudnn.allow_tf32 = False
                        logger.debug('使用FP32精度模式')
        except Exception as e:
            logger.warning(f'精度设置应用失败: {e}')
    def _apply_tensorrt_settings(self, config: OptimizationConfig):
        """应用TensorRT相关设置"""
        try:
            os.environ['TENSORRT_MAX_WORKSPACE_SIZE'] = str(config.max_workspace_size)
            if not self.gpu_info or self.gpu_info.compute_capability == (12, 0):
                    os.environ['TENSORRT_OPTIMIZATION_LEVEL'] = '5'
                    os.environ['TENSORRT_ENABLE_FP16'] = '1' if config.precision_mode in ['fp16', 'mixed'] else '0'
                    os.environ['TENSORRT_ENABLE_INT8'] = '0'
                    logger.debug('应用RTX 5070专用TensorRT优化设置')
        except Exception as e:
            logger.warning(f'TensorRT设置应用失败: {e}')
    def get_recommended_batch_size(self, model_size: str='medium') -> int:
        """\n        根据GPU和模型大小推荐批处理大小\n        \n        Args:\n            model_size: 模型大小 (\"small\", \"medium\", \"large\")\n            \n        Returns:\n            int: 推荐的批处理大小\n        """
        if not self.optimization_config:
            return 1
        else:
            base_batch_size = self.optimization_config.batch_size
            size_multipliers = {'small': 1.5, 'medium': 1.0, 'large': 0.5}
            multiplier = size_multipliers.get(model_size, 1.0)
            recommended_size = int(base_batch_size * multiplier)
            return max(1, recommended_size)
    def get_optimal_tensor_settings(self) -> Dict[str, Any]:
        """\n        获取最优的张量设置\n        \n        Returns:\n            Dict[str, Any]: 张量设置字典\n        """
        if not self.optimization_config or not self.gpu_info:
            return {'device': 'cpu', 'dtype': torch.float32, 'memory_format': torch.contiguous_format}
        else:
            settings = {'device': f'cuda:{self.gpu_info.device_id}', 'memory_format': torch.channels_last if self.optimization_config.tensor_core_enabled else torch.contiguous_format}
            if self.optimization_config.precision_mode == 'fp16':
                settings['dtype'] = torch.float16
                return settings
            else:
                if self.optimization_config.precision_mode == 'mixed':
                    settings['dtype'] = torch.float32
                    return settings
                else:
                    settings['dtype'] = torch.float32
                    return settings
    def _suppress_cuda_warnings(self):
        """抑制CUDA相关警告"""
        warnings.filterwarnings('ignore', message='.*CUDA initialization.*')
        warnings.filterwarnings('ignore', message='.*GPU.*not supported.*')
        warnings.filterwarnings('ignore', message='.*compute capability.*')
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        os.environ['TORCH_CUDA_ARCH_LIST'] = ''
        logger.debug('已抑制CUDA兼容性警告')
    def get_gpu_status(self) -> Dict[str, Any]:
        """\n        获取当前GPU状态信息\n        \n        Returns:\n            Dict[str, Any]: GPU状态字典\n        """
        if not self._cuda_available or self.gpu_info is None:
            return {'status': 'unavailable', 'message': 'CUDA不可用或GPU未检测'}
        else:
            try:
                device_id = self.gpu_info.device_id
                memory_allocated = torch.cuda.memory_allocated(device_id) // 1048576
                memory_reserved = torch.cuda.memory_reserved(device_id) // 1048576
                memory_free = self.gpu_info.memory_total - memory_allocated
                utilization = self._get_gpu_utilization()
                return {'status': 'available', 'name': self.gpu_info.name, 'compute_capability': self.gpu_info.compute_capability, 'architecture': self.gpu_info.architecture, 'memory_total': self.gpu_info.memory_total, 'memory_allocated': memory_allocated, 'memory_reserved': memory_reserved, 'memory_free': memory_free, 'utilization': utilization, 'driver_version': self.gpu_info.driver_version, 'cuda_version': self.gpu_info.cuda_version}
            except Exception as e:
                logger.error(f'获取GPU状态失败: {e}')
                return {'status': 'error', 'message': str(e)}
    def _get_gpu_utilization(self) -> Optional[float]:
        """获取GPU利用率"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            return None
    def save_config_to_file(self, config_path: Optional[str]=None) -> bool:
        """\n        保存GPU配置到文件\n        \n        Args:\n            config_path: 配置文件路径，如果为None则使用默认路径\n            \n        Returns:\n            bool: 保存成功返回True\n        """
        if not self.gpu_info or not self.optimization_config:
            logger.error('无GPU信息或配置可保存')
            return False
        else:
            if config_path is None:
                config_dir = Path('config')
                config_dir.mkdir(exist_ok=True)
                config_path = config_dir / 'gpu_config.json'
        try:
            config_data = {'gpu_info': {'name': self.gpu_info.name, 'compute_capability': self.gpu_info.compute_capability, 'memory_total': self.gpu_info.memory_total, 'architecture': self.gpu_info.architecture, 'cuda_cores': self.gpu_info.cuda_cores, 'driver_version': self.gpu_info.driver_version, 'cuda_version': self.gpu_info.cuda_version}, 'optimization_config': {'batch_size': self.optimization_config.batch_size, 'memory_fraction': self.optimization_config.memory_fraction, 'precision_mode': self.optimization_config.precision_mode, 'tensor_core_enabled': self.optimization_config.tensor_core_enabled, 'async_execution': self.optimization_config.async_execution, 'max_workspace_size': self.optimization_config.max_workspace_size, 'enable_cudnn_benchmark': self.optimization_config.enable_cudnn_benchmark, 'memory_pool_init_size': self.optimization_config.memory_pool_init_size}, 'timestamp': time.time()}
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            logger.info(f'GPU配置已保存到: {config_path}')
            return True
        except Exception as e:
            logger.error(f'保存GPU配置失败: {e}')
            return False
    def load_config_from_file(self, config_path: Optional[str]=None) -> bool:
        """\n        从文件加载GPU配置\n        \n        Args:\n            config_path: 配置文件路径，如果为None则使用默认路径\n            \n        Returns:\n            bool: 加载成功返回True\n        """
        if config_path is None:
            config_path = Path('config') / 'gpu_config.json'
        if not os.path.exists(config_path):
            logger.info('GPU配置文件不存在，将使用自动检测')
            return False
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            gpu_data = config_data.get('gpu_info', {})
            self.gpu_info = GPUInfo(name=gpu_data.get('name', 'Unknown'), compute_capability=tuple(gpu_data.get('compute_capability', (0, 0))), memory_total=gpu_data.get('memory_total', 0), memory_free=gpu_data.get('memory_total', 0), architecture=gpu_data.get('architecture', 'Unknown'), cuda_cores=gpu_data.get('cuda_cores', 0), device_id=0, driver_version=gpu_data.get('driver_version', 'Unknown'), cuda_version=gpu_data.get('cuda_version', 'Unknown'))
            opt_data = config_data.get('optimization_config', {})
            self.optimization_config = OptimizationConfig(batch_size=opt_data.get('batch_size', 2), memory_fraction=opt_data.get('memory_fraction', 0.7), precision_mode=opt_data.get('precision_mode', 'fp32'), tensor_core_enabled=opt_data.get('tensor_core_enabled', False), async_execution=opt_data.get('async_execution', False), max_workspace_size=opt_data.get('max_workspace_size', 1073741824), enable_cudnn_benchmark=opt_data.get('enable_cudnn_benchmark', False), memory_pool_init_size=opt_data.get('memory_pool_init_size', 512))
            logger.info(f'从文件加载GPU配置成功: {config_path}')
            return True
        except Exception as e:
            logger.error(f'加载GPU配置失败: {e}')
            return False
    def auto_configure(self) -> bool:
        """\n        自动配置GPU - 检测GPU并生成优化配置\n        \n        Returns:\n            bool: 配置成功返回True\n        """
        try:
            if self.load_config_from_file():
                logger.info('使用已保存的GPU配置')
                return True
            else:
                logger.info('开始自动GPU检测和配置...')
                gpu_info = self.detect_gpu()
                if gpu_info is None:
                    logger.error('GPU检测失败')
                    return False
                else:
                    config = self.get_optimal_config(gpu_info)
                    if config is None:
                        logger.error('优化配置生成失败')
                        return False
                    else:
                        if not self.apply_optimizations(config):
                            logger.error('优化配置应用失败')
                            return False
                        else:
                            self.save_config_to_file()
                            logger.info('GPU自动配置完成')
                            return True
        except Exception as e:
            logger.error(f'GPU自动配置失败: {e}')
            return False
    def cleanup(self):
        """清理GPU资源"""
        try:
            if self._cuda_available:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info('GPU资源清理完成')
        except Exception as e:
            logger.error(f'GPU资源清理失败: {e}')