
"""\n高清修复性能优化配置和工具\n"""
import torch
import logging
from typing import Dict, Any
from rope.performance_config import *
class EnhanceOptimizer:
    """高清修复优化器"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.optimization_settings = self._get_optimal_settings()
    def _get_optimal_settings(self) -> Dict[str, Any]:
        """根据GPU性能获取最优设置"""
        settings = {'tile_size_multiplier': ENHANCE_TILE_SIZE_MULTIPLIER, 'memory_cleanup_frequency': ENHANCE_MEMORY_CLEANUP_FREQUENCY, 'enable_single_tile_optimization': ENHANCE_ENABLE_SINGLE_TILE_OPTIMIZATION, 'batch_processing': True, 'memory_optimization': True}
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = gpu_memory / 1073741824
            gpu_name = torch.cuda.get_device_name(0)
            if gpu_memory_gb < 6:
                settings.update({'tile_size_multiplier': 1.5, 'memory_cleanup_frequency': 1, 'max_concurrent_tiles': 2, 'enable_aggressive_cleanup': True})
            else:
                if gpu_memory_gb < 12:
                    settings.update({'tile_size_multiplier': 2, 'memory_cleanup_frequency': 2, 'max_concurrent_tiles': 4, 'enable_aggressive_cleanup': False})
                else:
                    settings.update({'tile_size_multiplier': 3, 'memory_cleanup_frequency': 4, 'max_concurrent_tiles': 8, 'enable_aggressive_cleanup': False})
            self.logger.info(f'GPU检测: {gpu_name}, 显存: {gpu_memory_gb:.1f}GB')
        return settings
    def get_optimal_tile_size(self, base_tile_size: int=512) -> int:
        """获取最优tile大小"""
        multiplier = self.optimization_settings['tile_size_multiplier']
        optimal_size = int(base_tile_size * multiplier)
        return (optimal_size + 31) // 32 * 32
    def should_cleanup_memory(self, current_row: int) -> bool:
        """判断是否应该清理内存"""
        frequency = self.optimization_settings['memory_cleanup_frequency']
        return current_row % frequency == 0
    def should_use_single_tile(self, tile_count: int) -> bool:
        """判断是否应该使用单tile优化"""
        if not self.optimization_settings['enable_single_tile_optimization']:
            return False
        else:
            return tile_count <= 4
    def get_performance_tips(self) -> list:
        """获取性能优化建议"""
        tips = ['[提示] 高清修复性能优化建议:', '1. 使用较低的高清修复强度(50-70%)可以显著提升速度', '2. 对于实时应用，建议使用RealESRGAN-x2而非x4版本', '3. 确保GPU驱动程序是最新版本', '4. 关闭不必要的后台应用以释放GPU资源']
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1073741824
            if gpu_memory_gb < 8:
                tips.extend(['5. [注意] 检测到显存较少，建议:', '   - 降低输入分辨率', '   - 使用BSRGan-x2而非x4模型', '   - 启用更频繁的内存清理'])
                return tips
            else:
                tips.extend(['5. [OK] 显存充足，可以:', '   - 使用更高质量的模型', '   - 增加批处理大小', '   - 启用并行处理'])
        return tips
    def log_optimization_status(self):
        """记录优化状态"""
        self.logger.info('=== 高清修复优化状态 ===')
        for key, value in self.optimization_settings.items():
            self.logger.info(f'{key}: {value}')
        for tip in self.get_performance_tips():
            self.logger.info(tip)
enhance_optimizer = EnhanceOptimizer()
def get_optimal_tile_size(base_size: int=512) -> int:
    """获取最优tile大小"""
    return enhance_optimizer.get_optimal_tile_size(base_size)
def should_cleanup_memory(row: int) -> bool:
    """判断是否应该清理内存"""
    return enhance_optimizer.should_cleanup_memory(row)
def should_use_single_tile(tile_count: int) -> bool:
    """判断是否使用单tile优化"""
    return enhance_optimizer.should_use_single_tile(tile_count)
def log_performance_tips():
    """输出性能优化建议"""
    enhance_optimizer.log_optimization_status()
if __name__ == '__main__':
    log_performance_tips()