
import gc
import torch
import threading
import time
from collections import OrderedDict
from typing import Dict, Any, Optional, Callable
import logging
class ModelLRUCache:
    """模型LRU缓存管理器"""
    def __init__(self, max_models: int=5):
        self.cache = OrderedDict()
        self.max_models = max_models
        self.lock = threading.Lock()
        self.model_usage_count = {}
    def get_model(self, model_name: str) -> Optional[Any]:
        """获取模型，如果存在则移到最后（最近使用）"""
        with self.lock:
            if model_name in self.cache:
                model = self.cache.pop(model_name)
                self.cache[model_name] = model
                self.model_usage_count[model_name] = self.model_usage_count.get(model_name, 0) + 1
                return model
            else:
                return None
    def add_model(self, model_name: str, model: Any) -> None:
        """添加模型到缓存"""
        with self.lock:
            if model_name in self.cache:
                self.cache.pop(model_name)
                self.cache[model_name] = model
                return
            else:
                if len(self.cache) >= self.max_models:
                    oldest_name, oldest_model = self.cache.popitem(last=False)
                    self._cleanup_model(oldest_name, oldest_model)
                self.cache[model_name] = model
                self.model_usage_count[model_name] = 1
    def remove_model(self, model_name: str) -> bool:
        """移除指定模型"""
        with self.lock:
            if model_name in self.cache:
                model = self.cache.pop(model_name)
                self._cleanup_model(model_name, model)
                return True
            else:
                return False
    def clear_all(self) -> None:
        """清空所有缓存的模型"""
        with self.lock:
            for name, model in self.cache.items():
                self._cleanup_model(name, model)
            self.cache.clear()
            self.model_usage_count.clear()
    def _cleanup_model(self, model_name: str, model: Any) -> None:
        """清理模型内存"""
        try:
            if hasattr(model, 'cleanup'):
                model.cleanup()
            del model
            if model_name in self.model_usage_count:
                del self.model_usage_count[model_name]
            logging.warning(f'已清理模型: {model_name}')
        except Exception as e:
            logging.error(f'清理模型 {model_name} 时出错: {e}')
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        with self.lock:
            return {'cached_models': list(self.cache.keys()), 'cache_size': len(self.cache), 'max_size': self.max_models, 'usage_count': self.model_usage_count.copy()}
class SmartMemoryManager:
    """智能内存管理器"""
    def __init__(self, models_instance, gpu_memory_threshold: float=0.85):
        self.models = models_instance
        self.gpu_memory_threshold = gpu_memory_threshold
        self.model_cache = ModelLRUCache(max_models=5)
        self.monitoring = False
        self.monitor_thread = None
        self.last_cleanup_time = 0
        self.cleanup_interval = 30
    def start_monitoring(self) -> None:
        """启动内存监控"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
            self.monitor_thread.start()
    def stop_monitoring(self) -> None:
        """停止内存监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logging.warning('内存监控已停止')
    def _monitor_memory(self) -> None:
        """内存监控循环"""
        while self.monitoring:
            try:
                self._check_and_cleanup_memory()
                time.sleep(10)
            except Exception as e:
                logging.error(f'内存监控出错: {e}')
                time.sleep(30)
    def _check_and_cleanup_memory(self) -> None:
        """检查并清理内存"""
        try:
            used, total = self.models.get_gpu_memory()
            usage_ratio = used / total if total > 0 else 0
            current_time = time.time()
            if usage_ratio > self.gpu_memory_threshold and current_time - self.last_cleanup_time > self.cleanup_interval:
                    logging.warning(f'GPU内存使用率过高: {usage_ratio:.2%}, 开始清理')
                    self._smart_cleanup()
                    self.last_cleanup_time = current_time
        except Exception as e:
            return None
    def _smart_cleanup(self) -> None:
        """智能清理内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        cache_info = self.model_cache.get_cache_info()
        if cache_info['cache_size'] > 2:
            usage_count = cache_info['usage_count']
            sorted_models = sorted(usage_count.items(), key=lambda x: x[1])
            models_to_remove = [name for name, _ in sorted_models[:cache_info['cache_size'] - 2]]
            for model_name in models_to_remove:
                self.model_cache.remove_model(model_name)
        logging.warning('智能内存清理完成')
    def force_cleanup(self) -> None:
        """强制清理所有缓存"""
        logging.warning('执行强制内存清理')
        self.model_cache.clear_all()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logging.warning('强制内存清理完成')
    def get_memory_info(self) -> Dict[str, Any]:
        """获取内存信息"""
        try:
            used, total = self.models.get_gpu_memory()
            usage_ratio = used / total if total > 0 else 0
            return {'gpu_memory_used': used, 'gpu_memory_total': total, 'gpu_usage_ratio': usage_ratio, 'cache_info': self.model_cache.get_cache_info(), 'monitoring': self.monitoring}
        except Exception as e:
            return {}
class BatchProcessor:
    """批处理器"""
    def __init__(self, batch_size: int=4):
        self.batch_size = batch_size
    def process_batch(self, items: list, process_func: Callable, **kwargs) -> list:
        """批量处理数据"""
        results = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            try:
                with torch.no_grad():
                    batch_results = process_func(batch, **kwargs)
                    if isinstance(batch_results, (list, tuple)):
                        results.extend(batch_results)
                    else:
                        results.append(batch_results)
            except Exception as e:
                logging.error(f'批处理出错: {e}')
                for item in batch:
                    try:
                        result = process_func([item], **kwargs)
                        results.append(result)
                    except Exception as item_e:
                        results.append(None)
        return results
class ImagePreprocessCache:
    """图像预处理缓存"""
    def __init__(self, max_size: int=100):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.lock = threading.Lock()
    def get_processed(self, image_hash: str, preprocess_func: Callable) -> Any:
        """获取预处理结果，如果不存在则计算并缓存"""
        with self.lock:
            if image_hash in self.cache:
                result = self.cache.pop(image_hash)
                self.cache[image_hash] = result
                return result
            else:
                processed = preprocess_func()
                if len(self.cache) < self.max_size:
                    self.cache[image_hash] = processed
                else:
                    if self.cache:
                        self.cache.popitem(last=False)
                        self.cache[image_hash] = processed
                return processed
    def clear_cache(self) -> None:
        """清空缓存"""
        with self.lock:
            self.cache.clear()
    def get_cache_size(self) -> int:
        """获取缓存大小"""
        with self.lock:
            return len(self.cache)