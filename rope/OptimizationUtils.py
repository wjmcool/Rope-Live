
"""\n优化工具模块\n提供各种性能优化功能，包括模型预热、批处理优化、内存优化等\n"""
import time
import torch
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional
import threading
import queue
import logging
from functools import wraps
from rope.performance_config import *
def timing_decorator(func):
    """计时装饰器，用于测量函数执行时间"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        try:
            pass
        except ImportError:
            pass
        logging.warning(f'{func.__name__} 执行时间: {execution_time:.4f}s')
        return result
    return wrapper
class ModelWarmer:
    """模型预热器"""
    @staticmethod
    def warm_up_model(model, input_shape: Tuple, device: str='cuda', iterations: int=3):
        """预热模型以优化首次推理性能"""
        logging.warning(f'开始预热模型，输入形状: {input_shape}，迭代次数: {iterations}')
        try:
            if isinstance(input_shape[0], (list, tuple)):
                dummy_inputs = []
                for shape in input_shape:
                    dummy_input = torch.randn(shape, device=device, dtype=torch.float32)
                    dummy_inputs.append(dummy_input)
            else:
                dummy_inputs = [torch.randn(input_shape, device=device, dtype=torch.float32)]
            for i in range(iterations):
                start_time = time.time()
                if hasattr(model, 'run'):
                    input_dict = {}
                    input_names = [inp.name for inp in model.get_inputs()]
                    for j, inp in enumerate(dummy_inputs):
                        if j < len(input_names):
                            input_dict[input_names[j]] = inp.cpu().numpy()
                    _ = model.run(None, input_dict)
                else:
                    if hasattr(model, 'predict'):
                        feed_dict = {}
                        for j, inp in enumerate(dummy_inputs):
                            feed_dict[f'input_{j}'] = inp
                        _ = model.predict(feed_dict)
                    else:
                        with torch.no_grad():
                            if len(dummy_inputs) == 1:
                                _ = model(dummy_inputs[0])
                            else:
                                _ = model(*dummy_inputs)
                torch.cuda.synchronize() if device == 'cuda' else None
                warmup_time = time.time() - start_time
                logging.warning(f'预热迭代 {i + 1}/{iterations}: {warmup_time:.4f}s')
            logging.warning('模型预热完成')
        except Exception as e:
            logging.error(f'模型预热失败: {e}')
class BatchOptimizer:
    """批处理优化器"""
    def __init__(self, max_batch_size: int=BATCH_SIZE, max_wait_time: float=MAX_BATCH_WAIT_TIME):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.batch_queue = queue.Queue()
        self.result_queues = {}
        self.processing = False
        self.lock = threading.Lock()
    def add_to_batch(self, data: Any, request_id: str) -> queue.Queue:
        """添加数据到批处理队列"""
        result_queue = queue.Queue()
        with self.lock:
            self.result_queues[request_id] = result_queue
            self.batch_queue.put((data, request_id))
        return result_queue
        return result_queue
    def process_batch(self, process_func):
        """处理批次数据"""
        batch_data = []
        batch_ids = []
        start_time = time.time()
        while len(batch_data) < self.max_batch_size:
            if time.time() - start_time < self.max_wait_time:
                try:
                    data, request_id = self.batch_queue.get(timeout=0.01)
                    batch_data.append(data)
                    batch_ids.append(request_id)
                except queue.Empty:
                    if batch_data:
                        continue
                    break
            else:
                break
        if not batch_data:
            return
        try:
            results = process_func(batch_data)
            with self.lock:
                for i, request_id in enumerate(batch_ids):
                    if request_id in self.result_queues:
                        result_queue = self.result_queues[request_id]
                        result_queue.put(results[i] if isinstance(results, list) else results)
                        del self.result_queues[request_id]
        except Exception as e:
            logging.error(f'批处理失败: {e}')
            with self.lock:
                for request_id in batch_ids:
                    if request_id in self.result_queues:
                        self.result_queues[request_id].put(None)
                        del self.result_queues[request_id]

class ImageOptimizer:
    """图像优化器"""
    @staticmethod
    def optimize_image_for_inference(image: np.ndarray, target_size: Tuple[int, int]=None) -> np.ndarray:
        """优化图像以提高推理性能"""
        if target_size:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        return image
    @staticmethod
    def batch_preprocess_images(images: List[np.ndarray], target_size: Tuple[int, int]=None) -> np.ndarray:
        """批量预处理图像"""
        processed_images = []
        for img in images:
            processed_img = ImageOptimizer.optimize_image_for_inference(img, target_size)
            processed_images.append(processed_img)
        if processed_images:
            batch = np.stack(processed_images, axis=0)
            return batch
        else:
            return np.array([])
    @staticmethod
    def create_image_pyramid(image: np.ndarray, scales: List[float]=[0.5, 1.0, 1.5]) -> List[np.ndarray]:
        """创建图像金字塔以支持多尺度处理"""
        pyramid = []
        h, w = image.shape[:2]
        for scale in scales:
            new_h, new_w = (int(h * scale), int(w * scale))
            if new_h > 0 and new_w > 0:
                    scaled_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    pyramid.append(scaled_img)
        return pyramid
class MemoryOptimizer:
    """内存优化器"""
    @staticmethod
    def optimize_tensor_memory(tensor: torch.Tensor) -> torch.Tensor:
        """优化张量内存使用"""
        if torch.cuda.is_available() and (not tensor.is_cuda):
                tensor = tensor.cuda()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return tensor
    @staticmethod
    def clear_gpu_cache():
        """清理GPU缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """获取内存使用情况"""
        import psutil
        memory_info = {'cpu_percent': psutil.virtual_memory().percent, 'cpu_available_gb': psutil.virtual_memory().available / 1073741824}
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / 1073741824
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1073741824
            memory_info.update({'gpu_allocated_gb': gpu_memory_allocated, 'gpu_total_gb': gpu_memory_total, 'gpu_percent': gpu_memory_allocated / gpu_memory_total * 100})
        return memory_info
class InferenceOptimizer:
    """推理优化器"""
    @staticmethod
    def optimize_onnx_session(model_path: str, providers: List=None) -> Any:
        """优化ONNX推理会话"""
        import onnxruntime as ort
        if providers is None:
            providers = get_optimized_ort_providers()
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = min(4, os.cpu_count())
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        session = ort.InferenceSession(model_path, sess_options, providers=providers)
        logging.warning(f'ONNX会话已优化: {model_path}')
        return session
    @staticmethod
    @timing_decorator
    def run_inference_with_timing(model, inputs: Dict) -> Any:
        """带计时的推理执行"""
        if hasattr(model, 'run'):
            return model.run(None, inputs)
        else:
            if hasattr(model, 'predict'):
                return model.predict(inputs)
            else:
                with torch.no_grad():
                    return model(**inputs)
class ThreadPoolOptimizer:
    """线程池优化器"""
    def __init__(self, max_workers: int=MAX_WORKER_THREADS):
        from concurrent.futures import ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_workers = max_workers
    def submit_task(self, func, *args, **kwargs):
        """提交任务到线程池"""
        return self.executor.submit(func, *args, **kwargs)
    def shutdown(self, wait: bool=True):
        """关闭线程池"""
        self.executor.shutdown(wait=wait)
        logging.warning('线程池已关闭')
thread_pool_optimizer = ThreadPoolOptimizer()
model_warmer = ModelWarmer()
batch_optimizer = BatchOptimizer()
image_optimizer = ImageOptimizer()
memory_optimizer = MemoryOptimizer()
inference_optimizer = InferenceOptimizer()
def cleanup_optimizers():
    """清理优化器资源"""
    thread_pool_optimizer.shutdown()
    logging.warning('所有优化器已清理')