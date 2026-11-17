
"""\n性能和内存优化配置文件\n"""
global MODEL_CACHE_SIZE
global IMAGE_CACHE_SIZE
global BATCH_SIZE
import os
import torch
GPU_MEMORY_THRESHOLD = 0.85
GPU_MEMORY_CHECK_INTERVAL = 5.0
AUTO_MEMORY_CLEANUP = True
MODEL_CACHE_SIZE = 5
MODEL_USAGE_THRESHOLD = 10
CACHE_MODELS = True
BATCH_SIZE = 6
MAX_BATCH_WAIT_TIME = 0.05
ENABLE_BATCH_PROCESSING = True
ENHANCE_TILE_SIZE_MULTIPLIER = 2
ENHANCE_MEMORY_CLEANUP_FREQUENCY = 2
ENHANCE_ENABLE_SINGLE_TILE_OPTIMIZATION = True
IMAGE_CACHE_SIZE = 100
IMAGE_CACHE_ENABLED = True
TRT_MAX_WORKSPACE_SIZE = 3221225472
TRT_ENGINE_CACHE_ENABLED = True
TRT_TIMING_CACHE_ENABLED = True
TRT_OPTIMIZATION_LEVEL = 5
MAX_WORKER_THREADS = min(8, os.cpu_count())
THREAD_POOL_SIZE = 4
FORCE_GC_INTERVAL = 30.0
CUDA_CACHE_CLEAR_INTERVAL = 60.0
PERFORMANCE_LOG_INTERVAL = 10.0
def adjust_config_for_gpu():
    """根据GPU性能动态调整配置"""
    global MODEL_CACHE_SIZE
    global IMAGE_CACHE_SIZE
    global BATCH_SIZE
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_gb = gpu_memory / 1073741824
        if gpu_memory_gb < 6:
            BATCH_SIZE = 3
            MODEL_CACHE_SIZE = 3
            IMAGE_CACHE_SIZE = 50
        else:
            if gpu_memory_gb < 12:
                BATCH_SIZE = 6
                MODEL_CACHE_SIZE = 5
                IMAGE_CACHE_SIZE = 100
            else:
                BATCH_SIZE = 10
                MODEL_CACHE_SIZE = 8
                IMAGE_CACHE_SIZE = 200
def get_optimized_ort_providers():
    """获取优化的ONNX Runtime提供程序配置"""
    providers = []
    if torch.cuda.is_available():
        cuda_provider_options = {'device_id': 0, 'arena_extend_strategy': 'kNextPowerOfTwo', 'gpu_mem_limit': int(torch.cuda.get_device_properties(0).total_memory * 0.8), 'cudnn_conv_algo_search': 'EXHAUSTIVE', 'do_copy_in_default_stream': True}
        providers.append(('CUDAExecutionProvider', cuda_provider_options))
    cpu_provider_options = {'intra_op_num_threads': min(4, os.cpu_count()), 'inter_op_num_threads': 1}
    providers.append(('CPUExecutionProvider', cpu_provider_options))
    return providers
def get_optimized_trt_config():
    """获取优化的TensorRT配置"""
    return {'trt_max_workspace_size': TRT_MAX_WORKSPACE_SIZE, 'trt_engine_cache_enable': TRT_ENGINE_CACHE_ENABLED, 'trt_engine_cache_path': 'tensorrt-engines', 'trt_timing_cache_enable': TRT_TIMING_CACHE_ENABLED, 'trt_timing_cache_path': 'tensorrt-engines', 'trt_dump_ep_context_model': True, 'trt_ep_context_file_path': 'tensorrt-engines', 'trt_layer_norm_fp32_fallback': True, 'trt_builder_optimization_level': TRT_OPTIMIZATION_LEVEL, 'trt_fp16_enable': True, 'trt_int8_enable': False}
adjust_config_for_gpu()