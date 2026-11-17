
global TRT_LOGGER
import numpy as np
import torch
try:
    from torch.cuda import nvtx
except Exception:
    class _DummyNvtx:
        def range_push(self, *args, **kwargs):
            return
        def range_pop(self, *args, **kwargs):
            return
    nvtx = _DummyNvtx()
from collections import OrderedDict
import platform
from queue import Queue
from threading import Lock
try:
    import tensorrt as trt
    import ctypes
except ModuleNotFoundError:
    pass
numpy_to_torch_dtype_dict = {np.uint8: torch.uint8, np.int8: torch.int8, np.int16: torch.int16, np.int32: torch.int32, np.int64: torch.int64, np.float16: torch.float16, np.float32: torch.float32, np.float64: torch.float64, np.complex64: torch.complex64, np.complex128: torch.complex128}
if np.version.full_version >= '1.24.0':
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool
if 'trt' in globals():
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
else:
    TRT_LOGGER = {}
class TensorRTPredictor:
    """\n    Implements inference for the TensorRT engine with a pool of execution contexts.\n    """
    def __init__(self, **kwargs):
        """\n        :param engine_path: The path to the serialized engine to load from disk.\n        :param pool_size: The size of the pool of execution contexts.\n        """
        self.engine = None
        self.context_pool = None
        self.lock = Lock()
        self.device = kwargs.get('device', 'cuda')
        custom_plugin_path = kwargs.get('custom_plugin_path', None)
        if custom_plugin_path is not None:
            if platform.system().lower() == 'linux':
                ctypes.CDLL(custom_plugin_path, mode=ctypes.RTLD_GLOBAL)
            else:
                ctypes.CDLL(custom_plugin_path, mode=ctypes.RTLD_GLOBAL, winmode=0)
        engine_path = kwargs.get('model_path', None)
        self.debug = kwargs.get('debug', False)
        self.pool_size = kwargs.get('pool_size', 10)
        assert engine_path, f'model:{engine_path} must exist!'
        with open(engine_path, 'rb') as f:
            with trt.Runtime(TRT_LOGGER) as runtime:
                assert runtime
                self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.inputs = []
        self.outputs = []
        self.tensors = OrderedDict()
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            binding = {'index': idx, 'name': name, 'dtype': dtype, 'shape': list(shape)}
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        self.allocate_max_buffers()
        self.context_pool = Queue(maxsize=self.pool_size)
        for _ in range(self.pool_size):
            self.context_pool.put(self.engine.create_execution_context())
    def allocate_max_buffers(self):
        nvtx.range_push('allocate_max_buffers')
        batch_size = 1
        for idx in range(self.engine.num_io_tensors):
            binding = self.engine.get_tensor_name(idx)
            shape = self.engine.get_tensor_shape(binding)
            is_input = self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT
            if (-1) in shape:
                if is_input:
                    shape = self.engine.get_tensor_profile_shape(binding, 0)[(-1)]
                    batch_size = shape[0]
                else:
                    shape[0] = batch_size
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            tensor = torch.empty(tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]).to(device=self.device)
            self.tensors[binding] = tensor
        nvtx.range_pop()
    def input_spec(self):
        """\n        Get the specs for the input tensor of the network. Useful to prepare memory allocations.\n        :return: Two items, the shape of the input tensor and its (numpy) datatype.\n        """
        specs = []
        for i, o in enumerate(self.inputs):
            specs.append((o['name'], o['shape'], o['dtype']))
            if self.debug:
                print(f"trt input {i} -> {o['name']} -> {o['shape']}")
        return specs
    def output_spec(self):
        """\n        Get the specs for the output tensors of the network. Useful to prepare memory allocations.\n        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.\n        """
        specs = []
        for i, o in enumerate(self.outputs):
            specs.append((o['name'], o['shape'], o['dtype']))
            if self.debug:
                print(f"trt output {i} -> {o['name']} -> {o['shape']}")
        return specs
    def adjust_buffer(self, feed_dict, context):
        """\n        Adjust input buffer sizes and set input shapes in the given execution context.\n        :param feed_dict: A dictionary of inputs as numpy arrays.\n        :param context: The TensorRT execution context to set input shapes.\n        """
        nvtx.range_push('adjust_buffer')
        for name, buf in feed_dict.items():
            input_tensor = self.tensors[name]
            current_shape = list(buf.shape)
            slices = tuple((slice(0, dim) for dim in current_shape))
            input_tensor[slices].copy_(buf)
            context.set_input_shape(name, current_shape)
        nvtx.range_pop()
    def predict(self, feed_dict):
        """\n        Execute inference on a batch of images in synchronous mode using execute_v2.\n        :param feed_dict: A dictionary of inputs as numpy arrays.\n        :return: A dictionary of outputs as PyTorch tensors.\n        """
        with self.lock:
            context = self.context_pool.get()
        try:
            nvtx.range_push('set_tensors')
            self.adjust_buffer(feed_dict, context)
            for name, tensor in self.tensors.items():
                assert tensor.dtype == torch.float32, f'Tensor \'{name}\' should be torch.float32 but is {tensor.dtype}'
                context.set_tensor_address(name, tensor.data_ptr())
            nvtx.range_pop()
            bindings = [tensor.data_ptr() for tensor in self.tensors.values()]
            nvtx.range_push('execute')
            noerror = context.execute_v2(bindings)
            if not noerror:
                raise ValueError('ERROR: inference failed.')
            else:
                nvtx.range_pop()
                return self.tensors
        finally:
            with self.lock:
                self.context_pool.put(context)
    def predict_async(self, feed_dict, stream):
        """\n        Execute inference on a batch of images in asynchronous mode using execute_async_v3.\n        :param feed_dict: A dictionary of inputs as numpy arrays.\n        :param stream: A CUDA stream for asynchronous execution.\n        :return: A dictionary of outputs as PyTorch tensors.\n        """
        with self.lock:
            context = self.context_pool.get()
        try:
            nvtx.range_push('set_tensors')
            self.adjust_buffer(feed_dict, context)
            for name, tensor in self.tensors.items():
                assert tensor.dtype == torch.float32, f'Tensor \'{name}\' should be torch.float32 but is {tensor.dtype}'
                context.set_tensor_address(name, tensor.data_ptr())
            nvtx.range_pop()
            nvtx.range_push('execute_async')
            noerror = context.execute_async_v3(stream)
            if not noerror:
                raise ValueError('ERROR: inference failed.')
            else:
                nvtx.range_pop()
                return self.tensors
        finally:
            with self.lock:
                self.context_pool.put(context)
    def cleanup(self):
        """\n        Clean up all resources associated with the TensorRTPredictor.\n        This method should be called explicitly before deleting the object.\n        """
        if hasattr(self, 'engine') and self.engine is not None:
                del self.engine
                self.engine = None
        if hasattr(self, 'context_pool') and self.context_pool is not None:
                while not self.context_pool.empty():
                    context = self.context_pool.get()
                    del context
                self.context_pool = None
        self.inputs = None
        self.outputs = None
        self.tensors = None
        self.pool_size = None
    def __del__(self):
        self.cleanup()