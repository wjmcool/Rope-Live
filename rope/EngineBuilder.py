
global TRT_LOGGER
import os
import sys
import logging
import platform
import ctypes
import numpy as np
from pathlib import Path
try:
    import tensorrt as trt
except ModuleNotFoundError:
    pass
logging.basicConfig(level=logging.INFO)
logging.getLogger('EngineBuilder').setLevel(logging.INFO)
log = logging.getLogger('EngineBuilder')
if 'trt' in globals():
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
else:
    TRT_LOGGER = {}
class EngineBuilder:
    """\n    Parses an ONNX graph and builds a TensorRT engine from it.\n    """
    def __init__(self, verbose=False, custom_plugin_path=None):
        """\n        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.\n        :param custom_plugin_path: Path to the custom plugin library (DLL or SO).\n        """
        if verbose:
            TRT_LOGGER.min_severity = trt.Logger.Severity.VERBOSE
        trt.init_libnvinfer_plugins(TRT_LOGGER, namespace='')
        self.builder = trt.Builder(TRT_LOGGER)
        self.config = self.builder.create_builder_config()
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3221225472)
        profile = self.builder.create_optimization_profile()
        self.config.add_optimization_profile(profile)
        self.batch_size = None
        self.network = None
        self.parser = None
        if custom_plugin_path is not None:
            if platform.system().lower() == 'linux':
                ctypes.CDLL(custom_plugin_path, mode=ctypes.RTLD_GLOBAL)
            else:
                ctypes.CDLL(custom_plugin_path, mode=ctypes.RTLD_GLOBAL, winmode=0)
    def create_network(self, onnx_path):
        """\n        Parse the ONNX graph and create the corresponding TensorRT network definition.\n        :param onnx_path: The path to the ONNX graph to load.\n        """
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, TRT_LOGGER)
        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, 'rb') as f:
            if not self.parser.parse(f.read()):
                log.error('Failed to load ONNX file: {}'.format(onnx_path))
                for error in range(self.parser.num_errors):
                    log.error(self.parser.get_error(error))
                sys.exit(1)
        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]
        log.info('Network Description')
        for input in inputs:
            self.batch_size = input.shape[0]
            log.info('Input \'{}\' with shape {} and dtype {}'.format(input.name, input.shape, input.dtype))
        for output in outputs:
            log.info('Output \'{}\' with shape {} and dtype {}'.format(output.name, output.shape, output.dtype))
    def create_engine(self, engine_path, precision):
        """\n        Build the TensorRT engine and serialize it to disk.\n        :param engine_path: The path where to serialize the engine to.\n        :param precision: The datatype to use for the engine, either \'fp32\', \'fp16\' or \'int8\'.\n        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        log.info('Building {} Engine in {}'.format(precision, engine_path))
        if precision == 'fp16':
            if not self.builder.platform_has_fast_fp16:
                log.warning('FP16 is not supported natively on this platform/device')
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)
        serialized_engine = self.builder.build_serialized_network(self.network, self.config)
        if serialized_engine is None:
            raise RuntimeError('Errore nella costruzione del motore TensorRT!')
        else:
            with open(engine_path, 'wb') as f:
                log.info('Serializing engine to file: {:}'.format(engine_path))
                f.write(serialized_engine)
def change_extension(file_path, new_extension, version=None):
    """\n    Change the extension of the file path and optionally prepend a version.\n    """
    new_extension = new_extension.lstrip('.')
    if version:
        new_file_path = Path(file_path).with_suffix(f'.{version}.{new_extension}')
        return str(new_file_path)
    else:
        new_file_path = Path(file_path).with_suffix(f'.{new_extension}')
        return str(new_file_path)
def onnx_to_trt(onnx_model_path, trt_model_path=None, precision='fp16', custom_plugin_path=None, verbose=False):
    if trt_model_path is None:
        trt_version = trt.__version__
        trt_model_path = change_extension(onnx_model_path, 'trt', version=trt_version)
    builder = EngineBuilder(verbose=verbose, custom_plugin_path=custom_plugin_path)
    builder.create_network(onnx_model_path)
    builder.create_engine(trt_model_path, precision)