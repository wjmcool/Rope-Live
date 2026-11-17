
import torch
import onnxruntime
import rope.FaceUtil as faceutil
import numpy as np
onnxruntime.set_default_logger_severity(4)
onnxruntime.log_verbosity_level = (-1)
class DFMModel:
    def __init__(self, model_path: str, providers, device='cuda'):
        self._model_path = model_path
        self.providers = providers
        self.device = device
        self.syncvec = torch.empty((1, 1), dtype=torch.float32, device=device)
        sess = self._sess = onnxruntime.InferenceSession(str(model_path), providers=self.providers)
        inputs = sess.get_inputs()
        if len(inputs) == 0 or 'in_face' not in inputs[0].name:
            raise Exception(f'Invalid model {model_path}')
        else:
            self._input_height, self._input_width = inputs[0].shape[1:3]
            self._model_type = 1
            if len(inputs) == 2:
                if 'morph_value' not in inputs[1].name:
                    raise Exception(f'Invalid model {model_path}')
                else:
                    self._model_type = 2
            else:
                if len(inputs) > 2:
                    raise Exception(f'Invalid model {model_path}')
            self.onnx_to_torch_dtype = {'tensor(float)': torch.float32, 'tensor(double)': torch.float64, 'tensor(int32)': torch.int32, 'tensor(int64)': torch.int64}
            self.onnx_to_numpy_dtype = {'tensor(float)': np.float32, 'tensor(double)': np.float64, 'tensor(int32)': np.int32, 'tensor(int64)': np.int64}
    def get_model_path(self):
        return self._model_path
    def get_input_res(self):
        return (self._input_width, self._input_height)
    def has_morph_value(self) -> bool:
        return self._model_type == 2
    def _get_onnx_device_info(self):
        """获取ONNX Runtime兼容的设备信息"""
        if self.device.startswith('cuda'):
            if ':' in self.device:
                device_id = int(self.device.split(':')[1])
                return ('cuda', device_id)
            else:
                device_id = 0
                return ('cuda', device_id)
        else:
            return ('cpu', 0)
    def convert(self, img, morph_factor=0.75, rct=False):
        """\n        img    torch.Tensor  CHW uint8,float32\n        morph_factor   float   used if model supports it\n        returns:\n         img        NHW3  same dtype as img\n         celeb_mask NHW1  same dtype as img\n         face_mask  NHW1  same dtype as img\n        """
        dtype = img.dtype
        img = self.to_ufloat32(img)
        img = torch.unsqueeze(img, 0)
        img = torch.nn.functional.interpolate(img, size=(self._input_height, self._input_width), mode='bilinear', align_corners=False)
        img = img[:, [2, 1, 0], :, :]
        img = img.permute(0, 2, 3, 1).contiguous()
        N, H, W, C = img.shape
        io_binding = self._sess.io_binding()
        device_type, device_id = self._get_onnx_device_info()
        io_binding.bind_input(name='in_face:0', device_type=device_type, device_id=device_id, element_type=np.float32, shape=img.shape, buffer_ptr=img.data_ptr())
        if self._model_type == 2:
            morph_factor_t = torch.tensor([morph_factor], dtype=torch.float32, device=self.device)
            io_binding.bind_input(name='morph_value:0', device_type=device_type, device_id=device_id, element_type=np.float32, shape=morph_factor_t.shape, buffer_ptr=morph_factor_t.data_ptr())
        outputs = self._sess.get_outputs()
        binding_outputs = []
        for idx, output in enumerate(outputs):
            shape = self.convert_shape(output.shape)
            torch_dtype = self.onnx_to_torch_dtype[output.type]
            tensor_output = torch.empty(shape, dtype=torch_dtype, device=self.device).contiguous()
            binding_outputs.append(tensor_output)
            io_binding.bind_output(name=output.name, device_type=device_type, device_id=device_id, element_type=self.onnx_to_numpy_dtype[output.type], shape=shape, buffer_ptr=binding_outputs[idx].data_ptr())
        if self.device == 'cuda':
            torch.cuda.synchronize()
        else:
            if self.device != 'cpu':
                self.syncvec.cpu()
        self._sess.run_with_iobinding(io_binding)
        out_face_mask = self.to_dtype(self.ch(self.resize(binding_outputs[0], (W, H)), 1), dtype)
        out_face_mask = torch.squeeze(out_face_mask, dim=0)
        out_celeb = self.to_dtype(self.ch(self.resize(binding_outputs[1], (W, H)), 3), dtype)
        out_celeb_mask = self.to_dtype(self.ch(self.resize(binding_outputs[2], (W, H)), 1), dtype)
        if rct:
            img = self.to_dtype(img, dtype)
            out_celeb = self.rct(out_celeb, img, out_celeb_mask, out_celeb_mask, 0.3)
            out_celeb = torch.squeeze(out_celeb, dim=0)
            if out_celeb.shape[(-1)] == 3:
                out_celeb = out_celeb[..., [2, 1, 0]]
        else:
            out_celeb = torch.squeeze(out_celeb, dim=0)
            if out_celeb.shape[(-1)] == 3:
                out_celeb = out_celeb[..., [2, 1, 0]]
        out_celeb_mask = torch.squeeze(out_celeb_mask, dim=0)
        return (out_celeb, out_celeb_mask, out_face_mask)
    def rct(self, img: torch.Tensor, like: torch.Tensor, mask: torch.Tensor=None, like_mask: torch.Tensor=None, mask_cutoff=0.5):
        """\n        Transfer color using the RCT method.\n\n        Args:\n            img (torch.Tensor): [N, H, W, 3] torch.uint8/torch.float32\n            like (torch.Tensor): [N, H, W, 3] torch.uint8/torch.float32\n            mask (torch.Tensor, optional): [N, H, W, 1] torch.uint8/torch.float32\n            like_mask (torch.Tensor, optional): [N, H, W, 1] torch.uint8/torch.float32\n            mask_cutoff (float, optional): Cutoff value for masks. Defaults to 0.5.\n\n        Returns:\n            torch.Tensor: The color-transferred image. [N, C, H, W]\n        """
        dtype = img.dtype
        N = img.shape[0]
        img = self.to_ufloat32(img).permute(0, 3, 1, 2)
        like_for_stat = self.to_ufloat32(like).permute(0, 3, 1, 2)
        img_lab = torch.stack([faceutil.rgb_to_lab(img[i], False) for i in range(N)])
        like_lab = torch.stack([faceutil.rgb_to_lab(like_for_stat[i], False) for i in range(N)])
        if like_mask is not None:
            like_mask = self.get_image(self.ch(self.to_ufloat32(like_mask), 1), 'NHW')
            like_mask = like_mask.unsqueeze(1).expand((-1), 3, (-1), (-1))
            like_lab = like_lab.clone()
            like_lab[like_mask < mask_cutoff] = 0
        img_for_stat = img_lab.clone()
        if mask is not None:
            mask = self.get_image(self.ch(self.to_ufloat32(mask), 1), 'NHW')
            mask = mask.unsqueeze(1).expand((-1), 3, (-1), (-1))
            img_for_stat = img_for_stat.clone()
            img_for_stat[mask < mask_cutoff] = 0
        img_out = torch.zeros_like(img_lab)
        for i in range(N):
            source_l_mean, source_l_std = (img_for_stat[i, 0].mean(), img_for_stat[i, 0].std())
            source_a_mean, source_a_std = (img_for_stat[i, 1].mean(), img_for_stat[i, 1].std())
            source_b_mean, source_b_std = (img_for_stat[i, 2].mean(), img_for_stat[i, 2].std())
            like_l_mean, like_l_std = (like_lab[i, 0].mean(), like_lab[i, 0].std())
            like_a_mean, like_a_std = (like_lab[i, 1].mean(), like_lab[i, 1].std())
            like_b_mean, like_b_std = (like_lab[i, 2].mean(), like_lab[i, 2].std())
            source_l = img_lab[i, 0]
            source_a = img_lab[i, 1]
            source_b = img_lab[i, 2]
            source_l = (source_l - source_l_mean) * (like_l_std / (source_l_std + 1e-06)) + like_l_mean
            source_a = (source_a - source_a_mean) * (like_a_std / (source_a_std + 1e-06)) + like_a_mean
            source_b = (source_b - source_b_mean) * (like_b_std / (source_b_std + 1e-06)) + like_b_mean
            source_l = torch.clamp(source_l, 0, 100)
            source_a = torch.clamp(source_a, (-127), 127)
            source_b = torch.clamp(source_b, (-127), 127)
            img_out[i] = torch.stack([source_l, source_a, source_b], dim=0)
        img_out = torch.stack([faceutil.lab_to_rgb(img_out[i], False) for i in range(N)])
        img_out = self.to_dtype(img_out, dtype).permute(0, 2, 3, 1)
        return img_out
    def convert_shape(self, shape):
        return tuple((int(dim) if isinstance(dim, int) or (isinstance(dim, str) and dim.isdigit()) else 1 for dim in shape))
    def to_dtype(self, img, dtype, from_tanh=False):
        if dtype == torch.float32:
            return self.to_ufloat32(img, from_tanh=from_tanh)
        else:
            if dtype == torch.uint8:
                return self.to_uint8(img, from_tanh=from_tanh)
            else:
                raise ValueError('unsupported dtype')
    def to_ufloat32(self, img, as_tanh=False, from_tanh=False):
        """\n        Convert to uniform float32\n        """
        if img.dtype == torch.uint8:
            img = img.to(torch.float32)
            if as_tanh:
                img /= 127.5
                img -= 1.0
                return img
            else:
                img /= 255.0
                return img
        else:
            if img.dtype in (torch.float32, torch.float64) and from_tanh:
                    img += 1.0
                    img /= 2.0
            return img
    def to_uint8(self, img, from_tanh=False):
        """\n        Convert to uint8\n\n        if current image dtype is float32/64, then image will be multiplied by *255\n        """
        if img.dtype in (torch.float32, torch.float64):
            if from_tanh:
                img += 1.0
                img /= 2.0
            img *= 255.0
            img = torch.clamp(img, 0, 255)
        return img.to(torch.uint8)
    def get_image(self, img, format):
        """\n        Returns a PyTorch tensor image with the desired format.\n\n        Args:\n            img (torch.Tensor): Input image tensor.\n            format (str): Desired format, e.g., \'NHWC\', \'HWCN\', \'NHW\'.\n\n        Returns:\n            torch.Tensor: The image tensor in the desired format.\n        """
        format = format.upper()
        N_slice = 0 if 'N' not in format else slice(None)
        H_slice = 0 if 'H' not in format else slice(None)
        W_slice = 0 if 'W' not in format else slice(None)
        C_slice = 0 if 'C' not in format else slice(None)
        img = img[N_slice, H_slice, W_slice, C_slice]
        f = ''
        if 'N' in format:
            f += 'N'
        if 'H' in format:
            f += 'H'
        if 'W' in format:
            f += 'W'
        if 'C' in format:
            f += 'C'
        if f != format:
            d = {s: i for i, s in enumerate(f)}
            transpose_order = [d[s] for s in format]
            img = img.permute(transpose_order)
        return img.contiguous()
    def ch(self, img, TC: int):
        """\n        Clips or expands the channel dimension to the target number of channels.\n\n        Args:\n            img (torch.Tensor): Input image tensor with shape (N, H, W, C).\n            TC (int): Target number of channels, must be >= 1.\n\n        Returns:\n            torch.Tensor: Image tensor with the target number of channels.\n        """
        N, H, W, C = img.shape
        if TC <= 0:
            raise ValueError(f'channels must be positive value, not {TC}')
        else:
            if TC > C:
                img = img[..., 0:1]
                img = img.repeat(1, 1, 1, TC)
                return img
            else:
                if TC < C:
                    img = img[..., :TC]
                return img
    def resize(self, img: torch.Tensor, size: tuple, interpolation: str='bilinear'):
        """\n        Resize a PyTorch tensor image to the target size (W, H).\n\n        Args:\n            img (torch.Tensor): Input image tensor with shape (N, H, W, C).\n            size (Tuple[int, int]): Target size as (width, height).\n            interpolation (str): Interpolation method, can be \'nearest\', \'bilinear\', or \'bicubic\'.\n\n        Returns:\n            torch.Tensor: Resized image tensor with shape (N, TH, TW, C).\n        """
        supported_interpolations = {'bilinear', 'nearest', 'bicubic'}
        if interpolation not in supported_interpolations:
            raise ValueError(f'Interpolation \'{interpolation}\' not supported. Choose from {supported_interpolations}.')
        else:
            N, H, W, C = img.shape
            TW, TH = size
            if W != TW or H != TH:
                img = img.permute(0, 3, 1, 2)
                img = torch.nn.functional.interpolate(img, size=(TH, TW), mode=interpolation, align_corners=False)
                img = img.permute(0, 2, 3, 1)
            return img