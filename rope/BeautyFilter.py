
import cv2
import numpy as np
import torch
from torchvision.transforms import v2
class BeautyFilter:
    """\n    美颜滤镜类 - 提供磨皮、美白、瘦脸、滤镜等功能\n    优化用于直播场景，性能优先\n    """
    def __init__(self, device='cuda'):
        self.device = device
    def apply_beauty(self, img, face_kps=None, parameters=None):
        """\n        应用美颜效果到图像\n        \n        Args:\n            img: torch.Tensor (C, H, W) 或 numpy.ndarray (H, W, C) - RGB格式\n            face_kps: 人脸关键点 (5点或106点)，在原图的坐标系中\n            parameters: 美颜参数字典\n            \n        Returns:\n            处理后的图像，格式与输入相同\n        """
        if parameters is None:
            return img
        else:
            if not parameters.get('BeautySwitch', False):
                return img
            else:
                is_torch = isinstance(img, torch.Tensor)
                if is_torch:
                    img_np = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                else:
                    img_np = img.copy()
                standard_512_kps = np.array([[192.98138, 239.94708], [318.90277, 239.94708], [256.0, 314.0], [201.0, 371.0], [311.0, 371.0]])
                if parameters.get('BeautyWhitenSlider', 0) > 0:
                    img_np = self.whiten(img_np, parameters['BeautyWhitenSlider'])
                if parameters.get('BeautyFaceSlimSlider', 0) > 0:
                    img_np = self.face_slim(img_np, standard_512_kps, parameters['BeautyFaceSlimSlider'])
                if parameters.get('BeautyFilterTypeTextSel', 'None') not in ['None', '无']:
                    img_np = self.apply_filter(img_np, parameters['BeautyFilterTypeTextSel'], parameters.get('BeautyFilterSlider', 50))
                if is_torch:
                    img_tensor = torch.from_numpy(img_np).to(self.device)
                    img_tensor = img_tensor.permute(2, 0, 1)
                    return img_tensor
                else:
                    return img_np
    def _create_face_mask(self, shape, kps):
        """\n        基于关键点创建一个简单的椭圆人脸遮罩，边缘羽化\n        """
        h, w = shape
        mask = np.zeros((h, w), dtype=np.float32)
        if len(kps) >= 5:
            left_eye = kps[0]
            right_eye = kps[1]
            nose = kps[2]
            left_mouth = kps[3]
            right_mouth = kps[4]
            center_x = (left_eye[0] + right_eye[0] + left_mouth[0] + right_mouth[0]) / 4
            center_y = (left_eye[1] + right_eye[1] + left_mouth[1] + right_mouth[1]) / 4 + (nose[1] - left_eye[1]) * 0.5
            face_width = np.linalg.norm(right_eye - left_eye) * 2.5
            face_height = np.linalg.norm(nose - (left_eye + right_eye) / 2) * 4.0
            axes = (int(face_width / 2), int(face_height / 2))
            center = (int(center_x), int(center_y))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, (-1))
            blur_size = int(max(face_width, face_height) * 0.3)
            if blur_size % 2 == 0:
                blur_size += 1
            mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
        return mask[:, :, np.newaxis]
    def whiten(self, img, strength):
        """\n        美白效果 - 提升亮度和饱和度\n        \n        Args:\n            img: numpy.ndarray (H, W, C) - RGB格式\n            strength: 0-100 强度\n            \n        Returns:\n            处理后的图像\n        """
        if strength <= 0:
            return img
        else:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            brightness_factor = 1.0 + strength / 100.0 * 0.8
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)
            saturation_factor = 1.0 - strength / 100.0 * 0.3
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
            hsv = hsv.astype(np.uint8)
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            return result
    def face_slim(self, img, face_kps, strength):
        """\n        瘦脸效果 - 基于关键点的局部变形\n        \n        Args:\n            img: numpy.ndarray (H, W, C) - RGB格式\n            face_kps: 人脸5关键点 numpy.ndarray (5, 2) 或 (106, 2)\n            strength: 0-100 强度\n            \n        Returns:\n            处理后的图像\n        """
        if strength <= 0 or face_kps is None:
            return img
        else:
            if isinstance(face_kps, torch.Tensor):
                face_kps = face_kps.cpu().numpy()
            if len(face_kps) == 5:
                return self._face_slim_5points(img, face_kps, strength)
            else:
                return self._face_slim_detailed(img, face_kps, strength)
    def _apply_local_warp(self, map_x, map_y, target_pt, src_pt, radius, strength):
        """\n        局部图像形变\n        :param map_x, map_y: 坐标映射网格\n        :param target_pt: 变形后的目标位置 (形变中心)\n        :param src_pt: 原图像中的采样位置\n        :param radius: 影响半径\n        :param strength: 变形强度 (0-1)\n        """
        dist = np.sqrt((map_x - target_pt[0]) ** 2 + (map_y - target_pt[1]) ** 2)
        mask = np.where(dist < radius, (1 - dist / radius) ** 2, 0)
        offset_x = (src_pt[0] - target_pt[0]) * mask * strength
        offset_y = (src_pt[1] - target_pt[1]) * mask * strength
        return (map_x + offset_x, map_y + offset_y)
    def _face_slim_5points(self, img, kps, strength):
        """\n        基于5点关键点的全方位瘦脸（包含额头、脸颊、下颌、下巴）\n        """
        h, w = img.shape[:2]
        left_eye = kps[0]
        right_eye = kps[1]
        nose = kps[2]
        left_mouth = kps[3]
        right_mouth = kps[4]
        eye_dist = np.linalg.norm(right_eye - left_eye)
        face_center_x = (left_eye[0] + right_eye[0]) / 2
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)
        for i in range(h):
            map_x[i, :] = np.arange(w)
        for j in range(w):
            map_y[:, j] = np.arange(h)
        strength_factor = strength / 100.0
        warps = []
        src_left_cheek = np.array([face_center_x - eye_dist * 1.2, nose[1] - eye_dist * 0.1])
        target_left_cheek = src_left_cheek + np.array([eye_dist * 0.15, 0])
        warps.append((src_left_cheek, target_left_cheek, eye_dist * 0.9))
        src_right_cheek = np.array([face_center_x + eye_dist * 1.2, nose[1] - eye_dist * 0.1])
        target_right_cheek = src_right_cheek + np.array([-eye_dist * 0.15, 0])
        warps.append((src_right_cheek, target_right_cheek, eye_dist * 0.9))
        src_left_jaw = np.array([face_center_x - eye_dist * 1.0, nose[1] + eye_dist * 0.4])
        target_left_jaw = src_left_jaw + np.array([eye_dist * 0.18, -eye_dist * 0.1])
        warps.append((src_left_jaw, target_left_jaw, eye_dist * 0.9))
        src_right_jaw = np.array([face_center_x + eye_dist * 1.0, nose[1] + eye_dist * 0.4])
        target_right_jaw = src_right_jaw + np.array([-eye_dist * 0.18, -eye_dist * 0.1])
        warps.append((src_right_jaw, target_right_jaw, eye_dist * 0.9))
        src_chin = np.array([face_center_x, left_mouth[1] + eye_dist * 0.55])
        target_chin = src_chin + np.array([0, -eye_dist * 0.15])
        warps.append((src_chin, target_chin, eye_dist * 0.8))
        src_left_fh = np.array([face_center_x - eye_dist * 0.9, left_eye[1] - eye_dist * 0.6])
        target_left_fh = src_left_fh + np.array([eye_dist * 0.12, 0])
        warps.append((src_left_fh, target_left_fh, eye_dist * 0.9))
        src_right_fh = np.array([face_center_x + eye_dist * 0.9, right_eye[1] - eye_dist * 0.6])
        target_right_fh = src_right_fh + np.array([-eye_dist * 0.12, 0])
        warps.append((src_right_fh, target_right_fh, eye_dist * 0.9))
        for src, target, radius in warps:
            map_x, map_y = self._apply_local_warp(map_x, map_y, target, src, radius, strength_factor)
        map_x = np.clip(map_x, 0, w - 1)
        map_y = np.clip(map_y, 0, h - 1)
        result = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        return result
    def _face_slim_detailed(self, img, kps, strength):
        """\n        基于106点关键点的精确瘦脸（如果有106点数据）\n        这里使用简化版本，实际可以根据106点做更精细的调整\n        """
        kps_5 = kps[:5] if len(kps) >= 5 else kps
        return self._face_slim_5points(img, kps_5, strength)
    def apply_filter(self, img, filter_type, strength):
        """\n        应用滤镜效果\n        \n        Args:\n            img: numpy.ndarray (H, W, C) - RGB格式\n            filter_type: 滤镜类型 (\'Warm\', \'Cool\', \'Natural\', \'Vivid\' 或中文名)\n            strength: 0-100 强度\n            \n        Returns:\n            处理后的图像\n        """
        if filter_type in ['None', '无'] or strength <= 0:
            return img
        else:
            alpha = strength / 100.0
            if filter_type in ['Warm', '暖色调']:
                filtered = img.copy().astype(np.float32)
                filtered[:, :, 0] = np.clip(filtered[:, :, 0] * (1 + alpha * 0.1), 0, 255)
                filtered[:, :, 1] = np.clip(filtered[:, :, 1] * (1 + alpha * 0.05), 0, 255)
                filtered = filtered.astype(np.uint8)
            else:
                if filter_type in ['Cool', '冷色调']:
                    filtered = img.copy().astype(np.float32)
                    filtered[:, :, 2] = np.clip(filtered[:, :, 2] * (1 + alpha * 0.1), 0, 255)
                    filtered[:, :, 0] = np.clip(filtered[:, :, 0] * (1 - alpha * 0.05), 0, 255)
                    filtered = filtered.astype(np.uint8)
                else:
                    if filter_type in ['Natural', '自然色']:
                        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
                        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + alpha * 0.15), 0, 255)
                        hsv = hsv.astype(np.uint8)
                        filtered = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                    else:
                        if filter_type in ['Vivid', '鲜艳色']:
                            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
                            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + alpha * 0.3), 0, 255)
                            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + alpha * 0.1), 0, 255)
                            hsv = hsv.astype(np.uint8)
                            filtered = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                        else:
                            filtered = img
            result = cv2.addWeighted(img, 1 - alpha, filtered, alpha, 0)
            return result
    def apply_beauty_to_face_region(self, img, face_bbox, face_kps, parameters):
        """\n        只对人脸区域应用美颜（性能优化版本）\n        \n        Args:\n            img: torch.Tensor (C, H, W) - RGB格式\n            face_bbox: 人脸边界框 [x1, y1, x2, y2]\n            face_kps: 人脸关键点\n            parameters: 美颜参数\n            \n        Returns:\n            处理后的完整图像\n        """
        if not parameters.get('BeautySwitch', False):
            return img
        else:
            img_np = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            h, w = img_np.shape[:2]
            x1, y1, x2, y2 = face_bbox
            margin = int((x2 - x1) * 0.2)
            x1 = max(0, int(x1 - margin))
            y1 = max(0, int(y1 - margin))
            x2 = min(w, int(x2 + margin))
            y2 = min(h, int(y2 + margin))
            face_region = img_np[y1:y2, x1:x2].copy()
            if face_kps is not None:
                face_kps_adjusted = face_kps.copy()
                face_kps_adjusted[:, 0] -= x1
                face_kps_adjusted[:, 1] -= y1
            else:
                face_kps_adjusted = None
            face_region_beauty = self.apply_beauty(face_region, face_kps_adjusted, parameters)
            if isinstance(face_region_beauty, torch.Tensor):
                face_region_beauty = face_region_beauty.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            mask = np.ones((y2 - y1, x2 - x1), dtype=np.float32)
            feather_size = min(20, (x2 - x1) // 10, (y2 - y1) // 10)
            if feather_size > 0:
                mask = cv2.GaussianBlur(mask, (feather_size * 2 + 1, feather_size * 2 + 1), feather_size / 2)
            mask = np.expand_dims(mask, axis=2)
            img_np[y1:y2, x1:x2] = (face_region_beauty * mask + img_np[y1:y2, x1:x2] * (1 - mask)).astype(np.uint8)
            img_tensor = torch.from_numpy(img_np).to(self.device)
            img_tensor = img_tensor.permute(2, 0, 1)
            return img_tensor