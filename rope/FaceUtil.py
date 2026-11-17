
import cv2
import math
from math import sin, cos, acos, degrees
import numpy as np
from skimage import transform as trans
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
from numpy.linalg import norm as l2norm
import kornia.geometry.transform as kgm
src1 = np.array([[51.642, 50.115], [57.617, 49.99], [35.74, 69.007], [51.157, 89.05], [57.025, 89.702]], dtype=np.float32)
src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111], [45.177, 86.19], [64.246, 86.758]], dtype=np.float32)
src3 = np.array([[39.73, 51.138], [72.27, 51.138], [56.0, 68.493], [42.463, 87.01], [69.537, 87.01]], dtype=np.float32)
src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111], [48.167, 86.758], [67.236, 86.19]], dtype=np.float32)
src5 = np.array([[54.796, 49.99], [60.771, 50.115], [76.673, 69.007], [55.388, 89.702], [61.257, 89.05]], dtype=np.float32)
src = np.array([src1, src2, src3, src4, src5])
src_map = {112: src, 224: src * 2}
arcface_src = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
arcface_src = np.expand_dims(arcface_src, axis=0)
arcface_src_cuda = torch.tensor([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=torch.float32)
if torch.cuda.is_available():
    try:
        arcface_src_cuda = arcface_src_cuda.to('cuda')
    except Exception:
        pass
def pad_image_by_size(img, image_size):
    if not isinstance(image_size, tuple):
        image_size = (image_size, image_size)
    w, h = (img.size(dim=2), img.size(dim=1))
    target_h, target_w = image_size
    if w < target_w or h < target_h:
        pad_right = max(target_w - w, 0)
        pad_bottom = max(target_h - h, 0)
        img = torch.nn.functional.pad(img, (0, pad_right, 0, pad_bottom), mode='constant', value=0)
    return img
def transform(img, center, output_size, scale, rotation):
    img = pad_image_by_size(img, output_size)
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=((-1) * cx, (-1) * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2, output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = v2.functional.affine(img, np.rad2deg(t.rotation), (t.translation[0], t.translation[1]), t.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center=(0, 0))
    cropped = v2.functional.crop(cropped, 0, 0, output_size, output_size)
    return (cropped, M)
def trans_points2d(pts, M):
    ones_column = np.ones((pts.shape[0], 1), dtype=np.float32)
    homogeneous_pts = np.hstack([pts, ones_column])
    transformed_pts = np.dot(homogeneous_pts, M.T)
    return transformed_pts[:, :2]
def trans_points3d(pts, M):
    scale = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
    ones_column = np.ones((pts.shape[0], 1), dtype=np.float32)
    homogeneous_pts = np.hstack([pts[:, :2], ones_column])
    transformed_2d = np.dot(homogeneous_pts, M.T)
    scaled_z = pts[:, 2] * scale
    transformed_pts = np.hstack([transformed_2d[:, :2], scaled_z.reshape((-1), 1)])
    return transformed_pts
def trans_points(pts, M):
    if pts.shape[1] == 2:
        return trans_points2d(pts, M)
    else:
        return trans_points3d(pts, M)
def estimate_affine_matrix_3d23d(X, Y):
    """ Using least-squares solution\n    Args:\n        X: [n, 3]. 3d points(fixed)\n        Y: [n, 3]. corresponding 3d points(moving). Y = PX\n    Returns:\n        P_Affine: (3, 4). Affine camera matrix (the third row is [0, 0, 0, 1]).\n    """
    X_homo = np.hstack((X, np.ones([X.shape[0], 1])))
    P = np.linalg.lstsq(X_homo, Y, rcond=None)[0].T
    return P
def P2sRt(P):
    """ decompositing camera matrix P\n    Args:\n        P: (3, 4). Affine Camera Matrix.\n    Returns:\n        s: scale factor.\n        R: (3, 3). rotation matrix.\n        t: (3,). translation.\n    """
    t = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)
    R = np.concatenate((r1, r2, r3), 0)
    return (s, R, t)
def matrix2angle(R):
    """ get three Euler angles from Rotation Matrix\n    Args:\n        R: (3,3). rotation matrix\n    Returns:\n        x: pitch\n        y: yaw\n        z: roll\n    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-06
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    rx, ry, rz = (x * 180 / np.pi, y * 180 / np.pi, z * 180 / np.pi)
    return (rx, ry, rz)
def warp_affine_torchvision(img, matrix, image_size, rotation_ratio=0.0, border_value=0.0, border_mode='replicate', interpolation_value=v2.functional.InterpolationMode.NEAREST, device='cpu'):
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    if isinstance(img, torch.Tensor):
        img_tensor = img.to(device).float()
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
    else:
        img_tensor = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
    t = trans.SimilarityTransform()
    t.params[0:2] = matrix
    rotation = t.rotation
    if rotation_ratio != 0:
        rotation *= rotation_ratio
    if border_mode == 'replicate':
        fill = [border_value] * img_tensor.shape[1]
    else:
        if border_mode == 'constant':
            fill = [border_value] * img_tensor.shape[1]
        else:
            raise ValueError('Unsupported border_mode. Use \'replicate\' or \'constant\'.')
    warped_img_tensor = v2.functional.affine(img_tensor, angle=rotation, translate=(t.translation[0], t.translation[1]), scale=t.scale, shear=(0.0, 0.0), interpolation=interpolation_value, center=(0, 0), fill=fill)
    warped_img_tensor = v2.functional.crop(warped_img_tensor, 0, 0, image_size[1], image_size[0])
    return warped_img_tensor.squeeze(0)
def umeyama(src, dst, estimate_scale):
    num = src.shape[0]
    dim = src.shape[1]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    A = np.dot(dst_demean.T, src_demean) / num
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = (-1)
    T = np.eye(dim + 1, dtype=np.double)
    U, S, V = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    else:
        if rank == dim - 1:
            if np.linalg.det(U) * np.linalg.det(V) > 0:
                T[:dim, :dim] = np.dot(U, V)
            else:
                s = d[dim - 1]
                d[dim - 1] = (-1)
                T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
                d[dim - 1] = s
        else:
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V.T))
        if estimate_scale:
            scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
        else:
            scale = 1.0
        T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
        T[:dim, :dim] *= scale
        return T
def get_matrix(lmk, templates):
    if templates.shape[0] == 1:
        return umeyama(lmk, templates[0], True)[0:2, :]
    else:
        test_lmk = np.insert(lmk, 2, values=np.ones(5), axis=1)
        min_error, best_matrix = (float('inf'), [])
        for i in np.arange(templates.shape[0]):
            matrix = umeyama(lmk, templates[i], True)[0:2, :]
            error = np.sum(np.sqrt(np.sum((np.dot(matrix, test_lmk.T).T - templates[i]) ** 2, axis=1)))
            if error < min_error:
                min_error, best_matrix = (error, matrix)
        return best_matrix
def align_crop(img, lmk, image_size, mode='arcfacemap', interpolation=v2.InterpolationMode.NEAREST):
    if mode != 'arcfacemap':
        if mode == 'arcface112':
            templates = float(image_size) / 112.0 * arcface_src
        else:
            factor = float(image_size) / 128.0
            templates = arcface_src * factor
            templates[:, 0] += factor * 8.0
    else:
        templates = float(image_size) / 112.0 * src_map[112]
    matrix = get_matrix(lmk, templates)
    warped = warp_affine_torchvision(img, matrix, (image_size, image_size), rotation_ratio=57.2958, border_value=0.0, border_mode='replicate', interpolation_value=v2.functional.InterpolationMode.NEAREST, device=img.device)
    return (warped, matrix)
def get_arcface_template(image_size=112, mode='arcface112'):
    if mode == 'arcface112':
        template = float(image_size) / 112.0 * arcface_src
        return template
    else:
        if mode == 'arcface128':
            factor = float(image_size) / 128.0
            template = arcface_src * factor
            template[:, 0] += factor * 8.0
            return template
        else:
            template = float(image_size) / 112.0 * src_map[112]
            return template
def estimate_norm_arcface_template(lmk, src=arcface_src):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return (min_M, min_index)
def estimate_norm(lmk, image_size=112, mode='arcface112'):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    if mode != 'arcfacemap':
        if mode == 'arcface112':
            src = float(image_size) / 112.0 * arcface_src
        else:
            factor = float(image_size) / 128.0
            src = arcface_src * factor
            src[:, 0] += factor * 8.0
    else:
        src = float(image_size) / 112.0 * src_map[112]
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return (min_M, min_index)
def warp_face_by_bounding_box(img, bboxes, image_size=112):
    img = pad_image_by_size(img, image_size)
    source_points = np.array([[bboxes[0], bboxes[1]], [bboxes[2], bboxes[1]], [bboxes[0], bboxes[3]], [bboxes[2], bboxes[3]]]).astype(np.float32)
    target_points = np.array([[0, 0], [image_size, 0], [0, image_size], [image_size, image_size]]).astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(source_points, target_points)
    img = v2.functional.affine(img, tform.rotation, (tform.translation[0], tform.translation[1]), tform.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center=(0, 0))
    img = v2.functional.crop(img, 0, 0, image_size, image_size)
    M = tform.params[0:2]
    return (img, M)
def warp_face_by_face_landmark_5(img, kpss, image_size=112, mode='arcface112', interpolation=v2.InterpolationMode.NEAREST):
    img = pad_image_by_size(img, image_size)
    M, pose_index = estimate_norm(kpss, image_size, mode=mode)
    t = trans.SimilarityTransform()
    t.params[0:2] = M
    img = v2.functional.affine(img, t.rotation * 57.2958, (t.translation[0], t.translation[1]), t.scale, 0, interpolation=interpolation, center=(0, 0))
    img = v2.functional.crop(img, 0, 0, image_size, image_size)
    return (img, M)
def getRotationMatrix2D(center, output_size, scale, rotation, is_clockwise=True):
    scale_ratio = scale
    if not is_clockwise:
        rotation = -rotation
    rot = float(rotation) * np.pi / 180.0
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=((-1) * cx, (-1) * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2, output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    return M
def invertAffineTransform(M):
    """\n    t = trans.SimilarityTransform()\n    t.params[0:2] = M\n    IM = t.inverse.params[0:2, :]\n    """
    M_H = np.vstack([M, np.array([0, 0, 1])])
    IM = np.linalg.inv(M_H)
    return IM
def warp_face_by_bounding_box_for_landmark_68(img, bbox, input_size):
    """\n    :param img: raw image\n    :param bbox: the bbox for the face\n    :param input_size: tuple input image size\n    :return:\n    """
    img = pad_image_by_size(img, input_size[0])
    scale = 195 / np.subtract(bbox[2:], bbox[:2]).max()
    translation = (256 - np.add(bbox[2:], bbox[:2]) * scale) * 0.5
    rotation = 0
    t1 = trans.SimilarityTransform(scale=scale)
    t2 = trans.SimilarityTransform(rotation=rotation)
    t3 = trans.SimilarityTransform(translation=translation)
    t = t1 + t2 + t3
    affine_matrix = np.array([[scale, 0, translation[0]], [0, scale, translation[1]]])
    crop_image = v2.functional.affine(img, t.rotation, (t.translation[0], t.translation[1]), t.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center=(0, 0))
    crop_image = v2.functional.crop(crop_image, 0, 0, input_size[1], input_size[0])
    if torch.mean(crop_image.to(dtype=torch.float32)[0, :, :]) < 30:
        crop_image = cv2.cvtColor(crop_image.permute(1, 2, 0).to('cpu').numpy(), cv2.COLOR_RGB2Lab)
        crop_image[:, :, 0] = cv2.createCLAHE(clipLimit=2).apply(crop_image[:, :, 0])
        crop_image = torch.from_numpy(cv2.cvtColor(crop_image, cv2.COLOR_Lab2RGB)).to(img.device).permute(2, 0, 1)
    return (crop_image, affine_matrix)
def warp_face_by_bounding_box_for_landmark_98(img, bbox_org, input_size):
    """\n    :param img: raw image\n    :param bbox: the bbox for the face\n    :param input_size: tuple input image size\n    :return:\n    """
    img = pad_image_by_size(img, input_size[0])
    bbox = bbox_org.copy()
    min_face = 20
    base_extend_range = [0.2, 0.3]
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    if bbox_width <= min_face or bbox_height <= min_face:
        return (None, None)
    else:
        add = int(max(bbox_width, bbox_height))
        bimg = torch.nn.functional.pad(img, (add, add, add, add), 'constant', 0)
        bbox += add
        face_width = (1 + 2 * base_extend_range[0]) * bbox_width
        center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
        bbox[0] = center[0] - face_width // 2
        bbox[1] = center[1] - face_width // 2
        bbox[2] = center[0] + face_width // 2
        bbox[3] = center[1] + face_width // 2
        bbox = bbox.astype(np.int32)
        crop_image = bimg[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]
        h, w = (crop_image.size(dim=1), crop_image.size(dim=2))
        t_resize = v2.Resize((input_size[1], input_size[0]), antialias=False)
        crop_image = t_resize(crop_image)
        return (crop_image, [h, w, bbox[1], bbox[0], add])
def create_bounding_box_from_face_landmark_106_98_68(face_landmark_106_98_68):
    min_x, min_y = np.min(face_landmark_106_98_68, axis=0)
    max_x, max_y = np.max(face_landmark_106_98_68, axis=0)
    bounding_box = np.array([min_x, min_y, max_x, max_y]).astype(np.int16)
    return bounding_box
def convert_face_landmark_68_to_5(face_landmark_68, face_landmark_68_score):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55], dtype=np.int32) - 1
    face_landmark_5 = np.stack([np.mean(face_landmark_68[lm_idx[[1, 2]], :], 0), np.mean(face_landmark_68[lm_idx[[3, 4]], :], 0), face_landmark_68[lm_idx[0], :], face_landmark_68[lm_idx[5], :], face_landmark_68[lm_idx[6], :]], axis=0)
    if np.any(face_landmark_68_score):
        face_landmark_5_score = np.stack([np.mean(face_landmark_68_score[lm_idx[[1, 2]], :], 0), np.mean(face_landmark_68_score[lm_idx[[3, 4]], :], 0), face_landmark_68_score[lm_idx[0], :], face_landmark_68_score[lm_idx[5], :], face_landmark_68_score[lm_idx[6], :]], axis=0)
        return (face_landmark_5, face_landmark_5_score)
    else:
        face_landmark_5_score = np.array([])
        return (face_landmark_5, face_landmark_5_score)
def convert_face_landmark_98_to_5(face_landmark_98, face_landmark_98_score):
    face_landmark_5 = np.array([face_landmark_98[96], face_landmark_98[97], face_landmark_98[54], face_landmark_98[76], face_landmark_98[82]])
    face_landmark_5_score = np.array([face_landmark_98_score[96], face_landmark_98_score[97], face_landmark_98_score[54], face_landmark_98_score[76], face_landmark_98_score[82]])
    return (face_landmark_5, face_landmark_5_score)
def convert_face_landmark_106_to_5(face_landmark_106):
    face_landmark_5 = np.array([face_landmark_106[38], face_landmark_106[88], face_landmark_106[86], face_landmark_106[52], face_landmark_106[61]])
    return face_landmark_5
def convert_face_landmark_203_to_5(face_landmark_203, use_mean_eyes=False):
    if use_mean_eyes:
        eye_left = np.mean(face_landmark_203[[0, 6, 12, 18]], axis=0)
        eye_right = np.mean(face_landmark_203[[24, 30, 36, 42]], axis=0)
    else:
        eye_left = face_landmark_203[197]
        eye_right = face_landmark_203[198]
    nose = face_landmark_203[201]
    lip_left = face_landmark_203[48]
    lip_right = face_landmark_203[66]
    face_landmark_5 = np.array([eye_left, eye_right, nose, lip_left, lip_right])
    return face_landmark_5
def convert_face_landmark_478_to_5(face_landmark_478, use_mean_eyes=False):
    if use_mean_eyes:
        eye_left = np.mean(face_landmark_478[[472, 471, 470, 469]], axis=0)
        eye_right = np.mean(face_landmark_478[[477, 476, 475, 474]], axis=0)
    else:
        eye_left = face_landmark_478[468]
        eye_right = face_landmark_478[473]
    nose = face_landmark_478[4]
    lip_left = face_landmark_478[61]
    lip_right = face_landmark_478[291]
    face_landmark_5 = np.array([eye_left, eye_right, nose, lip_left, lip_right])
    return face_landmark_5
def convert_face_landmark_x_to_5(pts, **kwargs):
    pts_score = kwargs.get('pts_score', [])
    use_mean_eyes = kwargs.get('use_mean_eyes', False)
    if pts.shape[0] == 5:
        return pts
    else:
        if pts.shape[0] == 68:
            pt5 = convert_face_landmark_68_to_5(face_landmark_68=pts, face_landmark_68_score=pts_score)
            return pt5
        else:
            if pts.shape[0] == 98:
                pt5 = convert_face_landmark_98_to_5(face_landmark_98=pts, face_landmark_98_score=pts_score)
                return pt5
            else:
                if pts.shape[0] == 106:
                    pt5 = convert_face_landmark_106_to_5(face_landmark_106=pts)
                    return pt5
                else:
                    if pts.shape[0] == 203:
                        pt5 = convert_face_landmark_203_to_5(face_landmark_203=pts, use_mean_eyes=use_mean_eyes)
                        return pt5
                    else:
                        if pts.shape[0] == 478:
                            pt5 = convert_face_landmark_478_to_5(face_landmark_478=pts, use_mean_eyes=use_mean_eyes)
                            return pt5
                        else:
                            raise Exception(f'Unknow shape: {pts.shape}')
def test_bbox_landmarks(img, bbox, kpss, caption='image', show_kpss_label=False):
    image = img.permute(1, 2, 0).to('cpu').numpy().copy()
    if len(bbox) > 0:
        box = bbox.astype(int)
        color = (255, 0, 0)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
    if len(kpss) > 0:
        for i in range(kpss.shape[0]):
            kps = kpss[i].astype(int)
            color = (0, 0, 255)
            cv2.circle(image, (kps[0], kps[1]), 1, color, 2)
            if show_kpss_label:
                if kpss.shape[0] == 5:
                    match i:
                        case 0:
                            text = 'LE'
                        case 1:
                            text = 'RE'
                        case 2:
                            text = 'NO'
                        case 3:
                            text = 'LM'
                        case 4:
                            text = 'RM'
                else:
                    text = str(i)
                    image = cv2.putText(image, text, (kps[0], kps[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA, False)
    cv2.imshow(caption, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def test_multi_bbox_landmarks(img, bboxes, kpss, caption='image', show_kpss_label=False):
    if len(bboxes) > 0 and len(kpss) > 0:
        for i in range(np.array(kpss).shape[0]):
            test_bbox_landmarks(img, bboxes[i], kpss[i], caption=caption, show_kpss_label=show_kpss_label)
    else:
        if len(bboxes) > 0:
            for i in range(np.array(bboxes).shape[0]):
                test_bbox_landmarks(img, bboxes[i], [], caption=caption, show_kpss_label=show_kpss_label)
        else:
            if len(kpss) > 0:
                for i in range(np.array(kpss).shape[0]):
                    test_bbox_landmarks(img, [], kpss[i], caption=caption, show_kpss_label=show_kpss_label)
def detect_img_color(img):
    frame = img.permute(1, 2, 0)
    b = frame[:, :, :1]
    g = frame[:, :, 1:2]
    r = frame[:, :, 2:]
    b_mean = torch.mean(b.to(float))
    g_mean = torch.mean(g.to(float))
    r_mean = torch.mean(r.to(float))
    if b_mean > g_mean and b_mean > r_mean:
            return 'BGR'
    if g_mean > r_mean and g_mean > b_mean:
            return 'GBR'
    return 'RGB'
def get_face_orientation(face_size, lmk):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    src = np.squeeze(arcface_src, axis=0)
    src = float(face_size) / 112.0 * src
    tform.estimate(lmk, src)
    angle_deg_to_front = np.rad2deg(tform.rotation)
    return angle_deg_to_front
def rgb_to_yuv(image, normalize=False):
    """\n    Convert an RGB image to YUV.\n    Args:\n        image (torch.Tensor): The input image tensor in RGB format (C, H, W) with values in the range [0, 255].\n    Returns:\n        torch.Tensor: The image tensor in YUV format (C, H, W).\n    """
    if normalize:
        image = torch.div(image, 255.0)
    conversion_matrix = torch.tensor([[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]], device=image.device, dtype=image.dtype)
    yuv_image = torch.tensordot(image.permute(1, 2, 0), conversion_matrix, dims=1).permute(2, 0, 1)
    return yuv_image
def yuv_to_rgb(image, normalize=False):
    """\n    Convert a YUV image to RGB.\n    Args:\n        image (torch.Tensor): The input image tensor in YUV format (C, H, W) with values in the range [0, 1].\n    Returns:\n        torch.Tensor: The image tensor in RGB format (C, H, W).\n    """
    conversion_matrix = torch.tensor([[1, 0, 1.13983], [1, -0.39465, -0.5806], [1, 2.03211, 0]], device=image.device, dtype=image.dtype)
    rgb_image = torch.tensordot(image.permute(1, 2, 0), conversion_matrix, dims=1).permute(2, 0, 1)
    rgb_image = torch.clamp(rgb_image, 0, 1)
    if normalize:
        rgb_image = torch.mul(rgb_image, 255.0)
    return rgb_image
def rgb_to_lab(rgb, normalize=False):
    if normalize:
        rgb = rgb / 255.0
    rgb = rgb.permute(1, 2, 0).contiguous()
    mask = rgb > 0.04045
    rgb_linear = torch.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    rgb_linear = rgb_linear.view((-1), 3)
    matrix_rgb_to_xyz = torch.tensor([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.072175], [0.0193339, 0.119192, 0.9503041]], dtype=rgb.dtype, device=rgb.device)
    xyz = torch.matmul(rgb_linear, matrix_rgb_to_xyz.T)
    white_point = torch.tensor([0.95047, 1.0, 1.08883], dtype=xyz.dtype, device=xyz.device)
    xyz = xyz / white_point
    epsilon = 0.008856
    kappa = 903.3
    mask = xyz > epsilon
    f_xyz = torch.where(mask, xyz ** 0.3333333333333333, (kappa * xyz + 16) / 116)
    L = 116 * f_xyz[:, 1] - 16
    a = 500 * (f_xyz[:, 0] - f_xyz[:, 1])
    b = 200 * (f_xyz[:, 1] - f_xyz[:, 2])
    lab = torch.stack([L, a, b], dim=1)
    lab = lab.view(rgb.shape[0], rgb.shape[1], 3)
    lab = lab.permute(2, 0, 1)
    return lab
def lab_to_rgb(lab, normalize=False):
    if lab.dim() != 3 or lab.shape[0] != 3:
        raise ValueError('LAB tensor must have shape (3, H, W)')
    else:
        lab = lab.permute(1, 2, 0).contiguous()
        L = lab[:, :, 0]
        a = lab[:, :, 1]
        b = lab[:, :, 2]
        epsilon = 0.008856
        kappa = 903.3
        fy = (L + 16) / 116
        fx = fy + a / 500
        fz = fy - b / 200
        fx3 = fx ** 3
        fz3 = fz ** 3
        x = torch.where(fx3 > epsilon, fx3, (116 * fx - 16) / kappa)
        y = torch.where(L > kappa * epsilon, ((L + 16) / 116) ** 3, L / kappa)
        z = torch.where(fz3 > epsilon, fz3, (116 * fz - 16) / kappa)
        white_point = torch.tensor([0.95047, 1.0, 1.08883], dtype=lab.dtype, device=lab.device)
        xyz = torch.stack([x, y, z], dim=2) * white_point
        xyz = xyz.view((-1), 3)
        matrix_xyz_to_rgb = torch.tensor([[3.2404542, -1.5371385, -0.4985314], [-0.969266, 1.8760108, 0.041556], [0.0556434, -0.2040259, 1.0572252]], dtype=lab.dtype, device=lab.device)
        rgb_linear = torch.matmul(xyz, matrix_xyz_to_rgb.T)
        mask = rgb_linear > 0.0031308
        rgb = torch.where(mask, 1.055 * rgb_linear ** 0.4166666666666667 - 0.055, 12.92 * rgb_linear)
        rgb = rgb.view(lab.shape[0], lab.shape[1], 3)
        rgb = torch.clamp(rgb, 0.0, 1.0)
        rgb = rgb.permute(2, 0, 1)
        if normalize:
            rgb = rgb * 255.0
        return rgb
def parse_pt2_from_pt101(pt101, use_lip=True):
    """\n    parsing the 2 points according to the 101 points, which cancels the roll\n    """
    pt_left_eye = np.mean(pt101[[39, 42, 45, 48]], axis=0)
    pt_right_eye = np.mean(pt101[[51, 54, 57, 60]], axis=0)
    if use_lip:
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt101[75] + pt101[81]) / 2
        pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
        return pt2
    else:
        pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)
        return pt2
def parse_pt2_from_pt98(pt98, use_lip=True, use_mean_eyes=False):
    """\n    parsing the 2 points according to the 98 points, which cancels the roll\n    """
    if use_mean_eyes:
        pt_left_eye = np.mean(pt98[[66, 60, 62, 64]], axis=0)
        pt_right_eye = np.mean(pt98[[74, 68, 70, 72]], axis=0)
    else:
        pt_left_eye = pt98[96]
        pt_right_eye = pt98[97]
    if use_lip:
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt98[76] + pt98[82]) / 2
        pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
        return pt2
    else:
        pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)
        return pt2
def parse_pt2_from_pt106(pt106, use_lip=True, use_mean_eyes=False):
    """\n    parsing the 2 points according to the 106 points, which cancels the roll\n    """
    if use_mean_eyes:
        pt_left_eye = np.mean(pt106[[33, 35, 40, 39]], axis=0)
        pt_right_eye = np.mean(pt106[[87, 89, 94, 93]], axis=0)
    else:
        pt_left_eye = pt106[38]
        pt_right_eye = pt106[88]
    if use_lip:
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt106[52] + pt106[61]) / 2
        pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
        return pt2
    else:
        pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)
        return pt2
def parse_pt2_from_pt203(pt203, use_lip=True, use_mean_eyes=False):
    """\n    parsing the 2 points according to the 203 points, which cancels the roll\n    """
    if use_mean_eyes:
        pt_left_eye = np.mean(pt203[[0, 6, 12, 18]], axis=0)
        pt_right_eye = np.mean(pt203[[24, 30, 36, 42]], axis=0)
    else:
        pt_left_eye = pt203[197]
        pt_right_eye = pt203[198]
    if use_lip:
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt203[48] + pt203[66]) / 2
        pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
        return pt2
    else:
        pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)
        return pt2
def parse_pt2_from_pt478(pt478, use_lip=True, use_mean_eyes=False):
    """\n    parsing the 2 points according to the 203 points, which cancels the roll\n    """
    if use_mean_eyes:
        pt_left_eye = np.mean(pt478[[472, 471, 470, 469]], axis=0)
        pt_right_eye = np.mean(pt478[[477, 476, 475, 474]], axis=0)
    else:
        pt_left_eye = pt478[468]
        pt_right_eye = pt478[473]
    if use_lip:
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt478[61] + pt478[291]) / 2
        pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
        return pt2
    else:
        pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)
        return pt2
def parse_pt2_from_pt68(pt68, use_lip=True):
    """\n    parsing the 2 points according to the 68 points, which cancels the roll\n    """
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55], dtype=np.int32) - 1
    if use_lip:
        pt5 = np.stack([np.mean(pt68[lm_idx[[1, 2]], :], 0), np.mean(pt68[lm_idx[[3, 4]], :], 0), pt68[lm_idx[0], :], pt68[lm_idx[5], :], pt68[lm_idx[6], :]], axis=0)
        pt2 = np.stack([(pt5[0] + pt5[1]) / 2, (pt5[3] + pt5[4]) / 2], axis=0)
        return pt2
    else:
        pt2 = np.stack([np.mean(pt68[lm_idx[[1, 2]], :], 0), np.mean(pt68[lm_idx[[3, 4]], :], 0)], axis=0)
        return pt2
def parse_pt2_from_pt5(pt5, use_lip=True):
    """\n    parsing the 2 points according to the 5 points, which cancels the roll\n    """
    pt_left_eye = pt5[0]
    pt_right_eye = pt5[1]
    if use_lip:
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt5[3] + pt5[4]) / 2
        pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
        return pt2
    else:
        pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)
        return pt2
def parse_pt2_from_pt9(pt9, use_lip=True):
    """\n    parsing the 2 points according to the 9 points, which cancels the roll\n    [\'right eye right\', \'right eye left\', \'left eye right\', \'left eye left\', \'nose tip\', \'lip right\', \'lip left\', \'upper lip\', \'lower lip\']\n    """
    if use_lip:
        pt9 = np.stack([(pt9[2] + pt9[3]) / 2, (pt9[0] + pt9[1]) / 2, pt9[4], (pt9[5] + pt9[6]) / 2], axis=0)
        pt2 = np.stack([(pt9[0] + pt9[1]) / 2, pt9[3]], axis=0)
        return pt2
    else:
        pt2 = np.stack([(pt9[2] + pt9[3]) / 2, (pt9[0] + pt9[1]) / 2], axis=0)
        return pt2
def parse_pt2_from_pt_x(pts, use_lip=True, use_mean_eyes=False):
    if pts.shape[0] == 101:
        pt2 = parse_pt2_from_pt101(pts, use_lip=use_lip)
    else:
        if pts.shape[0] == 106:
            pt2 = parse_pt2_from_pt106(pts, use_lip=use_lip, use_mean_eyes=use_mean_eyes)
        else:
            if pts.shape[0] == 68:
                pt2 = parse_pt2_from_pt68(pts, use_lip=use_lip)
            else:
                if pts.shape[0] == 5:
                    pt2 = parse_pt2_from_pt5(pts, use_lip=use_lip)
                else:
                    if pts.shape[0] == 203:
                        pt2 = parse_pt2_from_pt203(pts, use_lip=use_lip, use_mean_eyes=use_mean_eyes)
                    else:
                        if pts.shape[0] == 98:
                            pt2 = parse_pt2_from_pt98(pts, use_lip=use_lip, use_mean_eyes=use_mean_eyes)
                        else:
                            if pts.shape[0] == 478:
                                pt2 = parse_pt2_from_pt478(pts, use_lip=use_lip, use_mean_eyes=use_mean_eyes)
                            else:
                                if pts.shape[0] > 101:
                                    pt2 = parse_pt2_from_pt101(pts[:101], use_lip=use_lip)
                                else:
                                    if pts.shape[0] == 9:
                                        pt2 = parse_pt2_from_pt9(pts, use_lip=use_lip)
                                    else:
                                        raise Exception(f'Unknow shape: {pts.shape}')
    if not use_lip:
        v = pt2[1] - pt2[0]
        pt2[1, 0] = pt2[0, 0] - v[1]
        pt2[1, 1] = pt2[0, 1] + v[0]
    return pt2
def parse_rect_from_landmark(pts, scale=1.5, need_square=True, vx_ratio=0, vy_ratio=0, use_deg_flag=False, **kwargs):
    """parsing center, size, angle from 101/68/5/x landmarks\n    vx_ratio: the offset ratio along the pupil axis x-axis, multiplied by size\n    vy_ratio: the offset ratio along the pupil axis y-axis, multiplied by size, which is used to contain more forehead area\n\n    judge with pts.shape\n    """
    pt2 = parse_pt2_from_pt_x(pts, use_lip=kwargs.get('use_lip', True), use_mean_eyes=kwargs.get('use_mean_eyes', False))
    uy = pt2[1] - pt2[0]
    l = np.linalg.norm(uy)
    if l <= 0.001:
        uy = np.array([0, 1], dtype=np.float32)
    else:
        uy /= l
    ux = np.array((uy[1], -uy[0]), dtype=np.float32)
    angle = acos(ux[0])
    if ux[1] < 0:
        angle = -angle
    M = np.array([ux, uy])
    center0 = np.mean(pts, axis=0)
    rpts = (pts - center0) @ M.T
    lt_pt = np.min(rpts, axis=0)
    rb_pt = np.max(rpts, axis=0)
    center1 = (lt_pt + rb_pt) / 2
    size = rb_pt - lt_pt
    if need_square:
        m = max(size[0], size[1])
        size[0] = m
        size[1] = m
    size *= scale
    center = center0 + ux * center1[0] + uy * center1[1]
    center = center + ux * (vx_ratio * size) + uy * (vy_ratio * size)
    if use_deg_flag:
        angle = degrees(angle)
    return (center, size, angle)
def _estimate_similar_transform_from_pts(pts, dsize, scale=1.5, vx_ratio=0, vy_ratio=(-0.1), flag_do_rot=True, **kwargs):
    """ calculate the affine matrix of the cropped image from sparse points, the original image to the cropped image, the inverse is the cropped image to the original image\n    pts: landmark, 101 or 68 points or other points, Nx2\n    scale: the larger scale factor, the smaller face ratio\n    vx_ratio: x shift\n    vy_ratio: y shift, the smaller the y shift, the lower the face region\n    rot_flag: if it is true, conduct correction\n    """
    center, size, angle = parse_rect_from_landmark(pts, scale=scale, vx_ratio=vx_ratio, vy_ratio=vy_ratio, use_lip=kwargs.get('use_lip', True), use_mean_eyes=kwargs.get('use_mean_eyes', False))
    s = dsize / size[0]
    tgt_center = np.array([dsize / 2, dsize / 2], dtype=np.float32)
    if flag_do_rot:
        costheta, sintheta = (cos(angle), sin(angle))
        cx, cy = (center[0], center[1])
        tcx, tcy = (tgt_center[0], tgt_center[1])
        M_INV = np.array([[s * costheta, s * sintheta, tcx - s * (costheta * cx + sintheta * cy)], [-s * sintheta, s * costheta, tcy - s * (-sintheta * cx + costheta * cy)]], dtype=np.float32)
    else:
        M_INV = np.array([[s, 0, tgt_center[0] - s * center[0]], [0, s, tgt_center[1] - s * center[1]]], dtype=np.float32)
    M_INV_H = np.vstack([M_INV, np.array([0, 0, 1])])
    M = np.linalg.inv(M_INV_H)
    return (M_INV, M[:2, ...])
def warp_face_by_face_landmark_x(img, pts, **kwargs):
    dsize = kwargs.get('dsize', 224)
    scale = kwargs.get('scale', 1.5)
    vy_ratio = kwargs.get('vy_ratio', (-0.1))
    interpolation = kwargs.get('interpolation', v2.InterpolationMode.BILINEAR)
    img = pad_image_by_size(img, dsize)
    M_o2c, M_c2o = _estimate_similar_transform_from_pts(pts, dsize=dsize, scale=scale, vy_ratio=vy_ratio, flag_do_rot=kwargs.get('flag_do_rot', True))
    t = trans.SimilarityTransform()
    t.params[0:2] = M_o2c
    img = v2.functional.affine(img, t.rotation * 57.2958, translate=(t.translation[0], t.translation[1]), scale=t.scale, shear=(0.0, 0.0), interpolation=interpolation, center=(0, 0))
    img = v2.functional.crop(img, 0, 0, dsize, dsize)
    return (img, M_o2c, M_c2o)
def create_faded_inner_mask(size, border_thickness, fade_thickness, blur_radius=3, device='cuda'):
    """\n    Create a mask with a thick black border and a faded white center towards the border (optimized version).\n    The white edges are smoothed using Gaussian blur.\n\n    Parameters:\n    - size: Tuple (height, width) for the mask size.\n    - border_thickness: The thickness of the outer black border.\n    - fade_thickness: The thickness over which the white center fades into the black border.\n    - blur_radius: The radius for the Gaussian blur to smooth the white edges.\n    - device: Device to perform the computation (\'cuda\' for GPU, \'cpu\' for CPU).\n\n    Returns:\n    - mask: A PyTorch tensor containing the mask.\n    """
    height, width = size
    try:
        from rope.RTX50Compatibility import compatible_zeros, compatible_arange, compatible_meshgrid
        mask = compatible_zeros((height, width), dtype=torch.float32, device=device)
        inner_start = border_thickness
        inner_end_x = width - border_thickness
        inner_end_y = height - border_thickness
        y_range = compatible_arange(height, device=device)
        x_range = compatible_arange(width, device=device)
        y_indices, x_indices = compatible_meshgrid(y_range, x_range, indexing='ij')
    except ImportError:
        mask = torch.zeros((height, width), dtype=torch.float32, device=device)
        inner_start = border_thickness
        inner_end_x = width - border_thickness
        inner_end_y = height - border_thickness
        y_indices, x_indices = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
    dist_to_left = x_indices - inner_start
    dist_to_right = inner_end_x - x_indices - 1
    dist_to_top = y_indices - inner_start
    dist_to_bottom = inner_end_y - y_indices - 1
    dist_to_border = torch.minimum(torch.minimum(dist_to_left, dist_to_right), torch.minimum(dist_to_top, dist_to_bottom))
    fade_region = (dist_to_border >= 0) & (dist_to_border < fade_thickness)
    mask[fade_region] = dist_to_border[fade_region] / fade_thickness
    white_region = dist_to_border >= fade_thickness
    mask[white_region] = 1.0
    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = torchvision.transforms.functional.gaussian_blur(mask, kernel_size=(blur_radius, blur_radius), sigma=blur_radius / 2)
    mask = mask.squeeze()
    return mask
def prepare_paste_back(mask_crop, crop_M_c2o, dsize, interpolation=v2.InterpolationMode.BILINEAR):
    """prepare mask for later image paste back\n    """
    t = trans.SimilarityTransform()
    t.params[0:2] = crop_M_c2o
    mask_crop = pad_image_by_size(mask_crop, (dsize[0], dsize[1]))
    mask_ori = v2.functional.affine(mask_crop, t.rotation * 57.2958, translate=(t.translation[0], t.translation[1]), scale=t.scale, shear=(0.0, 0.0), interpolation=interpolation, center=(0, 0))
    mask_ori = v2.functional.crop(mask_ori, 0, 0, dsize[0], dsize[1])
    mask_ori = torch.div(mask_ori.type(torch.float32), 255.0)
    return mask_ori
def paste_back(img_crop, M_c2o, img_ori, mask_ori, interpolation=v2.InterpolationMode.BILINEAR):
    """paste back the image\n    """
    dsize = (img_ori.shape[1], img_ori.shape[2])
    t = trans.SimilarityTransform()
    t.params[0:2] = M_c2o
    img_crop = pad_image_by_size(img_crop, (img_ori.shape[1], img_ori.shape[2]))
    output = v2.functional.affine(img_crop, t.rotation * 57.2958, translate=(t.translation[0], t.translation[1]), scale=t.scale, shear=(0.0, 0.0), interpolation=interpolation, center=(0, 0))
    output = v2.functional.crop(output, 0, 0, dsize[0], dsize[1])
    output = output.float()
    mask_ori = mask_ori.float()
    img_ori = img_ori.float()
    output.mul_(mask_ori)
    output.add_(img_ori.mul_(1 - mask_ori))
    output.clamp_(0, 255)
    output = output.to(torch.uint8)
    return output
def paste_back_kgm(img_crop, M_c2o, img_ori, mask_ori):
    """paste back the image\n    """
    dsize = (img_ori.shape[1], img_ori.shape[2])
    img_crop = pad_image_by_size(img_crop, (img_ori.shape[1], img_ori.shape[2]))
    img_crop = img_crop.float()
    img_back = transform_img_kgm(img_crop, M_c2o, dsize=dsize)
    img_back = torch.clip(mask_ori * img_back + (1 - mask_ori) * img_ori, 0, 255)
    return img_back.to(torch.uint8)
def transform_img_kgm(img, M, dsize, mode='bilinear', padding_mode='zeros', align_corners=True, fill_value=(0, 0, 0)):
    """ Conduct similarity or affine transformation to the image using PyTorch CUDA.\n\n    Args:\n    img (torch.Tensor): Input image tensor (C x H x W)\n    M (torch.Tensor): 2x3 or 3x3 transformation matrix\n    dsize (tuple[int, int]): size of the output image (height, width).\n    mode(str, optional): interpolation mode to calculate output values \'bilinear\' | \'nearest\'. Default: \"bilinear\"\n    align_corners (bool, optional): mode for grid_generation. Default: True\n    fill_value (Tensor, optional): tensor of shape that fills the padding area. Only supported for RGB. Default: zeros(3)\n\n    Returns:\n    torch.Tensor: Transformed image\n    """
    if isinstance(dsize, tuple) or isinstance(dsize, list):
        _dsize = tuple(dsize)
    else:
        _dsize = (dsize, dsize)
    if isinstance(M, np.ndarray):
        M = torch.from_numpy(M).to(img.device, dtype=img.dtype)
    M = M[:2, :]
    img_transformed = kgm.warp_affine(src=img.unsqueeze(0), M=M[None], dsize=(_dsize[0], _dsize[1]), mode=mode, padding_mode=padding_mode, align_corners=align_corners, fill_value=fill_value)
    img_transformed = img_transformed.squeeze(0)
    return img_transformed
def calculate_distance_ratio(lmk: np.ndarray, idx1: int, idx2: int, idx3: int, idx4: int, eps: float=1e-06) -> np.ndarray:
    return np.linalg.norm(lmk[:, idx1] - lmk[:, idx2], axis=1, keepdims=True) / (np.linalg.norm(lmk[:, idx3] - lmk[:, idx4], axis=1, keepdims=True) + eps)
def calc_eye_close_ratio(lmk: np.ndarray, target_eye_ratio: np.ndarray=None) -> np.ndarray:
    lefteye_close_ratio = calculate_distance_ratio(lmk, 6, 18, 0, 12)
    righteye_close_ratio = calculate_distance_ratio(lmk, 30, 42, 24, 36)
    if target_eye_ratio is not None:
        return np.concatenate([lefteye_close_ratio, righteye_close_ratio, target_eye_ratio], axis=1)
    else:
        return np.concatenate([lefteye_close_ratio, righteye_close_ratio], axis=1)
def calc_lip_close_ratio(lmk: np.ndarray) -> np.ndarray:
    return calculate_distance_ratio(lmk, 90, 102, 48, 66)
def headpose_pred_to_degree(pred):
    """\n    pred: (bs, 66) or (bs, 1) or others\n    """
    if pred.ndim > 1 and pred.shape[1] == 66:
        device = pred.device
        idx_tensor = [idx for idx in range(0, 66)]
        idx_tensor = torch.FloatTensor(idx_tensor).to(device)
        pred = torch.nn.functional.softmax(pred, dim=1)
        degree = torch.sum(pred * idx_tensor, axis=1) * 3 - 97.5
        return degree
    else:
        return pred
def get_rotation_matrix(pitch_, yaw_, roll_):
    """ the input is in degree\n    """
    pitch = pitch_ / 180 * np.pi
    yaw = yaw_ / 180 * np.pi
    roll = roll_ / 180 * np.pi
    device = pitch.device
    if pitch.ndim == 1:
        pitch = pitch.unsqueeze(1)
    if yaw.ndim == 1:
        yaw = yaw.unsqueeze(1)
    if roll.ndim == 1:
        roll = roll.unsqueeze(1)
    bs = pitch.shape[0]
    ones = torch.ones([bs, 1]).to(device)
    zeros = torch.zeros([bs, 1]).to(device)
    x, y, z = (pitch, yaw, roll)
    rot_x = torch.cat([ones, zeros, zeros, zeros, torch.cos(x), -torch.sin(x), zeros, torch.sin(x), torch.cos(x)], dim=1).reshape([bs, 3, 3])
    rot_y = torch.cat([torch.cos(y), zeros, torch.sin(y), zeros, ones, zeros, -torch.sin(y), zeros, torch.cos(y)], dim=1).reshape([bs, 3, 3])
    rot_z = torch.cat([torch.cos(z), -torch.sin(z), zeros, torch.sin(z), torch.cos(z), zeros, zeros, zeros, ones], dim=1).reshape([bs, 3, 3])
    rot = rot_z @ rot_y @ rot_x
    return rot.permute(0, 2, 1)
def transform_keypoint(kp_info: dict):
    """\n    transform the implicit keypoints with the pose, shift, and expression deformation\n    kp: BxNx3\n    """
    kp = kp_info['kp']
    pitch, yaw, roll = (kp_info['pitch'], kp_info['yaw'], kp_info['roll'])
    t, exp = (kp_info['t'], kp_info['exp'])
    scale = kp_info['scale']
    pitch = headpose_pred_to_degree(pitch)
    yaw = headpose_pred_to_degree(yaw)
    roll = headpose_pred_to_degree(roll)
    bs = kp.shape[0]
    if kp.ndim == 2:
        num_kp = kp.shape[1] // 3
    else:
        num_kp = kp.shape[1]
    rot_mat = get_rotation_matrix(pitch, yaw, roll)
    kp_transformed = kp.view(bs, num_kp, 3) @ rot_mat + exp.view(bs, num_kp, 3)
    kp_transformed *= scale[..., None]
    kp_transformed[:, :, 0:2] += t[:, None, 0:2]
    return kp_transformed
@torch.no_grad()
def update_delta_new_eyeball_direction(eyeball_direction_x, eyeball_direction_y, delta_new, **kwargs):
    if eyeball_direction_x > 0:
        delta_new[0, 11, 0] += eyeball_direction_x * 0.0007
        delta_new[0, 15, 0] += eyeball_direction_x * 0.001
    else:
        delta_new[0, 11, 0] += eyeball_direction_x * 0.001
        delta_new[0, 15, 0] += eyeball_direction_x * 0.0007
    delta_new[0, 11, 1] += eyeball_direction_y * (-0.001)
    delta_new[0, 15, 1] += eyeball_direction_y * (-0.001)
    blink = -eyeball_direction_y / 2.0
    delta_new[0, 11, 1] += blink * (-0.001)
    delta_new[0, 13, 1] += blink * 0.0003
    delta_new[0, 15, 1] += blink * (-0.001)
    delta_new[0, 16, 1] += blink * 0.0003
    return delta_new
@torch.no_grad()
def update_delta_new_smile(smile, delta_new, **kwargs):
    delta_new[0, 20, 1] += smile * (-0.01)
    delta_new[0, 14, 1] += smile * (-0.02)
    delta_new[0, 17, 1] += smile * 0.0065
    delta_new[0, 17, 2] += smile * 0.003
    delta_new[0, 13, 1] += smile * (-0.00275)
    delta_new[0, 16, 1] += smile * (-0.00275)
    delta_new[0, 3, 1] += smile * (-0.0035)
    delta_new[0, 7, 1] += smile * (-0.0035)
    return delta_new
@torch.no_grad()
def update_delta_new_wink(wink, delta_new, **kwargs):
    delta_new[0, 11, 1] += wink * 0.001
    delta_new[0, 13, 1] += wink * (-0.0003)
    delta_new[0, 17, 0] += wink * 0.0003
    delta_new[0, 17, 1] += wink * 0.0003
    delta_new[0, 3, 1] += wink * (-0.0003)
    return delta_new
@torch.no_grad()
def update_delta_new_eyebrow(eyebrow, delta_new, **kwargs):
    if eyebrow > 0:
        delta_new[0, 1, 1] += eyebrow * 0.001
        delta_new[0, 2, 1] += eyebrow * (-0.001)
        return delta_new
    else:
        delta_new[0, 1, 0] += eyebrow * (-0.001)
        delta_new[0, 2, 0] += eyebrow * 0.001
        delta_new[0, 1, 1] += eyebrow * 0.0003
        delta_new[0, 2, 1] += eyebrow * (-0.0003)
        return delta_new
@torch.no_grad()
def update_delta_new_lip_variation_zero(lip_variation_zero, delta_new, **kwargs):
    delta_new[0, 19, 0] += lip_variation_zero
    return delta_new
@torch.no_grad()
def update_delta_new_lip_variation_one(lip_variation_one, delta_new, **kwargs):
    delta_new[0, 14, 1] += lip_variation_one * 0.001
    delta_new[0, 3, 1] += lip_variation_one * (-0.0005)
    delta_new[0, 7, 1] += lip_variation_one * (-0.0005)
    delta_new[0, 17, 2] += lip_variation_one * (-0.0005)
    return delta_new
@torch.no_grad()
def update_delta_new_lip_variation_two(lip_variation_two, delta_new, **kwargs):
    delta_new[0, 20, 2] += lip_variation_two * (-0.001)
    delta_new[0, 20, 1] += lip_variation_two * (-0.001)
    delta_new[0, 14, 1] += lip_variation_two * (-0.001)
    return delta_new
@torch.no_grad()
def update_delta_new_lip_variation_three(lip_variation_three, delta_new, **kwargs):
    delta_new[0, 19, 1] += lip_variation_three * 0.001
    delta_new[0, 19, 2] += lip_variation_three * 0.0001
    delta_new[0, 17, 1] += lip_variation_three * (-0.0001)
    return delta_new
@torch.no_grad()
def update_delta_new_mov_x(mov_x, delta_new, **kwargs):
    delta_new[0, 5, 0] += mov_x
    return delta_new
@torch.no_grad()
def update_delta_new_mov_y(mov_y, delta_new, **kwargs):
    delta_new[0, 5, 1] += mov_y
    return delta_new
def calc_combined_eye_ratio(c_d_eyes_i, source_lmk, device='cuda'):
    c_s_eyes = calc_eye_close_ratio(source_lmk[None])
    c_s_eyes_tensor = torch.from_numpy(c_s_eyes).float().to(device)
    c_d_eyes_i_tensor = torch.Tensor([c_d_eyes_i[0][0]]).reshape(1, 1).to(device)
    combined_eye_ratio_tensor = torch.cat([c_s_eyes_tensor, c_d_eyes_i_tensor], dim=1)
    return combined_eye_ratio_tensor
def calc_combined_lip_ratio(c_d_lip_i, source_lmk, device='cuda'):
    c_s_lip = calc_lip_close_ratio(source_lmk[None])
    c_s_lip_tensor = torch.from_numpy(c_s_lip).float().to(device)
    c_d_lip_i_tensor = torch.Tensor([c_d_lip_i[0]]).to(device).reshape(1, 1)
    combined_lip_ratio_tensor = torch.cat([c_s_lip_tensor, c_d_lip_i_tensor], dim=1)
    return combined_lip_ratio_tensor
def concat_feat(kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
    """\n    kp_source: (bs, k, 3)\n    kp_driving: (bs, k, 3)\n    Return: (bs, 2k*3)\n    """
    bs_src = kp_source.shape[0]
    bs_dri = kp_driving.shape[0]
    assert bs_src == bs_dri, 'batch size must be equal'
    feat = torch.cat([kp_source.view(bs_src, (-1)), kp_driving.view(bs_dri, (-1))], dim=1)
    return feat
def apply_laplace_filter(img):
    laplace_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)
    img = img.unsqueeze(0).unsqueeze(0)
    laplacian = torch.nn.functional.conv2d(img, laplace_kernel, padding=1)
    return laplacian.squeeze(0).squeeze(0)
def jpegBlur(img, q):
    device = img.device
    if img.dim() != 3 or img.size(0) != 3:
        raise ValueError('Image must have shape [3, H, W].')
    else:
        if img.dtype == torch.float32:
            img_uint8 = img.type(torch.uint8).cpu()
        else:
            if img.dtype == torch.uint8:
                img_uint8 = img.cpu()
            else:
                raise ValueError('Unsupported image data type.')
        buffer = torchvision.io.encode_jpeg(img_uint8, quality=q)
        img_blurred = torchvision.io.decode_jpeg(buffer)
        img_blurred = img_blurred.to(device).type(torch.float32)
        return img_blurred
def histogram_matching(source_image, target_image, diffslider):
    device = source_image.device
    source_image_t = source_image.float().to(device) / 255.0
    target_image_t = target_image.float().to(device) / 255.0
    matched_target_image_t = target_image_t.clone()
    bin_edges = torch.linspace(0.0, 1.0, steps=257, device=device)
    for channel in range(3):
        source_channel = source_image_t[channel, :, :]
        target_channel = target_image_t[channel, :, :]
        source_hist = torch.histc(source_channel, bins=256, min=0.0, max=1.0)
        target_hist = torch.histc(target_channel, bins=256, min=0.0, max=1.0)
        source_pmf = source_hist / source_hist.sum()
        target_pmf = target_hist / target_hist.sum()
        source_cdf = torch.cumsum(source_pmf, dim=0)
        target_cdf = torch.cumsum(target_pmf, dim=0)
        target_channel_flat = target_channel.flatten()
        interp_t_values = interp1d(target_channel_flat, bin_edges[:(-1)], target_cdf, device=device)
        matched_channel_flat = interp1d_inverse(interp_t_values, source_cdf, bin_edges[:(-1)], device=device)
        matched_channel = matched_channel_flat.reshape(target_channel.shape)
        matched_target_image_t[channel, :, :] = matched_channel
    alpha = diffslider / 100.0
    final_image_t = (1 - alpha) * target_image_t + alpha * matched_target_image_t
    final_image_t = torch.clamp(final_image_t * 255.0, 0.0, 255.0)
    final_image_tensor = final_image_t.to(device).float()
    return final_image_tensor
def histogram_matching_withmask(source_image, target_image, mask, diffslider):
    device = source_image.device
    source_image_t = source_image.float().to(device) / 255.0
    target_image_t = target_image.float().to(device) / 255.0
    mask_t = mask.float().to(device)
    matched_target_image_t = target_image_t.clone()
    valid_mask = mask_t > 0.2
    if valid_mask.dim() == 3 and valid_mask.size(0) == 1:
            valid_mask = valid_mask.squeeze(0)
    bin_edges = torch.linspace(0.0, 1.0, steps=257, device=device)
    for channel in range(3):
        source_channel = source_image_t[channel, :, :]
        target_channel = target_image_t[channel, :, :]
        masked_source_values = source_channel[valid_mask]
        masked_target_values = target_channel[valid_mask]
        masked_source_values = masked_source_values[~torch.isnan(masked_source_values)]
        masked_source_values = masked_source_values[~torch.isinf(masked_source_values)]
        masked_target_values = masked_target_values[~torch.isnan(masked_target_values)]
        masked_target_values = masked_target_values[~torch.isinf(masked_target_values)]
        if masked_source_values.numel() == 0 or masked_target_values.numel() == 0:
            print(f'No valid masked pixels for channel {channel}. Skipping histogram matching for this channel.')
            continue
        else:
            masked_source_values = torch.clamp(masked_source_values, 0.0, 1.0)
            masked_target_values = torch.clamp(masked_target_values, 0.0, 1.0)
            source_hist = torch.histc(masked_source_values, bins=256, min=0.0, max=1.0)
            target_hist = torch.histc(masked_target_values, bins=256, min=0.0, max=1.0)
            source_hist += 1e-06
            target_hist += 1e-06
            source_hist_sum = source_hist.sum()
            target_hist_sum = target_hist.sum()
            if source_hist_sum == 0 or target_hist_sum == 0:
                print(f'Histogram sum is zero for channel {channel}. Skipping histogram matching for this channel.')
                continue
            else:
                source_pmf = source_hist / source_hist_sum
                target_pmf = target_hist / target_hist_sum
                source_cdf = torch.cumsum(source_pmf, dim=0)
                target_cdf = torch.cumsum(target_pmf, dim=0)
                source_cdf = torch.maximum(source_cdf, torch.cummax(source_cdf, dim=0)[0])
                target_cdf = torch.maximum(target_cdf, torch.cummax(target_cdf, dim=0)[0])
                if torch.isnan(source_cdf).any() or torch.isinf(source_cdf).any():
                    print(f'Channel {channel}: source_cdf contains NaN or Inf values. Skipping histogram matching for this channel.')
                    continue
                else:
                    target_channel_flat = target_channel.flatten()
                    interp_t_values = interp1d(target_channel_flat, bin_edges[:(-1)], target_cdf, device=device)
                    matched_channel_flat = interp1d_inverse(interp_t_values, source_cdf, bin_edges[:(-1)], device=device)
                    matched_channel = matched_channel_flat.reshape(target_channel.shape)
                    matched_target_image_t[channel, :, :][valid_mask] = matched_channel[valid_mask]
    alpha = diffslider / 100.0
    final_image_t = (1 - alpha) * target_image_t + alpha * matched_target_image_t
    final_image_t = torch.clamp(final_image_t * 255.0, 0.0, 255.0)
    final_image_tensor = final_image_t.to(device).float()
    return final_image_tensor
def interp1d(x, xp, fp, device='cpu'):
    assert torch.all(xp[1:] >= xp[:(-1)]), 'xp must be increasing'
    x = x.to(device).contiguous()
    xp = xp.to(device).contiguous()
    fp = fp.to(device)
    indices = torch.searchsorted(xp, x, right=True) - 1
    indices = indices.clamp(0, len(xp) - 2)
    x0 = xp[indices]
    x1 = xp[indices + 1]
    y0 = fp[indices]
    y1 = fp[indices + 1]
    slope = (y1 - y0) / (x1 - x0 + 1e-06)
    y = y0 + slope * (x - x0)
    y = torch.where(x < xp[0], fp[0], y)
    y = torch.where(x > xp[(-1)], fp[(-1)], y)
    return y
def interp1d_inverse(y, fp, xp, device='cpu'):
    assert torch.all(fp[1:] >= fp[:(-1)]), 'fp must be increasing'
    y = y.to(device).contiguous()
    fp = fp.to(device).contiguous()
    xp = xp.to(device)
    indices = torch.searchsorted(fp, y, right=True) - 1
    indices = indices.clamp(0, len(fp) - 2)
    y0 = fp[indices]
    y1 = fp[indices + 1]
    x0 = xp[indices]
    x1 = xp[indices + 1]
    slope = (x1 - x0) / (y1 - y0 + 1e-06)
    x = x0 + slope * (y - y0)
    x = torch.where(y < fp[0], xp[0], x)
    x = torch.where(y > fp[(-1)], xp[(-1)], x)
    return x
def histogram_matching_DFL_test(source_image, target_image, diffslider):
    source_image = source_image.type(torch.float32) / 255.0
    target_image = target_image.type(torch.float32) / 255.0
    source = rgb_to_lab(source_image, False)
    target = rgb_to_lab(target_image, False)
    target_l_mean, target_l_std = (target[0].mean(), target[0].std())
    target_a_mean, target_a_std = (target[1].mean(), target[1].std())
    target_b_mean, target_b_std = (target[2].mean(), target[2].std())
    source_l_mean, source_l_std = (source[0].mean(), source[0].std())
    source_a_mean, source_a_std = (source[1].mean(), source[1].std())
    source_b_mean, source_b_std = (source[2].mean(), source[2].std())
    target_l = (target[0] - target_l_mean) * (source_l_std / target_l_std) + source_l_mean
    target_a = (target[1] - target_a_mean) * (source_a_std / target_a_std) + source_a_mean
    target_b = (target[2] - target_b_mean) * (source_b_std / target_b_std) + source_b_mean
    target_l = torch.clamp(target_l, 0, 100)
    target_a = torch.clamp(target_a, (-127), 127)
    target_b = torch.clamp(target_b, (-127), 127)
    matched_target_image = torch.stack([target_l, target_a, target_b], 0)
    matched_target_image = lab_to_rgb(matched_target_image, False)
    final_image = (1 - diffslider / 100) * target_image + diffslider / 100 * matched_target_image
    final_image = torch.clamp(final_image * 255, 0, 255)
    return final_image
def histogram_matching_DFL_Orig(source_image, target_image, mask, diffslider):
    source_image = source_image.type(torch.float32) / 255.0
    target_image = target_image.type(torch.float32) / 255.0
    mask = mask.type(torch.float32).squeeze()
    mask_cutoff = 0.2
    mask = mask.unsqueeze(0)
    source_mask = mask.expand(source_image.shape[0], (-1), (-1))
    target_mask = mask.expand(target_image.shape[0], (-1), (-1))
    source = rgb_to_lab(source_image, False)
    target = rgb_to_lab(target_image, False)
    source_input = source.clone()
    if source_mask is not None:
        source_input[source_mask < mask_cutoff] = 0.0
    target_input = target.clone()
    if target_mask is not None:
        target_input[target_mask < mask_cutoff] = 0.0
    target_l_mean, target_l_std = (target_input[0].mean(), target_input[0].std())
    target_a_mean, target_a_std = (target_input[1].mean(), target_input[1].std())
    target_b_mean, target_b_std = (target_input[2].mean(), target_input[2].std())
    source_l_mean, source_l_std = (source_input[0].mean(), source_input[0].std())
    source_a_mean, source_a_std = (source_input[1].mean(), source_input[1].std())
    source_b_mean, source_b_std = (source_input[2].mean(), source_input[2].std())
    target_l = (target[0] - target_l_mean) * (source_l_std / target_l_std) + source_l_mean
    target_a = (target[1] - target_a_mean) * (source_a_std / target_a_std) + source_a_mean
    target_b = (target[2] - target_b_mean) * (source_b_std / target_b_std) + source_b_mean
    target_l = torch.clamp(target_l, 0, 100)
    target_a = torch.clamp(target_a, (-127), 127)
    target_b = torch.clamp(target_b, (-127), 127)
    matched_target_image = torch.stack([target_l, target_a, target_b], 0)
    matched_target_image = lab_to_rgb(matched_target_image, False)
    final_image = (1 - diffslider / 100) * target_image + diffslider / 100 * matched_target_image
    final_image = torch.clamp(final_image * 255, 0, 255)
    return final_image
def transform_t(img, center, output_size, scale, rotation):
    device = img.device
    dtype = img.dtype
    img = pad_image_by_size(img, output_size)
    scale_ratio = scale
    rot_rad = torch.tensor(rotation * torch.pi / 180.0, device=device, dtype=dtype)
    cos_theta = torch.cos(rot_rad) * scale_ratio
    sin_theta = torch.sin(rot_rad) * scale_ratio
    a = cos_theta
    b = sin_theta
    c = -sin_theta
    d = cos_theta
    cx, cy = center
    cx = cx * scale_ratio
    cy = cy * scale_ratio
    tx = -cx
    ty = -cy
    tx_final = output_size / 2
    ty_final = output_size / 2
    tx_total = tx_final + a * tx + b * ty
    ty_total = ty_final + c * tx + d * ty
    M = torch.tensor([[a, b, tx_total], [c, d, ty_total]], dtype=dtype, device=device)
    img_batch = img.unsqueeze(0)
    grid = torch.nn.functional.affine_grid(M.unsqueeze(0), img_batch.size(), align_corners=False)
    cropped_batch = torch.nn.functional.grid_sample(img_batch, grid, align_corners=False, mode='bilinear')
    cropped = cropped_batch.squeeze(0)
    return (cropped, M)
def trans_points2d_t(pts, M):
    if pts.dim() != 2 or pts.size(1) != 2:
        raise ValueError('pts deve essere un tensore 2D con dimensione (N, 2)')
    else:
        ones_column = torch.ones((pts.size(0), 1), dtype=pts.dtype, device=pts.device)
        homogeneous_pts = torch.cat([pts, ones_column], dim=1)
        transformed_pts = homogeneous_pts @ M.T
        return transformed_pts[:, :2]
def invertAffineTransform_t(M):
    if M.dim() == 2 and M.size() == (2, 3):
        M_H = torch.cat([M, torch.tensor([[0, 0, 1]], device=M.device, dtype=M.dtype)], dim=0)
        IM_H = torch.inverse(M_H)
        IM = IM_H[:2, :]
        return IM
    else:
        raise ValueError('M deve essere di dimensione (2, 3)')
def get_face_orientation_t(face_size, lmk):
    global arcface_src_cuda
    assert lmk.shape == (5, 2), 'lmk deve essere un tensore di forma (5, 2)'
    device = lmk.device
    if device != arcface_src_cuda.device:
        arcface_src_cuda = arcface_src_cuda.to(device)
    src_scaled = face_size / 112.0 * arcface_src_cuda
    centroid_lmk = lmk.mean(dim=0, keepdim=True)
    centroid_src = src_scaled.mean(dim=0, keepdim=True)
    lmk_centered = lmk - centroid_lmk
    src_centered = src_scaled - centroid_src
    norm_lmk = torch.norm(lmk_centered, dim=1).pow(2).sum().unsqueeze(0)
    norm_src = torch.norm(src_centered, dim=1).pow(2).sum().unsqueeze(0)
    scale = torch.sqrt(norm_src / norm_lmk)
    lmk_scaled = lmk_centered * scale
    covariance = torch.mm(src_centered.t(), lmk_scaled)
    U, S, V = torch.svd(covariance)
    R = torch.mm(U, V.t())
    det = torch.det(R)
    if det < 0:
        U[:, (-1)] *= (-1)
        R = torch.mm(U, V.t())
    angle_rad = torch.atan2(R[1, 0], R[0, 0])
    angle_deg = torch.rad2deg(angle_rad)
    return angle_deg