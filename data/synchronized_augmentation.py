"""
同步数据增强模块
确保光流图像和关键点之间的空间一致性
"""
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional
import random
from PIL import Image


class SynchronizedAugmentation:
    """同步的数据增强，保证光流图像和关键点的空间一致性"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化同步增强器
        
        Args:
            config: 增强配置
        """
        self.config = config
        
        # 几何变换参数
        self.horizontal_flip_prob = config.get('horizontal_flip', 0.0)
        self.rotation_range = config.get('rotation', 0)
        self.scale_range = config.get('scale_range', [1.0, 1.0])
        self.translation_range = config.get('translation_range', [0.0, 0.0])
        
        # 光流图像专用的颜色增强（不影响关键点）
        self.brightness = config.get('brightness', 0.0)
        self.contrast = config.get('contrast', 0.0)
        self.saturation = config.get('saturation', 0.0)
        self.hue = config.get('hue', 0.0)
        
        # 关键点专用的噪声增强（不影响光流）
        self.landmark_noise_std = config.get('landmark_noise_std', 0.0)
        
        # 图像尺寸
        self.target_size = config.get('target_size', (112, 112))
        
    def _generate_transform_params(self) -> Dict[str, Any]:
        """生成一致的变换参数"""
        params = {}
        
        # 水平翻转
        params['horizontal_flip'] = random.random() < self.horizontal_flip_prob
        
        # 旋转角度
        if self.rotation_range > 0:
            params['rotation_angle'] = random.uniform(-self.rotation_range, self.rotation_range)
        else:
            params['rotation_angle'] = 0.0
        
        # 缩放因子
        if self.scale_range[0] != self.scale_range[1]:
            params['scale_factor'] = random.uniform(self.scale_range[0], self.scale_range[1])
        else:
            params['scale_factor'] = self.scale_range[0]
        
        # 平移
        if self.translation_range[0] > 0 or self.translation_range[1] > 0:
            params['translation_x'] = random.uniform(-self.translation_range[0], self.translation_range[0])
            params['translation_y'] = random.uniform(-self.translation_range[1], self.translation_range[1])
        else:
            params['translation_x'] = 0.0
            params['translation_y'] = 0.0
        
        # 颜色变换参数（仅用于光流图像）
        params['color_params'] = {}
        if self.brightness > 0:
            params['color_params']['brightness'] = random.uniform(1-self.brightness, 1+self.brightness)
        if self.contrast > 0:
            params['color_params']['contrast'] = random.uniform(1-self.contrast, 1+self.contrast)
        if self.saturation > 0:
            params['color_params']['saturation'] = random.uniform(1-self.saturation, 1+self.saturation)
        if self.hue > 0:
            params['color_params']['hue'] = random.uniform(-self.hue, self.hue)
        
        return params
    
    def _apply_geometric_transform_to_image(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """对图像应用几何变换"""
        h, w = image.shape[:2]
        
        # 计算变换矩阵
        center = (w / 2, h / 2)
        
        # 旋转 + 缩放
        rotation_matrix = cv2.getRotationMatrix2D(center, params['rotation_angle'], params['scale_factor'])
        
        # 添加平移
        rotation_matrix[0, 2] += params['translation_x'] * w
        rotation_matrix[1, 2] += params['translation_y'] * h
        
        # 应用仿射变换
        transformed_image = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                         borderMode=cv2.BORDER_REFLECT_101)
        
        # 水平翻转
        if params['horizontal_flip']:
            transformed_image = cv2.flip(transformed_image, 1)
        
        return transformed_image
    
    def _apply_geometric_transform_to_landmarks(self, landmarks: np.ndarray, params: Dict[str, Any], 
                                              image_size: Tuple[int, int]) -> np.ndarray:
        """对关键点应用相同的几何变换"""
        h, w = image_size
        transformed_landmarks = landmarks.copy()
        
        # 水平翻转（需要先做，因为会改变坐标系）
        if params['horizontal_flip']:
            transformed_landmarks[:, 0] = w - transformed_landmarks[:, 0]
        
        # 计算变换中心
        center_x, center_y = w / 2, h / 2
        
        # 平移到原点
        transformed_landmarks[:, 0] -= center_x
        transformed_landmarks[:, 1] -= center_y
        
        # 旋转 + 缩放
        angle_rad = np.radians(params['rotation_angle'])
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        scale = params['scale_factor']
        
        # 应用旋转和缩放矩阵
        x = transformed_landmarks[:, 0]
        y = transformed_landmarks[:, 1]
        
        transformed_landmarks[:, 0] = scale * (cos_angle * x - sin_angle * y)
        transformed_landmarks[:, 1] = scale * (sin_angle * x + cos_angle * y)
        
        # 添加平移
        transformed_landmarks[:, 0] += params['translation_x'] * w
        transformed_landmarks[:, 1] += params['translation_y'] * h
        
        # 平移回原来的中心
        transformed_landmarks[:, 0] += center_x
        transformed_landmarks[:, 1] += center_y
        
        return transformed_landmarks
    
    def _apply_color_transform_to_image(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """对图像应用颜色变换（不影响关键点）"""
        if not params['color_params']:
            return image
        
        # 转换为PIL图像进行颜色变换
        pil_image = Image.fromarray(image)
        
        # 应用颜色变换
        if 'brightness' in params['color_params']:
            pil_image = transforms.functional.adjust_brightness(pil_image, params['color_params']['brightness'])
        
        if 'contrast' in params['color_params']:
            pil_image = transforms.functional.adjust_contrast(pil_image, params['color_params']['contrast'])
        
        if 'saturation' in params['color_params']:
            pil_image = transforms.functional.adjust_saturation(pil_image, params['color_params']['saturation'])
        
        if 'hue' in params['color_params']:
            pil_image = transforms.functional.adjust_hue(pil_image, params['color_params']['hue'])
        
        return np.array(pil_image)
    
    def _add_landmark_noise(self, landmarks: np.ndarray) -> np.ndarray:
        """为关键点添加噪声（不影响图像）"""
        if self.landmark_noise_std <= 0:
            return landmarks
        
        noise = np.random.normal(0, self.landmark_noise_std, landmarks.shape)
        return landmarks + noise
    
    def _resize_and_normalize_image(self, image: np.ndarray) -> torch.Tensor:
        """调整图像尺寸并标准化"""
        # 调整尺寸
        image = cv2.resize(image, self.target_size)
        
        # 转换为PIL图像
        pil_image = Image.fromarray(image)
        
        # 转换为张量并标准化
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        return transform(pil_image)
    
    def _normalize_landmarks(self, landmarks: np.ndarray, 
                           original_size: Tuple[int, int]) -> np.ndarray:
        """将关键点坐标标准化到目标尺寸"""
        original_h, original_w = original_size
        target_h, target_w = self.target_size
        
        # 计算缩放比例
        scale_x = target_w / original_w
        scale_y = target_h / original_h
        
        normalized_landmarks = landmarks.copy()
        normalized_landmarks[:, 0] *= scale_x
        normalized_landmarks[:, 1] *= scale_y
        
        return normalized_landmarks
    
    def __call__(self, optical_flow_image: np.ndarray, 
                 expression_landmarks: np.ndarray,
                 baseline_landmarks: np.ndarray,
                 training: bool = True) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """
        同步应用数据增强
        
        Args:
            optical_flow_image: 光流图像，形状为(H, W, 3)
            expression_landmarks: 表情关键点，形状为(146, 2)
            baseline_landmarks: 基线关键点，形状为(146, 2)
            training: 是否为训练模式
            
        Returns:
            增强后的光流图像张量和关键点
        """
        if not training:
            # 验证模式：只进行基础的尺寸调整和标准化
            image_tensor = self._resize_and_normalize_image(optical_flow_image)
            
            # 关键点标准化到目标尺寸
            original_size = optical_flow_image.shape[:2]
            expression_landmarks = self._normalize_landmarks(expression_landmarks, original_size)
            baseline_landmarks = self._normalize_landmarks(baseline_landmarks, original_size)
            
            return image_tensor, expression_landmarks, baseline_landmarks
        
        # 训练模式：应用同步增强
        original_size = optical_flow_image.shape[:2]
        
        # 生成一致的变换参数
        transform_params = self._generate_transform_params()
        
        # 1. 对图像和关键点应用相同的几何变换
        transformed_image = self._apply_geometric_transform_to_image(optical_flow_image, transform_params)
        transformed_expr_landmarks = self._apply_geometric_transform_to_landmarks(
            expression_landmarks, transform_params, original_size
        )
        transformed_base_landmarks = self._apply_geometric_transform_to_landmarks(
            baseline_landmarks, transform_params, original_size
        )
        
        # 2. 对图像应用颜色变换（不影响关键点）
        transformed_image = self._apply_color_transform_to_image(transformed_image, transform_params)
        
        # 3. 对关键点添加噪声（不影响图像）
        transformed_expr_landmarks = self._add_landmark_noise(transformed_expr_landmarks)
        transformed_base_landmarks = self._add_landmark_noise(transformed_base_landmarks)
        
        # 4. 最终的尺寸调整和标准化
        image_tensor = self._resize_and_normalize_image(transformed_image)
        
        # 关键点标准化到目标尺寸
        transformed_expr_landmarks = self._normalize_landmarks(transformed_expr_landmarks, original_size)
        transformed_base_landmarks = self._normalize_landmarks(transformed_base_landmarks, original_size)
        
        return image_tensor, transformed_expr_landmarks, transformed_base_landmarks


def create_synchronized_transforms(config: Dict[str, Any]) -> SynchronizedAugmentation:
    """
    创建同步数据增强器
    
    Args:
        config: 增强配置
        
    Returns:
        同步增强器
    """
    return SynchronizedAugmentation(config)
