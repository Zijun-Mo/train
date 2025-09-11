"""
数据预处理和增强
"""
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
from typing import Dict, Any, Tuple, Optional
import random


class OpticalFlowTransforms:
    """光流图像变换"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化光流图像变换
        
        Args:
            config: 增强配置
        """
        self.config = config
        self.train_transforms = self._build_train_transforms()
        self.val_transforms = self._build_val_transforms()
    
    def _build_train_transforms(self):
        """构建训练时的变换"""
        transforms_list = []
        
        # 基础变换
        transforms_list.extend([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
        ])
        
        # 数据增强
        if self.config.get('horizontal_flip', 0) > 0:
            transforms_list.append(
                transforms.RandomHorizontalFlip(p=self.config['horizontal_flip'])
            )
        
        if self.config.get('rotation', 0) > 0:
            transforms_list.append(
                transforms.RandomRotation(degrees=self.config['rotation'])
            )
        
        # 颜色变换
        color_jitter_params = {}
        if self.config.get('brightness', 0) > 0:
            color_jitter_params['brightness'] = self.config['brightness']
        if self.config.get('contrast', 0) > 0:
            color_jitter_params['contrast'] = self.config['contrast']
        if self.config.get('saturation', 0) > 0:
            color_jitter_params['saturation'] = self.config['saturation']
        if self.config.get('hue', 0) > 0:
            color_jitter_params['hue'] = self.config['hue']
        
        if color_jitter_params:
            transforms_list.append(
                transforms.ColorJitter(**color_jitter_params)
            )
        
        # 转换为张量并归一化
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        return transforms.Compose(transforms_list)
    
    def _build_val_transforms(self):
        """构建验证时的变换"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, image: np.ndarray, training: bool = True) -> torch.Tensor:
        """
        应用变换
        
        Args:
            image: 输入图像
            training: 是否为训练模式
            
        Returns:
            变换后的张量
        """
        if training:
            return self.train_transforms(image)
        else:
            return self.val_transforms(image)


class LandmarkTransforms:
    """关键点变换"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化关键点变换
        
        Args:
            config: 增强配置
        """
        self.noise_std = config.get('noise_std', 0.01)
        self.scale_range = config.get('scale_range', [0.95, 1.05])
    
    def add_noise(self, landmarks: np.ndarray) -> np.ndarray:
        """
        添加高斯噪声
        
        Args:
            landmarks: 关键点坐标，形状为(146, 2)
            
        Returns:
            添加噪声后的关键点
        """
        noise = np.random.normal(0, self.noise_std, landmarks.shape)
        return landmarks + noise
    
    def scale_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        随机缩放关键点
        
        Args:
            landmarks: 关键点坐标
            
        Returns:
            缩放后的关键点
        """
        scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
        
        # 计算中心点
        center = np.mean(landmarks, axis=0)
        
        # 相对于中心点缩放
        scaled_landmarks = center + (landmarks - center) * scale_factor
        
        return scaled_landmarks
    
    def __call__(self, expression_landmarks: np.ndarray, 
                 baseline_landmarks: np.ndarray, 
                 training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用变换
        
        Args:
            expression_landmarks: 表情关键点
            baseline_landmarks: 基线关键点
            training: 是否为训练模式
            
        Returns:
            变换后的关键点
        """
        if not training:
            return expression_landmarks, baseline_landmarks
        
        # 添加噪声
        if self.noise_std > 0:
            expression_landmarks = self.add_noise(expression_landmarks)
            baseline_landmarks = self.add_noise(baseline_landmarks)
        
        # 随机缩放
        if self.scale_range != [1.0, 1.0]:
            expression_landmarks = self.scale_landmarks(expression_landmarks)
            baseline_landmarks = self.scale_landmarks(baseline_landmarks)
        
        return expression_landmarks, baseline_landmarks


class DataNormalizer:
    """数据标准化器"""
    
    def __init__(self):
        self.landmark_stats = None
        self.score_stats = None
    
    def fit_landmarks(self, landmarks_list):
        """
        拟合关键点统计信息
        
        Args:
            landmarks_list: 关键点列表
        """
        all_landmarks = np.concatenate(landmarks_list, axis=0)
        
        self.landmark_stats = {
            'mean': np.mean(all_landmarks, axis=0),
            'std': np.std(all_landmarks, axis=0)
        }
    
    def fit_scores(self, scores_list):
        """
        拟合评分统计信息
        
        Args:
            scores_list: 评分列表
        """
        all_scores = np.array(scores_list)
        
        self.score_stats = {
            'mean': np.mean(all_scores, axis=0),
            'std': np.std(all_scores, axis=0)
        }
    
    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """标准化关键点"""
        if self.landmark_stats is None:
            return landmarks
        
        return (landmarks - self.landmark_stats['mean']) / (self.landmark_stats['std'] + 1e-8)
    
    def denormalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """反标准化关键点"""
        if self.landmark_stats is None:
            return landmarks
        
        return landmarks * self.landmark_stats['std'] + self.landmark_stats['mean']
    
    def normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """标准化评分"""
        if self.score_stats is None:
            return scores
        
        return (scores - self.score_stats['mean']) / (self.score_stats['std'] + 1e-8)
    
    def denormalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """反标准化评分"""
        if self.score_stats is None:
            return scores
        
        return scores * self.score_stats['std'] + self.score_stats['mean']


def load_optical_flow_image(image_path: str) -> np.ndarray:
    """
    加载光流图像
    
    Args:
        image_path: 图像路径
        
    Returns:
        图像数组，形状为(H, W, 3)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")
    
    # 转换为RGB格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image


def load_landmarks(landmarks_path: str) -> np.ndarray:
    """
    加载面部关键点数据
    
    Args:
        landmarks_path: 关键点文件路径
        
    Returns:
        关键点数组，形状为 (num_landmarks, 2) 或 (num_landmarks, 3)
    """
    if not os.path.exists(landmarks_path):
        raise FileNotFoundError(f"关键点文件不存在: {landmarks_path}")
    
    landmarks = np.load(landmarks_path)
    
    # 处理不同的关键点格式
    if landmarks.shape == (478, 3):
        # MediaPipe 完整面部网格格式，选择关键区域的点
        # 选择眼部、眉毛、鼻子、嘴部等重要区域的关键点
        # 这里我们取前146个点并只保留x,y坐标
        landmarks = landmarks[:146, :2]
    elif landmarks.shape == (478, 2):
        # MediaPipe 格式但只有2D坐标
        landmarks = landmarks[:146, :]
    elif landmarks.shape == (146, 2):
        # 已经是期望的格式
        pass
    elif landmarks.shape == (146, 3):
        # 146个点但有3D坐标，只取x,y
        landmarks = landmarks[:, :2]
    else:
        # 尝试自动适应其他格式
        if landmarks.ndim == 2 and landmarks.shape[1] >= 2:
            # 如果有足够的点，取前146个
            if landmarks.shape[0] >= 146:
                landmarks = landmarks[:146, :2]
            else:
                # 如果点数不够，重复填充到146个
                num_points = landmarks.shape[0]
                repeat_factor = 146 // num_points + 1
                landmarks = np.tile(landmarks[:, :2], (repeat_factor, 1))[:146, :]
        else:
            raise ValueError(f"不支持的关键点格式: {landmarks.shape}")
    
    # 确保最终格式为 (146, 2)
    if landmarks.shape != (146, 2):
        raise ValueError(f"关键点转换后形状不正确，期望(146, 2)，实际: {landmarks.shape}")
        
    return landmarks


def create_transforms(config: Dict[str, Any]) -> Tuple[OpticalFlowTransforms, LandmarkTransforms]:
    """
    创建数据变换
    
    Args:
        config: 增强配置
        
    Returns:
        光流和关键点变换器
    """
    optical_flow_config = config.get('optical_flow', {})
    landmark_config = config.get('landmarks', {})
    
    optical_flow_transforms = OpticalFlowTransforms(optical_flow_config)
    landmark_transforms = LandmarkTransforms(landmark_config)
    
    return optical_flow_transforms, landmark_transforms


def test_transforms():
    """测试数据变换"""
    # 测试配置
    config = {
        'optical_flow': {
            'horizontal_flip': 0.5,
            'rotation': 10,
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1
        },
        'landmarks': {
            'noise_std': 0.01,
            'scale_range': [0.95, 1.05]
        }
    }
    
    # 创建变换器
    optical_transforms, landmark_transforms = create_transforms(config)
    
    # 测试光流变换
    test_image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    transformed_image = optical_transforms(test_image, training=True)
    print(f"光流图像变换: {test_image.shape} -> {transformed_image.shape}")
    
    # 测试关键点变换
    test_expression = np.random.randn(146, 2).astype(np.float32)
    test_baseline = np.random.randn(146, 2).astype(np.float32)
    
    transformed_expr, transformed_base = landmark_transforms(
        test_expression, test_baseline, training=True
    )
    
    print(f"关键点变换: {test_expression.shape} -> {transformed_expr.shape}")
    print(f"基线变换: {test_baseline.shape} -> {transformed_base.shape}")
    
    # 测试数据标准化器
    normalizer = DataNormalizer()
    
    # 模拟拟合数据
    landmark_list = [np.random.randn(146, 2) for _ in range(100)]
    score_list = [np.random.randn(2) for _ in range(100)]
    
    normalizer.fit_landmarks(landmark_list)
    normalizer.fit_scores(score_list)
    
    # 测试标准化
    test_landmarks = np.random.randn(146, 2)
    normalized = normalizer.normalize_landmarks(test_landmarks)
    denormalized = normalizer.denormalize_landmarks(normalized)
    
    print(f"关键点标准化误差: {np.mean(np.abs(test_landmarks - denormalized))}")


if __name__ == "__main__":
    test_transforms()
