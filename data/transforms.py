"""
数据预处理工具函数
"""
import torch
import numpy as np
import cv2
import os
from typing import Dict, Any, Tuple, Optional


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


def test_data_loading():
    """测试数据加载函数"""
    print("测试数据加载函数...")
    
    # 创建测试数据
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    test_landmarks = np.random.randn(146, 2).astype(np.float32)
    
    print(f"测试图像形状: {test_image.shape}")
    print(f"测试关键点形状: {test_landmarks.shape}")
    
    # 测试数据标准化器
    normalizer = DataNormalizer()
    
    # 模拟拟合数据
    landmark_list = [np.random.randn(146, 2) for _ in range(100)]
    score_list = [np.random.randn(2) for _ in range(100)]
    
    normalizer.fit_landmarks(landmark_list)
    normalizer.fit_scores(score_list)
    
    # 测试标准化
    normalized = normalizer.normalize_landmarks(test_landmarks)
    denormalized = normalizer.denormalize_landmarks(normalized)
    
    print(f"关键点标准化误差: {np.mean(np.abs(test_landmarks - denormalized))}")
    print("数据加载测试完成!")


if __name__ == "__main__":
    test_data_loading()
