"""
数据集类定义
"""
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import random
from sklearn.model_selection import train_test_split

from data.transforms import load_optical_flow_image, load_landmarks
from data.synchronized_augmentation import SynchronizedAugmentation
from models.face_blendshapes import DualFaceBlendshapesExtractor


class FacialExpressionDataset(Dataset):
    """面部表情评估数据集"""
    
    def __init__(self, 
                 data_root: str,
                 video_ids: List[str],
                 face_blendshapes_model_path: str,
                 synchronized_augmenter: SynchronizedAugmentation,
                 training: bool = True,
                 preload_features: bool = False):
        """
        初始化数据集
        
        Args:
            data_root: 数据根目录
            video_ids: 视频ID列表
            face_blendshapes_model_path: face_blendshapes模型路径
            synchronized_augmenter: 同步数据增强器
            training: 是否为训练模式
            preload_features: 是否预加载特征
        """
        self.data_root = data_root
        self.video_ids = video_ids
        self.synchronized_augmenter = synchronized_augmenter
        self.training = training
        self.preload_features = preload_features
        
        # 初始化TFLite特征提取器
        self.feature_extractor = DualFaceBlendshapesExtractor(face_blendshapes_model_path)
        
        # 扫描数据
        self.data_items = self._scan_data()
        
        # 预加载特征（可选）
        self.preloaded_features = {}
        if preload_features:
            self._preload_features()
    
    def _scan_data(self) -> List[Dict[str, Any]]:
        """扫描数据目录，收集所有数据项"""
        data_items = []
        
        # 直接扫描data_root下的所有视频帧文件夹 (去掉了子目录层级)
        for video_frame in os.listdir(self.data_root):
            frame_path = os.path.join(self.data_root, video_frame)
            if not os.path.isdir(frame_path):
                continue
            
            # 解析视频ID和帧ID
            try:
                video_id, frame_id = video_frame.split('_')
                if video_id not in self.video_ids:
                    continue
            except ValueError:
                continue
            
            # 检查必要文件是否存在
            optical_flow_path = os.path.join(frame_path, 'optical_flow.jpg')
            expression_landmarks_path = os.path.join(frame_path, 'expression_landmarks.npy')
            baseline_landmarks_path = os.path.join(frame_path, 'baseline_landmarks.npy')
            label_path = os.path.join(frame_path, 'expression_g_column.json')
            
            if all(os.path.exists(p) for p in [optical_flow_path, expression_landmarks_path, 
                                               baseline_landmarks_path, label_path]):
                data_items.append({
                    'video_id': video_id,
                    'frame_id': frame_id,
                    'optical_flow_path': optical_flow_path,
                    'expression_landmarks_path': expression_landmarks_path,
                    'baseline_landmarks_path': baseline_landmarks_path,
                    'label_path': label_path
                })
        
        return data_items
    
    def _preload_features(self):
        """预加载关键点特征"""
        print("预加载关键点特征...")
        
        for i, item in enumerate(self.data_items):
            if i % 100 == 0:
                print(f"预加载进度: {i}/{len(self.data_items)}")
            
            # 加载关键点
            expression_landmarks = load_landmarks(item['expression_landmarks_path'])
            baseline_landmarks = load_landmarks(item['baseline_landmarks_path'])
            
            # 提取特征
            features = self.feature_extractor.extract_features(
                expression_landmarks, baseline_landmarks
            )
            
            # 存储特征
            item_key = f"{item['video_id']}_{item['frame_id']}"
            self.preloaded_features[item_key] = features
        
        print("特征预加载完成!")
    
    def __len__(self) -> int:
        return len(self.data_items)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个数据项
        
        Args:
            idx: 数据索引
            
        Returns:
            包含输入和标签的字典
        """
        item = self.data_items[idx]
        
        # 加载光流图像和关键点
        optical_flow_image = load_optical_flow_image(item['optical_flow_path'])
        expression_landmarks = load_landmarks(item['expression_landmarks_path'])
        baseline_landmarks = load_landmarks(item['baseline_landmarks_path'])
        
        # 应用同步数据增强
        optical_flow_tensor, expression_landmarks, baseline_landmarks = self.synchronized_augmenter(
            optical_flow_image, expression_landmarks, baseline_landmarks, self.training
        )
        
        # 提取关键点特征
        landmark_features = self.feature_extractor.extract_features(
            expression_landmarks, baseline_landmarks
        )
        
        # 转换为张量
        landmark_features_tensor = torch.from_numpy(landmark_features).float()
        
        # 加载标签
        with open(item['label_path'], 'r') as f:
            label_data = json.load(f)
        
        dynamics_score = float(label_data['dynamics'])
        synkinesis_score = float(label_data['synkinesis'])
        
        # 使用连续值模式
        target_tensor = torch.tensor([dynamics_score, synkinesis_score], dtype=torch.float32)
        
        return {
            'optical_flow': optical_flow_tensor,
            'landmark_features': landmark_features_tensor,
            'target': target_tensor,
            'video_id': item['video_id'],
            'frame_id': item['frame_id']
        }


def split_dataset(data_root: str, train_ratio: float = 0.7, 
                  val_ratio: float = 0.15, test_ratio: float = 0.15,
                  random_seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    按视频划分训练集、验证集和测试集
    
    Args:
        data_root: 数据根目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子
        
    Returns:
        训练集、验证集和测试集的视频ID列表
    """
    # 验证比例和是否为1
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"数据集划分比例之和必须为1.0，当前为{total_ratio}")
    
    # 获取所有视频ID
    video_ids = set()
    
    # 直接扫描data_root下的所有视频帧文件夹 (去掉了子目录层级)
    for video_frame in os.listdir(data_root):
        frame_path = os.path.join(data_root, video_frame)
        if not os.path.isdir(frame_path):
            continue
            
        try:
            # 更健壮的视频ID和帧ID解析
            # 期望格式：video_id_frame_id，其中可能包含多个下划线
            parts = video_frame.split('_')
            if len(parts) >= 2:
                # 取第一部分作为video_id，剩余部分作为frame_id
                video_id = parts[0]
                video_ids.add(video_id)
        except (ValueError, IndexError):
            continue
    
    video_ids = sorted(list(video_ids))
    
    # 首先划分训练集和临时集（验证+测试）
    train_video_ids, temp_video_ids = train_test_split(
        video_ids, train_size=train_ratio, random_state=random_seed
    )
    
    # 计算验证集在临时集中的比例
    val_ratio_in_temp = val_ratio / (val_ratio + test_ratio)
    
    # 将临时集划分为验证集和测试集
    val_video_ids, test_video_ids = train_test_split(
        temp_video_ids, train_size=val_ratio_in_temp, random_state=random_seed
    )
    
    return train_video_ids, val_video_ids, test_video_ids


def create_data_loaders(config: Dict[str, Any], 
                       synchronized_augmenter: SynchronizedAugmentation) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建数据加载器
    
    Args:
        config: 数据配置
        synchronized_augmenter: 同步数据增强器
        
    Returns:
        训练、验证和测试数据加载器
    """
    data_root = config['root_dir']
    face_blendshapes_model_path = config['face_blendshapes_model']
    train_ratio = config.get('train_ratio', 0.7)
    val_ratio = config.get('val_ratio', 0.15)
    test_ratio = config.get('test_ratio', 0.15)
    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 4)
    
    # 划分数据集
    train_video_ids, val_video_ids, test_video_ids = split_dataset(
        data_root, train_ratio, val_ratio, test_ratio
    )
    
    print(f"训练集视频数量: {len(train_video_ids)}")
    print(f"验证集视频数量: {len(val_video_ids)}")
    print(f"测试集视频数量: {len(test_video_ids)}")
    
    # 使用连续值输出格式（回归模式）
    output_format = 'continuous'
    print(f"使用输出格式: {output_format} (回归模式)")
    
    # 创建数据集
    train_dataset = FacialExpressionDataset(
        data_root=data_root,
        video_ids=train_video_ids,
        face_blendshapes_model_path=face_blendshapes_model_path,
        synchronized_augmenter=synchronized_augmenter,
        training=True,
        preload_features=config.get('preload_features', False)
    )
    
    val_dataset = FacialExpressionDataset(
        data_root=data_root,
        video_ids=val_video_ids,
        face_blendshapes_model_path=face_blendshapes_model_path,
        synchronized_augmenter=synchronized_augmenter,
        training=False,
        preload_features=config.get('preload_features', False)
    )
    
    test_dataset = FacialExpressionDataset(
        data_root=data_root,
        video_ids=test_video_ids,
        face_blendshapes_model_path=face_blendshapes_model_path,
        synchronized_augmenter=synchronized_augmenter,
        training=False,
        preload_features=config.get('preload_features', False)
    )
    
    print(f"训练集样本数量: {len(train_dataset)}")
    print(f"验证集样本数量: {len(val_dataset)}")
    print(f"测试集样本数量: {len(test_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader



