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

from data.transforms import OpticalFlowTransforms, LandmarkTransforms, load_optical_flow_image, load_landmarks
from models.face_blendshapes import DualFaceBlendshapesExtractor


class FacialExpressionDataset(Dataset):
    """面部表情评估数据集"""
    
    def __init__(self, 
                 data_root: str,
                 video_ids: List[str],
                 face_blendshapes_model_path: str,
                 optical_flow_transforms: OpticalFlowTransforms,
                 landmark_transforms: LandmarkTransforms,
                 training: bool = True,
                 preload_features: bool = False):
        """
        初始化数据集
        
        Args:
            data_root: 数据根目录
            video_ids: 视频ID列表
            face_blendshapes_model_path: face_blendshapes模型路径
            optical_flow_transforms: 光流图像变换器
            landmark_transforms: 关键点变换器
            training: 是否为训练模式
            preload_features: 是否预加载特征
        """
        self.data_root = data_root
        self.video_ids = video_ids
        self.optical_flow_transforms = optical_flow_transforms
        self.landmark_transforms = landmark_transforms
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
        
        # 加载光流图像
        optical_flow_image = load_optical_flow_image(item['optical_flow_path'])
        optical_flow_tensor = self.optical_flow_transforms(optical_flow_image, self.training)
        
        # 加载或获取关键点特征
        item_key = f"{item['video_id']}_{item['frame_id']}"
        if self.preload_features and item_key in self.preloaded_features:
            landmark_features = self.preloaded_features[item_key]
        else:
            # 加载关键点
            expression_landmarks = load_landmarks(item['expression_landmarks_path'])
            baseline_landmarks = load_landmarks(item['baseline_landmarks_path'])
            
            # 应用变换
            expression_landmarks, baseline_landmarks = self.landmark_transforms(
                expression_landmarks, baseline_landmarks, self.training
            )
            
            # 提取特征
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


def split_dataset(data_root: str, train_ratio: float = 0.8, 
                  random_seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    按视频划分训练集和验证集
    
    Args:
        data_root: 数据根目录
        train_ratio: 训练集比例
        random_seed: 随机种子
        
    Returns:
        训练集和验证集的视频ID列表
    """
    # 获取所有视频ID
    video_ids = set()
    
    # 直接扫描data_root下的所有视频帧文件夹 (去掉了子目录层级)
    for video_frame in os.listdir(data_root):
        frame_path = os.path.join(data_root, video_frame)
        if not os.path.isdir(frame_path):
            continue
            
        try:
            video_id, frame_id = video_frame.split('_')
            video_ids.add(video_id)
        except ValueError:
            continue
    
    video_ids = sorted(list(video_ids))
    
    # 划分训练集和验证集
    train_video_ids, val_video_ids = train_test_split(
        video_ids, train_size=train_ratio, random_state=random_seed
    )
    
    return train_video_ids, val_video_ids


def create_data_loaders(config: Dict[str, Any], 
                       optical_flow_transforms: OpticalFlowTransforms,
                       landmark_transforms: LandmarkTransforms) -> Tuple[DataLoader, DataLoader]:
    """
    创建数据加载器
    
    Args:
        config: 数据配置
        optical_flow_transforms: 光流变换器
        landmark_transforms: 关键点变换器
        
    Returns:
        训练和验证数据加载器
    """
    data_root = config['root_dir']
    face_blendshapes_model_path = config['face_blendshapes_model']
    train_ratio = config.get('train_ratio', 0.8)
    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 4)
    
    # 划分数据集
    train_video_ids, val_video_ids = split_dataset(data_root, train_ratio)
    
    print(f"训练集视频数量: {len(train_video_ids)}")
    print(f"验证集视频数量: {len(val_video_ids)}")
    
    # 使用连续值输出格式（回归模式）
    output_format = 'continuous'
    print(f"使用输出格式: {output_format} (回归模式)")
    
    # 创建数据集
    train_dataset = FacialExpressionDataset(
        data_root=data_root,
        video_ids=train_video_ids,
        face_blendshapes_model_path=face_blendshapes_model_path,
        optical_flow_transforms=optical_flow_transforms,
        landmark_transforms=landmark_transforms,
        training=True,
        preload_features=config.get('preload_features', False)
    )
    
    val_dataset = FacialExpressionDataset(
        data_root=data_root,
        video_ids=val_video_ids,
        face_blendshapes_model_path=face_blendshapes_model_path,
        optical_flow_transforms=optical_flow_transforms,
        landmark_transforms=landmark_transforms,
        training=False,
        preload_features=config.get('preload_features', False)
    )
    
    print(f"训练集样本数量: {len(train_dataset)}")
    print(f"验证集样本数量: {len(val_dataset)}")
    
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
    
    return train_loader, val_loader


def test_dataset():
    """测试数据集"""
    # 测试配置
    config = {
        'root_dir': '/home/jun/picture/output',
        'face_blendshapes_model': '/home/jun/picture/extracted_models/face_blendshapes.tflite',
        'train_ratio': 0.8,
        'batch_size': 4,
        'num_workers': 0,
        'preload_features': False
    }
    
    # 创建变换器（使用空配置进行测试）
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from data.transforms import create_transforms
    optical_transforms, landmark_transforms = create_transforms({})
    
    try:
        # 创建数据加载器
        train_loader, val_loader = create_data_loaders(
            config, optical_transforms, landmark_transforms
        )
        
        # 测试数据加载
        print("测试数据加载...")
        for i, batch in enumerate(train_loader):
            if i == 0:  # 只显示第一个batch的信息
                print(f"训练batch形状: 光流{batch['optical_flow'].shape}, "
                      f"关键点{batch['landmark_features'].shape}, "
                      f"目标{batch['target'].shape}")
            if i >= 2:  # 只测试前几个batch
                break
        
        for i, batch in enumerate(val_loader):
            if i == 0:  # 只显示第一个batch的信息
                print(f"验证batch形状: 光流{batch['optical_flow'].shape}, "
                      f"关键点{batch['landmark_features'].shape}, "
                      f"目标{batch['target'].shape}")
            if i >= 1:  # 只测试前几个batch
                break
        
        print("数据集测试成功！")
                
    except Exception as e:
        print(f"数据集测试失败: {e}")
        print("这可能是因为数据路径不存在或格式不正确")


if __name__ == "__main__":
    test_dataset()
