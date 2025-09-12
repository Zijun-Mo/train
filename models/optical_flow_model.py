"""
光流模型定义 - 基于ResNet-18
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any


class OpticalFlowModel(nn.Module):
    """光流特征提取模型"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化光流模型
        
        Args:
            config: 模型配置
        """
        super(OpticalFlowModel, self).__init__()
        
        self.input_size = config.get('input_size', [3, 112, 112])
        self.backbone = config.get('backbone', 'resnet18')
        self.pretrained = config.get('pretrained', True)
        self.output_dim = config.get('output_dim', 2)
        self.dropout = config.get('dropout', 0.5)
        
        # 构建backbone
        self._build_backbone()
        
        # 构建分类器
        self._build_classifier()
    
    def _build_backbone(self):
        """构建backbone网络"""
        if self.backbone == 'resnet18':
            self.backbone_net = models.resnet18(pretrained=self.pretrained)
            # 移除最后的全连接层
            self.backbone_net = nn.Sequential(*list(self.backbone_net.children())[:-1])
            self.feature_dim = 512
        elif self.backbone == 'resnet34':
            self.backbone_net = models.resnet34(pretrained=self.pretrained)
            self.backbone_net = nn.Sequential(*list(self.backbone_net.children())[:-1])
            self.feature_dim = 512
        elif self.backbone == 'resnet50':
            self.backbone_net = models.resnet50(pretrained=self.pretrained)
            self.backbone_net = nn.Sequential(*list(self.backbone_net.children())[:-1])
            self.feature_dim = 2048
        else:
            raise ValueError(f"不支持的backbone: {self.backbone}")
    
    def _build_classifier(self):
        """构建分类器"""
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(self.dropout),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout / 2),
            nn.Linear(64, self.output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入光流图像，形状为(batch_size, 3, 112, 112)
            
        Returns:
            输出特征，形状为(batch_size, output_dim)
        """
        # 通过backbone提取特征
        features = self.backbone_net(x)
        
        # 通过分类器输出结果
        output = self.classifier(features)
        
        return output
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取中间特征（用于特征融合）
        
        Args:
            x: 输入光流图像
            
        Returns:
            中间特征
        """
        with torch.no_grad():
            features = self.backbone_net(x)
            # 应用自适应池化和展平
            features = nn.AdaptiveAvgPool2d((1, 1))(features)
            features = torch.flatten(features, 1)
            return features
    
    def freeze_backbone(self):
        """冻结backbone参数"""
        for param in self.backbone_net.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """解冻backbone参数"""
        for param in self.backbone_net.parameters():
            param.requires_grad = True
    
    def freeze_classifier(self):
        """冻结分类器参数"""
        for param in self.classifier.parameters():
            param.requires_grad = False
    
    def unfreeze_classifier(self):
        """解冻分类器参数"""
        for param in self.classifier.parameters():
            param.requires_grad = True


def create_optical_flow_model(config: Dict[str, Any]) -> OpticalFlowModel:
    """
    创建光流模型
    
    Args:
        config: 模型配置
        
    Returns:
        光流模型实例
    """
    return OpticalFlowModel(config)
