"""
关键点模型定义 - 支持基础MLP和标准MLP架构
"""
import torch
import torch.nn as nn
from typing import Dict, Any, List


class LandmarkModel(nn.Module):
    """基础关键点特征处理模型"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化关键点模型
        
        Args:
            config: 模型配置
        """
        super(LandmarkModel, self).__init__()
        
        self.input_dim = config.get('input_dim', 104)  # 52 + 52
        self.hidden_dims = config.get('hidden_dims', [256, 128, 64])
        self.output_dim = config.get('output_dim', 2)
        self.dropout = config.get('dropout', 0.3)
        self.activation = config.get('activation', 'relu')
        
        # 构建网络
        self._build_network()
    
    def _build_network(self):
        """构建全连接网络"""
        layers = []
        
        # 输入层
        prev_dim = self.input_dim
        
        # 隐藏层
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self._get_activation(),
                nn.Dropout(self.dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self):
        """获取激活函数"""
        if self.activation == 'relu':
            return nn.ReLU(inplace=True)
        elif self.activation == 'leaky_relu':
            return nn.LeakyReLU(0.1, inplace=True)
        elif self.activation == 'elu':
            return nn.ELU(inplace=True)
        elif self.activation == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"不支持的激活函数: {self.activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征，形状为(batch_size, input_dim)
            
        Returns:
            输出特征，形状为(batch_size, output_dim)
        """
        return self.network(x)
    
    def freeze_all(self):
        """冻结所有参数"""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True
    
    def freeze_layers(self, layer_indices: List[int]):
        """
        冻结指定层的参数
        
        Args:
            layer_indices: 要冻结的层索引列表
        """
        for i, layer in enumerate(self.network):
            if i in layer_indices:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def unfreeze_layers(self, layer_indices: List[int]):
        """
        解冻指定层的参数
        
        Args:
            layer_indices: 要解冻的层索引列表
        """
        for i, layer in enumerate(self.network):
            if i in layer_indices:
                for param in layer.parameters():
                    param.requires_grad = True
    
    def get_trainable_parameters(self):
        """获取当前可训练的参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def extract_intermediate_features(self, x: torch.Tensor, layer_idx: int = -2) -> torch.Tensor:
        """
        提取中间层特征
        
        Args:
            x: 输入特征
            layer_idx: 提取第几层的特征（负数表示从后往前数）
            
        Returns:
            中间层特征
        """
        with torch.no_grad():
            for i, layer in enumerate(self.network):
                x = layer(x)
                if i == len(self.network) + layer_idx:
                    return x
            return x


class StandardBlock(nn.Module):
    """标准全连接块"""
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float, activation: str, use_batch_norm: bool = True):
        super(StandardBlock, self).__init__()
        
        layers = [nn.Linear(input_dim, output_dim)]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(output_dim))
        layers.extend([
            self._get_activation(activation),
            nn.Dropout(dropout)
        ])
        
        self.block = nn.Sequential(*layers)
    
    def _get_activation(self, activation: str):
        """获取激活函数"""
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'elu':
            return nn.ELU(inplace=True)
        elif activation == 'gelu':
            return nn.GELU()
        else:
            return nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class StandardLandmarkModel(nn.Module):
    """标准关键点模型，使用StandardBlock构建"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化标准关键点模型
        
        Args:
            config: 模型配置
        """
        super(StandardLandmarkModel, self).__init__()
        
        self.input_dim = config.get('input_dim', 104)
        self.hidden_dims = config.get('hidden_dims', [256, 128, 64])
        self.output_dim = config.get('output_dim', 2)
        self.dropout = config.get('dropout', 0.3)
        self.activation = config.get('activation', 'relu')
        self.use_batch_norm = config.get('use_batch_norm', True)
        
        # 构建网络
        self._build_network()
    
    def _build_network(self):
        """构建标准网络"""
        # 输入投影层
        layers = []
        layers.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(self.hidden_dims[0]))
        layers.extend([
            self._get_activation(),
            nn.Dropout(self.dropout)
        ])
        self.input_proj = nn.Sequential(*layers)
        
        # 构建标准块
        self.blocks = nn.ModuleList()
        for i in range(len(self.hidden_dims) - 1):
            block = StandardBlock(
                self.hidden_dims[i], 
                self.hidden_dims[i + 1],
                self.dropout,
                self.activation,
                self.use_batch_norm
            )
            self.blocks.append(block)
        
        # 输出层
        output_layers = [nn.Dropout(self.dropout)]
        if self.use_batch_norm:
            output_layers.append(nn.BatchNorm1d(self.hidden_dims[-1]))
        output_layers.append(nn.Linear(self.hidden_dims[-1], self.output_dim))
        self.output_layer = nn.Sequential(*output_layers)
    
    def _get_activation(self):
        """获取激活函数"""
        if self.activation == 'relu':
            return nn.ReLU(inplace=True)
        elif self.activation == 'leaky_relu':
            return nn.LeakyReLU(0.1, inplace=True)
        elif self.activation == 'elu':
            return nn.ELU(inplace=True)
        elif self.activation == 'gelu':
            return nn.GELU()
        else:
            return nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 输入投影
        x = self.input_proj(x)
        
        # 通过标准块
        for block in self.blocks:
            x = block(x)
        
        # 输出层
        output = self.output_layer(x)
        
        return output
    
    def freeze_all(self):
        """冻结所有参数"""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True
    
    def get_trainable_parameters(self):
        """获取当前可训练的参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_landmark_model(config: Dict[str, Any], enhanced: bool = False) -> nn.Module:
    """
    创建关键点模型
    
    Args:
        config: 模型配置
        enhanced: 是否使用增强版模型（目前只支持标准模型）
        
    Returns:
        关键点模型实例
    """
    model_type = config.get('model_type', 'mlp')
    
    if model_type == 'standard':
        return StandardLandmarkModel(config)
    else:
        return LandmarkModel(config)
