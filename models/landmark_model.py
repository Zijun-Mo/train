"""
关键点模型定义 - 支持传统MLP和Transformer架构
"""
import torch
import torch.nn as nn
import math
from typing import Dict, Any, List


class LandmarkModel(nn.Module):
    """关键点特征处理模型"""
    
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


class TransformerLandmarkModel(nn.Module):
    """基于Transformer的关键点特征处理模型"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Transformer关键点模型
        
        Args:
            config: 模型配置
        """
        super(TransformerLandmarkModel, self).__init__()
        
        self.input_dim = config.get('input_dim', 104)
        self.output_dim = config.get('output_dim', 2)
        self.dropout = config.get('dropout', 0.3)
        
        # Transformer配置
        transformer_config = config.get('transformer', {})
        self.num_heads = transformer_config.get('num_heads', 8)
        self.num_layers = transformer_config.get('num_layers', 3)
        self.dim_feedforward = transformer_config.get('dim_feedforward', 512)
        self.transformer_dropout = transformer_config.get('dropout', 0.1)
        
        # 计算模型维度（必须能被num_heads整除）
        self.model_dim = self._calculate_model_dim()
        
        # 输入投影层
        self.input_projection = nn.Linear(self.input_dim, self.model_dim)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(self.model_dim, self.transformer_dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.transformer_dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.num_layers
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Dropout(self.dropout),
            nn.Linear(self.model_dim, self.model_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.model_dim // 2, self.output_dim)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _calculate_model_dim(self):
        """计算合适的模型维度，确保能被num_heads整除"""
        # 从输入维度开始，找到第一个能被num_heads整除且不小于输入维度的数
        base_dim = max(self.input_dim, 128)  # 至少128维
        while base_dim % self.num_heads != 0:
            base_dim += 1
        return min(base_dim, 512)  # 最大不超过512维
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征，形状为(batch_size, input_dim)
            
        Returns:
            输出特征，形状为(batch_size, output_dim)
        """
        batch_size = x.size(0)
        
        # 输入投影 (batch_size, input_dim) -> (batch_size, model_dim)
        x = self.input_projection(x)
        
        # 添加序列维度以适配Transformer (batch_size, 1, model_dim)
        x = x.unsqueeze(1)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # Transformer编码 (batch_size, 1, model_dim)
        x = self.transformer_encoder(x)
        
        # 移除序列维度 (batch_size, model_dim)
        x = x.squeeze(1)
        
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
    
    def freeze_transformer(self):
        """只冻结Transformer部分"""
        for param in self.transformer_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_transformer(self):
        """解冻Transformer部分"""
        for param in self.transformer_encoder.parameters():
            param.requires_grad = True
    
    def get_trainable_parameters(self):
        """获取当前可训练的参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class EnhancedLandmarkModel(nn.Module):
    """增强版关键点模型，包含残差连接和注意力机制"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化增强版关键点模型
        
        Args:
            config: 模型配置
        """
        super(EnhancedLandmarkModel, self).__init__()
        
        self.input_dim = config.get('input_dim', 104)
        self.hidden_dims = config.get('hidden_dims', [256, 128, 64])
        self.output_dim = config.get('output_dim', 2)
        self.dropout = config.get('dropout', 0.3)
        self.activation = config.get('activation', 'relu')
        self.use_attention = config.get('use_attention', True)
        self.use_residual = config.get('use_residual', True)
        
        # 构建网络
        self._build_network()
    
    def _build_network(self):
        """构建增强网络"""
        # 输入投影层
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dims[0])
        
        # 构建残差块
        self.blocks = nn.ModuleList()
        for i in range(len(self.hidden_dims) - 1):
            block = ResidualBlock(
                self.hidden_dims[i], 
                self.hidden_dims[i + 1],
                self.dropout,
                self.activation
            )
            self.blocks.append(block)
        
        # 注意力机制
        if self.use_attention:
            self.attention = SelfAttention(self.hidden_dims[-1])
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dims[-1], self.output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 输入投影
        x = self.input_proj(x)
        
        # 通过残差块
        for block in self.blocks:
            x = block(x)
        
        # 应用注意力
        if self.use_attention:
            x = self.attention(x)
        
        # 输出层
        output = self.output_layer(x)
        
        return output


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float, activation: str):
        super(ResidualBlock, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 主路径
        self.main_path = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )
        
        # 残差连接（如果维度不同需要投影）
        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = nn.Identity()
        
        self.activation = self._get_activation(activation)
    
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
        # 主路径
        main_out = self.main_path(x)
        
        # 残差连接
        residual = self.residual_proj(x)
        
        # 相加并应用激活
        output = self.activation(main_out + residual)
        
        return output


class SelfAttention(nn.Module):
    """自注意力机制"""
    
    def __init__(self, dim: int):
        super(SelfAttention, self).__init__()
        
        self.dim = dim
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, dim)
        batch_size = x.size(0)
        
        # 为单个向量添加序列维度
        x = x.unsqueeze(1)  # (batch_size, 1, dim)
        
        q = self.query(x)  # (batch_size, 1, dim)
        k = self.key(x)    # (batch_size, 1, dim)
        v = self.value(x)  # (batch_size, 1, dim)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim ** 0.5)
        attention_weights = self.softmax(scores)
        
        # 应用注意力
        attended = torch.matmul(attention_weights, v)
        
        # 移除序列维度
        output = attended.squeeze(1)  # (batch_size, dim)
        
        return output


def create_landmark_model(config: Dict[str, Any], enhanced: bool = False) -> nn.Module:
    """
    创建关键点模型
    
    Args:
        config: 模型配置
        enhanced: 是否使用增强版模型
        
    Returns:
        关键点模型实例
    """
    model_type = config.get('model_type', 'mlp')
    
    if model_type == 'transformer':
        return TransformerLandmarkModel(config)
    elif enhanced:
        return EnhancedLandmarkModel(config)
    else:
        return LandmarkModel(config)


def test_landmark_model():
    """测试关键点模型"""
    config = {
        'input_dim': 104,
        'hidden_dims': [256, 128, 64],
        'output_dim': 2,
        'dropout': 0.3,
        'activation': 'relu'
    }
    
    # 测试基础模型
    print("测试基础关键点模型:")
    model = create_landmark_model(config, enhanced=False)
    
    batch_size = 4
    test_input = torch.randn(batch_size, 104)
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 测试Transformer模型
    print("\n测试Transformer关键点模型:")
    transformer_config = config.copy()
    transformer_config.update({
        'model_type': 'transformer',
        'transformer': {
            'num_heads': 8,
            'num_layers': 3,
            'dim_feedforward': 512,
            'dropout': 0.1
        }
    })
    
    transformer_model = create_landmark_model(transformer_config)
    
    with torch.no_grad():
        transformer_output = transformer_model(test_input)
    
    print(f"Transformer输出形状: {transformer_output.shape}")
    print(f"Transformer模型参数数量: {sum(p.numel() for p in transformer_model.parameters())}")
    
    # 测试增强版模型
    print("\n测试增强版关键点模型:")
    enhanced_config = config.copy()
    enhanced_config.update({
        'use_attention': True,
        'use_residual': True
    })
    
    enhanced_model = create_landmark_model(enhanced_config, enhanced=True)
    
    with torch.no_grad():
        enhanced_output = enhanced_model(test_input)
    
    print(f"增强版输出形状: {enhanced_output.shape}")
    print(f"增强版模型参数数量: {sum(p.numel() for p in enhanced_model.parameters())}")


if __name__ == "__main__":
    test_landmark_model()
