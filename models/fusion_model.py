"""
融合模型定义 - 结合光流和关键点特征
"""
import torch
import torch.nn as nn
from typing import Dict, Any


class FusionModel(nn.Module):
    """特征融合模型"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化融合模型
        
        Args:
            config: 模型配置
        """
        super(FusionModel, self).__init__()
        
        self.input_dim = config.get('input_dim', 4)  # 2 + 2
        self.hidden_dims = config.get('hidden_dims', [16, 8])
        self.output_dim = config.get('output_dim', 2)
        self.dropout = config.get('dropout', 0.2)
        self.activation = config.get('activation', 'relu')
        
        # 构建网络
        self._build_network()
    
    def _build_network(self):
        """构建融合网络"""
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
        elif self.activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"不支持的激活函数: {self.activation}")
    
    def forward(self, optical_flow_features: torch.Tensor, 
                landmark_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            optical_flow_features: 光流特征，形状为(batch_size, 2)
            landmark_features: 关键点特征，形状为(batch_size, 2)
            
        Returns:
            融合后的输出，形状为(batch_size, 2)
        """
        # 拼接特征
        combined_features = torch.cat([optical_flow_features, landmark_features], dim=1)
        
        # 通过融合网络
        output = self.network(combined_features)
        
        return output


class AdvancedFusionModel(nn.Module):
    """高级融合模型，包含交叉注意力机制"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化高级融合模型
        
        Args:
            config: 模型配置
        """
        super(AdvancedFusionModel, self).__init__()
        
        self.feature_dim = 2  # 每个模态的特征维度
        self.hidden_dims = config.get('hidden_dims', [16, 8])
        self.output_dim = config.get('output_dim', 2)
        self.dropout = config.get('dropout', 0.2)
        self.use_attention = config.get('use_attention', True)
        
        # 特征投影层
        self.optical_flow_proj = nn.Linear(self.feature_dim, 8)
        self.landmark_proj = nn.Linear(self.feature_dim, 8)
        
        # 交叉注意力机制
        if self.use_attention:
            self.cross_attention = CrossModalAttention(8)
        
        # 融合网络
        fusion_input_dim = 16 if self.use_attention else 4
        self.fusion_network = self._build_fusion_network(fusion_input_dim)
    
    def _build_fusion_network(self, input_dim: int):
        """构建融合网络"""
        layers = []
        
        prev_dim = input_dim
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, optical_flow_features: torch.Tensor, 
                landmark_features: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 特征投影
        optical_proj = self.optical_flow_proj(optical_flow_features)
        landmark_proj = self.landmark_proj(landmark_features)
        
        if self.use_attention:
            # 交叉注意力
            attended_optical, attended_landmark = self.cross_attention(optical_proj, landmark_proj)
            # 拼接注意力后的特征
            combined = torch.cat([attended_optical, attended_landmark], dim=1)
        else:
            # 直接拼接
            combined = torch.cat([optical_proj, landmark_proj], dim=1)
        
        # 融合网络
        output = self.fusion_network(combined)
        
        return output


class CrossModalAttention(nn.Module):
    """跨模态注意力机制"""
    
    def __init__(self, dim: int):
        super(CrossModalAttention, self).__init__()
        
        self.dim = dim
        
        # 查询、键、值投影
        self.optical_query = nn.Linear(dim, dim)
        self.optical_key = nn.Linear(dim, dim)
        self.optical_value = nn.Linear(dim, dim)
        
        self.landmark_query = nn.Linear(dim, dim)
        self.landmark_key = nn.Linear(dim, dim)
        self.landmark_value = nn.Linear(dim, dim)
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, optical_features: torch.Tensor, 
                landmark_features: torch.Tensor) -> tuple:
        """
        交叉注意力计算
        
        Args:
            optical_features: 光流特征
            landmark_features: 关键点特征
            
        Returns:
            注意力加权后的特征
        """
        batch_size = optical_features.size(0)
        
        # 光流特征关注关键点特征
        optical_q = self.optical_query(optical_features).unsqueeze(1)
        landmark_k = self.landmark_key(landmark_features).unsqueeze(1)
        landmark_v = self.landmark_value(landmark_features).unsqueeze(1)
        
        optical_attention_scores = torch.matmul(optical_q, landmark_k.transpose(-2, -1)) / (self.dim ** 0.5)
        optical_attention_weights = self.softmax(optical_attention_scores)
        attended_optical = torch.matmul(optical_attention_weights, landmark_v).squeeze(1)
        
        # 关键点特征关注光流特征
        landmark_q = self.landmark_query(landmark_features).unsqueeze(1)
        optical_k = self.optical_key(optical_features).unsqueeze(1)
        optical_v = self.optical_value(optical_features).unsqueeze(1)
        
        landmark_attention_scores = torch.matmul(landmark_q, optical_k.transpose(-2, -1)) / (self.dim ** 0.5)
        landmark_attention_weights = self.softmax(landmark_attention_scores)
        attended_landmark = torch.matmul(landmark_attention_weights, optical_v).squeeze(1)
        
        return attended_optical, attended_landmark


class CompleteFusionModel(nn.Module):
    """完整的融合模型，整合光流、关键点和融合网络"""
    
    def __init__(self, optical_flow_model, landmark_model, fusion_model):
        """
        初始化完整融合模型
        
        Args:
            optical_flow_model: 光流模型
            landmark_model: 关键点模型
            fusion_model: 融合模型
        """
        super(CompleteFusionModel, self).__init__()
        
        self.optical_flow_model = optical_flow_model
        self.landmark_model = landmark_model
        self.fusion_model = fusion_model
        
        # 梯度监控
        self.gradient_norms = {}
        self.register_gradient_hooks()
    
    def register_gradient_hooks(self):
        """注册梯度监控钩子"""
        def hook_fn(name):
            def hook(grad):
                if grad is not None:
                    self.gradient_norms[name] = grad.norm().item()
                return grad
            return hook
        
        # 监控关键层的梯度
        if hasattr(self.optical_flow_model, 'classifier'):
            for param in self.optical_flow_model.classifier.parameters():
                param.register_hook(hook_fn('optical_flow_classifier'))
                break
                
        if hasattr(self.landmark_model, 'network'):
            for param in self.landmark_model.network.parameters():
                param.register_hook(hook_fn('landmark_network'))
                break
                
        for param in self.fusion_model.parameters():
            param.register_hook(hook_fn('fusion_model'))
            break
    
    def get_gradient_info(self):
        """获取梯度信息用于监控梯度消失"""
        return self.gradient_norms.copy()
    
    def check_gradient_health(self, logger=None):
        """
        检查梯度健康状态并发出告警
        
        Args:
            logger: 日志记录器
            
        Returns:
            dict: 包含梯度健康状态的字典
        """
        import warnings
        
        health_status = {
            'healthy': True,
            'warnings': [],
            'gradient_stats': self.gradient_norms.copy()
        }
        
        # 梯度消失阈值 (通常小于1e-6被认为是梯度消失)
        vanishing_threshold = 1e-6
        # 梯度爆炸阈值 (大于100通常表示梯度爆炸)
        exploding_threshold = 100.0
        
        for layer_name, grad_norm in self.gradient_norms.items():
            # 检查梯度消失
            if grad_norm < vanishing_threshold:
                warning_msg = f"⚠️  梯度消失告警: {layer_name} 层梯度范数 {grad_norm:.2e} < {vanishing_threshold:.0e}"
                health_status['warnings'].append(warning_msg)
                health_status['healthy'] = False
                
                if logger:
                    logger.warning(warning_msg)
                else:
                    warnings.warn(warning_msg)
                    print(f"🚨 {warning_msg}")
            
            # 检查梯度爆炸  
            elif grad_norm > exploding_threshold:
                warning_msg = f"🔥 梯度爆炸告警: {layer_name} 层梯度范数 {grad_norm:.2f} > {exploding_threshold}"
                health_status['warnings'].append(warning_msg)
                health_status['healthy'] = False
                
                if logger:
                    logger.warning(warning_msg)
                else:
                    warnings.warn(warning_msg)
                    print(f"🚨 {warning_msg}")
            
            # 检查异常梯度 (NaN或Inf)
            elif not (grad_norm == grad_norm) or grad_norm == float('inf'):  # NaN检查
                warning_msg = f"💥 异常梯度告警: {layer_name} 层梯度为 {grad_norm} (NaN/Inf)"
                health_status['warnings'].append(warning_msg)
                health_status['healthy'] = False
                
                if logger:
                    logger.error(warning_msg)
                else:
                    warnings.warn(warning_msg)
                    print(f"🚨 {warning_msg}")
        
        # 如果梯度健康，输出状态信息
        if health_status['healthy'] and logger:
            grad_summary = ", ".join([f"{name}: {norm:.2e}" for name, norm in self.gradient_norms.items()])
            logger.debug(f"✅ 梯度健康状态良好 - {grad_summary}")
        
        return health_status
    
    def apply_gradient_clipping(self, max_norm=1.0):
        """应用梯度裁剪防止梯度爆炸"""
        total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
        return total_norm
    
    def handle_gradient_issues(self, logger=None, auto_fix=True):
        """
        自动处理梯度问题
        
        Args:
            logger: 日志记录器
            auto_fix: 是否自动修复梯度问题
            
        Returns:
            dict: 处理结果
        """
        health_status = self.check_gradient_health(logger)
        
        if not health_status['healthy'] and auto_fix:
            fix_actions = []
            
            # 检查是否有梯度爆炸
            has_exploding = any('爆炸' in warning for warning in health_status['warnings'])
            if has_exploding:
                # 应用梯度裁剪
                clipped_norm = self.apply_gradient_clipping(max_norm=1.0)
                fix_msg = f"🔧 已应用梯度裁剪: 原始范数 {clipped_norm:.2f} → 裁剪到 1.0"
                fix_actions.append(fix_msg)
                
                if logger:
                    logger.info(fix_msg)
                else:
                    print(fix_msg)
            
            # 检查是否有梯度消失
            has_vanishing = any('消失' in warning for warning in health_status['warnings'])
            if has_vanishing:
                # 建议调整学习率
                fix_msg = "💡 建议操作: 检测到梯度消失，考虑提高学习率或使用渐进式解冻"
                fix_actions.append(fix_msg)
                
                if logger:
                    logger.info(fix_msg)
                else:
                    print(fix_msg)
            
            # 检查是否有异常梯度
            has_nan = any('异常' in warning for warning in health_status['warnings'])
            if has_nan:
                # 零化异常梯度
                for param in self.parameters():
                    if param.grad is not None:
                        param.grad[param.grad != param.grad] = 0  # 将NaN设为0
                        param.grad[param.grad == float('inf')] = 0  # 将Inf设为0
                        param.grad[param.grad == float('-inf')] = 0  # 将-Inf设为0
                
                fix_msg = "🔧 已清理异常梯度 (NaN/Inf → 0)"
                fix_actions.append(fix_msg)
                
                if logger:
                    logger.warning(fix_msg)
                else:
                    print(fix_msg)
            
            health_status['fix_actions'] = fix_actions
        
        return health_status
    
    def forward(self, optical_flow_image: torch.Tensor, 
                landmark_features: torch.Tensor) -> torch.Tensor:
        """
        完整前向传播
        
        Args:
            optical_flow_image: 光流图像，形状为(batch_size, 3, 112, 112)
            landmark_features: 关键点特征，形状为(batch_size, 104)
            
        Returns:
            最终预测结果，形状为(batch_size, 2)
        """
        # 光流特征提取
        optical_features = self.optical_flow_model(optical_flow_image)
        
        # 关键点特征处理
        landmark_output = self.landmark_model(landmark_features)
        
        # 特征融合
        final_output = self.fusion_model(optical_features, landmark_output)
        
        return final_output
    
    def freeze_optical_flow_model(self):
        """冻结光流模型"""
        for param in self.optical_flow_model.parameters():
            param.requires_grad = False
    
    def freeze_landmark_model(self):
        """冻结关键点模型"""
        for param in self.landmark_model.parameters():
            param.requires_grad = False
    
    def freeze_landmark_fc_layers(self):
        """只冻结关键点模型的全连接层，用于TFLite特征提取器微调阶段"""
        # 关键点模型的全连接网络部分被冻结
        # TFLite部分不需要冻结因为它不参与梯度计算
        if hasattr(self.landmark_model, 'freeze_all'):
            self.landmark_model.freeze_all()
    
    def unfreeze_landmark_fc_layers(self):
        """解冻关键点模型的全连接层"""
        if hasattr(self.landmark_model, 'unfreeze_all'):
            self.landmark_model.unfreeze_all()
    
    def get_landmark_trainable_params(self):
        """获取关键点模型的可训练参数数量"""
        if hasattr(self.landmark_model, 'get_trainable_parameters'):
            return self.landmark_model.get_trainable_parameters()
        return sum(p.numel() for p in self.landmark_model.parameters() if p.requires_grad)
    
    def freeze_fusion_model(self):
        """冻结融合模型"""
        for param in self.fusion_model.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        """解冻所有模型"""
        for param in self.parameters():
            param.requires_grad = True
    
    def progressive_unfreeze(self, stage='early'):
        """
        渐进式解冻策略，缓解梯度消失问题
        
        Args:
            stage: 'early' - 只解冻后几层
                  'middle' - 解冻中间层
                  'full' - 全部解冻
        """
        if stage == 'early':
            # 只解冻分类器和融合模型
            for param in self.optical_flow_model.classifier.parameters():
                param.requires_grad = True
            for param in self.landmark_model.parameters():
                param.requires_grad = True
            for param in self.fusion_model.parameters():
                param.requires_grad = True
                
        elif stage == 'middle':
            # 解冻ResNet的后几层
            if hasattr(self.optical_flow_model, 'backbone_net'):
                # 解冻ResNet18的layer4和layer3
                for layer_name in ['layer4', 'layer3']:
                    if hasattr(self.optical_flow_model.backbone_net, layer_name):
                        layer = getattr(self.optical_flow_model.backbone_net, layer_name)
                        for param in layer.parameters():
                            param.requires_grad = True
        
        elif stage == 'full':
            self.unfreeze_all()
    
    def get_learning_rate_groups(self):
        """
        获取不同学习率的参数组，用于缓解梯度消失
        
        Returns:
            参数组列表，每组包含参数和建议学习率
        """
        groups = []
        
        # 融合模型 - 最高学习率
        fusion_params = list(self.fusion_model.parameters())
        if fusion_params:
            groups.append({
                'params': fusion_params,
                'lr_multiplier': 1.0,
                'name': 'fusion'
            })
        
        # 关键点模型 - 中等学习率
        landmark_params = list(self.landmark_model.parameters())
        if landmark_params:
            groups.append({
                'params': landmark_params,
                'lr_multiplier': 0.5,
                'name': 'landmark'
            })
        
        # 光流分类器 - 中等学习率
        classifier_params = list(self.optical_flow_model.classifier.parameters())
        if classifier_params:
            groups.append({
                'params': classifier_params,
                'lr_multiplier': 0.5,
                'name': 'optical_classifier'
            })
        
        # ResNet backbone - 最低学习率
        if hasattr(self.optical_flow_model, 'backbone_net'):
            backbone_params = list(self.optical_flow_model.backbone_net.parameters())
            if backbone_params:
                groups.append({
                    'params': backbone_params,
                    'lr_multiplier': 0.1,
                    'name': 'optical_backbone'
                })
        
        return groups


def create_fusion_model(config: Dict[str, Any], advanced: bool = False) -> nn.Module:
    """
    创建融合模型
    
    Args:
        config: 模型配置
        advanced: 是否使用高级融合模型
        
    Returns:
        融合模型实例
    """
    if advanced:
        return AdvancedFusionModel(config)
    else:
        return FusionModel(config)
