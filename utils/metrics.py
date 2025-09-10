"""
评估指标工具
"""
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsCalculator:
    """评估指标计算器"""
    
    def __init__(self, score_ranges: Dict[str, List[int]]):
        """
        初始化指标计算器
        
        Args:
            score_ranges: 评分范围，格式为{'dynamics': [0, 5], 'synkinesis': [0, 3]}
        """
        self.score_ranges = score_ranges
        self.reset()
    
    def reset(self):
        """重置累积的预测和真值"""
        self.predictions = []
        self.targets = []
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        更新预测和真值
        
        Args:
            pred: 预测值，形状为(batch_size, 2)
            target: 真值，形状为(batch_size, 2)
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        self.predictions.append(pred)
        self.targets.append(target)
    
    def compute(self, tolerance: float = 0.5) -> Dict[str, float]:
        """
        计算所有指标
        
        Args:
            tolerance: 准确率计算的容忍度
            
        Returns:
            包含所有指标的字典
        """
        if not self.predictions:
            return {}
        
        # 合并所有预测和真值
        pred_all = np.vstack(self.predictions)
        target_all = np.vstack(self.targets)
        
        metrics = {}
        
        # 整体指标
        metrics['overall_mse'] = mean_squared_error(target_all, pred_all)
        metrics['overall_mae'] = mean_absolute_error(target_all, pred_all)
        
        # 计算Pearson和Spearman相关系数
        try:
            pearson_corr, _ = pearsonr(target_all.flatten(), pred_all.flatten())
            metrics['overall_pearson'] = pearson_corr
        except:
            metrics['overall_pearson'] = 0.0
        
        try:
            spearman_corr, _ = spearmanr(target_all.flatten(), pred_all.flatten())
            metrics['overall_spearman'] = spearman_corr
        except:
            metrics['overall_spearman'] = 0.0
        
        # 分别计算每个评分的指标
        score_names = ['dynamics', 'synkinesis']
        for i, score_name in enumerate(score_names):
            pred_score = pred_all[:, i]
            target_score = target_all[:, i]
            
            # MSE和MAE
            metrics[f'{score_name}_mse'] = mean_squared_error(target_score, pred_score)
            metrics[f'{score_name}_mae'] = mean_absolute_error(target_score, pred_score)
            
            # 准确率（在容忍度范围内的预测比例）
            accuracy = np.mean(np.abs(pred_score - target_score) <= tolerance)
            metrics[f'{score_name}_accuracy'] = accuracy
            
            # Pearson和Spearman相关系数
            try:
                pearson_corr, _ = pearsonr(target_score, pred_score)
                metrics[f'{score_name}_pearson'] = pearson_corr
            except:
                metrics[f'{score_name}_pearson'] = 0.0
            
            try:
                spearman_corr, _ = spearmanr(target_score, pred_score)
                metrics[f'{score_name}_spearman'] = spearman_corr
            except:
                metrics[f'{score_name}_spearman'] = 0.0
        
        # 整体准确率
        overall_accuracy = np.mean(
            np.all(np.abs(pred_all - target_all) <= tolerance, axis=1)
        )
        metrics['overall_accuracy'] = overall_accuracy
        
        return metrics
    
    def plot_predictions(self, save_path: str = None) -> plt.Figure:
        """
        绘制预测vs真值的散点图
        
        Args:
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        if not self.predictions:
            return None
        
        pred_all = np.vstack(self.predictions)
        target_all = np.vstack(self.targets)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        score_names = ['Dynamics', 'Synkinesis']
        score_ranges = [self.score_ranges['dynamics'], self.score_ranges['synkinesis']]
        
        for i, (ax, name, score_range) in enumerate(zip(axes, score_names, score_ranges)):
            pred_score = pred_all[:, i]
            target_score = target_all[:, i]
            
            # 散点图
            ax.scatter(target_score, pred_score, alpha=0.6, s=20)
            
            # 完美预测线
            min_val, max_val = score_range
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
            
            # 设置标签和标题
            ax.set_xlabel(f'True {name} Score')
            ax.set_ylabel(f'Predicted {name} Score')
            ax.set_title(f'{name} Score Prediction')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 设置坐标轴范围
            ax.set_xlim(min_val - 0.5, max_val + 0.5)
            ax.set_ylim(min_val - 0.5, max_val + 0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_confusion_matrix(self, save_path: str = None) -> plt.Figure:
        """
        绘制混淆矩阵（将连续值离散化）
        
        Args:
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        if not self.predictions:
            return None
        
        pred_all = np.vstack(self.predictions)
        target_all = np.vstack(self.targets)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        score_names = ['Dynamics', 'Synkinesis']
        score_ranges = [self.score_ranges['dynamics'], self.score_ranges['synkinesis']]
        
        for i, (ax, name, score_range) in enumerate(zip(axes, score_names, score_ranges)):
            pred_score = pred_all[:, i]
            target_score = target_all[:, i]
            
            # 将连续值四舍五入到最近的整数
            pred_discrete = np.round(np.clip(pred_score, score_range[0], score_range[1])).astype(int)
            target_discrete = np.round(target_score).astype(int)
            
            # 创建混淆矩阵
            labels = list(range(score_range[0], score_range[1] + 1))
            confusion = np.zeros((len(labels), len(labels)))
            
            for true_val, pred_val in zip(target_discrete, pred_discrete):
                if true_val in labels and pred_val in labels:
                    true_idx = labels.index(true_val)
                    pred_idx = labels.index(pred_val)
                    confusion[true_idx, pred_idx] += 1
            
            # 归一化
            confusion = confusion / (confusion.sum(axis=1, keepdims=True) + 1e-8)
            
            # 绘制热力图
            sns.heatmap(confusion, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_xlabel(f'Predicted {name} Score')
            ax.set_ylabel(f'True {name} Score')
            ax.set_title(f'{name} Score Confusion Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def calculate_loss(pred: torch.Tensor, target: torch.Tensor, 
                  loss_type: str = 'mse', weights: List[float] = None) -> torch.Tensor:
    """
    计算损失函数
    
    Args:
        pred: 预测值，形状为(batch_size, 2)
        target: 真值，形状为(batch_size, 2)
        loss_type: 损失函数类型
        weights: 各个输出的权重
        
    Returns:
        损失值
    """
    if weights is None:
        weights = [1.0, 1.0]
    
    weights = torch.tensor(weights, device=pred.device, dtype=pred.dtype)
    
    if loss_type == 'mse':
        loss = torch.nn.functional.mse_loss(pred, target, reduction='none')
    elif loss_type == 'mae':
        loss = torch.nn.functional.l1_loss(pred, target, reduction='none')
    elif loss_type == 'smooth_l1':
        loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction='none')
    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}")
    
    # 应用权重
    weighted_loss = loss * weights.unsqueeze(0)
    
    # 返回平均损失
    return weighted_loss.mean()
