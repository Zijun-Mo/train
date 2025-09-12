"""
评估指标计算工具
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr


class MetricsCalculator:
    """指标计算器"""
    
    def __init__(self, score_ranges: Dict[str, List[float]], tolerance: float = 0.5):
        """
        初始化指标计算器
        
        Args:
            score_ranges: 评分范围，如 {'dynamics': [0, 5], 'synkinesis': [0, 3]}
            tolerance: 准确率计算的容忍度
        """
        self.score_ranges = score_ranges
        self.tolerance = tolerance
        self.reset()
    
    def reset(self):
        """重置所有指标"""
        self.predictions = []
        self.targets = []
        self.dynamics_preds = []
        self.dynamics_targets = []
        self.synkinesis_preds = []
        self.synkinesis_targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        更新指标
        
        Args:
            predictions: 预测值，形状为 (batch_size, 2)
            targets: 真实值，形状为 (batch_size, 2)
        """
        # 转换为numpy数组
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        # 累积所有预测和目标
        self.predictions.extend(pred_np)
        self.targets.extend(target_np)
        
        # 分别存储两个表情维度
        self.dynamics_preds.extend(pred_np[:, 0])
        self.dynamics_targets.extend(target_np[:, 0])
        self.synkinesis_preds.extend(pred_np[:, 1])
        self.synkinesis_targets.extend(target_np[:, 1])
    
    def compute_accuracy(self, preds: np.ndarray, targets: np.ndarray) -> float:
        """
        计算准确率（在容忍度范围内的预测被认为是正确的）
        
        Args:
            preds: 预测值
            targets: 真实值
            
        Returns:
            准确率
        """
        diff = np.abs(preds - targets)
        correct = diff <= self.tolerance
        return np.mean(correct)
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        计算所有指标
        
        Returns:
            包含所有指标的字典
        """
        if not self.predictions:
            return {}
        
        # 转换为numpy数组
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        dynamics_preds = np.array(self.dynamics_preds)
        dynamics_targets = np.array(self.dynamics_targets)
        synkinesis_preds = np.array(self.synkinesis_preds)
        synkinesis_targets = np.array(self.synkinesis_targets)
        
        metrics = {}
        
        # 整体指标
        metrics['mse'] = mean_squared_error(targets, preds)
        metrics['mae'] = mean_absolute_error(targets, preds)
        
        # 样本级准确率计算（方式二：只有两个维度都正确才算正确）
        dynamics_correct = np.abs(dynamics_preds - dynamics_targets) <= self.tolerance
        synkinesis_correct = np.abs(synkinesis_preds - synkinesis_targets) <= self.tolerance
        sample_correct = dynamics_correct & synkinesis_correct  # 两个维度都必须正确
        metrics['overall_accuracy'] = np.mean(sample_correct)
        
        # Dynamics 指标
        metrics['dynamics_mse'] = mean_squared_error(dynamics_targets, dynamics_preds)
        metrics['dynamics_mae'] = mean_absolute_error(dynamics_targets, dynamics_preds)
        metrics['dynamics_accuracy'] = self.compute_accuracy(dynamics_preds, dynamics_targets)
        metrics['dynamics_std'] = np.std(dynamics_preds - dynamics_targets)
        
        # Synkinesis 指标
        metrics['synkinesis_mse'] = mean_squared_error(synkinesis_targets, synkinesis_preds)
        metrics['synkinesis_mae'] = mean_absolute_error(synkinesis_targets, synkinesis_preds)
        metrics['synkinesis_accuracy'] = self.compute_accuracy(synkinesis_preds, synkinesis_targets)
        metrics['synkinesis_std'] = np.std(synkinesis_preds - synkinesis_targets)
        
        # 相关系数（如果数据足够）
        if len(dynamics_preds) > 1:
            try:
                dynamics_pearson, _ = pearsonr(dynamics_targets, dynamics_preds)
                metrics['dynamics_pearson'] = dynamics_pearson if not np.isnan(dynamics_pearson) else 0.0
                
                dynamics_spearman, _ = spearmanr(dynamics_targets, dynamics_preds)
                metrics['dynamics_spearman'] = dynamics_spearman if not np.isnan(dynamics_spearman) else 0.0
            except:
                metrics['dynamics_pearson'] = 0.0
                metrics['dynamics_spearman'] = 0.0
        
        if len(synkinesis_preds) > 1:
            try:
                synkinesis_pearson, _ = pearsonr(synkinesis_targets, synkinesis_preds)
                metrics['synkinesis_pearson'] = synkinesis_pearson if not np.isnan(synkinesis_pearson) else 0.0
                
                synkinesis_spearman, _ = spearmanr(synkinesis_targets, synkinesis_preds)
                metrics['synkinesis_spearman'] = synkinesis_spearman if not np.isnan(synkinesis_spearman) else 0.0
            except:
                metrics['synkinesis_pearson'] = 0.0
                metrics['synkinesis_spearman'] = 0.0
        
        return metrics
    
    def compute(self, tolerance=None) -> Dict[str, float]:
        """
        计算所有指标（为了向后兼容训练器调用）
        
        Args:
            tolerance: 可选的容忍度覆盖值
            
        Returns:
            包含所有指标的字典
        """
        # 如果提供了新的容忍度，临时更新
        original_tolerance = self.tolerance
        if tolerance is not None:
            self.tolerance = tolerance
        
        try:
            metrics = self.compute_metrics()
            return metrics
        finally:
            # 恢复原始容忍度
            self.tolerance = original_tolerance


def calculate_loss(predictions: torch.Tensor, 
                  targets: torch.Tensor, 
                  loss_type: str = 'mse',
                  weights: List[float] = [1.0, 1.0]) -> torch.Tensor:
    """
    计算损失函数
    
    Args:
        predictions: 预测值，形状为 (batch_size, 2)
        targets: 真实值，形状为 (batch_size, 2)
        loss_type: 损失函数类型 ('mse', 'mae', 'huber', 'smooth_l1')
        weights: 每个输出维度的权重
        
    Returns:
        损失值
    """
    weights = torch.tensor(weights, device=predictions.device, dtype=predictions.dtype)
    
    if loss_type == 'mse':
        loss = F.mse_loss(predictions, targets, reduction='none')
    elif loss_type == 'mae':
        loss = F.l1_loss(predictions, targets, reduction='none')
    elif loss_type == 'huber':
        loss = F.smooth_l1_loss(predictions, targets, reduction='none', beta=1.0)
    elif loss_type == 'smooth_l1':
        loss = F.smooth_l1_loss(predictions, targets, reduction='none')
    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}")
    
    # 应用权重
    weighted_loss = loss * weights
    
    # 返回平均损失
    return weighted_loss.mean()


def evaluate_model_on_dataset(model, data_loader, device, metrics_calculator, loss_config):
    """
    在整个数据集上评估模型
    
    Args:
        model: 要评估的模型
        data_loader: 数据加载器
        device: 设备
        metrics_calculator: 指标计算器
        loss_config: 损失配置
        
    Returns:
        评估指标字典
    """
    model.eval()
    metrics_calculator.reset()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in data_loader:
            # 数据移动到设备
            optical_flow = batch['optical_flow'].to(device)
            landmark_features = batch['landmark_features'].to(device)
            target = batch['target'].to(device)
            
            # 前向传播
            if hasattr(model, 'optical_flow_model') and hasattr(model, 'landmark_model'):
                # 融合模型
                pred = model(optical_flow, landmark_features)
            else:
                # 单模型
                if optical_flow.shape[1] == 3:  # 光流图像
                    pred = model(optical_flow)
                else:  # 关键点特征
                    pred = model(landmark_features)
            
            # 计算损失
            loss = calculate_loss(
                pred, target,
                loss_config.get('type', 'mse'),
                loss_config.get('weights', [1.0, 1.0])
            )
            
            total_loss += loss.item()
            num_batches += 1
            
            # 更新指标
            metrics_calculator.update(pred, target)
    
    # 计算平均损失
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # 计算所有指标
    metrics = metrics_calculator.compute_metrics()
    metrics['loss'] = avg_loss
    
    return metrics


def print_metrics_summary(metrics: Dict[str, float], title: str = "Metrics Summary"):
    """
    打印指标摘要
    
    Args:
        metrics: 指标字典
        title: 标题
    """
    print(f"\n{title}")
    print("=" * len(title))
    
    if 'loss' in metrics:
        print(f"Loss: {metrics['loss']:.4f}")
    
    if 'mse' in metrics:
        print(f"MSE: {metrics['mse']:.4f}")
    
    if 'mae' in metrics:
        print(f"MAE: {metrics['mae']:.4f}")
    
    if 'overall_accuracy' in metrics:
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    
    print("\nDynamics:")
    if 'dynamics_mse' in metrics:
        print(f"  MSE: {metrics['dynamics_mse']:.4f}")
    if 'dynamics_mae' in metrics:
        print(f"  MAE: {metrics['dynamics_mae']:.4f}")
    if 'dynamics_accuracy' in metrics:
        print(f"  Accuracy: {metrics['dynamics_accuracy']:.4f}")
    if 'dynamics_std' in metrics:
        print(f"  Std: {metrics['dynamics_std']:.4f}")
    if 'dynamics_pearson' in metrics:
        print(f"  Pearson: {metrics['dynamics_pearson']:.4f}")
    
    print("\nSynkinesis:")
    if 'synkinesis_mse' in metrics:
        print(f"  MSE: {metrics['synkinesis_mse']:.4f}")
    if 'synkinesis_mae' in metrics:
        print(f"  MAE: {metrics['synkinesis_mae']:.4f}")
    if 'synkinesis_accuracy' in metrics:
        print(f"  Accuracy: {metrics['synkinesis_accuracy']:.4f}")
    if 'synkinesis_std' in metrics:
        print(f"  Std: {metrics['synkinesis_std']:.4f}")
    if 'synkinesis_pearson' in metrics:
        print(f"  Pearson: {metrics['synkinesis_pearson']:.4f}")


def test_metrics():
    """测试指标计算"""
    print("测试指标计算...")
    
    # 创建指标计算器
    calculator = MetricsCalculator(
        score_ranges={'dynamics': [0, 5], 'synkinesis': [0, 3]},
        tolerance=0.5
    )
    
    # 模拟一些预测和目标数据，用于演示样本级准确率
    print("\n演示样本级准确率计算:")
    print("=" * 50)
    
    # 手动创建一些样本来展示效果
    import torch
    
    # 样本1: dynamics正确, synkinesis正确 -> 整体正确
    preds1 = torch.tensor([[2.1, 1.8]], dtype=torch.float32)  # 预测
    targets1 = torch.tensor([[2.0, 2.0]], dtype=torch.float32)  # 真实
    calculator.update(preds1, targets1)
    print("样本1: dynamics误差=0.1✓, synkinesis误差=0.2✓ -> 样本正确")
    
    # 样本2: dynamics正确, synkinesis错误 -> 整体错误
    preds2 = torch.tensor([[3.1, 1.2]], dtype=torch.float32)
    targets2 = torch.tensor([[3.0, 2.0]], dtype=torch.float32)
    calculator.update(preds2, targets2)
    print("样本2: dynamics误差=0.1✓, synkinesis误差=0.8✗ -> 样本错误")
    
    # 样本3: dynamics错误, synkinesis正确 -> 整体错误
    preds3 = torch.tensor([[1.2, 0.9]], dtype=torch.float32)
    targets3 = torch.tensor([[2.0, 1.0]], dtype=torch.float32)
    calculator.update(preds3, targets3)
    print("样本3: dynamics误差=0.8✗, synkinesis误差=0.1✓ -> 样本错误")
    
    # 样本4: dynamics正确, synkinesis正确 -> 整体正确
    preds4 = torch.tensor([[4.3, 1.6]], dtype=torch.float32)
    targets4 = torch.tensor([[4.0, 1.5]], dtype=torch.float32)
    calculator.update(preds4, targets4)
    print("样本4: dynamics误差=0.3✓, synkinesis误差=0.1✓ -> 样本正确")
    
    # 添加更多随机样本
    batch_size = 6
    for i in range(2):  # 2个额外batch
        preds = torch.randn(batch_size, 2) * 0.3 + torch.tensor([2.5, 1.5])  # 中心在合理范围
        targets = torch.randn(batch_size, 2) * 0.3 + torch.tensor([2.5, 1.5])
        calculator.update(preds, targets)
    
    # 计算指标
    metrics = calculator.compute_metrics()
    
    # 打印结果
    print(f"\n预期结果: 4个手动样本中有2个正确，整体准确率应该≤0.5")
    print_metrics_summary(metrics, "Sample-Level Accuracy Test")
    
    print(f"\n关键观察:")
    print(f"- 整体准确率 (样本级): {metrics['overall_accuracy']:.3f}")
    print(f"- Dynamics准确率: {metrics['dynamics_accuracy']:.3f}")  
    print(f"- Synkinesis准确率: {metrics['synkinesis_accuracy']:.3f}")
    print(f"- 注意: 整体准确率 ≤ min(dynamics准确率, synkinesis准确率)")
    print(f"- 这反映了只有两个维度都正确时样本才被认为是正确的")
    
    print("\n测试完成!")


def evaluate_model_on_dataset_partial(model, data_loader, device, metrics_calculator, loss_config, max_batches=10):
    """
    在数据集的部分数据上快速评估模型（用于训练集的简略评估）
    
    Args:
        model: 要评估的模型
        data_loader: 数据加载器
        device: 设备
        metrics_calculator: 指标计算器
        loss_config: 损失配置
        max_batches: 最大评估批次数
        
    Returns:
        包含评估指标的字典
    """
    model.eval()
    metrics_calculator.reset()
    
    total_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
                
            # 移动数据到设备
            optical_flow = batch['optical_flow'].to(device, non_blocking=True)
            landmark_features = batch['landmark_features'].to(device, non_blocking=True)
            target = batch['target'].to(device, non_blocking=True)
            
            # 前向传播
            if hasattr(model, 'optical_flow_model') and hasattr(model, 'landmark_model'):
                # 完整模型
                pred = model(optical_flow, landmark_features)
            elif hasattr(model, 'optical_flow_model'):
                # 只有光流模型
                pred = model(optical_flow)
            else:  # 关键点特征
                pred = model(landmark_features)
            
            # 计算损失
            loss = calculate_loss(
                pred, target,
                loss_config.get('type', 'mse'),
                loss_config.get('weights', [1.0, 1.0])
            )
            
            total_loss += loss.item()
            batch_count += 1
            
            # 更新指标
            metrics_calculator.update(pred, target)
    
    # 计算指标
    metrics = metrics_calculator.compute_metrics()
    
    # 添加平均损失
    if batch_count > 0:
        metrics['loss'] = total_loss / batch_count
    else:
        metrics['loss'] = 0.0
    
    return metrics


if __name__ == "__main__":
    test_metrics()
