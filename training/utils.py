"""
训练工具函数
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, StepLR, ReduceLROnPlateau, 
    CosineAnnealingWarmRestarts
)
import os
import time
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np


def setup_device(device_config: Dict[str, Any]) -> torch.device:
    """
    设置计算设备
    
    Args:
        device_config: 设备配置
        
    Returns:
        PyTorch设备对象
    """
    device_type = device_config.get('type', 'auto')
    
    if device_type == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用，但配置要求使用CUDA")
        device = torch.device('cuda')
    elif device_type == 'cpu':
        device = torch.device('cpu')
    else:
        raise ValueError(f"不支持的设备类型: {device_type}")
    
    if device.type == 'cuda':
        gpu_ids = device_config.get('gpu_ids', [0])
        if len(gpu_ids) > 1:
            print(f"使用多GPU: {gpu_ids}")
        else:
            print(f"使用GPU: {gpu_ids[0]}")
            torch.cuda.set_device(gpu_ids[0])
    else:
        print("使用CPU")
    
    return device


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    创建优化器
    
    Args:
        model: 模型
        config: 优化器配置
        
    Returns:
        优化器实例
    """
    optimizer_type = config.get('type', 'adam').lower()
    learning_rate = float(config.get('learning_rate', 0.001))
    weight_decay = float(config.get('weight_decay', 0.0001))
    
    if optimizer_type == 'adam':
        betas = config.get('betas', [0.9, 0.999])
        eps = float(config.get('eps', 1e-8))
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adamw':
        betas = config.get('betas', [0.9, 0.999])
        eps = float(config.get('eps', 1e-8))
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'sgd':
        momentum = float(config.get('momentum', 0.9))
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, 
                    config: Dict[str, Any]) -> Optional[object]:
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        config: 调度器配置
        
    Returns:
        调度器实例
    """
    scheduler_type = config.get('scheduler', 'none').lower()
    
    if scheduler_type == 'none' or scheduler_type is None:
        return None
    elif scheduler_type == 'cosine':
        T_max = int(config.get('epochs', 100))
        eta_min = float(config.get('eta_min', 0))
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif scheduler_type == 'step':
        step_size = int(config.get('step_size', 30))
        gamma = float(config.get('gamma', 0.1))
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'plateau':
        factor = float(config.get('factor', 0.1))
        patience = int(config.get('patience', 10))
        return ReduceLROnPlateau(optimizer, factor=factor, patience=patience)
    elif scheduler_type == 'cosine_warm':
        T_0 = int(config.get('T_0', 10))
        T_mult = int(config.get('T_mult', 2))
        return CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult)
    else:
        raise ValueError(f"不支持的调度器类型: {scheduler_type}")


def apply_warmup(optimizer: torch.optim.Optimizer, 
                epoch: int, 
                warmup_epochs: int, 
                base_lr: float):
    """
    应用学习率预热
    
    Args:
        optimizer: 优化器
        epoch: 当前epoch
        warmup_epochs: 预热epoch数
        base_lr: 基础学习率
    """
    if epoch < warmup_epochs:
        warmup_lr = base_lr * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, 
                 monitor: str = 'val_loss', mode: str = 'min'):
        """
        初始化早停机制
        
        Args:
            patience: 容忍epoch数
            min_delta: 最小改进量
            monitor: 监控指标
            mode: 监控模式（'min'或'max'）
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        self.compare = self._get_compare_fn()
    
    def _get_compare_fn(self):
        """获取比较函数"""
        if self.mode == 'min':
            return lambda current, best: current < best - self.min_delta
        else:
            return lambda current, best: current > best + self.min_delta
    
    def __call__(self, current_score: float) -> bool:
        """
        检查是否应该早停
        
        Args:
            current_score: 当前分数
            
        Returns:
            是否应该早停
        """
        if self.best_score is None:
            self.best_score = current_score
        elif self.compare(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class ModelCheckpoint:
    """模型检查点管理"""
    
    def __init__(self, save_dir: str, monitor: str = 'val_loss', 
                 mode: str = 'min', save_best: bool = True, save_last: bool = True):
        """
        初始化检查点管理器
        
        Args:
            save_dir: 保存目录
            monitor: 监控指标
            mode: 监控模式
            save_best: 是否保存最佳模型
            save_last: 是否保存最后一个模型
        """
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best = save_best
        self.save_last = save_last
        
        self.best_score = None
        self.best_epoch = 0
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.compare = self._get_compare_fn()
    
    def _get_compare_fn(self):
        """获取比较函数"""
        if self.mode == 'min':
            return lambda current, best: current < best
        else:
            return lambda current, best: current > best
    
    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler: Optional[object], epoch: int, 
                       metrics: Dict[str, float]):
        """
        保存检查点
        
        Args:
            model: 模型
            optimizer: 优化器
            scheduler: 调度器
            epoch: 当前epoch
            metrics: 指标字典
        """
        current_score = metrics.get(self.monitor)
        
        # 保存最后一个模型
        if self.save_last:
            last_path = os.path.join(self.save_dir, 'last_checkpoint.pth')
            self._save_model(model, optimizer, scheduler, epoch, metrics, last_path)
        
        # 保存最佳模型
        if self.save_best and current_score is not None:
            if self.best_score is None or self.compare(current_score, self.best_score):
                self.best_score = current_score
                self.best_epoch = epoch
                
                best_path = os.path.join(self.save_dir, 'best_checkpoint.pth')
                self._save_model(model, optimizer, scheduler, epoch, metrics, best_path)
                
                print(f"保存最佳模型: epoch {epoch}, {self.monitor} = {current_score:.4f}")
    
    def _save_model(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                   scheduler: Optional[object], epoch: int, 
                   metrics: Dict[str, float], save_path: str):
        """保存模型到指定路径"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, save_path)


def load_checkpoint(model: nn.Module, checkpoint_path: str,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[object] = None,
                   device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    加载检查点
    
    Args:
        model: 模型
        checkpoint_path: 检查点路径
        optimizer: 优化器（可选）
        scheduler: 调度器（可选）
        device: 设备（可选）
        
    Returns:
        检查点信息
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 加载调度器状态
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {})
    }


def setup_logging(log_dir: str, log_level: str = 'INFO') -> logging.Logger:
    """
    设置日志
    
    Args:
        log_dir: 日志目录
        log_level: 日志级别
        
    Returns:
        日志记录器
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志记录器
    logger = logging.getLogger('training')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 文件处理器
    file_handler = logging.FileHandler(
        os.path.join(log_dir, 'training.log'),
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, log_level.upper()))
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    统计模型参数数量
    
    Args:
        model: 模型
        
    Returns:
        总参数数量和可训练参数数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def print_model_info(model: nn.Module, logger: Optional[logging.Logger] = None):
    """
    打印模型信息
    
    Args:
        model: 模型
        logger: 日志记录器
    """
    total_params, trainable_params = count_parameters(model)
    
    info = [
        f"模型总参数数量: {total_params:,}",
        f"可训练参数数量: {trainable_params:,}",
        f"模型结构: {model.__class__.__name__}"
    ]
    
    for line in info:
        if logger:
            logger.info(line)
        else:
            print(line)


class Timer:
    """训练计时器"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
    
    def stop(self):
        """停止计时"""
        self.end_time = time.time()
    
    def elapsed(self) -> float:
        """获取耗时（秒）"""
        if self.start_time is None:
            return 0.0
        
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time
    
    def format_time(self, seconds: float) -> str:
        """格式化时间"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
