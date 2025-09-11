"""
主训练器类
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json
import numpy as np
from typing import Dict, Any, Optional
import logging
from tqdm import tqdm

from models.optical_flow_model import create_optical_flow_model
from models.landmark_model import create_landmark_model
from models.fusion_model import create_fusion_model, CompleteFusionModel
from utils.metrics import MetricsCalculator, calculate_loss
from training.utils import (
    setup_device, create_optimizer, create_scheduler, apply_warmup,
    EarlyStopping, ModelCheckpoint, Timer, print_model_info
)


class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理numpy类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class FacialExpressionTrainer:
    """面部表情评估训练器"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        初始化训练器
        
        Args:
            config: 配置对象
            logger: 日志记录器
        """
        self.config = config
        self.logger = logger
        
        # 设置设备
        self.device = setup_device(config.device_config)
        
        # 设置随机种子
        self._set_seed(config.get('seed', 42))
        
        # 初始化模型
        self._init_models()
        
        # 初始化训练组件
        self._init_training_components()
        
        # 初始化指标计算器
        self.metrics_calculator = MetricsCalculator(
            config.get('evaluation', {}).get('score_ranges', {
                'dynamics': [0, 5],
                'synkinesis': [0, 3]
            })
        )
        
        # 训练历史记录
        self.train_history = []
        self.val_history = []
        self.stage_history = []  # 记录每个阶段的epoch范围
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        
        # 确保结果可复现
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _init_models(self):
        """初始化模型"""
        model_config = self.config.model_config
        
        # 创建子模型
        self.optical_flow_model = create_optical_flow_model(model_config['optical_flow'])
        self.landmark_model = create_landmark_model(model_config['landmark'])
        self.fusion_model = create_fusion_model(model_config['fusion'])
        
        # 创建完整模型
        self.complete_model = CompleteFusionModel(
            self.optical_flow_model,
            self.landmark_model,
            self.fusion_model
        )
        
        # 移动到设备
        self.complete_model.to(self.device)
        
        # 打印模型信息
        if self.logger:
            print_model_info(self.complete_model, self.logger)
        else:
            print_model_info(self.complete_model)
    
    def _init_training_components(self):
        """初始化训练组件"""
        training_config = self.config.training_config
        
        # 早停
        early_stopping_config = training_config.get('early_stopping', {})
        self.early_stopping = EarlyStopping(**early_stopping_config)
        
        # 梯度监控配置
        self.gradient_monitoring = training_config.get('gradient_monitoring', {})
        self.gradient_monitoring_enabled = self.gradient_monitoring.get('enabled', True)
        self.gradient_check_frequency = self.gradient_monitoring.get('check_frequency', 10)
        self.gradient_auto_fix = self.gradient_monitoring.get('auto_fix', True)
        
        # 检查点
        checkpoint_config = training_config.get('checkpoint', {})
        save_dir = self.config.get_experiment_dir()
        
        # 过滤掉ModelCheckpoint不支持的参数
        filtered_checkpoint_config = {
            k: v for k, v in checkpoint_config.items() 
            if k not in ['load_best_after_training']
        }
        
        self.checkpoint = ModelCheckpoint(save_dir, **filtered_checkpoint_config)
        
        # 损失函数配置
        self.loss_config = training_config.get('loss', {})
    
    def train_stage(self, stage_name: str, train_loader: DataLoader, 
                   val_loader: DataLoader, stage_config: Dict[str, Any]):
        """
        训练单个阶段
        
        Args:
            stage_name: 阶段名称
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            stage_config: 阶段配置
        """
        if self.logger:
            self.logger.info(f"开始训练阶段: {stage_name}")
        else:
            print(f"开始训练阶段: {stage_name}")
        
        # 设置当前训练阶段
        self.current_stage = stage_name
        
        # 记录阶段开始
        stage_start_epoch = len(self.train_history)
        
        # 配置模型参数冻结
        self._configure_model_freezing(stage_name)
        
        # 创建优化器和调度器
        optimizer_config = self.config.training_config.get('optimizer', {})
        optimizer_config.update(stage_config)
        optimizer = create_optimizer(self.complete_model, optimizer_config)
        
        scheduler = create_scheduler(optimizer, stage_config)
        
        # 训练参数
        epochs = stage_config.get('epochs', 50)
        warmup_epochs = stage_config.get('warmup_epochs', 0)
        base_lr = stage_config.get('learning_rate', 0.001)
        
        # 训练循环
        for epoch in range(epochs):
            # 预热学习率
            if warmup_epochs > 0:
                apply_warmup(optimizer, epoch, warmup_epochs, base_lr)
            
            # 训练一个epoch
            train_metrics = self._train_epoch(train_loader, optimizer, epoch)
            
            # 验证
            val_metrics = self._validate_epoch(val_loader, epoch)
            
            # 更新学习率
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['val_loss'])
                else:
                    scheduler.step()
            
            # 记录历史
            self.train_history.append(train_metrics)
            self.val_history.append(val_metrics)
            
            # 日志记录
            self._log_epoch_results(stage_name, epoch, train_metrics, val_metrics)
            
            # 保存检查点
            all_metrics = {**train_metrics, **val_metrics}
            self.checkpoint.save_checkpoint(
                self.complete_model, optimizer, scheduler, epoch, all_metrics
            )
            
            # 早停检查（传递当前epoch）
            if self.early_stopping(val_metrics['val_loss'], epoch):
                if self.logger:
                    self.logger.info(f"早停触发，停止训练")
                else:
                    print("早停触发，停止训练")
                break
        
        # 记录阶段结束
        stage_end_epoch = len(self.train_history) - 1
        self.stage_history.append({
            'stage_name': stage_name,
            'start_epoch': stage_start_epoch,
            'end_epoch': stage_end_epoch,
            'num_epochs': stage_end_epoch - stage_start_epoch + 1
        })
        
        # 每个阶段完成后加载最佳模型
        self._load_best_model_after_stage(stage_name)
    
    def _configure_model_freezing(self, stage_name: str):
        """配置模型参数冻结"""
        # 先解冻所有参数
        self.complete_model.unfreeze_all()
        
        if stage_name == 'optical_flow_pretrain':
            # 只训练光流模型
            self.complete_model.freeze_landmark_model()
            self.complete_model.freeze_fusion_model()
        elif stage_name == 'landmark_fc_train':
            # 只训练关键点模型的全连接层（第一个关键点训练阶段）
            self.complete_model.freeze_optical_flow_model()
            self.complete_model.freeze_fusion_model()
            # 关键点模型保持解冻状态，因为这个阶段要训练FC层
        elif stage_name == 'landmark_finetune':
            # 微调整个关键点路径（第二个关键点训练阶段）
            self.complete_model.freeze_optical_flow_model()
            self.complete_model.freeze_fusion_model()
            # 这个阶段如果需要可以添加更精细的控制
        elif stage_name == 'fusion_train':
            # 只训练融合模型
            self.complete_model.freeze_optical_flow_model()
            self.complete_model.freeze_landmark_model()
        # end_to_end阶段不冻结任何参数
    
    def _train_epoch(self, train_loader: DataLoader, 
                    optimizer: torch.optim.Optimizer, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.complete_model.train()
        self.metrics_calculator.reset()
        
        total_loss = 0.0
        num_batches = len(train_loader)
        
        timer = Timer()
        timer.start()
        
        with tqdm(train_loader, desc=f"Epoch {epoch} Train") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # 数据移动到设备
                optical_flow = batch['optical_flow'].to(self.device)
                landmark_features = batch['landmark_features'].to(self.device)
                target = batch['target'].to(self.device)
                
                # 前向传播 - 根据当前训练阶段选择不同的输出路径
                optimizer.zero_grad()
                
                if hasattr(self, 'current_stage'):
                    if self.current_stage == 'optical_flow_pretrain':
                        # 光流预训练阶段：直接使用光流模型输出
                        pred = self.complete_model.optical_flow_model(optical_flow)
                    elif self.current_stage in ['landmark_fc_train', 'landmark_finetune']:
                        # 关键点训练阶段：直接使用关键点模型输出
                        pred = self.complete_model.landmark_model(landmark_features)
                    else:
                        # 融合训练和端到端阶段：使用完整模型
                        pred = self.complete_model(optical_flow, landmark_features)
                else:
                    # 默认使用完整模型
                    pred = self.complete_model(optical_flow, landmark_features)
                
                # 计算损失
                loss = calculate_loss(
                    pred, target,
                    self.loss_config.get('type', 'mse'),
                    self.loss_config.get('weights', [1.0, 1.0])
                )
                
                # 反向传播
                loss.backward()
                
                # 梯度健康检查和自动修复
                if (self.gradient_monitoring_enabled and 
                    batch_idx % self.gradient_check_frequency == 0 and 
                    hasattr(self.complete_model, 'handle_gradient_issues')):
                    
                    gradient_status = self.complete_model.handle_gradient_issues(
                        logger=self.logger, 
                        auto_fix=self.gradient_auto_fix
                    )
                    
                    # 如果梯度不健康，记录详细信息
                    if not gradient_status['healthy']:
                        if self.logger:
                            self.logger.warning(f"批次 {batch_idx}: 检测到梯度异常")
                            for warning in gradient_status['warnings']:
                                self.logger.warning(warning)
                            if 'fix_actions' in gradient_status:
                                for action in gradient_status['fix_actions']:
                                    self.logger.info(action)
                        
                        # 如果有严重的梯度问题，考虑跳过这个batch
                        has_nan = any('异常' in warning for warning in gradient_status['warnings'])
                        if has_nan and self.logger:
                            self.logger.error("检测到NaN/Inf梯度，建议检查数据和模型")
                
                optimizer.step()
                
                # 累积损失
                total_loss += loss.item()
                
                # 更新指标
                self.metrics_calculator.update(pred, target)
                
                # 更新进度条
                pbar.set_postfix({'loss': loss.item()})
        
        timer.stop()
        
        # 计算指标（使用配置中的容忍度）
        tolerance = self.config.get('validation', {}).get('tolerance', 0.5)
        metrics = self.metrics_calculator.compute(tolerance=tolerance)
        metrics['train_loss'] = total_loss / num_batches
        metrics['train_time'] = timer.elapsed()
        
        return metrics
    
    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """验证一个epoch"""
        self.complete_model.eval()
        self.metrics_calculator.reset()
        
        total_loss = 0.0
        num_batches = len(val_loader)
        
        timer = Timer()
        timer.start()
        
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch} Val") as pbar:
                for batch in pbar:
                    # 数据移动到设备
                    optical_flow = batch['optical_flow'].to(self.device)
                    landmark_features = batch['landmark_features'].to(self.device)
                    target = batch['target'].to(self.device)
                    
                    # 前向传播 - 根据当前训练阶段选择不同的输出路径
                    if hasattr(self, 'current_stage'):
                        if self.current_stage == 'optical_flow_pretrain':
                            # 光流预训练阶段：直接使用光流模型输出
                            pred = self.complete_model.optical_flow_model(optical_flow)
                        elif self.current_stage in ['landmark_fc_train', 'landmark_finetune']:
                            # 关键点训练阶段：直接使用关键点模型输出
                            pred = self.complete_model.landmark_model(landmark_features)
                        else:
                            # 融合训练和端到端阶段：使用完整模型
                            pred = self.complete_model(optical_flow, landmark_features)
                    else:
                        # 默认使用完整模型
                        pred = self.complete_model(optical_flow, landmark_features)
                    
                    # 计算损失
                    loss = calculate_loss(
                        pred, target,
                        self.loss_config.get('type', 'mse'),
                        self.loss_config.get('weights', [1.0, 1.0])
                    )
                    
                    # 累积损失
                    total_loss += loss.item()
                    
                    # 更新指标
                    self.metrics_calculator.update(pred, target)
                    
                    # 更新进度条
                    pbar.set_postfix({'loss': loss.item()})
        
        timer.stop()
        
        # 计算指标（使用配置中的容忍度）
        tolerance = self.config.get('validation', {}).get('tolerance', 0.5)
        metrics = self.metrics_calculator.compute(tolerance=tolerance)
        metrics['val_loss'] = total_loss / num_batches
        metrics['val_time'] = timer.elapsed()
        
        return metrics
    
    def _log_epoch_results(self, stage_name: str, epoch: int, 
                          train_metrics: Dict[str, float], 
                          val_metrics: Dict[str, float]):
        """记录epoch结果"""
        log_msg = (
            f"Stage: {stage_name} | Epoch: {epoch} | "
            f"Train Loss: {train_metrics['train_loss']:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Train Acc: {train_metrics.get('overall_accuracy', 0):.4f} | "
            f"Val Acc: {val_metrics.get('overall_accuracy', 0):.4f}"
        )
        
        if self.logger:
            self.logger.info(log_msg)
        else:
            print(log_msg)
    
    def train_all_stages(self, train_loader: DataLoader, val_loader: DataLoader):
        """训练所有阶段"""
        stages_config = self.config.training_config.get('stages', {})
        
        stage_order = [
            'optical_flow_pretrain',
            'landmark_fc_train',      # 新增：关键点FC层训练
            'landmark_finetune',      # 新增：关键点模型微调
            'fusion_train',
            'end_to_end'
        ]
        
        for stage_name in stage_order:
            if stage_name in stages_config:
                stage_config = stages_config[stage_name]
                
                if self.logger:
                    self.logger.info(f"=" * 60)
                    self.logger.info(f"开始训练阶段: {stage_name}")
                    self.logger.info(f"=" * 60)
                else:
                    print(f"=" * 60)
                    print(f"开始训练阶段: {stage_name}")
                    print(f"=" * 60)
                
                # 为每个阶段重置检查点管理器，确保每个阶段都有独立的最佳模型跟踪
                self._reset_checkpoint_for_stage(stage_name, stage_config)
                
                # 重置早停机制，使用阶段特定的best_model_start_epoch
                self._reset_early_stopping_for_stage(stage_name, stage_config)
                
                # 训练该阶段
                self.train_stage(stage_name, train_loader, val_loader, stage_config)
        
        # 训练完成后，加载最佳模型
        self._load_best_model_after_training()
    
    def _load_best_model_after_training(self):
        """训练完成后加载最佳模型"""
        # 检查配置是否启用此功能
        checkpoint_config = self.config.training_config.get('checkpoint', {})
        if not checkpoint_config.get('load_best_after_training', True):
            return
            
        try:
            best_checkpoint_path = os.path.join(self.config.get_experiment_dir(), 'best_checkpoint.pth')
            
            if os.path.exists(best_checkpoint_path):
                if self.logger:
                    self.logger.info("训练完成，正在加载最佳模型...")
                else:
                    print("训练完成，正在加载最佳模型...")
                
                # 加载最佳模型状态
                checkpoint = torch.load(best_checkpoint_path, map_location=self.device, weights_only=False)
                self.complete_model.load_state_dict(checkpoint['model_state_dict'])
                
                best_epoch = checkpoint.get('epoch', 0)
                best_metrics = checkpoint.get('metrics', {})
                best_val_loss = best_metrics.get('val_loss', 'N/A')
                
                if self.logger:
                    self.logger.info(f"✅ 已加载最佳模型: epoch {best_epoch}, val_loss = {best_val_loss}")
                else:
                    print(f"✅ 已加载最佳模型: epoch {best_epoch}, val_loss = {best_val_loss}")
                    
            else:
                if self.logger:
                    self.logger.warning("未找到最佳模型检查点，保持当前模型状态")
                else:
                    print("⚠️  警告: 未找到最佳模型检查点，保持当前模型状态")
                    
        except Exception as e:
            if self.logger:
                self.logger.error(f"加载最佳模型失败: {e}")
            else:
                print(f"❌ 错误: 加载最佳模型失败: {e}")
    
    def _load_best_model_after_stage(self, stage_name: str):
        """每个训练阶段完成后加载最佳模型"""
        # 检查配置是否启用此功能
        checkpoint_config = self.config.training_config.get('checkpoint', {})
        if not checkpoint_config.get('load_best_after_training', True):
            return
            
        try:
            best_checkpoint_path = os.path.join(self.config.get_experiment_dir(), 'best_checkpoint.pth')
            
            if os.path.exists(best_checkpoint_path):
                if self.logger:
                    self.logger.info(f"阶段 {stage_name} 完成，正在加载该阶段最佳模型...")
                else:
                    print(f"阶段 {stage_name} 完成，正在加载该阶段最佳模型...")
                
                # 加载最佳模型状态
                checkpoint = torch.load(best_checkpoint_path, map_location=self.device, weights_only=False)
                self.complete_model.load_state_dict(checkpoint['model_state_dict'])
                
                best_epoch = checkpoint.get('epoch', 0)
                best_metrics = checkpoint.get('metrics', {})
                best_val_loss = best_metrics.get('val_loss', 'N/A')
                best_val_acc = best_metrics.get('overall_accuracy', 'N/A')
                
                if self.logger:
                    self.logger.info(f"✅ 已加载阶段 {stage_name} 最佳模型: epoch {best_epoch}")
                    self.logger.info(f"   最佳验证损失: {best_val_loss}, 最佳验证准确率: {best_val_acc}")
                else:
                    print(f"✅ 已加载阶段 {stage_name} 最佳模型: epoch {best_epoch}")
                    print(f"   最佳验证损失: {best_val_loss}, 最佳验证准确率: {best_val_acc}")
                    
            else:
                if self.logger:
                    self.logger.warning(f"阶段 {stage_name}: 未找到最佳模型检查点，保持当前模型状态")
                else:
                    print(f"⚠️  阶段 {stage_name}: 未找到最佳模型检查点，保持当前模型状态")
                    
        except Exception as e:
            if self.logger:
                self.logger.error(f"阶段 {stage_name} 加载最佳模型失败: {e}")
            else:
                print(f"❌ 阶段 {stage_name} 加载最佳模型失败: {e}")
    
    def _reset_checkpoint_for_stage(self, stage_name: str, stage_config: Dict[str, Any]):
        """为新阶段重置检查点管理器"""
        checkpoint_config = self.config.training_config.get('checkpoint', {}).copy()
        save_dir = self.config.get_experiment_dir()
        
        # 使用阶段特定的best_model_start_epoch，如果没有则使用全局配置
        stage_best_model_start_epoch = stage_config.get('best_model_start_epoch')
        if stage_best_model_start_epoch is not None:
            checkpoint_config['best_model_start_epoch'] = stage_best_model_start_epoch
        
        # 过滤掉ModelCheckpoint不支持的参数
        filtered_checkpoint_config = {
            k: v for k, v in checkpoint_config.items() 
            if k not in ['load_best_after_training']
        }
        
        # 重置检查点管理器，清除之前阶段的最佳记录
        self.checkpoint = ModelCheckpoint(save_dir, **filtered_checkpoint_config)
        
        best_start_epoch = filtered_checkpoint_config.get('best_model_start_epoch', 0)
        if self.logger:
            self.logger.info(f"已为阶段 {stage_name} 重置检查点管理器 (从第 {best_start_epoch} 个epoch开始保存最佳模型)")
        else:
            print(f"已为阶段 {stage_name} 重置检查点管理器 (从第 {best_start_epoch} 个epoch开始保存最佳模型)")
    
    def _reset_early_stopping_for_stage(self, stage_name: str, stage_config: Dict[str, Any]):
        """为新阶段重置早停机制"""
        early_stopping_config = self.config.training_config.get('early_stopping', {}).copy()
        
        # 使用阶段特定的best_model_start_epoch作为early stopping的start_epoch
        stage_best_model_start_epoch = stage_config.get('best_model_start_epoch')
        if stage_best_model_start_epoch is not None:
            early_stopping_config['start_epoch'] = stage_best_model_start_epoch
        else:
            # 如果阶段没有特定设置，使用全局检查点配置中的值
            global_best_start = self.config.training_config.get('checkpoint', {}).get('best_model_start_epoch', 0)
            early_stopping_config['start_epoch'] = global_best_start
        
        # 重置早停机制
        self.early_stopping = EarlyStopping(**early_stopping_config)
        
        early_start_epoch = early_stopping_config.get('start_epoch', 0)
        if self.logger:
            self.logger.info(f"已为阶段 {stage_name} 重置早停机制 (从第 {early_start_epoch} 个epoch开始启用早停)")
        else:
            print(f"已为阶段 {stage_name} 重置早停机制 (从第 {early_start_epoch} 个epoch开始启用早停)")
    def save_training_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.config.get_experiment_dir(), 'training_history.json')
        
        history = {
            'train': self.train_history,
            'val': self.val_history
        }
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2, cls=NumpyEncoder)
        
        if self.logger:
            self.logger.info(f"训练历史已保存到: {history_path}")
        else:
            print(f"训练历史已保存到: {history_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        from training.utils import load_checkpoint
        
        checkpoint_info = load_checkpoint(
            self.complete_model, checkpoint_path, device=self.device
        )
        
        if self.logger:
            self.logger.info(f"已加载检查点: {checkpoint_path}")
            self.logger.info(f"检查点信息: {checkpoint_info}")
        else:
            print(f"已加载检查点: {checkpoint_path}")
            print(f"检查点信息: {checkpoint_info}")
        
        return checkpoint_info
