# 面部表情动态评估训练系统使用指南

## 1. 快速开始

### 1.1 环境配置

1. 创建虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置数据路径：
编辑`config.yaml`文件中的`data.root_dir`，指向您的数据目录。

### 1.2 开始训练

```bash
python main.py
```

## 2. 配置文件详解

### 2.1 基本配置结构

配置文件`config.yaml`包含以下主要部分：
- `data`: 数据相关配置
- `model`: 模型架构配置
- `training`: 训练策略配置
- `validation`: 验证配置
- `logging`: 日志和保存配置

### 2.2 最佳模型保存配置

#### 2.2.1 全局配置

```yaml
training:
  checkpoint:
    save_best: true                  # 是否保存最佳模型
    save_last: true                  # 是否保存最后一个模型
    monitor: "val_loss"              # 监控指标
    mode: "min"                      # 监控模式 (min/max)
    best_model_start_epoch: 5        # 全局默认：从第5个epoch开始保存最佳模型
    load_best_after_training: true   # 每个阶段完成后是否自动加载最佳模型
```

#### 2.2.2 阶段特定配置

每个训练阶段可以单独设置`best_model_start_epoch`参数：

```yaml
training:
  stages:
    optical_flow_pretrain:
      epochs: 50
      learning_rate: 0.001
      best_model_start_epoch: 5      # 从第5个epoch开始保存最佳模型
      
    landmark_fc_train:
      epochs: 20
      learning_rate: 0.01
      best_model_start_epoch: 3      # 从第3个epoch开始保存最佳模型
      
    landmark_finetune:
      epochs: 20
      learning_rate: 0.001
      best_model_start_epoch: 3      # 从第3个epoch开始保存最佳模型
      
    fusion_train:
      epochs: 30
      learning_rate: 0.01
      best_model_start_epoch: 5      # 从第5个epoch开始保存最佳模型
      
    end_to_end:
      epochs: 50
      learning_rate: 0.0001
      best_model_start_epoch: 10     # 从第10个epoch开始保存最佳模型
```

#### 2.2.3 参数说明

- **`best_model_start_epoch`**: 从第几个epoch开始保存最佳模型
  - 默认值: 0（传统行为，从第一个epoch开始保存）
  - 建议值: 根据训练阶段特点设置
    - 预训练阶段: 5-10（模型需要更多时间稳定）
    - 微调阶段: 3-5（模型相对稳定，可以较早开始保存）
    - 端到端训练: 10-15（整体模型调优需要较长的稳定期）

#### 2.2.4 设置建议

**为什么需要延迟保存最佳模型和延迟早停？**

1. **避免早期不稳定**: 训练初期，模型参数随机化，损失波动较大
2. **提高模型质量**: 等待模型相对稳定后再开始保存，确保"最佳"模型真的是好模型
3. **防止过早停止**: 早停机制也延迟启动，避免在模型还未稳定时就误判为不再改进
4. **节省存储空间**: 减少不必要的模型保存操作
5. **提高训练效率**: 避免频繁的文件I/O操作

**延迟机制的协同工作：**

- **最佳模型保存**: 从指定epoch开始监控和保存性能最佳的模型
- **早停机制**: 从相同的epoch开始监控是否应该提前停止训练
- **自动同步**: 早停的启动时间自动与`best_model_start_epoch`保持一致

**各阶段推荐设置：**

| 训练阶段 | 推荐值 | 原因 |
|---------|--------|------|
| optical_flow_pretrain | 5-10 | 预训练模型需要时间适应新任务 |
| landmark_fc_train | 3-5 | 全连接层训练相对快速 |
| landmark_finetune | 3-5 | 在已训练基础上微调 |
| fusion_train | 5-8 | 融合不同特征需要稳定期 |
| end_to_end | 10-15 | 整体优化需要较长稳定期 |

## 3. 训练监控

### 3.1 训练日志

训练过程中会显示如下信息：

```
Stage: optical_flow_pretrain | Epoch: 0 | Train Loss: 2.1234 | Val Loss: 2.0987 | Train Acc: 0.1234 | Val Acc: 0.1456
注意: 将从第 5 个epoch开始保存最佳模型
已为阶段 optical_flow_pretrain 重置早停机制 (从第 5 个epoch开始启用早停)

Stage: optical_flow_pretrain | Epoch: 5 | Train Loss: 1.5678 | Val Loss: 1.4321 | Train Acc: 0.3456 | Val Acc: 0.3789
保存最佳模型: epoch 5, val_loss = 1.4321

Stage: optical_flow_pretrain | Epoch: 6 | Train Loss: 1.4567 | Val Loss: 1.3210 | Train Acc: 0.4567 | Val Acc: 0.4890
保存最佳模型: epoch 6, val_loss = 1.3210

Stage: optical_flow_pretrain | Epoch: 15 | Train Loss: 1.6789 | Val Loss: 1.5432 | Train Acc: 0.3210 | Val Acc: 0.3456
早停触发，停止训练
```

### 3.2 检查点文件

训练过程中会生成以下文件：

- `best_checkpoint.pth`: 当前阶段的最佳模型
- `last_checkpoint.pth`: 最新的模型检查点
- `training_history.json`: 完整的训练历史记录
- `training_curves.png`: 综合训练曲线（带延迟保存标记）
- `stage_losses.png`: 分阶段损失分析（智能最佳点标注）

### 3.3 可视化图像解读

#### 分阶段损失图 (`stage_losses.png`)
- **红色星形标记** ⭐: 实际保存的最佳模型点
- **橙色圆点标记** 🔸: 早期更优但因延迟策略未保存的点
- **灰色虚线** ┆: 开始保存最佳模型的时间点
- **黄色标注框**: 详细的epoch和损失信息

#### 综合训练曲线 (`training_curves.png`)
- **彩色垂直虚线**: 各阶段开始保存最佳模型的时间点
- **不同颜色**: 区分不同训练阶段的延迟保存起始点

## 4. 常见问题

### 4.1 Q: 如果不想延迟保存最佳模型怎么办？

A: 将`best_model_start_epoch`设置为0即可恢复传统行为。

### 4.2 Q: 可以在训练过程中修改这个参数吗？

A: 不建议。参数在每个阶段开始时读取，阶段进行中修改不会生效。

### 4.3 Q: 如何选择合适的延迟epoch数？

A: 观察训练曲线，选择损失开始相对稳定的epoch作为起始点。一般来说：
- 简单任务：3-5个epoch
- 复杂任务：10-15个epoch
- 预训练阶段：5-10个epoch

### 4.4 Q: 延迟保存会影响早停机制吗？

A: 不会影响，反而是协同工作的。早停机制会自动使用与最佳模型保存相同的延迟时间，确保两个机制都在模型稳定后才开始工作。

### 4.5 Q: 为什么早停机制也要延迟启动？

A: 这样可以避免在模型训练不稳定期就误判为"不再改进"而过早停止训练。延迟启动确保模型有足够的时间达到相对稳定的状态。

### 4.6 Q: 可以让早停机制的延迟时间与最佳模型保存不同吗？

A: 当前实现中两者自动保持一致，这是最佳实践。如果需要不同的延迟时间，可以修改代码中的相关逻辑。

## 5. 高级配置

### 5.1 自定义监控指标

```yaml
training:
  checkpoint:
    monitor: "overall_accuracy"      # 监控整体准确率而非损失
    mode: "max"                      # 最大化准确率
    best_model_start_epoch: 8        # 从第8个epoch开始保存
```

### 5.2 禁用某些检查点功能

```yaml
training:
  checkpoint:
    save_best: false                 # 不保存最佳模型
    save_last: true                  # 只保存最后一个模型
    load_best_after_training: false  # 不自动加载最佳模型
```

## 6. 故障排除

### 6.1 训练中断恢复

如果训练意外中断，可以从最后的检查点恢复：

```python
python main.py --resume experiments/experiment_YYYYMMDD_HHMMSS/last_checkpoint.pth
```

### 6.2 验证配置正确性

使用提供的测试脚本验证配置：

```bash
python test_best_model_config.py
```

这将显示各阶段的配置参数和生效值。

### 6.3 查看训练效果

运行演示脚本查看延迟保存的效果：

```bash
python demo_best_model_config.py
```