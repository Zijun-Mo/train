# 项目文件结构总览

```
train/
├── README.md                    # 项目概述和架构说明
├── USAGE.md                     # 详细使用指南
├── config.yaml                  # 主配置文件
├── requirements.txt             # Python依赖包
├── install.sh                   # 环境安装脚本
├── main.py                      # 主训练脚本
├── test_components.py           # 组件测试脚本
│
├── models/                      # 模型定义模块
│   ├── __init__.py
│   ├── face_blendshapes.py      # TFLite模型包装器
│   ├── optical_flow_model.py    # 光流模型(ResNet-18)
│   ├── landmark_model.py        # 关键点模型(全连接)
│   └── fusion_model.py          # 融合模型
│
├── data/                        # 数据处理模块
│   ├── __init__.py
│   ├── dataset.py               # 数据集类定义
│   └── transforms.py            # 数据预处理和增强
│
├── training/                    # 训练模块
│   ├── __init__.py
│   ├── trainer.py               # 主训练器类
│   └── utils.py                 # 训练工具函数
│
└── utils/                       # 通用工具模块
    ├── __init__.py
    ├── config.py                # 配置文件解析
    └── metrics.py               # 评估指标计算
```

## 核心功能说明

### 🎯 主要特性
- **多阶段训练策略**：从预训练到端到端微调
- **多模态融合**：结合光流和关键点特征
- **灵活配置**：通过YAML文件管理所有参数
- **完整监控**：训练过程可视化和指标跟踪
- **断点续训**：支持从检查点恢复训练

### 🏗️ 模型架构
- **光流分支**：ResNet-18 → 2维特征向量
- **关键点分支**：104维输入 → 全连接网络 → 2维特征向量
- **融合网络**：4维输入 → 最终2维评分输出

### 📊 输出评分
- **dynamics**：动态评分 (0-5)
- **synkinesis**：联动评分 (0-3)

### 🔄 训练流程
1. **光流预训练**：微调ResNet-18，直接预测评分
2. **关键点训练**：训练全连接层处理104维特征
3. **融合训练**：训练融合网络结合两个模态
4. **端到端微调**：全局优化所有参数

## 使用方式

### 快速开始
```bash
# 1. 安装环境
./install.sh

# 2. 测试组件
python3 test_components.py

# 3. 开始训练
python3 main.py
```

### 进阶使用
```bash
# 自定义配置训练
python3 main.py --config custom_config.yaml

# 从检查点恢复
python3 main.py --resume path/to/checkpoint.pth

# 仅测试数据加载
python3 main.py --test-data
```

## 配置要点

### 数据路径设置
```yaml
data:
  root_dir: "/home/jun/picture/output"
  face_blendshapes_model: "/home/jun/picture/extracted_models/face_blendshapes.tflite"
```

### 训练参数调优
```yaml
training:
  stages:
    optical_flow_pretrain:
      epochs: 50
      learning_rate: 0.001
    # ... 其他阶段
```

### 模型架构定制
```yaml
model:
  optical_flow:
    backbone: "resnet18"
    dropout: 0.5
  landmark:
    hidden_dims: [256, 128, 64]
    dropout: 0.3
```

## 输出说明

训练完成后，在 `experiments/experiment_YYYYMMDD_HHMMSS/` 目录下会生成：
- `best_checkpoint.pth`：最佳模型
- `training_history.json`：训练历史
- `training_curves.png`：训练曲线图
- `training_report.txt`：训练报告
- `training.log`：详细日志

## 环境要求

- **Python**: 3.8+
- **PyTorch**: 1.9+
- **TensorFlow**: 2.6+ (用于TFLite)
- **其他依赖**：见 requirements.txt

## 硬件建议

- **最低配置**：CPU + 8GB RAM
- **推荐配置**：GPU + 16GB RAM
- **存储空间**：至少10GB可用空间

## 性能优化建议

1. **使用GPU**：显著加速训练
2. **预加载特征**：减少I/O开销
3. **合理batch_size**：平衡内存和效率
4. **数据并行**：多GPU训练支持

## 故障排除

1. **环境问题**：运行 `test_components.py` 诊断
2. **数据问题**：使用 `--test-data` 参数检查
3. **内存问题**：减小batch_size或关闭特征预加载
4. **收敛问题**：调整学习率和训练策略

---

✨ **项目已完整实现，可以开始训练！** ✨
