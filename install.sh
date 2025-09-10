#!/bin/bash

# 面部表情动态评估模型训练环境安装脚本

echo "开始安装面部表情动态评估模型训练环境..."

# 检查Python版本
python_version=$(python3 --version 2>&1)
echo "当前Python版本: $python_version"

# 检查是否有pip
if ! command -v pip3 &> /dev/null; then
    echo "错误: pip3 未找到，请先安装pip"
    exit 1
fi

# 更新pip
echo "更新pip..."
pip3 install --upgrade pip

# 安装依赖
echo "安装Python依赖包..."
pip3 install -r requirements.txt

# 检查GPU支持
echo "检查GPU支持..."
python3 -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('将使用CPU进行训练')
"

# 检查TensorFlow Lite
echo "检查TensorFlow Lite..."
python3 -c "
try:
    import tensorflow as tf
    print(f'TensorFlow版本: {tf.__version__}')
    print('TensorFlow Lite可用')
except ImportError:
    print('错误: TensorFlow未正确安装')
"

# 创建实验目录
echo "创建实验目录..."
mkdir -p experiments
mkdir -p logs

# 测试数据加载（如果数据存在）
echo "测试环境配置..."
if [ -d "/home/jun/picture/output" ]; then
    echo "发现数据目录，测试数据加载..."
    python3 main.py --test-data
else
    echo "数据目录不存在，跳过数据加载测试"
    echo "请确保数据位于 /home/jun/picture/output 目录下"
fi

echo "安装完成!"
echo ""
echo "使用方法:"
echo "1. 确保数据位于正确路径: /home/jun/picture/output"
echo "2. 确保预训练模型位于: /home/jun/picture/extracted_models/face_blendshapes.tflite"
echo "3. 修改config.yaml中的配置参数"
echo "4. 运行训练: python3 main.py"
echo "5. 查看训练结果: experiments/目录下"
echo ""
echo "更多选项:"
echo "- 测试数据加载: python3 main.py --test-data"
echo "- 从检查点恢复: python3 main.py --resume path/to/checkpoint.pth"
echo "- 使用自定义配置: python3 main.py --config custom_config.yaml"
