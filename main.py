"""
主训练脚本
"""
import os
import sys
import argparse
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import load_config
from utils.metrics import MetricsCalculator
from data.transforms import create_transforms
from data.dataset import create_data_loaders
from training.trainer import FacialExpressionTrainer
from training.utils import setup_logging


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='面部表情动态评估模型训练')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--resume', 
        type=str, 
        default=None,
        help='恢复训练的检查点路径'
    )
    parser.add_argument(
        '--test-data', 
        action='store_true',
        help='仅测试数据加载，不进行训练'
    )
    
    args = parser.parse_args()
    
    try:
        # 加载配置
        print("加载配置文件...")
        config = load_config(args.config)
        print(f"实验目录: {config.get_experiment_dir()}")
        
        # 设置日志
        print("设置日志...")
        logger = setup_logging(
            config.get_experiment_dir(),
            config.get('logging.level', 'INFO')
        )
        logger.info("开始面部表情动态评估模型训练")
        logger.info(f"配置文件: {args.config}")
        logger.info(f"实验目录: {config.get_experiment_dir()}")
        
        # 创建数据变换器
        print("创建数据变换器...")
        augmentation_config = config.get('augmentation', {})
        optical_flow_transforms, landmark_transforms = create_transforms(augmentation_config)
        logger.info("数据变换器创建完成")
        
        # 创建数据加载器
        print("创建数据加载器...")
        data_config = config.data_config
        train_loader, val_loader = create_data_loaders(
            data_config, optical_flow_transforms, landmark_transforms
        )
        logger.info(f"训练集大小: {len(train_loader.dataset)}")
        logger.info(f"验证集大小: {len(val_loader.dataset)}")
        logger.info(f"训练批次数: {len(train_loader)}")
        logger.info(f"验证批次数: {len(val_loader)}")
        
        # 如果只是测试数据，则退出
        if args.test_data:
            print("测试数据加载...")
            test_data_loading(train_loader, val_loader, logger)
            return
        
        # 创建训练器
        print("创建训练器...")
        trainer = FacialExpressionTrainer(config, logger)
        logger.info("训练器创建完成")
        
        # 恢复训练（如果指定）
        if args.resume:
            print(f"恢复训练: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # 开始训练
        print("开始训练...")
        logger.info("开始训练所有阶段")
        start_time = datetime.now()
        
        trainer.train_all_stages(train_loader, val_loader)
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        # 保存训练历史
        trainer.save_training_history()
        
        # 训练完成
        logger.info("训练完成!")
        logger.info(f"总训练时间: {training_time}")
        print(f"训练完成! 总用时: {training_time}")
        print(f"结果保存在: {config.get_experiment_dir()}")
        
        # 生成评估报告
        generate_evaluation_report(config, trainer, logger)
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        if 'logger' in locals():
            logger.info("训练被用户中断")
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        if 'logger' in locals():
            logger.error(f"训练过程中发生错误: {e}", exc_info=True)
        raise


def test_data_loading(train_loader, val_loader, logger):
    """测试数据加载"""
    try:
        logger.info("测试训练数据加载...")
        for i, batch in enumerate(train_loader):
            logger.info(f"训练批次 {i}:")
            logger.info(f"  光流图像: {batch['optical_flow'].shape}")
            logger.info(f"  关键点特征: {batch['landmark_features'].shape}")
            logger.info(f"  目标: {batch['target'].shape}")
            logger.info(f"  视频ID: {batch['video_id'][:3]}...")
            
            if i >= 2:  # 只测试前几个批次
                break
        
        logger.info("测试验证数据加载...")
        for i, batch in enumerate(val_loader):
            logger.info(f"验证批次 {i}:")
            logger.info(f"  光流图像: {batch['optical_flow'].shape}")
            logger.info(f"  关键点特征: {batch['landmark_features'].shape}")
            logger.info(f"  目标: {batch['target'].shape}")
            
            if i >= 1:  # 只测试前几个批次
                break
        
        logger.info("数据加载测试完成")
        print("数据加载测试完成")
        
    except Exception as e:
        logger.error(f"数据加载测试失败: {e}", exc_info=True)
        print(f"数据加载测试失败: {e}")


def generate_evaluation_report(config, trainer, logger):
    """生成评估报告"""
    try:
        logger.info("生成评估报告...")
        
        # 创建指标计算器
        metrics_calculator = MetricsCalculator(
            config.get('evaluation', {}).get('score_ranges', {
                'dynamics': [0, 5],
                'synkinesis': [0, 3]
            })
        )
        
        # 生成训练曲线图
        if config.get('logging.visualize.enabled', True):
            plot_path = os.path.join(config.get_experiment_dir(), 'training_curves.png')
            plot_training_curves(trainer.train_history, trainer.val_history, plot_path)
            logger.info(f"训练曲线已保存: {plot_path}")
            
            # 生成分阶段loss图
            stage_plot_path = os.path.join(config.get_experiment_dir(), 'stage_losses.png')
            plot_stage_losses(trainer.train_history, trainer.val_history, trainer.stage_history, stage_plot_path)
            logger.info(f"分阶段损失曲线已保存: {stage_plot_path}")
        
        # 生成最终报告
        report_path = os.path.join(config.get_experiment_dir(), 'training_report.txt')
        generate_final_report(trainer, report_path, logger)
        
        logger.info("评估报告生成完成")
        
    except Exception as e:
        logger.error(f"生成评估报告失败: {e}", exc_info=True)


def plot_training_curves(train_history, val_history, save_path):
    """绘制训练曲线"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        epochs = range(len(train_history))
        
        # 提取损失
        train_losses = [h.get('train_loss', 0) for h in train_history]
        val_losses = [h.get('val_loss', 0) for h in val_history]
        
        # 提取准确率
        train_accs = [h.get('overall_accuracy', 0) for h in train_history]
        val_accs = [h.get('overall_accuracy', 0) for h in val_history]
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy')
        ax2.plot(epochs, val_accs, 'r-', label='Val Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        print("matplotlib不可用，跳过训练曲线绘制")
    except Exception as e:
        print(f"绘制训练曲线失败: {e}")


def plot_stage_losses(train_history, val_history, stage_history, save_path):
    """绘制分阶段训练损失曲线"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 设置字体以避免中文显示问题
        plt.rcParams['font.family'] = 'DejaVu Sans'
        
        # 提取损失数据
        train_losses = [h.get('train_loss', 0) for h in train_history]
        val_losses = [h.get('val_loss', 0) for h in val_history]
        
        # 创建3x2子图以容纳5个阶段
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        axes = axes.flatten()
        
        # 定义阶段信息
        stage_titles = {
            'optical_flow_pretrain': 'Stage 1: Optical Flow Pretrain',
            'landmark_fc_train': 'Stage 2a: Landmark FC Training',
            'landmark_finetune': 'Stage 2b: Landmark Finetune',
            'fusion_train': 'Stage 3: Fusion Training',
            'end_to_end': 'Stage 4: End-to-End Tuning'
        }
        
        # 绘制每个阶段
        for i, stage_info in enumerate(stage_history):
            if i >= 5:  # 最多显示5个阶段
                break
                
            start_epoch = stage_info['start_epoch']
            end_epoch = stage_info['end_epoch']
            stage_name = stage_info['stage_name']
            
            # 获取该阶段的数据
            stage_train_losses = train_losses[start_epoch:end_epoch + 1]
            stage_val_losses = val_losses[start_epoch:end_epoch + 1]
            relative_epochs = list(range(len(stage_train_losses)))
            
            # 绘制该阶段的损失曲线
            ax = axes[i]
            ax.plot(relative_epochs, stage_train_losses, 'b-', label='Train Loss', linewidth=2)
            ax.plot(relative_epochs, stage_val_losses, 'r-', label='Val Loss', linewidth=2)
            
            # 设置标题和标签
            title = stage_titles.get(stage_name, f'Stage {i+1}')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch (relative)')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 标注最佳验证损失
            if stage_val_losses:
                min_idx = np.argmin(stage_val_losses)
                min_loss = stage_val_losses[min_idx]
                ax.scatter(min_idx, min_loss, color='red', s=50, zorder=5)
                ax.annotate(f'Best: {min_loss:.4f}', 
                           xy=(min_idx, min_loss),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                           fontsize=8)
        
        # 隐藏多余的子图（第6个）
        for i in range(len(stage_history), 6):
            axes[i].set_visible(False)
        
        plt.suptitle('Multi-Stage Training Loss Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Stage loss plot saved to: {save_path}")
        
    except ImportError:
        print("matplotlib not available, skipping stage loss plotting")
    except Exception as e:
        print(f"Failed to plot stage losses: {e}")


def generate_final_report(trainer, report_path, logger):
    """生成最终报告"""
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("面部表情动态评估模型训练报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 训练总结
            f.write("训练总结:\n")
            f.write(f"- 总训练轮数: {len(trainer.train_history)}\n")
            
            if trainer.train_history:
                best_train_loss = min(h.get('train_loss', float('inf')) for h in trainer.train_history)
                best_val_loss = min(h.get('val_loss', float('inf')) for h in trainer.val_history)
                best_val_acc = max(h.get('overall_accuracy', 0) for h in trainer.val_history)
                
                f.write(f"- 最佳训练损失: {best_train_loss:.4f}\n")
                f.write(f"- 最佳验证损失: {best_val_loss:.4f}\n")
                f.write(f"- 最佳验证准确率: {best_val_acc:.4f}\n")
            
            f.write("\n模型架构:\n")
            f.write("- 光流模型: ResNet-18\n")
            f.write("- 关键点模型: 全连接网络\n")
            f.write("- 融合模型: 多模态融合网络\n")
            
            f.write("\n训练阶段:\n")
            f.write("1. 光流模型预训练\n")
            f.write("2a. 关键点全连接层训练\n")
            f.write("2b. 关键点模型微调\n")
            f.write("3. 融合模型训练\n")
            f.write("4. 端到端微调\n")
            
        logger.info(f"最终报告已保存: {report_path}")
        
    except Exception as e:
        logger.error(f"生成最终报告失败: {e}")


if __name__ == "__main__":
    main()
