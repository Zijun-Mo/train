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
            plot_training_curves(trainer.train_history, trainer.val_history, plot_path, config)
            logger.info(f"训练曲线已保存: {plot_path}")
            
            # 生成分阶段loss图
            stage_plot_path = os.path.join(config.get_experiment_dir(), 'stage_losses.png')
            plot_stage_losses(trainer.train_history, trainer.val_history, trainer.stage_history, stage_plot_path, config)
            logger.info(f"分阶段损失曲线已保存: {stage_plot_path}")
        
        # 生成最终报告
        report_path = os.path.join(config.get_experiment_dir(), 'training_report.txt')
        generate_final_report(trainer, report_path, logger)
        
        logger.info("评估报告生成完成")
        
    except Exception as e:
        logger.error(f"生成评估报告失败: {e}", exc_info=True)


def plot_training_curves(train_history, val_history, save_path, config):
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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
        
        # 标注每个阶段的延迟保存开始点
        try:
            stages_config = config.get('training', {}).get('stages', {})
            global_best_start = config.get('training', {}).get('checkpoint', {}).get('best_model_start_epoch', 0)
            
            # 计算每个阶段的累积epoch
            cumulative_epochs = 0
            stage_colors = ['purple', 'green', 'blue', 'orange', 'brown']
            
            for i, (stage_name, stage_config) in enumerate(stages_config.items()):
                stage_epochs = stage_config.get('epochs', 0)
                best_start_epoch = stage_config.get('best_model_start_epoch', global_best_start)
                
                # 阶段内的best_start位置
                actual_best_start = cumulative_epochs + best_start_epoch
                
                if actual_best_start < len(epochs) and best_start_epoch > 0:
                    color = stage_colors[i % len(stage_colors)]
                    ax1.axvline(x=actual_best_start, color=color, linestyle=':', alpha=0.7, linewidth=1.5)
                    ax1.text(actual_best_start, ax1.get_ylim()[1] * (0.9 - i*0.1), 
                            f'{stage_name}\nsave starts', 
                            ha='center', va='top', fontsize=7, rotation=90,
                            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
                
                cumulative_epochs += stage_epochs
                
        except Exception as e:
            print(f"Warning: Could not add stage markers to loss plot: {e}")
        
        ax1.set_title('Training and Validation Loss\n(with delayed save markers)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
        ax2.plot(epochs, val_accs, 'r-', label='Val Accuracy', linewidth=2)
        ax2.set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Training curves with delayed save markers saved to: {save_path}")
        
    except ImportError:
        print("matplotlib不可用，跳过训练曲线绘制")
    except Exception as e:
        print(f"绘制训练曲线失败: {e}")


def plot_stage_losses(train_history, val_history, stage_history, save_path, config):
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
        
        # 从配置文件中获取每个阶段的best_model_start_epoch
        stages_config = config.get('training', {}).get('stages', {})
        global_best_start = config.get('training', {}).get('checkpoint', {}).get('best_model_start_epoch', 0)
        
        def get_stage_best_start_epoch(stage_name):
            """获取指定阶段的best_model_start_epoch"""
            stage_config = stages_config.get(stage_name, {})
            return stage_config.get('best_model_start_epoch', global_best_start)
        
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
            
            # 从配置文件获取该阶段的best_model_start_epoch
            best_start_epoch = get_stage_best_start_epoch(stage_name)
            
            # 标注实际会保存的最佳模型点（考虑延迟保存）
            if stage_val_losses and len(stage_val_losses) > best_start_epoch:
                # 只在best_start_epoch之后的epoch中寻找最佳点
                valid_losses = stage_val_losses[best_start_epoch:]
                valid_epochs = relative_epochs[best_start_epoch:]
                
                if valid_losses:
                    # 找到有效范围内的最小损失
                    min_idx_in_valid = np.argmin(valid_losses)
                    actual_best_epoch = valid_epochs[min_idx_in_valid]
                    actual_best_loss = valid_losses[min_idx_in_valid]
                    
                    # 标注实际保存的最佳点
                    ax.scatter(actual_best_epoch, actual_best_loss, color='red', s=80, zorder=5, 
                              marker='*', edgecolors='darkred', linewidth=1)
                    ax.annotate(f'Best Saved: {actual_best_loss:.4f}\n(epoch {actual_best_epoch})', 
                               xy=(actual_best_epoch, actual_best_loss),
                               xytext=(10, 15), textcoords='offset points',
                               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                               fontsize=8, ha='left')
                    
                    # 如果best_start_epoch > 0，用灰色虚线标出延迟保存的开始点
                    if best_start_epoch > 0:
                        ax.axvline(x=best_start_epoch, color='gray', linestyle='--', alpha=0.7, linewidth=1)
                        ax.text(best_start_epoch, ax.get_ylim()[1] * 0.9, 
                               f'Save starts\n(epoch {best_start_epoch})', 
                               ha='center', va='top', fontsize=7, 
                               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
                    
                    # 如果延迟保存前有更小的损失点，用不同颜色标出（但注明不会保存）
                    if best_start_epoch > 0:
                        early_losses = stage_val_losses[:best_start_epoch]
                        early_epochs = relative_epochs[:best_start_epoch]
                        if early_losses:
                            early_min_idx = np.argmin(early_losses)
                            early_min_epoch = early_epochs[early_min_idx]
                            early_min_loss = early_losses[early_min_idx]
                            
                            # 只在确实有更小损失时才标注
                            if early_min_loss < actual_best_loss:
                                ax.scatter(early_min_epoch, early_min_loss, color='orange', s=60, 
                                          zorder=4, marker='o', alpha=0.8)
                                ax.annotate(f'Not saved: {early_min_loss:.4f}\n(too early)', 
                                           xy=(early_min_epoch, early_min_loss),
                                           xytext=(-15, -20), textcoords='offset points',
                                           bbox=dict(boxstyle='round', facecolor='orange', alpha=0.6),
                                           fontsize=7, ha='center',
                                           arrowprops=dict(arrowstyle='->', color='orange', alpha=0.8))
                                           
            elif stage_val_losses:
                # 如果整个阶段都没有达到best_start_epoch，显示最小损失但标注为未保存
                min_idx = np.argmin(stage_val_losses)
                min_loss = stage_val_losses[min_idx]
                ax.scatter(min_idx, min_loss, color='orange', s=60, zorder=4, marker='o', alpha=0.8)
                ax.annotate(f'No model saved\n(all epochs < {best_start_epoch})', 
                           xy=(min_idx, min_loss),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round', facecolor='orange', alpha=0.6),
                           fontsize=8)
        
        # 隐藏多余的子图（第6个）
        for i in range(len(stage_history), 6):
            axes[i].set_visible(False)
        
        plt.suptitle('Multi-Stage Training Loss Analysis\n(Best points reflect actual model saving with delay)', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Stage loss plot with delayed best model markers saved to: {save_path}")
        
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
