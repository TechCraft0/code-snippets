import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_curves(metrics_history, save_path):
    """绘制训练过程中的损失和指标曲线"""
    
    # 创建3x2子图布局
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Training Curves', fontsize=18)
    
    # 1. 详细训练损失分解（主损失+辅助头损失+边缘损失）
    if 'train_total_loss' in metrics_history and metrics_history['train_total_loss']:
        axes[0, 0].plot(metrics_history['train_iters'], metrics_history['train_total_loss'], 'b-', label='Total Loss', linewidth=2)
        if 'train_seg_loss' in metrics_history and metrics_history['train_seg_loss']:
            axes[0, 0].plot(metrics_history['train_iters'], metrics_history['train_seg_loss'], 'g-', label='Seg Loss', alpha=0.8)
        if 'train_edge_loss' in metrics_history and metrics_history['train_edge_loss']:
            axes[0, 0].plot(metrics_history['train_iters'], metrics_history['train_edge_loss'], 'c-', label='Edge Loss', alpha=0.8)
        if 'train_main_loss' in metrics_history and metrics_history['train_main_loss']:
            axes[0, 0].plot(metrics_history['train_iters'], metrics_history['train_main_loss'], 'r-', label='Main Head', alpha=0.7)
        # 绘制辅助头损失
        colors = ['orange', 'purple', 'brown', 'pink']
        for i in range(4):
            aux_key = f'train_aux{i}_loss'
            if aux_key in metrics_history and metrics_history[aux_key]:
                axes[0, 0].plot(metrics_history['train_iters'], metrics_history[aux_key], 
                              color=colors[i], label=f'Aux Head {i}', alpha=0.6, linestyle='--')
        axes[0, 0].set_title('Training Loss Breakdown (with Edge Loss)')
        axes[0, 0].set_xlabel('Iterations')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
    
    # 2. 验证/测试损失对比
    if 'val_loss' in metrics_history and metrics_history['val_loss']:
        axes[0, 1].plot(metrics_history['val_iters'], metrics_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        if 'test_loss' in metrics_history and metrics_history['test_loss']:
            axes[0, 1].plot(metrics_history['test_iters'], metrics_history['test_loss'], 'g-', label='Test Loss', linewidth=2)
        axes[0, 1].set_title('Validation/Test Loss')
        axes[0, 1].set_xlabel('Iterations')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
    
    # 3. mIoU曲线
    if 'val_miou' in metrics_history and metrics_history['val_miou']:
        axes[1, 0].plot(metrics_history['val_iters'], metrics_history['val_miou'], 'r-', label='Val mIoU', linewidth=2)
        if 'test_miou' in metrics_history and metrics_history['test_miou']:
            axes[1, 0].plot(metrics_history['test_iters'], metrics_history['test_miou'], 'g-', label='Test mIoU', linewidth=2)
        axes[1, 0].set_title('Mean IoU')
        axes[1, 0].set_xlabel('Iterations')
        axes[1, 0].set_ylabel('mIoU')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
    
    # 4. 像素准确率曲线
    if 'val_pixel_acc' in metrics_history and metrics_history['val_pixel_acc']:
        axes[1, 1].plot(metrics_history['val_iters'], metrics_history['val_pixel_acc'], 'r-', label='Val Pixel Acc', linewidth=2)
        if 'test_pixel_acc' in metrics_history and metrics_history['test_pixel_acc']:
            axes[1, 1].plot(metrics_history['test_iters'], metrics_history['test_pixel_acc'], 'g-', label='Test Pixel Acc', linewidth=2)
        axes[1, 1].set_title('Pixel Accuracy')
        axes[1, 1].set_xlabel('Iterations')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    
    # 5. 学习率曲线
    if 'learning_rate' in metrics_history and metrics_history['learning_rate']:
        axes[2, 0].plot(metrics_history['train_iters'], metrics_history['learning_rate'], 'purple', linewidth=2)
        axes[2, 0].set_title('Learning Rate Schedule')
        axes[2, 0].set_xlabel('Iterations')
        axes[2, 0].set_ylabel('Learning Rate')
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].set_yscale('log')
    
    # 6. 综合损失对比（训练vs验证vs测试）
    if 'train_total_loss' in metrics_history and metrics_history['train_total_loss']:
        # 对训练损失进行平滑处理以便对比
        train_loss_smooth = np.convolve(metrics_history['train_total_loss'], np.ones(10)/10, mode='valid')
        train_iters_smooth = metrics_history['train_iters'][9:]
        axes[2, 1].plot(train_iters_smooth, train_loss_smooth, 'b-', label='Train Loss (smoothed)', alpha=0.8)
        
        if 'val_loss' in metrics_history and metrics_history['val_loss']:
            axes[2, 1].plot(metrics_history['val_iters'], metrics_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        if 'test_loss' in metrics_history and metrics_history['test_loss']:
            axes[2, 1].plot(metrics_history['test_iters'], metrics_history['test_loss'], 'g-', label='Test Loss', linewidth=2)
        
        axes[2, 1].set_title('Loss Comparison')
        axes[2, 1].set_xlabel('Iterations')
        axes[2, 1].set_ylabel('Loss')
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()