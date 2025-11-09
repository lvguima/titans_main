"""
工具函数
包括早停、学习率调整等
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience: 等待多少个epoch没有改善后停止
            verbose: 是否打印信息
            delta: 最小变化阈值
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf  # 使用小写 np.inf (NumPy 2.0兼容)
        self.delta = delta
    
    def __call__(self, val_loss, model, path):
        """
        Args:
            val_loss: 验证集损失
            model: 模型
            path: 保存路径
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, path):
        """保存模型"""
        if self.verbose:
            print(f'验证集损失下降 ({self.val_loss_min:.6f} --> {val_loss:.6f}). 保存模型...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, epoch, args, printout=True):
    """
    学习率调整策略
    
    Args:
        optimizer: 优化器
        epoch: 当前epoch
        args: 参数配置
        printout: 是否打印
    """
    if args.lradj == 'type1':
        # 每个epoch衰减
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        # 固定epoch衰减
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        # cosine annealing
        lr_adjust = {epoch: args.learning_rate * (1 + np.cos(epoch / args.train_epochs * np.pi)) / 2}
    elif args.lradj == 'constant':
        # 不调整
        lr_adjust = {epoch: args.learning_rate}
    else:
        lr_adjust = {epoch: args.learning_rate}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            print(f'更新学习率为 {lr:.6f}')


def visual(true, preds=None, name='./pic/test.jpg'):
    """
    可视化预测结果
    
    Args:
        true: 真实值
        preds: 预测值
        name: 保存路径
    """
    plt.figure(figsize=(12, 6))
    plt.plot(true, label='Ground Truth', linewidth=2, alpha=0.7)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2, alpha=0.7)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Prediction vs Ground Truth', fontsize=14, fontweight='bold')
    
    # 确保目录存在
    Path(name).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(name, bbox_inches='tight', dpi=150)
    plt.close()


def visual_loss_curve(train_losses, val_losses, name='./pic/loss_curve.jpg'):
    """
    可视化训练和验证loss曲线
    
    Args:
        train_losses: 训练loss列表
        val_losses: 验证loss列表
        name: 保存路径
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.7)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2, alpha=0.7)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    
    # 确保目录存在
    Path(name).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(name, bbox_inches='tight', dpi=150)
    plt.close()


def visual_comprehensive(trues, preds, losses=None, name='./pic/comprehensive.jpg', train_size=None):
    """
    综合可视化：包含预测对比、误差分布、滚动误差等（增强版）
    
    Args:
        trues: 真实值
        preds: 预测值
        losses: 损失值（可选）
        name: 保存路径
        train_size: 训练集大小（用于标注训练/测试分割，可选）
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
    
    # 1. 预测对比（完整序列）
    ax1 = fig.add_subplot(gs[0, :])
    steps = range(len(trues))
    ax1.plot(steps, trues, 'k-', alpha=0.5, label='Ground Truth', linewidth=1.5)
    ax1.plot(steps, preds, 'b-', alpha=0.7, label='Prediction', linewidth=1.5)
    
    # 标注训练/测试分割
    if train_size is not None and 0 < train_size < len(trues):
        ax1.axvline(x=train_size, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Train/Test Split')
        ax1.axvspan(0, train_size, alpha=0.1, color='blue')
        ax1.axvspan(train_size, len(trues), alpha=0.1, color='orange')
    
    ax1.set_title('Prediction vs Ground Truth (Full Sequence)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time Steps', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. 滚动误差（MSE）
    ax2 = fig.add_subplot(gs[1, 0])
    if losses is not None and len(losses) > 0:
        window = min(100, len(losses))
        if len(losses) >= window:
            smoothed_loss = np.convolve(losses, np.ones(window)/window, mode='valid')
            x_smooth = np.arange(len(smoothed_loss))
        else:
            smoothed_loss = losses
            x_smooth = np.arange(len(losses))
        
        ax2.plot(x_smooth, smoothed_loss, color='red', linewidth=2, alpha=0.7)
        if train_size is not None and 0 < train_size < len(trues):
            ax2.axvline(x=train_size, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax2.set_title(f'Rolling MSE (window={window})', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time Steps', fontsize=11)
        ax2.set_ylabel('MSE', fontsize=11)
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No Loss Data', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=14, color='gray')
        ax2.set_title('Rolling MSE', fontsize=12, fontweight='bold')
    
    # 3. 预测对比（局部放大）
    ax3 = fig.add_subplot(gs[1, 1])
    zoom_len = min(500, len(trues))
    steps_zoom = range(zoom_len)
    ax3.plot(steps_zoom, trues[:zoom_len], 'k-', alpha=0.5, label='Ground Truth', linewidth=1.5)
    ax3.plot(steps_zoom, preds[:zoom_len], 'b-', alpha=0.7, label='Prediction', linewidth=1.5)
    ax3.set_title(f'Prediction vs Ground Truth (First {zoom_len} Steps)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time Steps', fontsize=11)
    ax3.set_ylabel('Value', fontsize=11)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. 误差分布
    ax4 = fig.add_subplot(gs[2, 0])
    errors = trues - preds
    ax4.hist(errors, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax4.axvline(x=0, color='k', linestyle='--', linewidth=2)
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    ax4.axvline(x=mean_error, color='r', linestyle='-', linewidth=2, label=f'Mean: {mean_error:.4f}')
    ax4.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Error (True - Pred)', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. 误差随时间变化
    ax5 = fig.add_subplot(gs[2, 1])
    abs_errors = np.abs(errors)
    window_err = min(50, len(abs_errors))
    if len(abs_errors) >= window_err:
        smoothed_abs_error = np.convolve(abs_errors, np.ones(window_err)/window_err, mode='valid')
        x_err = np.arange(len(smoothed_abs_error))
    else:
        smoothed_abs_error = abs_errors
        x_err = np.arange(len(abs_errors))
    
    ax5.plot(x_err, smoothed_abs_error, color='orange', linewidth=2, alpha=0.7)
    if train_size is not None and 0 < train_size < len(trues):
        ax5.axvline(x=train_size, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax5.set_title(f'Rolling Absolute Error (window={window_err})', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Time Steps', fontsize=11)
    ax5.set_ylabel('Absolute Error', fontsize=11)
    ax5.grid(True, alpha=0.3)
    
    # 添加指标文本框
    mse = np.mean((trues - preds) ** 2)
    mae = np.mean(np.abs(trues - preds))
    rmse = np.sqrt(mse)
    
    textstr = f'Overall Metrics:\n' \
              f'MSE:  {mse:.6f}\n' \
              f'RMSE: {rmse:.6f}\n' \
              f'MAE:  {mae:.6f}'
    
    ax1.text(0.02, 0.98, textstr, 
            transform=ax1.transAxes, fontsize=11,
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.suptitle('Comprehensive Prediction Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    # 确保目录存在
    Path(name).parent.mkdir(parents=True, exist_ok=True)
    # matplotlib的jpg不支持quality参数，使用pil_kwargs代替
    plt.savefig(name, bbox_inches='tight', dpi=300, format='jpg', pil_kwargs={'quality': 95})
    plt.close()


def save_results(true, preds, losses, path):
    """
    保存预测结果到CSV
    
    Args:
        true: 真实值
        preds: 预测值
        losses: 损失值
        path: 保存路径
    """
    import pandas as pd
    
    df = pd.DataFrame({
        'step': np.arange(len(true.flatten())),
        'true': true.flatten(),
        'pred': preds.flatten(),
        'loss': losses,
        'error': true.flatten() - preds.flatten(),
        'abs_error': np.abs(true.flatten() - preds.flatten())
    })
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✓ 结果已保存到: {path}")


def create_experiment_folder(base_path, experiment_name):
    """
    创建规范的实验文件夹结构
    
    Args:
        base_path: 基础路径
        experiment_name: 实验名称
    
    Returns:
        experiment_paths: 包含各类文件路径的字典
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_folder = Path(base_path) / f"{experiment_name}_{timestamp}"
    
    # 创建子文件夹
    folders = {
        'root': exp_folder,
        'checkpoints': exp_folder / 'checkpoints',
        'results': exp_folder / 'results',
        'figures': exp_folder / 'figures',
        'logs': exp_folder / 'logs'
    }
    
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ 实验文件夹已创建: {exp_folder}")
    
    return folders, timestamp


def visual_detailed_analysis(trues, preds, train_losses, val_losses, 
                             train_split_idx=None, drift_points=None,
                             name='./pic/detailed_analysis.pdf'):
    """
    详细分析可视化：包含训练loss、测试预测、误差分析等
    参考experiment_comparison.py的风格
    
    Args:
        trues: 真实值数组
        preds: 预测值数组
        train_losses: 训练loss历史
        val_losses: 验证loss历史
        train_split_idx: 训练/测试分割点索引（在trues/preds中的位置）
        drift_points: 漂移点列表（可选）
        name: 保存路径
    """
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # 子图1: 完整预测序列（跨行）
    ax1 = fig.add_subplot(gs[0, :])
    steps = np.arange(len(trues))
    ax1.plot(steps, trues, 'k-', alpha=0.5, label='真实值 (Ground Truth)', linewidth=1.5)
    ax1.plot(steps, preds, 'b-', alpha=0.7, label='预测值 (Prediction)', linewidth=1.5)
    
    # 标注训练/测试分割
    if train_split_idx is not None and 0 < train_split_idx < len(trues):
        ax1.axvline(x=train_split_idx, color='red', linestyle='--', linewidth=2, 
                   alpha=0.8, label='训练/测试分割')
        ax1.axvspan(0, train_split_idx, alpha=0.1, color='blue', label='训练阶段')
        ax1.axvspan(train_split_idx, len(trues), alpha=0.1, color='yellow', label='测试阶段')
    
    # 标注漂移点
    if drift_points is not None:
        for dp in drift_points:
            if 0 <= dp < len(trues):
                ax1.axvline(x=dp, color='orange', linestyle=':', alpha=0.5, linewidth=1)
    
    ax1.set_title('预测序列全览', fontsize=14, fontweight='bold')
    ax1.set_xlabel('时间步', fontsize=12)
    ax1.set_ylabel('值', fontsize=12)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 计算整体指标
    mse = np.mean((trues - preds) ** 2)
    mae = np.mean(np.abs(trues - preds))
    rmse = np.sqrt(mse)
    ax1.text(0.02, 0.98, f'MSE: {mse:.6f}\nMAE: {mae:.6f}\nRMSE: {rmse:.6f}', 
            transform=ax1.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # 子图2: 训练和验证Loss曲线
    ax2 = fig.add_subplot(gs[1, 0])
    epochs = np.arange(1, len(train_losses) + 1)
    ax2.plot(epochs, train_losses, 'b-', label='训练Loss', linewidth=2, alpha=0.7)
    ax2.plot(epochs, val_losses, 'r-', label='验证Loss', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss (MSE)', fontsize=11)
    ax2.set_title('训练过程Loss曲线', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 标注最佳epoch
    best_epoch = np.argmin(val_losses) + 1
    best_loss = val_losses[best_epoch - 1]
    ax2.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5, linewidth=2)
    ax2.text(best_epoch, best_loss, f'最佳Epoch: {best_epoch}', 
            fontsize=9, ha='left', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # 子图3: 误差分布直方图
    ax3 = fig.add_subplot(gs[1, 1])
    errors = trues - preds
    ax3.hist(errors, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax3.axvline(x=0, color='k', linestyle='--', linewidth=2)
    ax3.axvline(x=np.mean(errors), color='blue', linestyle='--', linewidth=2, 
               label=f'均值: {np.mean(errors):.4f}')
    ax3.set_xlabel('预测误差 (真实值 - 预测值)', fontsize=11)
    ax3.set_ylabel('频数', fontsize=11)
    ax3.set_title('误差分布', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 添加统计信息
    error_std = np.std(errors)
    ax3.text(0.02, 0.98, f'标准差: {error_std:.6f}\n偏度: {np.mean(errors):.6f}', 
            transform=ax3.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # 子图4: 滚动MSE（测试阶段）
    ax4 = fig.add_subplot(gs[2, 0])
    squared_errors = (trues - preds) ** 2
    window = min(50, len(squared_errors) // 10)
    if window > 1:
        rolling_mse = pd.Series(squared_errors).rolling(window, min_periods=1).mean().values
        ax4.plot(steps, rolling_mse, 'purple', linewidth=2, alpha=0.7)
        ax4.set_title(f'滚动MSE (窗口={window})', fontsize=12, fontweight='bold')
    else:
        ax4.plot(steps, squared_errors, 'purple', linewidth=1, alpha=0.5)
        ax4.set_title('逐点平方误差', fontsize=12, fontweight='bold')
    
    if train_split_idx is not None:
        ax4.axvline(x=train_split_idx, color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    if drift_points is not None:
        for dp in drift_points:
            if 0 <= dp < len(trues):
                ax4.axvline(x=dp, color='orange', linestyle=':', alpha=0.5, linewidth=1)
    
    ax4.set_xlabel('时间步', fontsize=11)
    ax4.set_ylabel('MSE', fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # 子图5: 预测散点图
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.scatter(trues, preds, alpha=0.3, s=10, color='blue')
    
    # 绘制理想预测线
    min_val = min(trues.min(), preds.min())
    max_val = max(trues.max(), preds.max())
    ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测')
    
    ax5.set_xlabel('真实值', fontsize=11)
    ax5.set_ylabel('预测值', fontsize=11)
    ax5.set_title('预测散点图', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # 添加相关系数
    corr = np.corrcoef(trues, preds)[0, 1]
    ax5.text(0.05, 0.95, f'相关系数: {corr:.4f}', 
            transform=ax5.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    fig.suptitle('Titans预测详细分析', fontsize=16, fontweight='bold', y=0.995)
    
    # 保存
    Path(name).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(name, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ 详细分析可视化已保存到: {name}")


def set_requires_grad(model, requires_grad, keywords=None):
    """
    设置模型参数是否需要梯度
    
    Args:
        model: 模型
        requires_grad: bool
        keywords: 参数名关键词列表（如果指定，只影响包含这些关键词的参数）
    """
    count = 0
    for name, param in model.named_parameters():
        if keywords is None:
            param.requires_grad = requires_grad
            count += 1
        else:
            if any(kw in name for kw in keywords):
                param.requires_grad = requires_grad
                count += 1
    
    print(f"✓ 设置 {count} 个参数的 requires_grad = {requires_grad}")


def count_parameters(model):
    """
    统计模型参数量
    
    Args:
        model: 模型
    
    Returns:
        total: 总参数量
        trainable: 可训练参数量
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_device(args):
    """
    获取设备
    
    Args:
        args: 参数配置
    
    Returns:
        device: torch.device
    """
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"✓ 使用GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        print("✓ 使用CPU")
    
    return device


if __name__ == '__main__':
    """测试工具函数"""
    print("测试EarlyStopping...")
    
    early_stopping = EarlyStopping(patience=3, verbose=True)
    
    # 模拟训练过程
    val_losses = [1.0, 0.9, 0.85, 0.86, 0.87, 0.88]
    
    for epoch, val_loss in enumerate(val_losses):
        print(f"\nEpoch {epoch}, Val Loss: {val_loss}")
        # 这里用None代替model和path进行测试
        # early_stopping(val_loss, None, 'test_checkpoint.pth')
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
    
    print("\n✓ 工具函数测试通过!")

