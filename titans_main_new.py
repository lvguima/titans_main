"""
Titans-PyTorch 主程序 (重构版)
基于模块化的持续学习框架，支持灵活的Backbone和Memory Unit组合
"""

import argparse
import torch
import numpy as np
import random
import os
from pathlib import Path
from datetime import datetime


def set_seed(seed=2021):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Titans持续学习框架：模块化时间序列预测'
    )
    
    # ==================== 基础配置 ====================
    parser.add_argument('--is_training', type=int, default=1,
                        help='运行模式: 1=预训练, 0=在线测试')
    parser.add_argument('--model_id', type=str, default='continual_forecaster',
                        help='模型标识符')
    parser.add_argument('--seed', type=int, default=2021,
                        help='随机种子')
    parser.add_argument('--des', type=str, default='experiment',
                        help='实验描述（用于文件夹命名）')
    
    # ==================== 数据配置 ====================
    parser.add_argument('--data', type=str, default='synthetic',
                        help='数据集名称')
    parser.add_argument('--root_path', type=str, default='./dataset/',
                        help='数据集根目录')
    parser.add_argument('--data_path', type=str, default='realistic_drift_data.csv',
                        help='数据文件名')
    parser.add_argument('--features', type=str, default='M',
                        help='预测任务: M=多对多, S=单对单, MS=多对单')
    parser.add_argument('--target', type=str, default='target',
                        help='目标列名（S或MS任务）')
    parser.add_argument('--freq', type=str, default='h',
                        help='时间频率: s/t/h/d/w/m')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='模型保存路径')
    
    # ==================== 时间序列配置 ====================
    parser.add_argument('--seq_len', type=int, default=64,
                        help='输入序列长度')
    parser.add_argument('--label_len', type=int, default=0,
                        help='Decoder起始token长度')
    parser.add_argument('--pred_len', type=int, default=1,
                        help='预测长度')
    parser.add_argument('--input_dim', type=int, default=3,
                        help='输入特征维度（自动检测）')
    parser.add_argument('--output_dim', type=int, default=1,
                        help='输出特征维度（自动检测）')
    
    # ==================== 模型架构选择 ====================
    parser.add_argument('--backbone_type', type=str, default='titans',
                        choices=['lstm', 'transformer', 'titans'],
                        help='Backbone类型')
    parser.add_argument('--memory_type', type=str, default='titans_mlp',
                        choices=['titans_mlp', 'titans_attention', 'none'],
                        help='Memory Unit类型')
    parser.add_argument('--fusion_type', type=str, default='add',
                        choices=['add', 'concat', 'gated'],
                        help='特征融合方式')
    
    # ==================== Backbone配置 ====================
    parser.add_argument('--d_model', type=int, default=384,
                        help='Backbone隐藏维度')
    parser.add_argument('--e_layers', type=int, default=4,
                        help='Backbone层数')
    parser.add_argument('--n_heads', type=int, default=6,
                        help='注意力头数（Transformer/Titans）')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout比率')
    
    # ==================== Memory Unit配置 ====================
    parser.add_argument('--memory_chunk_size', type=int, default=1,
                        help='Memory chunk size')
    parser.add_argument('--neural_memory_batch_size', type=int, default=256,
                        help='多少个token后更新一次记忆权重')
    parser.add_argument('--memory_model_type', type=str, default='mlp',
                        choices=['mlp', 'attention'],
                        help='Memory内部模型类型')
    
    # ==================== 训练配置 ====================
    parser.add_argument('--train_epochs', type=int, default=10,
                        help='预训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='预训练学习率')
    parser.add_argument('--patience', type=int, default=7,
                        help='早停patience')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='梯度裁剪阈值')
    
    # ==================== 优化器配置 ====================
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd'],
                        help='优化器类型')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='权重衰减')
    parser.add_argument('--loss', type=str, default='mse',
                        choices=['mse', 'mae', 'huber'],
                        help='损失函数')
    
    # ==================== 在线测试配置 ====================
    parser.add_argument('--test_mode', type=str, default='memory_only',
                        choices=['memory_only', 'full_model'],
                        help='测试模式: memory_only=仅M学习, full_model=M和P都学习')
    parser.add_argument('--online_lr', type=float, default=1e-5,
                        help='在线学习率（full_model模式）')
    
    # ==================== 日志和保存配置 ====================
    parser.add_argument('--log_interval', type=int, default=100,
                        help='日志打印间隔')
    parser.add_argument('--save_pred', action='store_true', default=True,
                        help='保存预测结果')
    parser.add_argument('--save_fig', action='store_true', default=True,
                        help='保存可视化图表')
    parser.add_argument('--result_path', type=str, default='./results/',
                        help='结果保存路径')
    parser.add_argument('--fig_path', type=str, default='./figs/',
                        help='图表保存路径')
    
    # ==================== GPU配置 ====================
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='是否使用GPU')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU设备号')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False,
                        help='使用多GPU')
    parser.add_argument('--devices', type=str, default='0,1',
                        help='GPU设备列表')
    
    args = parser.parse_args()
    return args


def print_args(args):
    """打印参数配置"""
    print("\n" + "="*70)
    print("实验配置")
    print("="*70)
    
    print("\n【基础配置】")
    print(f"  运行模式: {'预训练' if args.is_training else '在线测试'}")
    print(f"  模型ID: {args.model_id}")
    print(f"  实验描述: {args.des}")
    print(f"  随机种子: {args.seed}")
    
    print("\n【数据配置】")
    print(f"  数据集: {args.data}")
    print(f"  数据路径: {args.root_path}{args.data_path}")
    print(f"  序列长度: {args.seq_len}")
    print(f"  预测长度: {args.pred_len}")
    print(f"  特征维度: 输入{args.input_dim} -> 输出{args.output_dim}")
    
    print("\n【模型架构】")
    print(f"  Backbone: {args.backbone_type}")
    print(f"  Memory Unit: {args.memory_type}")
    print(f"  融合方式: {args.fusion_type}")
    print(f"  Backbone维度: {args.d_model}")
    print(f"  Backbone层数: {args.e_layers}")
    print(f"  注意力头数: {args.n_heads}")
    
    print("\n【Memory配置】")
    print(f"  Memory模型类型: {args.memory_model_type}")
    print(f"  Memory chunk size: {args.memory_chunk_size}")
    print(f"  Neural memory batch size: {args.neural_memory_batch_size}")
    
    if args.is_training:
        print("\n【训练配置】")
        print(f"  训练轮数: {args.train_epochs}")
        print(f"  批大小: {args.batch_size}")
        print(f"  学习率: {args.learning_rate}")
        print(f"  优化器: {args.optimizer}")
        print(f"  损失函数: {args.loss}")
        print(f"  梯度裁剪: {args.clip_grad}")
        print(f"  早停patience: {args.patience}")
    else:
        print("\n【在线测试配置】")
        print(f"  测试模式: {args.test_mode}")
        if args.test_mode == 'full_model':
            print(f"  在线学习率: {args.online_lr}")
    
    print("\n【设备配置】")
    print(f"  使用GPU: {args.use_gpu}")
    if args.use_gpu:
        print(f"  GPU设备: {args.gpu}")
        print(f"  多GPU: {args.use_multi_gpu}")
    
    print("="*70 + "\n")


def main():
    """主函数"""
    # 解析参数
    args = get_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 打印配置
    print_args(args)
    
    # 创建保存目录
    Path(args.checkpoints).mkdir(parents=True, exist_ok=True)
    Path(args.result_path).mkdir(parents=True, exist_ok=True)
    Path(args.fig_path).mkdir(parents=True, exist_ok=True)
    
    # 导入新的训练器
    from utils.trainer_new import ContinualTrainer
    
    # 创建训练器
    trainer = ContinualTrainer(args)
    
    if args.is_training:
        # ==================== 预训练阶段 ====================
        print("\n" + "="*70)
        print("阶段1: 预训练（同时训练Backbone和Memory Unit）")
        print("="*70)
        
        start_time = datetime.now()
        trainer.train()
        end_time = datetime.now()
        
        print(f"\n预训练完成！用时: {end_time - start_time}")
        
    else:
        # ==================== 在线测试阶段 ====================
        freeze_backbone = (args.test_mode == 'memory_only')
        
        print("\n" + "="*70)
        if freeze_backbone:
            print("阶段2: 在线测试 - 模式A（仅Memory Unit学习）")
        else:
            print("阶段2: 在线测试 - 模式B（Memory Unit + Backbone都学习）")
        print("="*70)
        
        start_time = datetime.now()
        mae, mse, rmse, mape, mspe, rse = trainer.online_test(
            freeze_backbone=freeze_backbone,
            load_checkpoint=True
        )
        end_time = datetime.now()
        
        print(f"\n在线测试完成！用时: {end_time - start_time}")
        
        # 打印最终指标
        print("\n" + "="*70)
        print("最终测试指标")
        print("="*70)
        print(f"  MAE:  {mae:.6f}")
        print(f"  MSE:  {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAPE: {mape:.6f}")
        print(f"  MSPE: {mspe:.6f}")
        print(f"  RSE:  {rse:.6f}")
        print("="*70)


if __name__ == '__main__':
    main()

