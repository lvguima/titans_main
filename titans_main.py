"""
Titans-PyTorch 主程序
基于MAC (Memory-As-Context) 架构的时间序列预测模型
"""

import argparse
import torch
import numpy as np
import random
import os
from pathlib import Path
from datetime import datetime

# 设置随机种子
def set_seed(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser(description='Titans-PyTorch: Time Series Forecasting with MAC Architecture')
    
    # ==================== 基础配置 ====================
    parser.add_argument('--task_name', type=str, default='forecasting',
                        help='任务名称: forecasting, classification, anomaly_detection')
    parser.add_argument('--is_training', type=int, default=1, 
                        help='是否训练模式: 1-训练, 0-测试')
    parser.add_argument('--model_id', type=str, default='titans_mac', 
                        help='模型标识符')
    parser.add_argument('--seed', type=int, default=2021, 
                        help='随机种子')
    
    # ==================== 数据配置 ====================
    parser.add_argument('--data', type=str, default='synthetic', 
                        help='数据集名称: synthetic, ETTh1, ETTh2, ETTm1, ETTm2, custom')
    parser.add_argument('--root_path', type=str, default='./dataset/', 
                        help='数据集根目录')
    parser.add_argument('--data_path', type=str, default='realistic_drift_data.csv', 
                        help='数据文件名')
    parser.add_argument('--features', type=str, default='M',
                        help='预测任务类型: M-多变量预测多变量, S-单变量预测单变量, MS-多变量预测单变量')
    parser.add_argument('--target', type=str, default='target', 
                        help='目标列名（S或MS任务中使用）')
    parser.add_argument('--freq', type=str, default='h',
                        help='时间特征编码频率: s-秒, t-分, h-时, d-日, w-周, m-月')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', 
                        help='模型保存路径')
    
    # ==================== 时间序列配置 ====================
    parser.add_argument('--seq_len', type=int, default=64, 
                        help='输入序列长度（历史窗口大小）')
    parser.add_argument('--label_len', type=int, default=0, 
                        help='decoder起始token长度（用于某些架构）')
    parser.add_argument('--pred_len', type=int, default=1, 
                        help='预测序列长度（1表示单步预测）')
    parser.add_argument('--input_dim', type=int, default=3, 
                        help='输入特征维度')
    parser.add_argument('--output_dim', type=int, default=1, 
                        help='输出特征维度')
    
    # ==================== 稀疏标签配置 ====================
    parser.add_argument('--sparse_label', action='store_true', default=False,
                        help='是否使用稀疏标签训练')
    parser.add_argument('--sparse_step', type=int, default=10, 
                        help='稀疏标签步长（每隔多少步更新一次）')
    
    # ==================== Titans模型配置 ====================
    parser.add_argument('--dim', type=int, default=384, 
                        help='模型维度')
    parser.add_argument('--depth', type=int, default=6, 
                        help='Transformer层数')
    parser.add_argument('--segment_len', type=int, default=32, 
                        help='分段长度')
    parser.add_argument('--dim_head', type=int, default=64, 
                        help='每个注意力头的维度')
    parser.add_argument('--heads', type=int, default=6, 
                        help='注意力头数')
    parser.add_argument('--dropout', type=float, default=0.1, 
                        help='Dropout概率')
    
    # ==================== 记忆机制配置 ====================
    parser.add_argument('--num_persist_mem_tokens', type=int, default=4, 
                        help='持久记忆token数量')
    parser.add_argument('--num_longterm_mem_tokens', type=int, default=4, 
                        help='长期记忆token数量')
    parser.add_argument('--neural_memory_layers', type=str, default='2,4', 
                        help='神经记忆层位置（逗号分隔，如 2,4）')
    parser.add_argument('--neural_memory_segment_len', type=int, default=16, 
                        help='神经记忆分段长度')
    parser.add_argument('--neural_memory_batch_size', type=int, default=2048, 
                        help='神经记忆批大小')
    parser.add_argument('--neural_mem_weight_residual', action='store_true', default=True,
                        help='神经记忆权重残差连接')
    
    # ==================== 神经记忆模型配置 ====================
    parser.add_argument('--memory_model_type', type=str, default='attention',
                        help='记忆模型类型: attention, mlp, factorized_mlp, swiglu_mlp, gated_residual',
                        choices=['attention', 'mlp', 'factorized_mlp', 'swiglu_mlp', 'gated_residual'])
    parser.add_argument('--memory_dim', type=int, default=64, 
                        help='记忆模型维度')
    parser.add_argument('--memory_scale', type=float, default=8.0, 
                        help='记忆模型缩放因子')
    parser.add_argument('--memory_expansion_factor', type=int, default=2, 
                        help='记忆模型扩展因子')
    parser.add_argument('--memory_dim_head', type=int, default=64, 
                        help='记忆模型注意力头维度')
    parser.add_argument('--memory_heads', type=int, default=6, 
                        help='记忆模型注意力头数')
    parser.add_argument('--memory_momentum', action='store_true', default=True,
                        help='记忆模型是否使用momentum')
    parser.add_argument('--memory_momentum_order', type=int, default=2, 
                        help='Momentum阶数')
    parser.add_argument('--memory_max_lr', type=float, default=0.0001, 
                        help='记忆模型最大学习率')
    parser.add_argument('--memory_use_accelerated_scan', action='store_true', default=False,
                        help='使用加速扫描（需要CUDA）')
    
    # ==================== MAC架构配置 ====================
    parser.add_argument('--use_mac_fusion', action='store_true', default=True,
                        help='使用MAC融合（记忆作为上下文）')
    parser.add_argument('--use_flex_attn', action='store_true', default=False,
                        help='使用Flex Attention')
    parser.add_argument('--sliding_window_attn', action='store_true', default=False,
                        help='使用滑动窗口注意力')
    
    # ==================== 训练配置 ====================
    parser.add_argument('--train_epochs', type=int, default=3, 
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='批大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                        help='学习率')
    parser.add_argument('--train_update_freq', type=int, default=1, 
                        help='训练更新频率（梯度累积步数，1=每个batch更新，4=每4个batch累积后更新，相当于batch_size*4）')
    parser.add_argument('--patience', type=int, default=7, 
                        help='早停耐心值')
    parser.add_argument('--clip_grad', type=float, default=1.0, 
                        help='梯度裁剪阈值')
    
    # ==================== 在线学习配置 ====================
    parser.add_argument('--online_learning', type=int, default=1,
                        help='测试时启用在线学习（NeuralMemory持续适应）: 1-启用, 0-禁用')
    parser.add_argument('--online_lr', type=float, default=1e-4, 
                        help='在线学习学习率（仅在online_update_memory_only=False时用于更新backbone）')
    parser.add_argument('--online_update_memory_only', action='store_true', default=True,
                        help='在线学习仅更新记忆（True=信任NeuralMemory自动更新，False=额外用optimizer更新backbone）')
    
    # ==================== 优化器配置 ====================
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='优化器: adam, adamw, sgd')
    parser.add_argument('--weight_decay', type=float, default=0.0, 
                        help='权重衰减')
    parser.add_argument('--loss', type=str, default='mse', 
                        help='损失函数: mse, mae, huber')
    parser.add_argument('--lradj', type=str, default='type1', 
                        help='学习率调整策略')
    
    # ==================== GPU配置 ====================
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='使用GPU')
    parser.add_argument('--gpu', type=int, default=0, 
                        help='GPU设备ID')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False,
                        help='使用多GPU')
    parser.add_argument('--devices', type=str, default='0,1', 
                        help='多GPU设备ID（逗号分隔）')
    
    # ==================== 日志与可视化 ====================
    parser.add_argument('--log_interval', type=int, default=100, 
                        help='日志打印间隔')
    parser.add_argument('--save_pred', action='store_true', default=True,
                        help='保存预测结果')
    parser.add_argument('--save_fig', action='store_true', default=True,
                        help='保存可视化图表')
    parser.add_argument('--result_path', type=str, default='./results/', 
                        help='结果保存路径')
    parser.add_argument('--fig_path', type=str, default='./figures/', 
                        help='图表保存路径')
    
    # ==================== 实验配置 ====================
    parser.add_argument('--num_workers', type=int, default=0, 
                        help='数据加载进程数')
    parser.add_argument('--itr', type=int, default=1, 
                        help='实验重复次数')
    parser.add_argument('--des', type=str, default='experiment', 
                        help='实验描述')
    
    args = parser.parse_args()
    
    # 后处理
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    # 解析神经记忆层位置
    if isinstance(args.neural_memory_layers, str):
        args.neural_memory_layers = tuple([int(x) for x in args.neural_memory_layers.split(',')])
    
    # 创建必要的目录
    Path(args.checkpoints).mkdir(parents=True, exist_ok=True)
    Path(args.result_path).mkdir(parents=True, exist_ok=True)
    if args.save_fig:
        Path(args.fig_path).mkdir(parents=True, exist_ok=True)
    
    return args


def print_args(args):
    """打印参数配置"""
    print('\n' + '='*70)
    print(' '*25 + 'Titans-PyTorch 参数配置')
    print('='*70)
    
    arg_dict = vars(args)
    categories = {
        '基础配置': ['task_name', 'is_training', 'model_id', 'seed'],
        '数据配置': ['data', 'root_path', 'data_path', 'features', 'target'],
        '时间序列配置': ['seq_len', 'label_len', 'pred_len', 'input_dim', 'output_dim'],
        '模型配置': ['dim', 'depth', 'segment_len', 'dim_head', 'heads', 'dropout'],
        '记忆机制配置': ['num_persist_mem_tokens', 'num_longterm_mem_tokens', 
                      'neural_memory_layers', 'neural_memory_segment_len', 
                      'neural_memory_batch_size', 'memory_model_type'],
        '训练配置': ['train_epochs', 'batch_size', 'learning_rate', 'optimizer'],
        'GPU配置': ['use_gpu', 'gpu', 'use_multi_gpu']
    }
    
    for category, keys in categories.items():
        print(f'\n【{category}】')
        for key in keys:
            if key in arg_dict:
                print(f'  {key:30s} : {arg_dict[key]}')
    
    print('='*70 + '\n')


def main():
    # 解析参数
    args = get_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 打印参数
    print_args(args)
    
    # 导入训练器
    from utils.trainer import Trainer
    
    # 创建训练器
    trainer = Trainer(args)
    
    # 训练或测试
    if args.is_training:
        print(f'>>> 开始训练: {args.model_id} <<<')
        trainer.train()
        print(f'>>> 开始测试: {args.model_id} <<<')
        trainer.test()
    else:
        print(f'>>> 仅测试模式: {args.model_id} <<<')
        trainer.test()
    
    print('\n实验完成！')


if __name__ == '__main__':
    main()

