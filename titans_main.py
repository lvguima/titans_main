"""
Titans-PyTorch 主程序 (v2)

基于元学习的持续学习框架，实现"学习如何学习"的机制。

支持三种实验模式：
  模式1 (Baseline): 标准在线学习（无LMM）
                    使用 --memory_type none
  
  模式2 (Simple TTT): 带LMM，固定更新策略
                     使用 --memory_type lmm_mlp --use_meta_learning 0 --meta_learner_type fixed
  
  模式3 (Full Meta-TTT): 带LMM，元学习动态策略（终极目标）
                        使用 --memory_type lmm_mlp --use_meta_learning 1 --meta_learner_type adaptive

核心创新：
1. **内外双循环**: 内循环进行快速记忆更新，外循环优化元策略
2. **元学习器**: 动态生成LMM的更新策略（学习率、动量、遗忘率）
3. **模块化**: Backbone(P) + LMM + Meta-Learner 可独立替换
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
                        help='是否进行预训练（1=是，0=否）')
    parser.add_argument('--do_online_test', type=int, default=1,
                        help='是否进行在线测试（1=是，0=否）')
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
    parser.add_argument('--memory_type', type=str, default='lmm_mlp',
                        choices=['lmm_mlp', 'lmm_attention', 'titans_mlp', 'titans_attention', 'none'],
                        help='Memory Unit类型（lmm_mlp推荐，向后兼容titans_mlp）')
    parser.add_argument('--fusion_type', type=str, default='add',
                        choices=['add', 'concat', 'gated'],
                        help='特征融合方式')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader 的线程数')
    
    # ==================== 元学习配置 (新增) ====================
    parser.add_argument('--use_meta_learning', type=int, default=0,
                        help='是否启用元学习 (0=模式1/2, 1=模式3)')
    parser.add_argument('--meta_learner_type', type=str, default='fixed',
                        choices=['fixed', 'adaptive'],
                        help='元学习器类型: fixed=固定策略(模式2), adaptive=自适应策略(模式3)')
    parser.add_argument('--meta_learner_hidden_dim', type=int, default=128,
                        help='元学习器隐藏层维度')
    parser.add_argument('--fixed_theta', type=float, default=1e-3,
                        help='固定学习率（当meta_learner_type=fixed时）')
    parser.add_argument('--fixed_eta', type=float, default=0.9,
                        help='固定动量（当meta_learner_type=fixed时）')
    parser.add_argument('--fixed_alpha', type=float, default=0.1,
                        help='固定遗忘率（当meta_learner_type=fixed时）')
    # ==================== Backbone配置 ====================
    parser.add_argument('--d_model', type=int, default=256,
                        help='Backbone隐藏维度（8GB GPU建议256，16GB+可使用384）')
    parser.add_argument('--e_layers', type=int, default=3,
                        help='Backbone层数（8GB GPU建议3，16GB+可使用4）')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='注意力头数（Transformer/Titans，必须能整除d_model）')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout比率')
    
    # ==================== Memory Unit配置 ====================
    parser.add_argument('--memory_chunk_size', type=int, default=1,
                        help='Memory chunk size')
    parser.add_argument('--neural_memory_batch_size', type=int, default=512,
                        help='多少个token后更新一次记忆权重（增大此值可减少更新频率，提高稳定性。建议512-1024）')
    parser.add_argument('--memory_model_type', type=str, default='mlp',
                        choices=['mlp', 'attention'],
                        help='Memory内部模型类型')
    
    # ==================== 训练配置 ====================
    parser.add_argument('--train_epochs', type=int, default=5,
                        help='预训练轮数')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='批大小（8GB GPU建议2，如果仍不足可减小到1）')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='预训练学习率（NeuralMemory训练不稳定，建议使用较小学习率）')
    parser.add_argument('--patience', type=int, default=3,
                        help='早停patience')
    parser.add_argument('--clip_grad', type=float, default=0.5,
                        help='梯度裁剪阈值（NeuralMemory训练不稳定，建议使用较小值）')
    
    # ==================== 优化器配置 ====================
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd'],
                        help='优化器类型')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='权重衰减')
    parser.add_argument('--loss', type=str, default='mse',
                        choices=['mse', 'mae', 'huber'],
                        help='损失函数')
    parser.add_argument('--lradj', type=str, default='type1',
                        choices=['type1', 'type2', 'type3', 'constant'],
                        help='学习率调整策略: type1=每epoch衰减, type2=固定epoch衰减, type3=cosine, constant=不变')
    
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
    
    print("\n【元学习配置】")
    if args.memory_type == 'none':
        print(f"  模式: 模式1 (Baseline - 无Memory)")
    elif args.use_meta_learning == 0:
        print(f"  模式: 模式2 (Simple TTT - 固定策略)")
        print(f"  元学习器: {args.meta_learner_type}")
        if args.meta_learner_type == 'fixed':
            print(f"  固定学习率: {args.fixed_theta}")
            print(f"  固定动量: {args.fixed_eta}")
            print(f"  固定遗忘率: {args.fixed_alpha}")
    else:
        print(f"  模式: 模式3 (Full Meta-TTT - 自适应策略)")
        print(f"  元学习器: {args.meta_learner_type}")
        print(f"  元学习器隐藏维度: {args.meta_learner_hidden_dim}")
    
    print("\n【Memory配置】")
    print(f"  Memory模型类型: {args.memory_model_type}")
    print(f"  Memory chunk size: {args.memory_chunk_size}")
    print(f"  Neural memory batch size: {args.neural_memory_batch_size}")
    
    print("\n【训练配置】")
    print(f"  训练轮数: {args.train_epochs}")
    print(f"  批大小: {args.batch_size}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  优化器: {args.optimizer}")
    print(f"  损失函数: {args.loss}")
    print(f"  梯度裁剪: {args.clip_grad}")
    print(f"  早停patience: {args.patience}")
    
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
    """
    主函数
    
    运行示例：
    
    # 模式1: Baseline (无Memory)
    python titans_main.py --memory_type none --des baseline
    
    # 模式2: Simple TTT (固定策略)
    python titans_main.py --memory_type lmm_mlp --use_meta_learning 0 \
        --meta_learner_type fixed --des simple_ttt
    
    # 模式3: Full Meta-TTT (自适应策略)
    python titans_main.py --memory_type lmm_mlp --use_meta_learning 1 \
        --meta_learner_type adaptive --des full_meta_ttt
    """
    # 解析参数
    args = get_args()
    
    # 验证参数兼容性
    if args.d_model % args.n_heads != 0:
        print(f"\n❌ 错误: d_model ({args.d_model}) 必须能被 n_heads ({args.n_heads}) 整除！")
        print(f"   建议: 将 n_heads 改为 {args.d_model} 的因数（如 4, 8, 16, 32）")
        print(f"   或者: 将 d_model 改为能被 {args.n_heads} 整除的值")
        # 自动修复：将 n_heads 调整为最接近的能整除 d_model 的值
        import math
        # 找到 d_model 的最大因数，且 <= 原 n_heads 或接近原 n_heads
        factors = [i for i in range(1, args.d_model + 1) if args.d_model % i == 0]
        # 选择最接近原 n_heads 的因数
        best_n_heads = min(factors, key=lambda x: abs(x - args.n_heads))
        print(f"   自动修复: 将 n_heads 调整为 {best_n_heads}")
        args.n_heads = best_n_heads
    
    # 设置PyTorch内存管理（避免内存碎片化）
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 清理GPU缓存（避免内存碎片化）
    if torch.cuda.is_available():
        # 多次清理确保彻底
        for _ in range(3):
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 检查内存状态
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        
        print(f"\n✓ 已清理GPU缓存")
        print(f"✓ GPU内存管理: expandable_segments=True")
        print(f"✓ GPU内存状态: 已分配 {allocated:.2f} GB / 已保留 {reserved:.2f} GB")
        
        # 如果内存占用过高，给出警告
        if allocated > 1.0 or reserved > 2.0:
            print(f"\n⚠ 警告: GPU内存占用较高！")
            print(f"⚠ 已分配: {allocated:.2f} GB, 已保留: {reserved:.2f} GB")
            print(f"⚠ 必须重启Python进程以清理内存碎片！")
            print(f"⚠ 如果重启后仍溢出，使用: --batch_size 1 --neural_memory_batch_size 128")
            print(f"⚠ 或减小模型: --d_model 128 --e_layers 2")
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 打印配置
    print_args(args)
    
    # 创建保存目录
    Path(args.checkpoints).mkdir(parents=True, exist_ok=True)
    Path(args.result_path).mkdir(parents=True, exist_ok=True)
    Path(args.fig_path).mkdir(parents=True, exist_ok=True)
    
    # 导入训练器
    from utils.trainer import ContinualTrainer
    
    # 创建训练器
    trainer = ContinualTrainer(args)
    
    # ==================== 阶段1: 预训练 ====================
    if args.is_training:
        print("\n" + "="*70)
        print("阶段1: 预训练（同时训练Backbone和Memory Unit）")
        print("="*70)
        
        start_time = datetime.now()
        trainer.train()
        end_time = datetime.now()
        
        print(f"\n预训练完成！用时: {end_time - start_time}")
    else:
        print("\n" + "="*70)
        print("跳过预训练阶段（将加载已有checkpoint）")
        print("="*70)
    
    # ==================== 阶段2: 在线测试 ====================
    if args.do_online_test:
        freeze_backbone = (args.test_mode == 'memory_only')
        
        print("\n" + "="*70)
        if freeze_backbone:
            print("阶段2: 在线测试 - 模式A（仅Memory Unit学习，Backbone冻结）")
        else:
            print("阶段2: 在线测试 - 模式B（Memory Unit + Backbone都学习）")
        print("="*70)
        
        start_time = datetime.now()
        try:
            mae, mse, rmse, mape, mspe, rse = trainer.online_test(
                freeze_backbone=freeze_backbone,
                load_checkpoint=True  # 加载预训练的checkpoint
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
        except Exception as e:
            print(f"\n❌ 在线测试失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n" + "="*70)
        print("跳过在线测试阶段")
        print("="*70)


if __name__ == '__main__':
    main()

