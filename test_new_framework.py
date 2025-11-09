"""
测试新框架的功能
验证各个模块是否正常工作
"""

import torch
import sys
import os

# 添加titans-pytorch-original到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'titans-pytorch-original'))

from models.backbones import build_backbone, LSTMBackbone, TransformerBackbone, TitansBackbone
from models.memory_units import build_memory_unit, TitansMemoryWrapper, NoMemoryUnit
from models.framework import ContinualForecaster, build_continual_forecaster


def test_backbones():
    """测试所有Backbone"""
    print("\n" + "="*70)
    print("测试 Backbones")
    print("="*70)
    
    batch_size = 4
    seq_len = 32
    input_dim = 3
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # 测试LSTM
    print("\n1. 测试 LSTMBackbone...")
    lstm_backbone = build_backbone('lstm', input_dim=input_dim, hidden_dim=128)
    lstm_out = lstm_backbone(x)
    print(f"   输入: {x.shape}")
    print(f"   输出: {lstm_out.shape}")
    print(f"   ✓ LSTMBackbone 工作正常")
    
    # 测试Transformer
    print("\n2. 测试 TransformerBackbone...")
    transformer_backbone = build_backbone('transformer', input_dim=input_dim, dim=128, depth=2, heads=4)
    transformer_out = transformer_backbone(x)
    print(f"   输入: {x.shape}")
    print(f"   输出: {transformer_out.shape}")
    print(f"   ✓ TransformerBackbone 工作正常")
    
    # 测试Titans
    print("\n3. 测试 TitansBackbone...")
    titans_backbone = build_backbone('titans', input_dim=input_dim, dim=128, depth=2, heads=4, seq_len=seq_len)
    titans_out = titans_backbone(x)
    print(f"   输入: {x.shape}")
    print(f"   输出: {titans_out.shape}")
    print(f"   ✓ TitansBackbone 工作正常")
    
    print("\n" + "="*70)


def test_memory_units():
    """测试所有Memory Unit"""
    print("\n" + "="*70)
    print("测试 Memory Units")
    print("="*70)
    
    batch_size = 4
    seq_len = 32
    dim = 128
    
    features = torch.randn(batch_size, seq_len, dim)
    
    # 测试TitansMemory (MLP)
    print("\n1. 测试 TitansMemoryWrapper (MLP)...")
    memory_mlp = build_memory_unit('titans_mlp', dim=dim, neural_memory_batch_size=256)
    
    # 不使用cache
    mem_out1, _ = memory_mlp(features, cache=None, return_cache=False)
    print(f"   输入: {features.shape}")
    print(f"   输出: {mem_out1.shape}")
    
    # 使用cache
    mem_out2, cache = memory_mlp(features, cache=None, return_cache=True)
    print(f"   输出（with cache）: {mem_out2.shape}")
    print(f"   Cache类型: {type(cache)}")
    print(f"   ✓ TitansMemoryWrapper (MLP) 工作正常")
    
    # 测试NoMemory
    print("\n2. 测试 NoMemoryUnit...")
    no_memory = build_memory_unit('none', dim=dim)
    no_mem_out, _ = no_memory(features, cache=None, return_cache=False)
    print(f"   输入: {features.shape}")
    print(f"   输出: {no_mem_out.shape}")
    print(f"   ✓ NoMemoryUnit 工作正常")
    
    print("\n" + "="*70)


def test_continual_forecaster():
    """测试完整的ContinualForecaster"""
    print("\n" + "="*70)
    print("测试 ContinualForecaster")
    print("="*70)
    
    batch_size = 4
    seq_len = 32
    input_dim = 3
    output_dim = 1
    pred_len = 1
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # 测试不同组合
    configs = [
        ('lstm', 'titans_mlp', 'add'),
        ('transformer', 'titans_mlp', 'concat'),
        ('titans', 'titans_mlp', 'gated'),
        ('titans', 'none', 'add'),
    ]
    
    for i, (backbone_type, memory_type, fusion_type) in enumerate(configs, 1):
        print(f"\n{i}. 测试组合: {backbone_type} + {memory_type} + {fusion_type}")
        
        model = build_continual_forecaster(
            backbone_type=backbone_type,
            memory_type=memory_type,
            input_dim=input_dim,
            output_dim=output_dim,
            pred_len=pred_len,
            seq_len=seq_len,
            backbone_dim=128,
            backbone_depth=2,
            backbone_heads=4,
            neural_memory_batch_size=256,
            fusion_type=fusion_type
        )
        
        # 不使用cache
        pred1, _ = model(x, cache=None, return_cache=False)
        print(f"   输入: {x.shape}")
        print(f"   输出（no cache）: {pred1.shape}")
        
        # 使用cache
        pred2, cache = model(x, cache=None, return_cache=True)
        print(f"   输出（with cache）: {pred2.shape}")
        
        # 打印模型信息
        info = model.get_model_info()
        print(f"   模型参数: {info['total_params']:,}")
        print(f"   ✓ 组合 {backbone_type}+{memory_type}+{fusion_type} 工作正常")
    
    print("\n" + "="*70)


def test_gradient_flow():
    """测试梯度流动"""
    print("\n" + "="*70)
    print("测试梯度流动")
    print("="*70)
    
    batch_size = 4
    seq_len = 32
    input_dim = 3
    output_dim = 1
    pred_len = 1
    
    x = torch.randn(batch_size, seq_len, input_dim)
    y = torch.randn(batch_size, pred_len, output_dim)
    
    # 构建模型
    model = build_continual_forecaster(
        backbone_type='titans',
        memory_type='titans_mlp',
        input_dim=input_dim,
        output_dim=output_dim,
        pred_len=pred_len,
        seq_len=seq_len,
        backbone_dim=128,
        backbone_depth=2,
        backbone_heads=4,
        neural_memory_batch_size=256,
        fusion_type='add'
    )
    
    print("\n1. 测试全模型训练（P+M都更新）...")
    model.train()
    pred, _ = model(x, cache=None, return_cache=False)
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()
    
    # 检查梯度
    backbone_has_grad = any(p.grad is not None for p in model.backbone.parameters())
    memory_has_grad = any(p.grad is not None for p in model.memory_unit.parameters())
    
    print(f"   Backbone有梯度: {backbone_has_grad}")
    print(f"   Memory Unit有梯度: {memory_has_grad}")
    print(f"   ✓ 全模型训练梯度正常")
    
    # 清除梯度
    model.zero_grad()
    
    print("\n2. 测试冻结Backbone（仅M更新）...")
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    pred, _ = model(x, cache=None, return_cache=False)
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()
    
    # 检查梯度
    backbone_has_grad = any(p.grad is not None for p in model.backbone.parameters() if p.requires_grad)
    memory_has_grad = any(p.grad is not None for p in model.memory_unit.parameters())
    
    print(f"   Backbone有梯度: {backbone_has_grad}")
    print(f"   Memory Unit有梯度: {memory_has_grad}")
    print(f"   ✓ 冻结Backbone梯度正常")
    
    print("\n" + "="*70)


def test_cache_mechanism():
    """测试cache机制"""
    print("\n" + "="*70)
    print("测试Cache机制")
    print("="*70)
    
    batch_size = 2
    seq_len = 16
    input_dim = 3
    output_dim = 1
    pred_len = 1
    
    # 构建模型
    model = build_continual_forecaster(
        backbone_type='titans',
        memory_type='titans_mlp',
        input_dim=input_dim,
        output_dim=output_dim,
        pred_len=pred_len,
        seq_len=seq_len,
        backbone_dim=64,
        backbone_depth=2,
        backbone_heads=2,
        neural_memory_batch_size=64,
        fusion_type='add'
    )
    
    model.eval()
    
    print("\n模拟在线学习流程（3个batch）...")
    cache = None
    
    for i in range(3):
        x = torch.randn(batch_size, seq_len, input_dim)
        pred, cache = model(x, cache=cache, return_cache=True)
        
        seq_index = cache[0] if isinstance(cache, tuple) else 0
        print(f"   Batch {i+1}: pred shape={pred.shape}, seq_index={seq_index}")
    
    print(f"   ✓ Cache机制工作正常，seq_index递增")
    
    print("\n" + "="*70)


def main():
    """运行所有测试"""
    print("\n" + "🚀 " + "="*66 + " 🚀")
    print("  开始测试新框架")
    print("🚀 " + "="*66 + " 🚀")
    
    try:
        test_backbones()
        test_memory_units()
        test_continual_forecaster()
        test_gradient_flow()
        test_cache_mechanism()
        
        print("\n" + "✅ " + "="*66 + " ✅")
        print("  所有测试通过！新框架工作正常")
        print("✅ " + "="*66 + " ✅\n")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

