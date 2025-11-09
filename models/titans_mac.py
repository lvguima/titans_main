"""
Titans MAC (Memory-As-Context) 时间序列预测模型
基于titans_pytorch封装的自定义版本，适配时间序列预测任务
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# 添加父目录到路径以导入titans_pytorch
parent_dir = Path(__file__).parent.parent
titans_pytorch_path = parent_dir / 'titans-pytorch-original'
sys.path.insert(0, str(titans_pytorch_path))

from titans_pytorch import MemoryAsContextTransformer, MemoryAttention


class TimeSeriesTitansTransformer(nn.Module):
    """
    自定义时间序列Titans Transformer
    将连续时间序列输入适配到MAC Transformer架构
    """
    
    def __init__(
        self, 
        input_dim,
        output_dim,
        pred_len,
        dim,
        depth,
        segment_len,
        num_persist_mem_tokens=0,
        num_longterm_mem_tokens=0,
        neural_memory_layers=None,
        neural_memory_segment_len=None,
        neural_memory_batch_size=None,
        neural_mem_weight_residual=False,
        neural_mem_gate_attn_output=False,
        use_flex_attn=False,
        sliding_window_attn=False,
        dim_head=64,
        heads=8,
        neural_memory_model=None,
        neural_memory_kwargs=None,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pred_len = pred_len
        self.dim = dim
        
        # 1. 输入投影层（在transformer外部处理）
        self.input_projection = nn.Linear(input_dim, dim)
        
        # 2. 创建特殊的token_emb来处理已投影的数据
        # 原始库期望输入是2D的[batch, seq_len]，然后token_emb转为3D
        # 我们的策略：在forward中将3D投影后的数据flatten成2D传入，
        # 然后在这个特殊的embedding中reshape回3D
        class ReshapeEmbedding(nn.Module):
            """
            特殊的embedding，用于处理已经投影好的3D数据
            输入：[batch*seq_len, dim] (被flatten的3D数据)
            输出：[batch, seq_len, dim] (恢复3D形状)
            """
            def __init__(self):
                super().__init__()
                self.stored_shape = None  # 用于存储原始形状
            
            def forward(self, x):
                # x: [batch, seq_len] 但实际是被reshape的 [batch*seq_len, dim]的第一维
                # 我们需要从stored_shape恢复
                if self.stored_shape is not None:
                    batch, seq_len, dim = self.stored_shape
                    # x这里其实是个dummy，我们直接返回stored_data
                    result = self.stored_data
                    self.stored_shape = None  # 清空
                    return result
                else:
                    # 不应该到这里
                    raise RuntimeError("ReshapeEmbedding: stored_shape not set")
        
        self.reshape_emb = ReshapeEmbedding()
        
        self.transformer = MemoryAsContextTransformer(
            num_tokens=dim,  # 设为dim，这样to_logits输出[batch, seq_len, dim]而不是[batch, seq_len, 1]
            dim=dim,
            depth=depth,
            segment_len=segment_len,
            num_persist_mem_tokens=num_persist_mem_tokens,
            num_longterm_mem_tokens=num_longterm_mem_tokens,
            neural_memory_layers=neural_memory_layers,
            neural_memory_segment_len=neural_memory_segment_len,
            neural_memory_batch_size=neural_memory_batch_size,
            neural_mem_weight_residual=neural_mem_weight_residual,
            neural_mem_gate_attn_output=neural_mem_gate_attn_output,
            use_flex_attn=use_flex_attn,
            sliding_window_attn=sliding_window_attn,
            dim_head=dim_head,
            heads=heads,
            neural_memory_model=neural_memory_model,
            neural_memory_kwargs=neural_memory_kwargs or {},
            token_emb=self.reshape_emb,  # 使用特殊的reshape embedding
        )
        
        # 3. 输出预测头：模型维度 → 预测值
        self.prediction_head = nn.Linear(dim, output_dim * pred_len)
    
    def forward(self, x, cache=None, return_cache=False):
        """
        Args:
            x: 输入序列 [batch, seq_len, input_dim] (连续特征)
            cache: NeuralMemory缓存状态 (seq_index, kv_caches, neural_mem_caches)
            return_cache: 是否返回更新后的cache
        
        Returns:
            如果return_cache=False: 预测值 [batch, pred_len, output_dim]
            如果return_cache=True: (预测值, 更新后的cache)
        """
        # Step 1: 投影连续特征到模型维度
        # x: [batch, seq_len, input_dim] → [batch, seq_len, dim]
        batch, seq_len = x.shape[:2]
        x_proj = self.input_projection(x)  # [batch, seq_len, dim]
        
        # Step 2: 原始库期望2D输入[batch, seq_len]，我们需要"伪装"
        # 将投影后的3D数据存储到reshape_emb中，然后传入一个dummy的2D索引
        self.reshape_emb.stored_data = x_proj
        self.reshape_emb.stored_shape = (batch, seq_len, self.dim)
        
        # 创建dummy的2D输入（原始库会在Line 717解包这个的shape）
        x_dummy = torch.zeros(batch, seq_len, dtype=torch.long, device=x.device)
        
        # Step 3: MAC Transformer前向传播
        if return_cache:
            # 在线学习模式：需要维护cache状态
            logits, next_cache = self.transformer(
                x_dummy,  # 传入2D dummy，token_emb会从stored_data中取出真实的3D数据
                cache=cache, 
                return_cache=True,
                disable_flex_attn=True  # 时间序列通常不需要flex attention
            )
            
            # 处理longterm_mem token的特殊情况（原始库在某些位置返回None）
            if logits is None:
                # 当处理longterm_mem tokens时，直接返回cache
                return None, next_cache
        else:
            # 正常训练/推理模式
            logits = self.transformer(x_dummy, disable_flex_attn=True)
            next_cache = None
        
        # Step 3: 提取最后一个token的表示
        # logits shape: [batch, seq_len, dim]
        last_hidden = logits[:, -1, :]  # [batch, dim]
        
        # Step 4: 预测
        pred = self.prediction_head(last_hidden)  # [batch, output_dim * pred_len]
        pred = pred.view(-1, self.pred_len, self.output_dim)  # [batch, pred_len, output_dim]
        
        if return_cache:
            return pred, next_cache
        return pred


class TitansMAC(nn.Module):
    """
    Titans MAC模型包装器
    提供灵活的输入输出维度配置，适配不同的时间序列数据集
    """
    
    def __init__(self, args):
        """
        初始化Titans MAC模型
        
        Args:
            args: 参数对象，包含所有配置
        """
        super(TitansMAC, self).__init__()
        
        self.args = args
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.pred_len = args.pred_len
        
        # 创建神经记忆模型
        neural_memory_model = self._create_memory_model(args)
        
        # 补全neural_memory_kwargs（添加原始库中使用的重要参数）
        neural_memory_kwargs = {
            'dim_head': args.memory_dim_head,
            'heads': args.memory_heads,
            'momentum': args.memory_momentum,
            'momentum_order': args.memory_momentum_order,
            'default_step_transform_max_lr': args.memory_max_lr,
            'use_accelerated_scan': args.memory_use_accelerated_scan,
            # 🔑 新增：原始实验中证明有效的参数
            'attn_pool_chunks': True,  # 使用注意力池化chunk representations
            'qk_rmsnorm': True,  # QK归一化，稳定训练
            'per_parameter_lr_modulation': True,  # 每层学习率调制
            'spectral_norm_surprises': True,  # 梯度谱归一化（Muon优化器启发）
        }
        
        # 创建自定义时间序列Titans模型
        self.model = TimeSeriesTitansTransformer(
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            pred_len=args.pred_len,
            dim=args.dim,
            depth=args.depth,
            segment_len=args.segment_len,
            num_persist_mem_tokens=args.num_persist_mem_tokens,
            num_longterm_mem_tokens=args.num_longterm_mem_tokens,
            neural_memory_layers=args.neural_memory_layers,
            neural_memory_segment_len=args.neural_memory_segment_len,
            neural_memory_batch_size=args.neural_memory_batch_size,
            neural_mem_weight_residual=args.neural_mem_weight_residual,
            neural_mem_gate_attn_output=not args.use_mac_fusion,  # MAC模式下为False
            use_flex_attn=args.use_flex_attn,
            sliding_window_attn=args.sliding_window_attn,
            dim_head=args.dim_head,
            heads=args.heads,
            neural_memory_model=neural_memory_model,
            neural_memory_kwargs=neural_memory_kwargs,
        )
        
        # 打印模型信息
        self._print_model_info()
        
        # 🔍 调试：打印NeuralMemory层的配置
        self._debug_neural_memory()
    
    def _create_memory_model(self, args):
        """创建神经记忆模型"""
        from titans_pytorch import (
            MemoryAttention, 
            MemoryMLP, 
            FactorizedMemoryMLP, 
            MemorySwiGluMLP, 
            GatedResidualMemoryMLP
        )
        
        # 根据不同的记忆模型类型创建（每种模型的参数不同）
        if args.memory_model_type == 'attention':
            # MemoryAttention(dim, scale, expansion_factor)
            return MemoryAttention(
                dim=args.memory_dim,
                scale=args.memory_scale,
                expansion_factor=args.memory_expansion_factor
            )
        elif args.memory_model_type == 'mlp':
            # MemoryMLP(dim, depth, expansion_factor)
            return MemoryMLP(
                dim=args.memory_dim,
                depth=2,  # 默认2层
                expansion_factor=args.memory_expansion_factor
            )
        elif args.memory_model_type == 'factorized_mlp':
            # FactorizedMemoryMLP(dim, depth, expansion_factor)
            return FactorizedMemoryMLP(
                dim=args.memory_dim,
                depth=2,
                expansion_factor=args.memory_expansion_factor
            )
        elif args.memory_model_type == 'swiglu_mlp':
            # MemorySwiGluMLP(dim, depth=1, expansion_factor)
            return MemorySwiGluMLP(
                dim=args.memory_dim,
                depth=1,  # 默认1（SwiGLU论文推荐）
                expansion_factor=args.memory_expansion_factor
            )
        elif args.memory_model_type == 'gated_residual':
            # GatedResidualMemoryMLP(dim, depth, k=32)
            return GatedResidualMemoryMLP(
                dim=args.memory_dim,
                depth=2,
                k=32
            )
        else:
            # 默认使用MLP
            return MemoryMLP(
                dim=args.memory_dim,
                depth=2,
                expansion_factor=args.memory_expansion_factor
            )
    
    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f'\n{"="*70}')
        print(f"Titans MAC 模型信息:")
        print(f"{'='*70}")
        print(f"  输入维度: {self.input_dim}")
        print(f"  输出维度: {self.output_dim}")
        print(f"  预测长度: {self.pred_len}")
        print(f"  模型维度: {self.args.dim}")
        print(f"  层数: {self.args.depth}")
        print(f"  注意力头数: {self.args.heads}")
        print(f"  神经记忆层: {self.args.neural_memory_layers}")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"{'='*70}\n")
    
    def _debug_neural_memory(self):
        """调试：检查NeuralMemory是否被正确创建"""
        print("="*70)
        print("🔍 NeuralMemory调试信息:")
        print("="*70)
        
        # 检查transformer的layers
        has_neural_mem = False
        for i, layer_modules in enumerate(self.model.transformer.layers):
            mem = layer_modules[4]  # mem在第5个位置（index=4）
            if mem is not None:
                has_neural_mem = True
                batch_size = mem.batch_size if hasattr(mem, 'batch_size') else 'None'
                chunk_size = mem.chunk_size if hasattr(mem, 'chunk_size') else 'None'
                print(f"  Layer {i+1}: ✅ 有NeuralMemory")
                print(f"    - batch_size: {batch_size}")
                print(f"    - chunk_size: {chunk_size}")
                print(f"    - 记忆参数: {sum(p.numel() for p in mem.parameters()):,}")
        
        if not has_neural_mem:
            print("  ⚠️ 警告：没有找到任何NeuralMemory层！")
        
        print("="*70 + "\n")
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, cache=None, return_cache=False):
        """
        前向传播
        
        Args:
            x_enc: 输入序列 [batch_size, seq_len, input_dim]
            x_mark_enc: 时间特征（可选）
            x_dec: decoder输入（可选，某些架构需要）
            x_mark_dec: decoder时间特征（可选）
            cache: NeuralMemory缓存状态 (seq_index, kv_caches, neural_mem_caches)，用于在线学习
            return_cache: 是否返回更新后的cache状态
        
        Returns:
            如果return_cache=False: 输出预测 [batch_size, pred_len, output_dim]
            如果return_cache=True: (输出预测, 更新后的cache)
        """
        # 🔑 关键：将cache参数传递给底层TimeSeriesTitansTransformer
        if return_cache:
            # 在线学习模式：需要维护cache状态
            output, next_cache = self.model(x_enc, cache=cache, return_cache=True)
            
            # 调整输出维度
            if self.pred_len == 1 and output.dim() == 2:
                output = output.unsqueeze(1)  # [batch_size, 1, output_dim]
            
            return output, next_cache
        else:
            # 正常训练/推理：不使用cache
            output = self.model(x_enc)
            
            # 调整输出维度
            if self.pred_len == 1 and output.dim() == 2:
                output = output.unsqueeze(1)  # [batch_size, 1, output_dim]
            elif output.dim() == 2:
                output = output.unsqueeze(1)
            
            return output
    
    def freeze_non_memory_params(self):
        """
        冻结非记忆参数（backbone/预测结构），仅保留记忆参数可训练
        用于在线学习时只更新记忆单元
        """
        for name, param in self.named_parameters():
            if 'neural_memory' in name or 'mem' in name or 'longterm_mems' in name or 'persist_mem' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✓ 已冻结非记忆参数")
        print(f"  可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    def unfreeze_all(self):
        """解冻所有参数（backbone + 记忆）"""
        for param in self.parameters():
            param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✓ 已解冻所有参数: {trainable_params:,}")
    
    def get_memory_params(self):
        """获取记忆单元的参数（用于单独优化）"""
        return [p for name, p in self.named_parameters() 
                if ('neural_memory' in name or 'mem' in name) and p.requires_grad]
    
    def get_backbone_params(self):
        """获取backbone/预测结构的参数"""
        return [p for name, p in self.named_parameters() 
                if not ('neural_memory' in name or 'mem' in name) and p.requires_grad]


def build_model(args):
    """
    构建模型的工厂函数
    
    Args:
        args: 参数配置
    
    Returns:
        模型实例
    """
    model = TitansMAC(args)
    return model


if __name__ == '__main__':
    """测试模型"""
    import argparse
    
    # 创建测试参数
    args = argparse.Namespace()
    
    # 基础配置
    args.input_dim = 7
    args.output_dim = 7
    args.pred_len = 1
    
    # 模型配置
    args.dim = 256
    args.depth = 4
    args.segment_len = 16
    args.dim_head = 64
    args.heads = 4
    args.dropout = 0.1
    
    # 记忆配置
    args.num_persist_mem_tokens = 4
    args.num_longterm_mem_tokens = 4
    args.neural_memory_layers = (1, 3)
    args.neural_memory_segment_len = 8
    args.neural_memory_batch_size = 32
    args.neural_mem_weight_residual = True
    
    # 记忆模型配置
    args.memory_model_type = 'attention'
    args.memory_dim = 64
    args.memory_scale = 8.0
    args.memory_expansion_factor = 2
    args.memory_dim_head = 64
    args.memory_heads = 4
    args.memory_momentum = True
    args.memory_momentum_order = 2
    args.memory_max_lr = 0.0001
    args.memory_use_accelerated_scan = False
    
    # MAC配置
    args.use_mac_fusion = True
    args.use_flex_attn = False
    args.sliding_window_attn = False
    
    # 创建模型
    print("创建Titans MAC模型...")
    model = TitansMAC(args)
    
    # 测试前向传播
    batch_size = 8
    seq_len = 96
    x = torch.randn(batch_size, seq_len, args.input_dim)
    
    print(f"\n输入形状: {x.shape}")
    output = model(x)
    print(f"输出形状: {output.shape}")
    
    print("\n✓ 模型测试通过!")

