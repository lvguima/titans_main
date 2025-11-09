"""
Memory Units for Continual Learning

这个模块定义了不同的记忆单元(Memory Unit, M)
每个Memory Unit负责在线学习和适应数据漂移
"""

import torch
import torch.nn as nn
import sys
import os

# 添加titans-pytorch-original到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'titans-pytorch-original'))

from titans_pytorch import NeuralMemory
from titans_pytorch.memory_models import MemoryMLP, MemoryAttention


class TitansMemoryWrapper(nn.Module):
    """
    封装Titans的NeuralMemory模块
    
    这个Wrapper提供统一的接口，使得NeuralMemory可以与不同的Backbone协同工作
    
    参数:
        dim: 特征维度（需要与Backbone输出维度一致）
        chunk_size: 记忆单元的chunk大小
        neural_memory_batch_size: 多少个token后更新一次记忆权重
        memory_model_type: 'mlp' 或 'attention'
        其他NeuralMemory的参数
    """
    def __init__(
        self,
        dim: int,
        chunk_size: int = 1,
        neural_memory_batch_size: int = 256,
        memory_model_type: str = 'mlp',
        heads: int = 1,
        dim_head: int = None,
        mlp_depth: int = 2,
        mlp_expansion_factor: float = 4.0,
        momentum: bool = True,
        momentum_order: int = 1,
        attn_pool_chunks: bool = False,
        qk_rmsnorm: bool = False,
        per_parameter_lr_modulation: bool = False,
        spectral_norm_surprises: bool = False,
        max_grad_norm: float = None,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size
        self.neural_memory_batch_size = neural_memory_batch_size
        self.memory_model_type = memory_model_type
        
        # 构建memory model
        if memory_model_type.lower() == 'mlp':
            memory_model = MemoryMLP(
                dim=dim_head or dim,
                depth=mlp_depth,
                expansion_factor=mlp_expansion_factor
            )
        elif memory_model_type.lower() == 'attention':
            memory_model = MemoryAttention(
                dim=dim_head or dim,
                expansion_factor=mlp_expansion_factor
            )
        else:
            raise ValueError(f"Unknown memory_model_type: {memory_model_type}")
        
        # 创建NeuralMemory
        self.neural_memory = NeuralMemory(
            dim=dim,
            chunk_size=chunk_size,
            batch_size=neural_memory_batch_size,
            heads=heads,
            dim_head=dim_head,
            model=memory_model,
            momentum=momentum,
            momentum_order=momentum_order,
            attn_pool_chunks=attn_pool_chunks,
            qk_rmsnorm=qk_rmsnorm,
            per_parameter_lr_modulation=per_parameter_lr_modulation,
            spectral_norm_surprises=spectral_norm_surprises,
            max_grad_norm=max_grad_norm,
            **kwargs
        )
        
    def forward(self, features, cache=None, return_cache=False):
        """
        Args:
            features: [batch, seq_len, dim] - 来自Backbone的特征
            cache: NeuralMemState - 上一步的记忆状态
            return_cache: bool - 是否返回cache供下一步使用
            
        Returns:
            retrieved_memory: [batch, seq_len, dim] - 从记忆中检索的信息
            next_cache: NeuralMemState (如果return_cache=True)
        """
        if return_cache:
            # 持续学习模式：维护cache跨batch累积
            retrieved_memory, next_cache = self.neural_memory(
                seq=features,
                state=cache,
                return_surprises=False
            )
            return retrieved_memory, next_cache
        else:
            # 独立batch模式：每个batch独立处理
            retrieved_memory, _ = self.neural_memory(
                seq=features,
                state=None,
                return_surprises=False
            )
            return retrieved_memory, None
    
    def get_config(self):
        """返回当前Memory Unit的配置信息"""
        return {
            'type': 'TitansMemory',
            'dim': self.dim,
            'chunk_size': self.chunk_size,
            'neural_memory_batch_size': self.neural_memory_batch_size,
            'memory_model_type': self.memory_model_type,
            'heads': self.neural_memory.heads,
        }


class NoMemoryUnit(nn.Module):
    """
    空记忆单元 - 用于消融实验
    
    这个模块不进行任何记忆操作，只返回零张量
    可以用来验证记忆机制的有效性
    """
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.dim = dim
        
    def forward(self, features, cache=None, return_cache=False):
        """
        Args:
            features: [batch, seq_len, dim]
            
        Returns:
            zeros: [batch, seq_len, dim] - 零张量
            cache: None
        """
        batch, seq_len = features.shape[:2]
        zeros = torch.zeros_like(features)
        
        if return_cache:
            return zeros, None
        return zeros, None
    
    def get_config(self):
        return {
            'type': 'NoMemory',
            'dim': self.dim
        }


def build_memory_unit(memory_type: str, **kwargs):
    """
    工厂函数：根据类型构建Memory Unit
    
    Args:
        memory_type: 'titans_mlp', 'titans_attention', 'none'
        **kwargs: 传递给具体Memory Unit的参数
        
    Returns:
        memory_unit: nn.Module
    """
    memory_type = memory_type.lower()
    
    if memory_type == 'titans_mlp':
        return TitansMemoryWrapper(memory_model_type='mlp', **kwargs)
    elif memory_type == 'titans_attention':
        return TitansMemoryWrapper(memory_model_type='attention', **kwargs)
    elif memory_type == 'none':
        return NoMemoryUnit(**kwargs)
    else:
        raise ValueError(f"Unknown memory type: {memory_type}. "
                        f"Supported types: 'titans_mlp', 'titans_attention', 'none'")

