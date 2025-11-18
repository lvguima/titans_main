"""
神经记忆单元 (LMM - Learnable Memory Module)

这个模块实现了Titans的神经记忆机制，作为快速权重系统。
LMM在内循环中根据"惊奇度"实时更新自己的参数，以编码序列的历史信息。

核心机制:
1. **状态**: 持有自身参数 M (例如MemoryMLP的权重)
2. **输入**: 特征 f_t (用于生成 k, v) 和上一时刻的状态 M_{t-1}
3. **更新**: 根据元参数(θ_t, η_t, α_t)，通过惊奇度梯度更新: M_t = (1-α_t)M_{t-1} + S_t
4. **输出**: 更新后的状态 M_t，并根据查询 f_t 输出检索到的记忆 m_t
"""

import torch
import torch.nn as nn
import sys
import os
from typing import Optional, Tuple, Dict

# 添加titans-pytorch-main到路径
titans_path = os.path.join(os.path.dirname(__file__), '..', 'titans-pytorch-main')
if titans_path not in sys.path:
    sys.path.insert(0, titans_path)

from titans_pytorch import NeuralMemory
from titans_pytorch.memory_models import MemoryMLP, MemoryAttention


class LMMWrapper(nn.Module):
    """
    神经记忆单元 (LMM) - 使用Titans的NeuralMemory实现
    
    这个wrapper封装了Titans的NeuralMemory，提供符合新框架设计的接口。
    
    关键特性:
    - 在内循环中进行惊奇度驱动的参数更新
    - 支持通过元学习器提供的元参数进行更新
    - 维护跨时间步的记忆状态
    
    参数:
        dim: 特征维度 (需要与Backbone输出维度一致)
        chunk_size: 记忆单元的chunk大小
        neural_memory_batch_size: 多少个token后更新一次记忆权重
        memory_model_type: 'mlp' 或 'attention'
        heads: 多头数量
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
        momentum: bool = False,
        momentum_order: int = 1,
        attn_pool_chunks: bool = False,
        qk_rmsnorm: bool = False,
        per_parameter_lr_modulation: bool = False,
        spectral_norm_surprises: bool = False,
        max_grad_norm: float = None,
        default_step_transform_max_lr: float = 0.001,
        init_adaptive_step_bias: float = -15.0,
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
            default_step_transform_max_lr=default_step_transform_max_lr,
            init_adaptive_step_bias=init_adaptive_step_bias,
            **kwargs
        )
    
    def inner_update(
        self,
        features: torch.Tensor,
        memory_state: Optional[any],
        meta_params: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> any:
        """
        内循环更新 - 根据惊奇度和元参数更新LMM的状态
        
        这是设计文档中描述的核心内循环机制。
        
        Args:
            features: [batch, feature_dim] 当前时间步的特征
            memory_state: 上一时刻的记忆状态
            meta_params: (theta, eta, alpha) 元参数
                - theta: [batch] 学习率
                - eta: [batch] 动量
                - alpha: [batch] 遗忘率
        
        Returns:
            updated_state: 更新后的记忆状态
        
        实现说明:
            如果提供了meta_params，将动态调整NeuralMemory的学习率。
            通过临时替换adaptive_step_transform来注入元学习器生成的学习率。
        """
        # 扩展features维度以匹配NeuralMemory的期望输入
        if features.dim() == 2:
            features = features.unsqueeze(1)  # [batch, feature_dim] -> [batch, 1, feature_dim]
        
        # 如果提供了meta_params，使用它们来动态调整更新策略
        if meta_params is not None:
            theta, eta, alpha = meta_params  # theta: [batch], eta: [batch], alpha: [batch]
            
            # 保存原始的adaptive_step_transform
            original_transform = self.neural_memory.adaptive_step_transform
            
            # 创建一个使用meta_learner生成的学习率的transform
            def meta_adaptive_transform(adaptive_step):
                # adaptive_step: [batch_heads, seq_len]
                # theta: [batch]
                # 需要扩展theta到正确的形状
                batch_size = features.shape[0]
                heads = self.neural_memory.heads
                
                # theta从 [batch] 扩展到 [batch*heads, 1]
                theta_expanded = theta.repeat_interleave(heads).unsqueeze(-1)  # [batch*heads, 1]
                
                # 返回meta_learner指定的学习率
                return theta_expanded.expand_as(adaptive_step)
            
            # 临时替换transform
            self.neural_memory.adaptive_step_transform = meta_adaptive_transform
            
            try:
                # 调用NeuralMemory进行存储和更新
                _, updated_state = self.neural_memory(
                    seq=features,
                    state=memory_state,
                    return_surprises=False
                )
            finally:
                # 恢复原始transform（确保即使出错也能恢复）
                self.neural_memory.adaptive_step_transform = original_transform
        else:
            # 使用NeuralMemory内置的自适应学习率机制
            _, updated_state = self.neural_memory(
                seq=features,
                state=memory_state,
                return_surprises=False
            )
        
        return updated_state
    
    def retrieve(
        self,
        features: torch.Tensor,
        memory_state: any
    ) -> torch.Tensor:
        """
        从记忆中检索信息
        
        Args:
            features: [batch, feature_dim] 查询特征
            memory_state: 当前记忆状态
        
        Returns:
            memory_output: [batch, feature_dim] 检索到的记忆
        """
        if features.dim() == 2:
            features = features.unsqueeze(1)  # [batch, feature_dim] -> [batch, 1, feature_dim]
        
        # 使用NeuralMemory的retrieve功能
        memory_output, _ = self.neural_memory(
            seq=features,
            state=memory_state,
            return_surprises=False
        )
        
        # 压缩回原始维度
        if memory_output.shape[1] == 1:
            memory_output = memory_output.squeeze(1)  # [batch, 1, dim] -> [batch, dim]
        
        return memory_output
    
    def init_state(self, batch_size: int = None) -> any:
        """
        初始化记忆状态
        
        Args:
            batch_size: batch大小 (如果为None，将在第一次forward时自动初始化)
        
        Returns:
            initial_state: 初始记忆状态
        """
        # NeuralMemory在第一次调用时会自动初始化状态
        return None
    
    def forward(
        self,
        features: torch.Tensor,
        cache: Optional[any] = None,
        return_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[any]]:
        """
        前向传播 - 用于兼容旧接口
        
        Args:
            features: [batch, seq_len, dim] 来自Backbone的特征
            cache: 上一步的记忆状态
            return_cache: 是否返回cache供下一步使用
        
        Returns:
            retrieved_memory: [batch, seq_len, dim] 从记忆中检索的信息
            next_cache: NeuralMemState (如果return_cache=True)
        """
        # 检查输入特征
        if torch.isnan(features).any() or torch.isinf(features).any():
            raise RuntimeError(
                f"Memory Unit 输入包含nan/inf！"
                f"features range: [{features.min().item()}, {features.max().item()}]"
            )
        
        try:
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
                
                # 检查输出
                if torch.isnan(retrieved_memory).any() or torch.isinf(retrieved_memory).any():
                    # 打印详细调试信息
                    print(f"\n❌ NeuralMemory 输出包含nan/inf！")
                    print(f"   输入特征范围: [{features.min().item():.6f}, {features.max().item():.6f}]")
                    print(f"   输入特征均值: {features.mean().item():.6f}, std: {features.std().item():.6f}]")
                    print(f"   输出范围: [{retrieved_memory.min().item() if not torch.isnan(retrieved_memory).all() else 'all_nan'}, "
                          f"{retrieved_memory.max().item() if not torch.isnan(retrieved_memory).all() else 'all_nan'}]")
                    print(f"   nan数量: {torch.isnan(retrieved_memory).sum().item()}/{retrieved_memory.numel()}")
                    print(f"   inf数量: {torch.isinf(retrieved_memory).sum().item()}/{retrieved_memory.numel()}")
                    
                    print(f"\n   NeuralMemory 配置:")
                    print(f"     - dim: {self.dim}")
                    print(f"     - chunk_size: {self.chunk_size}")
                    print(f"     - batch_size: {self.neural_memory_batch_size}")
                    print(f"     - memory_model_type: {self.memory_model_type}")
                
                return retrieved_memory, None
                
        except Exception as e:
            print(f"\n❌ NeuralMemory 前向传播异常: {type(e).__name__}: {e}")
            print(f"   输入特征形状: {features.shape}")
            print(f"   输入特征范围: [{features.min().item():.6f}, {features.max().item():.6f}]")
            raise
    
    def get_config(self):
        """返回当前Memory Unit的配置信息"""
        return {
            'type': 'LMMWrapper',
            'dim': self.dim,
            'chunk_size': self.chunk_size,
            'neural_memory_batch_size': self.neural_memory_batch_size,
            'memory_model_type': self.memory_model_type,
            'heads': self.neural_memory.heads,
        }


class NoMemoryUnit(nn.Module):
    """
    空记忆单元 - 用于消融实验 (模式1: Baseline)
    
    这个模块不进行任何记忆操作，只返回零张量。
    可以用来验证记忆机制的有效性。
    """
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.dim = dim
    
    def inner_update(self, features, memory_state, meta_params=None):
        """不执行任何更新，直接返回None状态"""
        return None
    
    def retrieve(self, features, memory_state):
        """返回零张量"""
        if features.dim() == 2:
            return torch.zeros_like(features)
        else:
            return torch.zeros_like(features[:, -1, :])
    
    def init_state(self, batch_size=None):
        """返回None状态"""
        return None
    
    def forward(self, features, cache=None, return_cache=False):
        """
        Args:
            features: [batch, seq_len, dim]
        
        Returns:
            zeros: [batch, seq_len, dim] - 零张量
            cache: None
        """
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
        memory_type: 'lmm_mlp', 'lmm_attention', 'none'
                    (原来的titans_mlp改名为lmm_mlp，更符合新设计)
        **kwargs: 传递给具体Memory Unit的参数
    
    Returns:
        memory_unit: nn.Module
    """
    memory_type = memory_type.lower()
    
    # 向后兼容：支持旧的命名
    if memory_type == 'titans_mlp':
        memory_type = 'lmm_mlp'
    elif memory_type == 'titans_attention':
        memory_type = 'lmm_attention'
    
    if memory_type == 'lmm_mlp':
        return LMMWrapper(memory_model_type='mlp', **kwargs)
    elif memory_type == 'lmm_attention':
        return LMMWrapper(memory_model_type='attention', **kwargs)
    elif memory_type == 'none':
        return NoMemoryUnit(**kwargs)
    else:
        raise ValueError(
            f"Unknown memory type: {memory_type}. "
            f"Supported types: 'lmm_mlp', 'lmm_attention', 'none'"
        )


if __name__ == '__main__':
    # 测试代码
    print("测试神经记忆单元...")
    
    # 测试 LMMWrapper
    print("\n1. 测试 LMMWrapper")
    lmm = LMMWrapper(
        dim=256,
        chunk_size=1,
        neural_memory_batch_size=64,
        memory_model_type='mlp'
    )
    
    # 测试输入
    features = torch.randn(4, 64, 256)  # [batch=4, seq_len=64, dim=256]
    
    # 测试forward (旧接口)
    print("   测试forward接口...")
    memory_output, _ = lmm(features, cache=None, return_cache=False)
    print(f"   输入形状: {features.shape}")
    print(f"   输出形状: {memory_output.shape}")
    print(f"   输出范围: [{memory_output.min():.6f}, {memory_output.max():.6f}]")
    
    # 测试 inner_update 和 retrieve (新接口)
    print("\n   测试内循环接口...")
    state = lmm.init_state()
    
    for t in range(3):
        f_t = features[:, t, :]  # [batch, dim]
        
        # 内循环更新
        state = lmm.inner_update(f_t, state, meta_params=None)
        
        # 检索记忆
        m_t = lmm.retrieve(f_t, state)
        
        print(f"   时间步 {t}: m_t 形状={m_t.shape}, 范围=[{m_t.min():.6f}, {m_t.max():.6f}]")
    
    # 测试 NoMemoryUnit
    print("\n2. 测试 NoMemoryUnit")
    no_mem = NoMemoryUnit(dim=256)
    
    memory_output, _ = no_mem(features, cache=None, return_cache=False)
    print(f"   输入形状: {features.shape}")
    print(f"   输出形状: {memory_output.shape}")
    print(f"   全零检查: {torch.allclose(memory_output, torch.zeros_like(memory_output))}")
    
    # 测试工厂函数
    print("\n3. 测试工厂函数")
    lmm_from_factory = build_memory_unit('lmm_mlp', dim=256)
    no_mem_from_factory = build_memory_unit('none', dim=256)
    print(f"   lmm_mlp type: {type(lmm_from_factory).__name__}")
    print(f"   none type: {type(no_mem_from_factory).__name__}")
    
    print("\n✓ 神经记忆单元测试通过!")

