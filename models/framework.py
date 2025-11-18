"""
Continual Forecasting Framework (v2)

这个模块定义了基于元学习的持续学习预测核心框架。
它将Backbone(P)、Memory Unit(LMM)和Meta-Learner组合成一个完整的预测系统。

核心创新:
1. **内外双循环**: 内循环进行快速记忆更新，外循环优化元策略
2. **元学习**: Meta-Learner学习如何动态调整LMM的更新策略
3. **模块化**: 三大组件(P, LMM, Meta-Learner)可独立替换

支持三种实验模式:
- 模式1 (Baseline): 标准在线学习（无LMM）
- 模式2 (Simple TTT): 带LMM，固定更新策略
- 模式3 (Full Meta-TTT): 带LMM，元学习动态策略（终极目标）
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class ContinualForecaster(nn.Module):
    """
    持续学习时间序列预测器 (v2 - 元学习版本)
    
    架构流程:
        Input [batch, seq_len, input_dim]
          ↓
        Backbone P: 特征提取 → f_t
          ↓
        Meta-Learner: 生成元参数 (θ_t, η_t, α_t)
          ↓
        LMM 内循环: 根据元参数更新记忆 → M_t
          ↓
        Memory Retrieval: 检索记忆 → m_t
          ↓
        Feature Fusion: (f_t + m_t)
          ↓
        Prediction Head: 最终预测
          ↓
        Output [batch, pred_len, output_dim]
    
    参数:
        backbone: 预测主干网络 (P)
        memory_unit: 记忆单元 (LMM)
        meta_learner: 元学习器 (Meta-Learner) - 可选，如果为None则使用固定策略
        output_dim: 输出特征维度
        pred_len: 预测长度
        fusion_type: 特征融合方式 ('add', 'concat', 'gated')
        use_meta_learning: 是否启用元学习 (模式3)
    """
    def __init__(
        self,
        backbone: nn.Module,
        memory_unit: nn.Module,
        meta_learner: Optional[nn.Module] = None,
        output_dim: int = 1,
        pred_len: int = 1,
        fusion_type: str = 'add',
        dropout: float = 0.1,
        use_meta_learning: bool = False
    ):
        super().__init__()
        self.backbone = backbone
        self.memory_unit = memory_unit
        self.meta_learner = meta_learner
        self.output_dim = output_dim
        self.pred_len = pred_len
        self.fusion_type = fusion_type
        self.use_meta_learning = use_meta_learning
        
        # 获取backbone输出维度
        if hasattr(backbone, 'hidden_dim'):
            self.feature_dim = backbone.hidden_dim
        elif hasattr(backbone, 'dim'):
            self.feature_dim = backbone.dim
        else:
            raise ValueError("Backbone必须有'hidden_dim'或'dim'属性")
        
        # 特征融合层
        if fusion_type == 'add':
            # 直接相加，不需要额外参数
            self.fusion = None
            fused_dim = self.feature_dim
        elif fusion_type == 'concat':
            # 拼接后降维
            self.fusion = nn.Sequential(
                nn.Linear(self.feature_dim * 2, self.feature_dim),
                nn.LayerNorm(self.feature_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            fused_dim = self.feature_dim
        elif fusion_type == 'gated':
            # 门控融合
            self.fusion = nn.Sequential(
                nn.Linear(self.feature_dim * 2, self.feature_dim),
                nn.Sigmoid()
            )
            fused_dim = self.feature_dim
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        
        # 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, output_dim * pred_len)
        )
    
    def forward_with_inner_loop(
        self,
        sequence_x: torch.Tensor,
        cache: Optional[any] = None,
        return_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[any]]:
        """
        带内循环的前向传播 - 实现设计文档中的内外双循环机制
        
        内循环: 对序列中的每一步t执行记忆更新
        外循环: 通过反向传播优化Backbone和Meta-Learner
        
        Args:
            sequence_x: [batch, seq_len, input_dim] - 输入时间序列
            cache: 记忆状态cache (用于持续学习)
            return_cache: 是否返回下一步的cache
        
        Returns:
            pred: [batch, pred_len, output_dim] - 预测结果
            next_cache: 下一步的cache (如果return_cache=True)
        """
        batch_size, seq_len = sequence_x.shape[:2]
        
        # 0. 初始化LMM状态
        if cache is None:
            memory_state = self.memory_unit.init_state(batch_size)
        else:
            memory_state = cache
        
        # 🚀 性能优化：批量提取特征，避免重复调用Backbone
        # 原来：逐个token调用backbone（慢242倍）
        # 优化：一次性处理整个序列（加速10-50倍）
        
        # 1. 一次性提取所有特征（关键优化！）
        features_all = self.backbone(sequence_x)  # [batch, seq_len, feature_dim]
        
        # 2. 如果启用元学习，一次性生成所有元参数
        if self.use_meta_learning and self.meta_learner is not None:
            # Meta-Learner处理整个序列，输出每个时间步的元参数
            theta_all, eta_all, alpha_all = self.meta_learner(features_all)
            # theta_all: [batch, seq_len], eta_all: [batch, seq_len], alpha_all: [batch, seq_len]
        else:
            theta_all = eta_all = alpha_all = None
        
        # 存储每一步的记忆输出（用于最终预测）
        all_memory_outputs = []
        
        # ==== 内循环 (Inner Loop): 逐步更新LMM状态 ====
        # 注意：LMM必须逐步更新，因为状态是递归的
        for t in range(seq_len):
            # 3. 直接索引预计算的特征（无需再调用backbone）
            f_t = features_all[:, t, :]  # [batch, feature_dim]
            
            # 4. 索引当前时间步的元参数
            if self.use_meta_learning and self.meta_learner is not None:
                theta_t = theta_all[:, t]  # [batch]
                eta_t = eta_all[:, t]      # [batch]
                alpha_t = alpha_all[:, t]  # [batch]
                meta_params = (theta_t, eta_t, alpha_t)
            else:
                meta_params = None
            
            # 5. LMM 执行内循环更新
            memory_state = self.memory_unit.inner_update(
                f_t, memory_state, meta_params
            )
            
            # 6. 从更新后的LMM中检索记忆
            memory_output_t = self.memory_unit.retrieve(f_t, memory_state)  # [batch, feature_dim]
            
            # 收集记忆输出
            all_memory_outputs.append(memory_output_t)
        
        # 8. 拼接结果（features_all已经是正确形状，直接使用）
        features = features_all  # [batch, seq_len, feature_dim]
        memory_outputs = torch.stack(all_memory_outputs, dim=1)  # [batch, seq_len, feature_dim]
        
        # 检查是否有nan/inf
        if torch.isnan(features).any() or torch.isinf(features).any():
            raise RuntimeError("Backbone输出包含nan/inf！")
        if torch.isnan(memory_outputs).any() or torch.isinf(memory_outputs).any():
            raise RuntimeError("Memory输出包含nan/inf！")
        
        # 6. 特征融合
        if self.fusion_type == 'add':
            fused = features + memory_outputs
        elif self.fusion_type == 'concat':
            concat_features = torch.cat([features, memory_outputs], dim=-1)
            fused = self.fusion(concat_features)
        elif self.fusion_type == 'gated':
            concat_features = torch.cat([features, memory_outputs], dim=-1)
            gate = self.fusion(concat_features)
            fused = features * gate + memory_outputs * (1 - gate)
        
        # 检查融合后的特征
        if torch.isnan(fused).any() or torch.isinf(fused).any():
            raise RuntimeError("特征融合后包含nan/inf！")
        
        # 7. 使用最后一个时间步的特征进行预测
        last_feature = fused[:, -1, :]  # [batch, feature_dim]
        
        # 8. 预测头
        pred_flat = self.prediction_head(last_feature)  # [batch, output_dim * pred_len]
        pred = pred_flat.view(batch_size, self.pred_len, self.output_dim)
        
        # 检查最终预测
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            raise RuntimeError("预测输出包含nan/inf！")
        
        if return_cache:
            return pred, memory_state
        return pred, None
    
    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[any] = None,
        return_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[any]]:
        """
        前向传播 - 自动选择是否使用内循环
        
        Args:
            x: [batch, seq_len, input_dim] - 输入时间序列
            cache: 记忆状态cache (用于持续学习)
            return_cache: 是否返回下一步的cache
        
        Returns:
            pred: [batch, pred_len, output_dim] - 预测结果
            next_cache: 下一步的cache (如果return_cache=True)
        """
        # 检查是否使用NoMemoryUnit (模式1: Baseline)
        is_no_memory = self.memory_unit.__class__.__name__ == 'NoMemoryUnit'
        
        if is_no_memory or not self.use_meta_learning:
            # 模式1或模式2: 使用简化的forward (兼容旧实现)
            return self._forward_simple(x, cache, return_cache)
        else:
            # 模式3: 使用带内循环的forward
            return self.forward_with_inner_loop(x, cache, return_cache)
    
    def _forward_simple(
        self,
        x: torch.Tensor,
        cache: Optional[any] = None,
        return_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[any]]:
        """
        简化的前向传播 - 用于模式1和模式2
        
        这个方法与原来的实现兼容，不使用逐步的内循环。
        """
        batch_size, seq_len = x.shape[:2]
        
        # 1. Backbone提取特征
        features = self.backbone(x)  # [batch, seq_len, feature_dim]
        
        # 检查backbone输出
        if torch.isnan(features).any() or torch.isinf(features).any():
            raise RuntimeError("Backbone输出包含nan/inf！")
        
        # 2. Memory Unit处理
        try:
            if return_cache:
                memory_output, next_cache = self.memory_unit(
                    features,
                    cache=cache,
                    return_cache=True
                )
            else:
                memory_output, _ = self.memory_unit(
                    features,
                    cache=None,
                    return_cache=False
                )
                next_cache = None
        except RuntimeError as e:
            if "nan" in str(e).lower() or "inf" in str(e).lower():
                raise RuntimeError(f"Memory Unit产生nan/inf: {e}")
            raise
        
        # 检查memory输出
        if memory_output is None:
            raise RuntimeError("Memory Unit输出为None！")
        if torch.isnan(memory_output).any() or torch.isinf(memory_output).any():
            raise RuntimeError("Memory Unit输出包含nan/inf！")
        
        # 3. 特征融合
        if self.fusion_type == 'add':
            fused = features + memory_output
        elif self.fusion_type == 'concat':
            concat_features = torch.cat([features, memory_output], dim=-1)
            fused = self.fusion(concat_features)
        elif self.fusion_type == 'gated':
            concat_features = torch.cat([features, memory_output], dim=-1)
            gate = self.fusion(concat_features)
            fused = features * gate + memory_output * (1 - gate)
        
        # 检查融合后的特征
        if torch.isnan(fused).any() or torch.isinf(fused).any():
            raise RuntimeError("特征融合后包含nan/inf！")
        
        # 4. 使用最后一个时间步的特征进行预测
        last_feature = fused[:, -1, :]  # [batch, feature_dim]
        
        # 5. 预测头
        pred_flat = self.prediction_head(last_feature)  # [batch, output_dim * pred_len]
        pred = pred_flat.view(batch_size, self.pred_len, self.output_dim)
        
        # 检查最终预测
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            raise RuntimeError("预测输出包含nan/inf！")
        
        if return_cache:
            return pred, next_cache
        return pred, None
    
    def get_model_info(self):
        """返回模型的详细信息"""
        info = {
            'backbone': self.backbone.__class__.__name__,
            'memory_unit': self.memory_unit.__class__.__name__,
            'meta_learner': self.meta_learner.__class__.__name__ if self.meta_learner else 'None',
            'feature_dim': self.feature_dim,
            'output_dim': self.output_dim,
            'pred_len': self.pred_len,
            'fusion_type': self.fusion_type,
            'use_meta_learning': self.use_meta_learning,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
        
        # 添加各组件的配置
        if hasattr(self.memory_unit, 'get_config'):
            info['memory_config'] = self.memory_unit.get_config()
        
        if self.meta_learner and hasattr(self.meta_learner, 'get_config'):
            info['meta_learner_config'] = self.meta_learner.get_config()
        
        return info


def build_continual_forecaster(
    backbone_type: str,
    memory_type: str,
    input_dim: int,
    output_dim: int,
    pred_len: int,
    seq_len: int = 64,
    backbone_dim: int = 384,
    backbone_depth: int = 4,
    backbone_heads: int = 6,
    neural_memory_batch_size: int = 256,
    memory_chunk_size: int = 1,
    memory_model_type: str = 'mlp',
    fusion_type: str = 'add',
    # 元学习相关参数
    use_meta_learning: bool = False,
    meta_learner_type: str = 'fixed',  # 'adaptive' or 'fixed'
    meta_learner_hidden_dim: int = 128,
    **kwargs
) -> ContinualForecaster:
    """
    工厂函数：构建完整的持续学习预测器
    
    Args:
        backbone_type: 'lstm', 'transformer', 'titans'
        memory_type: 'lmm_mlp', 'lmm_attention', 'none'
        input_dim: 输入特征维度
        output_dim: 输出特征维度
        pred_len: 预测长度
        seq_len: 输入序列长度
        backbone_dim: Backbone的隐藏维度
        use_meta_learning: 是否启用元学习 (模式3)
        meta_learner_type: 元学习器类型 ('adaptive'或'fixed')
        其他配置参数...
    
    Returns:
        model: ContinualForecaster
    """
    from models.backbones import build_backbone
    from models.memory import build_memory_unit
    from models.meta_learner import build_meta_learner
    
    # 构建Backbone
    backbone_kwargs = {
        'input_dim': input_dim,
    }
    
    if backbone_type == 'lstm':
        backbone_kwargs.update({
            'hidden_dim': backbone_dim,
            'num_layers': backbone_depth,
        })
    elif backbone_type == 'transformer':
        backbone_kwargs.update({
            'dim': backbone_dim,
            'depth': backbone_depth,
            'heads': backbone_heads,
        })
    elif backbone_type == 'titans':
        backbone_kwargs.update({
            'dim': backbone_dim,
            'depth': backbone_depth,
            'heads': backbone_heads,
            'seq_len': seq_len,
        })
    
    backbone = build_backbone(backbone_type, **backbone_kwargs)
    
    # 构建Memory Unit
    memory_kwargs = {
        'dim': backbone_dim,
        'chunk_size': memory_chunk_size,
        'neural_memory_batch_size': neural_memory_batch_size,
    }
    
    if memory_type.startswith('lmm') or memory_type.startswith('titans'):
        memory_kwargs.update({
            'heads': 1,
            'mlp_depth': 2,
            'mlp_expansion_factor': 4.0,
            'max_grad_norm': 0.5,
            'default_step_transform_max_lr': 0.001,
            'init_adaptive_step_bias': -15.0,
            'momentum': False,
            'qk_rmsnorm': False,
            'attn_pool_chunks': False,
        })
    
    memory_unit = build_memory_unit(memory_type, **memory_kwargs)
    
    # 构建Meta-Learner (如果启用元学习)
    meta_learner = None
    if use_meta_learning and memory_type != 'none':
        meta_kwargs = {
            'meta_learner_type': meta_learner_type,
            'input_dim': backbone_dim,
        }
        if meta_learner_type.lower() == 'adaptive':
            meta_kwargs['hidden_dim'] = meta_learner_hidden_dim
        meta_learner = build_meta_learner(**meta_kwargs)
    
    # 构建完整模型
    model = ContinualForecaster(
        backbone=backbone,
        memory_unit=memory_unit,
        meta_learner=meta_learner,
        output_dim=output_dim,
        pred_len=pred_len,
        fusion_type=fusion_type,
        use_meta_learning=use_meta_learning
    )
    
    return model


if __name__ == '__main__':
    print("测试持续学习预测框架...")
    
    # 测试模式1: Baseline (无Memory)
    print("\n1. 测试模式1: Baseline")
    model1 = build_continual_forecaster(
        backbone_type='lstm',
        memory_type='none',
        input_dim=3,
        output_dim=1,
        pred_len=1,
        seq_len=64,
        backbone_dim=256,
        use_meta_learning=False
    )
    x = torch.randn(2, 64, 3)
    pred1, _ = model1(x)
    print(f"   模型: {model1.get_model_info()['backbone']} + {model1.get_model_info()['memory_unit']}")
    print(f"   输出形状: {pred1.shape}")
    
    # 测试模式2: Simple TTT (Fixed策略)
    print("\n2. 测试模式2: Simple TTT")
    model2 = build_continual_forecaster(
        backbone_type='transformer',
        memory_type='lmm_mlp',
        input_dim=3,
        output_dim=1,
        pred_len=1,
        seq_len=64,
        backbone_dim=256,
        use_meta_learning=False,
        meta_learner_type='fixed'
    )
    pred2, _ = model2(x)
    print(f"   模型: {model2.get_model_info()['backbone']} + {model2.get_model_info()['memory_unit']}")
    print(f"   输出形状: {pred2.shape}")
    
    # 测试模式3: Full Meta-TTT (自适应策略)
    print("\n3. 测试模式3: Full Meta-TTT")
    model3 = build_continual_forecaster(
        backbone_type='titans',
        memory_type='lmm_mlp',
        input_dim=3,
        output_dim=1,
        pred_len=1,
        seq_len=64,
        backbone_dim=256,
        use_meta_learning=True,
        meta_learner_type='adaptive'
    )
    pred3, _ = model3(x)
    print(f"   模型: {model3.get_model_info()['backbone']} + {model3.get_model_info()['memory_unit']} + {model3.get_model_info()['meta_learner']}")
    print(f"   输出形状: {pred3.shape}")
    
    print("\n✓ 框架测试通过!")
