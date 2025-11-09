"""
Continual Forecasting Framework

这个模块定义了持续学习预测的核心框架
它将Backbone(P)和Memory Unit(M)组合成一个完整的预测系统
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class ContinualForecaster(nn.Module):
    """
    持续学习时间序列预测器
    
    这个模型将预测主干(Backbone P)和记忆单元(Memory Unit M)解耦，
    使得两者可以独立替换和训练。
    
    架构流程:
        Input [batch, seq_len, input_dim]
          ↓
        Backbone P: 特征提取
          ↓
        Features [batch, seq_len, backbone_dim]
          ↓
        Memory Unit M: 记忆检索与更新
          ↓
        Memory Output [batch, seq_len, memory_dim]
          ↓
        特征融合 (features + memory)
          ↓
        Prediction Head: 最终预测
          ↓
        Output [batch, pred_len, output_dim]
    
    参数:
        backbone: 预测主干网络
        memory_unit: 记忆单元
        output_dim: 输出特征维度
        pred_len: 预测长度
        fusion_type: 特征融合方式 ('add', 'concat', 'gated')
    """
    def __init__(
        self,
        backbone: nn.Module,
        memory_unit: nn.Module,
        output_dim: int,
        pred_len: int = 1,
        fusion_type: str = 'add',
        dropout: float = 0.1
    ):
        super().__init__()
        self.backbone = backbone
        self.memory_unit = memory_unit
        self.output_dim = output_dim
        self.pred_len = pred_len
        self.fusion_type = fusion_type
        
        # 获取backbone输出维度
        # 假设backbone有hidden_dim或dim属性
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
        
    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[any] = None,
        return_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[any]]:
        """
        前向传播
        
        Args:
            x: [batch, seq_len, input_dim] - 输入时间序列
            cache: 记忆状态cache (用于持续学习)
            return_cache: 是否返回下一步的cache
            
        Returns:
            pred: [batch, pred_len, output_dim] - 预测结果
            next_cache: 下一步的cache (如果return_cache=True)
        """
        batch_size, seq_len = x.shape[:2]
        
        # 1. Backbone提取特征
        features = self.backbone(x)  # [batch, seq_len, feature_dim]
        
        # 2. Memory Unit处理
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
        
        # 4. 使用最后一个时间步的特征进行预测
        last_feature = fused[:, -1, :]  # [batch, feature_dim]
        
        # 5. 预测头
        pred_flat = self.prediction_head(last_feature)  # [batch, output_dim * pred_len]
        pred = pred_flat.view(batch_size, self.pred_len, self.output_dim)
        
        if return_cache:
            return pred, next_cache
        return pred, None
    
    def get_model_info(self):
        """返回模型的详细信息"""
        info = {
            'backbone': self.backbone.__class__.__name__,
            'memory_unit': self.memory_unit.__class__.__name__,
            'feature_dim': self.feature_dim,
            'output_dim': self.output_dim,
            'pred_len': self.pred_len,
            'fusion_type': self.fusion_type,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
        
        # 添加memory unit的配置
        if hasattr(self.memory_unit, 'get_config'):
            info['memory_config'] = self.memory_unit.get_config()
        
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
    **kwargs
) -> ContinualForecaster:
    """
    工厂函数：构建完整的持续学习预测器
    
    Args:
        backbone_type: 'lstm', 'transformer', 'titans'
        memory_type: 'titans_mlp', 'titans_attention', 'none'
        input_dim: 输入特征维度
        output_dim: 输出特征维度
        pred_len: 预测长度
        seq_len: 输入序列长度
        backbone_dim: Backbone的隐藏维度
        其他配置参数...
        
    Returns:
        model: ContinualForecaster
    """
    from models.backbones import build_backbone
    from models.memory_units import build_memory_unit
    
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
            'seq_len': seq_len,  # 只有Titans需要seq_len
        })
    
    backbone = build_backbone(backbone_type, **backbone_kwargs)
    
    # 构建Memory Unit
    memory_kwargs = {
        'dim': backbone_dim,  # Memory的维度必须与Backbone输出一致
        'chunk_size': memory_chunk_size,
        'neural_memory_batch_size': neural_memory_batch_size,
    }
    
    if memory_type.startswith('titans'):
        memory_kwargs.update({
            'heads': 1,  # 可以根据需要调整
            'mlp_depth': 2,
            'mlp_expansion_factor': 4.0,
        })
    
    memory_unit = build_memory_unit(memory_type, **memory_kwargs)
    
    # 构建完整模型
    model = ContinualForecaster(
        backbone=backbone,
        memory_unit=memory_unit,
        output_dim=output_dim,
        pred_len=pred_len,
        fusion_type=fusion_type
    )
    
    return model

