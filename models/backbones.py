"""
Time Series Forecasting Backbones

这个模块定义了不同的预测主干网络(Backbone, P)
每个Backbone负责从时间序列数据中提取特征
"""

import torch
import torch.nn as nn
from torch.nn import Parameter
from einops import rearrange, repeat


class LSTMBackbone(nn.Module):
    """
    基于LSTM的时间序列预测主干
    
    参数:
        input_dim: 输入特征维度
        hidden_dim: LSTM隐藏层维度
        num_layers: LSTM层数
        seq_len: 输入序列长度
        dropout: Dropout比率
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        seq_len: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim]
            
        Returns:
            features: [batch, seq_len, hidden_dim]
        """
        batch_size = x.shape[0]
        
        # LSTM forward
        features, (h_n, c_n) = self.lstm(x)  # [batch, seq_len, hidden_dim]
        
        # Layer normalization
        features = self.layer_norm(features)
        
        return features


class TransformerBackbone(nn.Module):
    """
    基于Transformer的时间序列预测主干
    
    参数:
        input_dim: 输入特征维度
        dim: Transformer特征维度
        depth: Transformer层数
        heads: 多头注意力头数
        dim_head: 每个注意力头的维度
        mlp_dim: FFN隐藏层维度
        dropout: Dropout比率
    """
    def __init__(
        self,
        input_dim: int,
        dim: int = 384,
        depth: int = 4,
        heads: int = 6,
        dim_head: int = 64,
        mlp_dim: int = 1536,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dim = dim
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, dim)
        
        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN架构
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )
        
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim]
            
        Returns:
            features: [batch, seq_len, dim]
        """
        # 投影到transformer维度
        x = self.input_projection(x)  # [batch, seq_len, dim]
        
        # Transformer encoding
        features = self.transformer(x)  # [batch, seq_len, dim]
        
        # 最终归一化
        features = self.norm(features)
        
        return features


class TitansBackbone(nn.Module):
    """
    基于原始Titans MAC Transformer的主干
    
    这个Backbone使用了Titans库中的MemoryAsContextTransformer，
    但移除了其内置的NeuralMemory部分，只使用其Transformer架构
    
    参数:
        input_dim: 输入特征维度
        dim: 特征维度
        depth: Transformer层数
        heads: 多头注意力头数
        其他参数参考原始MAC Transformer
    """
    def __init__(
        self,
        input_dim: int,
        dim: int = 384,
        depth: int = 6,
        heads: int = 6,
        dim_head: int = 64,
        seq_len: int = 64,
        dropout: float = 0.1,
        causal: bool = True,
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dim = dim
        self.seq_len = seq_len
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, dim)
        
        # 位置编码 (可学习的)
        self.pos_embedding = Parameter(torch.randn(1, seq_len, dim))
        
        # Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )
        
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim]
            
        Returns:
            features: [batch, seq_len, dim]
        """
        batch_size, seq_len = x.shape[:2]
        
        # 输入投影
        x = self.input_projection(x)  # [batch, seq_len, dim]
        
        # 添加位置编码
        if seq_len <= self.seq_len:
            x = x + self.pos_embedding[:, :seq_len, :]
        else:
            # 如果序列长度超过预设，进行插值
            pos_emb = nn.functional.interpolate(
                self.pos_embedding.transpose(1, 2),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            x = x + pos_emb
        
        x = self.dropout(x)
        
        # Transformer处理
        features = self.transformer(x)  # [batch, seq_len, dim]
        
        # 最终归一化
        features = self.norm(features)
        
        return features


def build_backbone(backbone_type: str, **kwargs):
    """
    工厂函数：根据类型构建Backbone
    
    Args:
        backbone_type: 'lstm', 'transformer', 'titans'
        **kwargs: 传递给具体Backbone的参数
        
    Returns:
        backbone: nn.Module
    """
    backbone_type = backbone_type.lower()
    
    if backbone_type == 'lstm':
        return LSTMBackbone(**kwargs)
    elif backbone_type == 'transformer':
        return TransformerBackbone(**kwargs)
    elif backbone_type == 'titans':
        return TitansBackbone(**kwargs)
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}. "
                        f"Supported types: 'lstm', 'transformer', 'titans'")

