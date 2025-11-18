"""
元学习器模块 (Meta-Learner)

这是实现"学习如何学习"的关键组件。
元学习器根据当前上下文动态生成LMM的更新策略(元参数)。

核心思想:
- 输入: Backbone提取的特征 f_t
- 输出: 元参数 (θ_t, η_t, α_t)
  - θ_t: 学习率 (learning rate)
  - η_t: 动量系数 (momentum)
  - α_t: 遗忘率 (forgetting rate)
"""

import torch
import torch.nn as nn


class MetaLearner(nn.Module):
    """
    元学习器 - 生成LMM的更新策略
    
    参数:
        input_dim: 输入特征维度 (来自Backbone)
        hidden_dim: 元网络隐藏层维度
        theta_range: 学习率范围 (min, max)
        eta_range: 动量范围 (min, max)
        alpha_range: 遗忘率范围 (min, max)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        theta_range: tuple = (1e-5, 1e-2),  # 学习率范围
        eta_range: tuple = (0.8, 0.99),     # 动量范围
        alpha_range: tuple = (0.0, 1.0),    # 遗忘率范围
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.theta_range = theta_range
        self.eta_range = eta_range
        self.alpha_range = alpha_range
        
        # 共享的特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 三个独立的输出头
        # 1. 学习率预测头
        self.theta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出 [0, 1]
        )
        
        # 2. 动量预测头
        self.eta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出 [0, 1]
        )
        
        # 3. 遗忘率预测头
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出 [0, 1]
        )
    
    def forward(self, features):
        """
        生成元参数
        
        Args:
            features: [batch, seq_len, input_dim] 或 [batch, input_dim]
                     来自Backbone的特征
        
        Returns:
            如果输入是3D:
                theta: [batch, seq_len] 学习率
                eta: [batch, seq_len] 动量
                alpha: [batch, seq_len] 遗忘率
            如果输入是2D:
                theta: [batch] 学习率
                eta: [batch] 动量
                alpha: [batch] 遗忘率
        """
        # 处理输入维度
        original_shape = features.shape
        is_sequence = features.dim() == 3
        
        if is_sequence:
            # [batch, seq_len, dim] -> 展平处理每个时间步
            batch_size, seq_len, dim = features.shape
            features = features.reshape(batch_size * seq_len, dim)  # [batch*seq_len, dim]
        
        # 提取共享特征
        shared_features = self.feature_extractor(features)  # [batch*seq_len, hidden_dim] 或 [batch, hidden_dim]
        
        # 生成三个元参数
        theta_normalized = self.theta_head(shared_features).squeeze(-1)  # [batch*seq_len] 或 [batch]
        eta_normalized = self.eta_head(shared_features).squeeze(-1)
        alpha_normalized = self.alpha_head(shared_features).squeeze(-1)
        
        # 缩放到指定范围
        theta = self._scale_to_range(theta_normalized, *self.theta_range)
        eta = self._scale_to_range(eta_normalized, *self.eta_range)
        alpha = self._scale_to_range(alpha_normalized, *self.alpha_range)
        
        # 如果是序列输入，恢复形状
        if is_sequence:
            theta = theta.reshape(batch_size, seq_len)  # [batch, seq_len]
            eta = eta.reshape(batch_size, seq_len)
            alpha = alpha.reshape(batch_size, seq_len)
        
        return theta, eta, alpha
    
    def _scale_to_range(self, normalized_value, min_val, max_val):
        """
        将 [0, 1] 范围的值缩放到 [min_val, max_val]
        
        Args:
            normalized_value: [batch] 范围在 [0, 1]
            min_val: 最小值
            max_val: 最大值
        
        Returns:
            scaled_value: [batch] 范围在 [min_val, max_val]
        """
        return min_val + (max_val - min_val) * normalized_value
    
    def get_config(self):
        """返回配置信息"""
        return {
            'type': 'MetaLearner',
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'theta_range': self.theta_range,
            'eta_range': self.eta_range,
            'alpha_range': self.alpha_range,
        }


class FixedMetaLearner(nn.Module):
    """
    固定元参数的"元学习器" - 用于模式2 (Simple TTT)
    
    这个模块不进行学习,只返回固定的超参数。
    用于与真正的元学习器进行对比实验。
    
    参数:
        fixed_theta: 固定的学习率
        fixed_eta: 固定的动量
        fixed_alpha: 固定的遗忘率
    """
    def __init__(
        self,
        input_dim: int,  # 为了接口兼容
        fixed_theta: float = 1e-3,
        fixed_eta: float = 0.9,
        fixed_alpha: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.fixed_theta = fixed_theta
        self.fixed_eta = fixed_eta
        self.fixed_alpha = fixed_alpha
        
        # 注册为buffer (不参与训练,但会保存到checkpoint)
        self.register_buffer('theta', torch.tensor(fixed_theta))
        self.register_buffer('eta', torch.tensor(fixed_eta))
        self.register_buffer('alpha', torch.tensor(fixed_alpha))
    
    def forward(self, features):
        """
        返回固定的元参数
        
        Args:
            features: [batch, seq_len, input_dim] 或 [batch, input_dim]
        
        Returns:
            如果输入是3D:
                theta: [batch, seq_len] 固定学习率
                eta: [batch, seq_len] 固定动量
                alpha: [batch, seq_len] 固定遗忘率
            如果输入是2D:
                theta: [batch] 固定学习率
                eta: [batch] 固定动量
                alpha: [batch] 固定遗忘率
        """
        # 处理不同维度
        if features.dim() == 3:
            batch_size, seq_len = features.shape[:2]
            # 扩展到 [batch, seq_len]
            theta = self.theta.expand(batch_size, seq_len)
            eta = self.eta.expand(batch_size, seq_len)
            alpha = self.alpha.expand(batch_size, seq_len)
        else:
            batch_size = features.shape[0]
            # 扩展到 [batch]
            theta = self.theta.expand(batch_size)
            eta = self.eta.expand(batch_size)
            alpha = self.alpha.expand(batch_size)
        
        return theta, eta, alpha
    
    def get_config(self):
        """返回配置信息"""
        return {
            'type': 'FixedMetaLearner',
            'input_dim': self.input_dim,
            'fixed_theta': self.fixed_theta,
            'fixed_eta': self.fixed_eta,
            'fixed_alpha': self.fixed_alpha,
        }


def build_meta_learner(
    meta_learner_type: str,
    input_dim: int,
    **kwargs
):
    """
    工厂函数: 构建元学习器
    
    Args:
        meta_learner_type: 'adaptive' 或 'fixed'
        input_dim: 输入特征维度
        **kwargs: 其他参数
    
    Returns:
        meta_learner: nn.Module
    """
    meta_learner_type = meta_learner_type.lower()
    
    if meta_learner_type == 'adaptive':
        return MetaLearner(input_dim=input_dim, **kwargs)
    elif meta_learner_type == 'fixed':
        return FixedMetaLearner(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(
            f"Unknown meta_learner_type: {meta_learner_type}. "
            f"Supported types: 'adaptive', 'fixed'"
        )


if __name__ == '__main__':
    # 测试代码
    print("测试元学习器...")
    
    # 测试 MetaLearner
    print("\n1. 测试 MetaLearner (自适应)")
    meta_learner = MetaLearner(input_dim=256, hidden_dim=128)
    
    # 测试输入
    features = torch.randn(4, 64, 256)  # [batch=4, seq_len=64, dim=256]
    theta, eta, alpha = meta_learner(features)
    
    print(f"   输入形状: {features.shape}")
    print(f"   theta 形状: {theta.shape}, 范围: [{theta.min():.6f}, {theta.max():.6f}]")
    print(f"   eta 形状: {eta.shape}, 范围: [{eta.min():.6f}, {eta.max():.6f}]")
    print(f"   alpha 形状: {alpha.shape}, 范围: [{alpha.min():.6f}, {alpha.max():.6f}]")
    
    # 测试 FixedMetaLearner
    print("\n2. 测试 FixedMetaLearner (固定)")
    fixed_meta_learner = FixedMetaLearner(
        input_dim=256,
        fixed_theta=1e-3,
        fixed_eta=0.9,
        fixed_alpha=0.1
    )
    
    theta_fixed, eta_fixed, alpha_fixed = fixed_meta_learner(features)
    print(f"   theta: {theta_fixed[0]:.6f} (固定)")
    print(f"   eta: {eta_fixed[0]:.6f} (固定)")
    print(f"   alpha: {alpha_fixed[0]:.6f} (固定)")
    
    # 测试工厂函数
    print("\n3. 测试工厂函数")
    meta_learner_adaptive = build_meta_learner('adaptive', input_dim=256)
    meta_learner_fixed = build_meta_learner('fixed', input_dim=256)
    print(f"   adaptive type: {type(meta_learner_adaptive).__name__}")
    print(f"   fixed type: {type(meta_learner_fixed).__name__}")
    
    print("\n✓ 元学习器测试通过!")
