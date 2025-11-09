# Titans 持续学习框架重构指南

## 📋 重构概述

本次重构彻底解耦了预测主干（Backbone P）和记忆单元（Memory Unit M），构建了一个模块化、可扩展的持续学习框架。

### 核心改进

1. **模块化架构**：P和M完全解耦，可独立替换
2. **清晰的实验逻辑**：从3种混乱的测试模式简化为2种明确的在线学习模式
3. **可扩展性**：轻松添加新的Backbone或Memory Unit
4. **符合原始设计**：正确遵循Titans NeuralMemory的自适应学习机制

---

## 🏗️ 新架构说明

### 文件结构

```
models/
├── backbones.py           # 预测主干库（P）
│   ├── LSTMBackbone
│   ├── TransformerBackbone
│   └── TitansBackbone
├── memory_units.py        # 记忆单元库（M）
│   ├── TitansMemoryWrapper
│   └── NoMemoryUnit
└── framework.py           # 统一容器
    └── ContinualForecaster

utils/
└── trainer_new.py         # 新训练器
    └── ContinualTrainer

titans_main_new.py         # 新入口文件
```

### 架构流程

```
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
```

---

## 🔄 学习流程设计

### 1. 预训练阶段（is_training=1）

**目标**：在训练集上同时优化P和M，学习任务的基础模式

**行为**：
- P和M的参数都通过反向传播更新
- 每个batch独立处理（不维护cache）
- 使用标准监督学习

**命令**：
```bash
python titans_main_new.py --is_training 1
```

---

### 2. 在线测试阶段（is_training=0）

在线测试阶段有两种模式，通过`--test_mode`参数控制：

#### 模式A：仅记忆单元学习（memory_only）

**目标**：测试轻量级适应能力，避免灾难性遗忘

**行为**：
- ✅ Backbone P **冻结**（`param.requires_grad=False`）
- ✅ Memory Unit M 通过内置机制自动更新
- ✅ cache跨batch传递，实现记忆累积学习
- ✅ 不使用外部optimizer

**核心机制**：
```python
# NeuralMemory在forward时自动完成：
# 1. Store: 计算grad(MSE(M(k), v))，更新记忆权重
# 2. Retrieve: 返回M(q)作为context
# 3. 通过cache维持seq_index、weights等状态
```

**命令**：
```bash
python titans_main_new.py --is_training 0 --test_mode memory_only
```

---

#### 模式B：全模型学习（full_model）

**目标**：测试最大适应能力

**行为**：
- ✅ Memory Unit M 自动更新（内置机制）
- ✅ Backbone P 通过反向传播 + optimizer更新
- ✅ cache跨batch传递，实现记忆累积学习

**命令**：
```bash
python titans_main_new.py --is_training 0 --test_mode full_model --online_lr 1e-5
```

---

## 🎯 使用示例

### 基础训练与测试

```bash
# 1. 预训练
python titans_main_new.py \
    --is_training 1 \
    --backbone_type titans \
    --memory_type titans_mlp \
    --data synthetic \
    --seq_len 64 \
    --pred_len 1 \
    --train_epochs 10 \
    --batch_size 32

# 2. 在线测试 - 模式A（仅M学习）
python titans_main_new.py \
    --is_training 0 \
    --test_mode memory_only \
    --des experiment

# 3. 在线测试 - 模式B（M和P都学习）
python titans_main_new.py \
    --is_training 0 \
    --test_mode full_model \
    --online_lr 1e-5 \
    --des experiment
```

### 切换不同的Backbone

```bash
# 使用LSTM作为Backbone
python titans_main_new.py \
    --is_training 1 \
    --backbone_type lstm \
    --memory_type titans_mlp \
    --d_model 256

# 使用Transformer作为Backbone
python titans_main_new.py \
    --is_training 1 \
    --backbone_type transformer \
    --memory_type titans_mlp \
    --d_model 384 \
    --e_layers 4 \
    --n_heads 6
```

### 切换不同的Memory Unit

```bash
# 使用基于Attention的Memory
python titans_main_new.py \
    --is_training 1 \
    --backbone_type titans \
    --memory_type titans_attention \
    --memory_model_type attention

# 不使用Memory（消融实验）
python titans_main_new.py \
    --is_training 1 \
    --backbone_type titans \
    --memory_type none
```

---

## 🔧 扩展新模块

### 添加新的Backbone

在`models/backbones.py`中添加：

```python
class MyCustomBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # 必须有这个属性！
        # ... 你的实现
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            features: [batch, seq_len, hidden_dim]
        """
        # ... 你的实现
        return features
```

然后在`build_backbone`函数中添加：

```python
def build_backbone(backbone_type: str, **kwargs):
    if backbone_type == 'my_custom':
        return MyCustomBackbone(**kwargs)
    # ...
```

### 添加新的Memory Unit

在`models/memory_units.py`中添加：

```python
class MyCustomMemoryWrapper(nn.Module):
    def __init__(self, dim, **kwargs):
        super().__init__()
        self.dim = dim
        # ... 你的实现
    
    def forward(self, features, cache=None, return_cache=False):
        """
        Args:
            features: [batch, seq_len, dim]
            cache: 上一步的状态
            return_cache: 是否返回cache
        Returns:
            memory_output: [batch, seq_len, dim]
            next_cache: 下一步的cache
        """
        # ... 你的实现
        return memory_output, next_cache
    
    def get_config(self):
        return {'type': 'MyCustomMemory', 'dim': self.dim}
```

---

## 📊 关键参数说明

### Backbone参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--backbone_type` | Backbone类型 | `titans` |
| `--d_model` | Backbone隐藏维度 | `384` |
| `--e_layers` | Backbone层数 | `4` |
| `--n_heads` | 注意力头数 | `6` |

### Memory参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--memory_type` | Memory类型 | `titans_mlp` |
| `--memory_chunk_size` | Memory chunk size | `1` |
| `--neural_memory_batch_size` | 多少token后更新记忆 | `256` |
| `--memory_model_type` | Memory内部模型类型 | `mlp` |

### 融合参数

| 参数 | 说明 | 选项 |
|------|------|------|
| `--fusion_type` | 特征融合方式 | `add`, `concat`, `gated` |

---

## 🆚 新旧对比

### 旧版本问题

1. ❌ P和M高度耦合在`TimeSeriesTitansTransformer`中
2. ❌ 3种测试模式逻辑混乱，存在冗余
3. ❌ "静态推理"模式设计错误（`torch.no_grad()`抑制了NeuralMemory）
4. ❌ 无法轻松替换Backbone或Memory Unit
5. ❌ 大量`if/else`逻辑，难以维护

### 新版本优势

1. ✅ P和M完全解耦，独立可替换
2. ✅ 2种明确的在线学习模式（模式A、B）
3. ✅ 正确遵循Titans的设计哲学
4. ✅ 模块化、可扩展
5. ✅ 代码清晰、易维护

---

## 🚀 迁移步骤

如果你已经有旧版本的实验：

### 步骤1：使用新入口文件

```bash
# 旧命令
python titans_main.py --is_training 1

# 新命令
python titans_main_new.py --is_training 1
```

### 步骤2：调整测试模式参数

```bash
# 旧命令
python titans_main.py --is_training 0 --online_learning 0  # 静态推理
python titans_main.py --is_training 0 --online_learning 1 --online_update_memory_only 1  # 仅M
python titans_main.py --is_training 0 --online_learning 1 --online_update_memory_only 0  # M+P

# 新命令
python titans_main_new.py --is_training 0 --test_mode memory_only  # 仅M
python titans_main_new.py --is_training 0 --test_mode full_model   # M+P
```

### 步骤3：转换checkpoint（如需要）

新旧模型结构不同，checkpoint不兼容。建议重新训练。

---

## ❓ 常见问题

### Q1: 为什么模式A不使用optimizer？

**A**: 因为NeuralMemory的核心设计就是通过`torch.func.grad`在forward时自动更新。它有自己的自适应学习率、动量、遗忘机制。外部optimizer反而会干扰这个机制。

### Q2: cache的作用是什么？

**A**: cache维护了NeuralMemory的状态（seq_index、weights、momentums等）。只有跨batch传递cache，才能实现真正的"持续学习"。否则每个batch都会重置状态。

### Q3: 如何验证Memory Unit是否在学习？

**A**: 
1. 打印cache的`seq_index`，应该会随batch递增
2. 观察loss，模式A应该比"无Memory"模式更好
3. 模式B应该比模式A更好（因为P也在学习）

### Q4: 我可以完全不使用Memory Unit吗？

**A**: 可以！设置`--memory_type none`即可。这可以作为消融实验的baseline。

---

## 📝 总结

本次重构从根本上解决了之前架构的问题，使代码库变得：

- ✅ **清晰**：职责分离，逻辑明确
- ✅ **正确**：符合Titans的设计哲学
- ✅ **灵活**：模块化，易扩展
- ✅ **可维护**：减少冗余，易理解

现在你可以自由地：
- 尝试不同的Backbone（LSTM、Transformer、Titans...）
- 尝试不同的Memory机制（MLP、Attention...）
- 对比不同的在线学习策略（仅M、M+P）
- 轻松添加你自己的新模块

**开始你的实验吧！** 🎉

