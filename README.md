# Titans 元学习时间序列预测框架

基于元学习的持续学习框架，实现"学习如何学习"的机制。

## 🚀 快速开始

### 安装依赖
```bash
pip install torch numpy pandas matplotlib tensordict einops einx
```

### 运行三种模式

**模式1: Baseline (无Memory)**
```bash
python titans_main.py --memory_type none --des baseline --train_epochs 2 --batch_size 2
```

**模式2: Simple TTT (固定策略)**
```bash
python titans_main.py --memory_type lmm_mlp --use_meta_learning 0 --meta_learner_type fixed --des simple_ttt --train_epochs 2 --batch_size 2
```

**模式3: Full Meta-TTT (自适应策略)** ⭐
```bash
python titans_main.py --memory_type lmm_mlp --use_meta_learning 1 --meta_learner_type adaptive --des full_meta_ttt --train_epochs 3 --batch_size 2 --learning_rate 5e-6
```

## 📚 核心文档

1. **`NEW_FRAMEWORK_DESIGN.md`** - 完整的设计文档
2. **`QUICKSTART.md`** - 详细的快速开始指南（包含故障排除）

## 🎯 三种模式对比

| 特性 | 模式1 | 模式2 | 模式3 |
|-----|------|------|------|
| Backbone | ✅ | ✅ | ✅ |
| LMM | ❌ | ✅ | ✅ |
| Meta-Learner | ❌ | 固定 | 自适应 |
| 元学习 | ❌ | ❌ | ✅ |

## 🏗️ 架构

```
Input → Backbone (P) → Meta-Learner → LMM (内循环) → Fusion → Output
         ↑                  ↑
         └──── 外循环梯度流 ────┘
```

## 📊 目录结构

```
titans_main/
├── models/
│   ├── backbones.py        # Backbone (P)
│   ├── memory.py           # LMM
│   ├── meta_learner.py     # Meta-Learner
│   └── framework.py        # 核心框架
├── utils/
│   └── trainer.py          # 训练器（内外双循环）
├── dataset/                # 数据集
├── configs/                # 配置文件
└── titans_main.py          # 主程序
```

## ⚡ 核心创新

1. **内外双循环**: 内循环快速适应，外循环优化策略
2. **元学习器**: 动态生成LMM更新策略 (θ, η, α)
3. **模块化**: Backbone + LMM + Meta-Learner 完全解耦

## 🔧 注意事项

- **GPU内存**: 建议8GB+，如不足使用 `--batch_size 1 --d_model 128`
- **Windows用户**: 使用单行命令（不支持 `\` 续行）
- **模式3**: Meta-Learner会学习如何动态调整LMM的学习率

## 📈 预期结果

```
模式3 (Full Meta-TTT) > 模式2 (Simple TTT) > 模式1 (Baseline)
```

## ✅ 重要更新

- ✅ Meta-Params已完全集成到LMM更新（2025-11-17）
- ✅ 模式3可以真正验证元学习机制
- ✅ 所有三种模式都可以正常运行

---

**详细文档**: 参见 `QUICKSTART.md` 和 `NEW_FRAMEWORK_DESIGN.md`

