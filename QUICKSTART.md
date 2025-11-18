# 🚀 快速开始指南

## 安装依赖

```bash
pip install torch numpy pandas matplotlib tensordict einops einx
```

## 🎯 三种实验模式

### 模式1: Baseline

无Memory，标准在线学习，用于建立性能基准。

**Bash/Linux/Mac:**
```bash
python titans_main.py \
    --memory_type none \
    --des baseline \
    --train_epochs 2 \
    --batch_size 2 \
    --d_model 256 \
    --e_layers 3
```

**PowerShell/Windows:**
```powershell
python titans_main.py --memory_type none --des baseline --train_epochs 2 --batch_size 2 --d_model 256 --e_layers 3
```

**预期**: 
- 训练应该顺利完成
- 这是最简单的模式，用于验证基础流程
- 记录性能指标作为基准

---

### 模式2: Simple TTT

带LMM，固定更新策略。

**Bash/Linux/Mac:**
```bash
python titans_main.py \
    --memory_type lmm_mlp \
    --use_meta_learning 0 \
    --meta_learner_type fixed \
    --des simple_ttt \
    --train_epochs 2 \
    --batch_size 2 \
    --d_model 256 \
    --e_layers 3 \
    --fixed_theta 0.001 \
    --fixed_eta 0.9 \
    --fixed_alpha 0.1
```

**PowerShell/Windows:**
```powershell
python titans_main.py --memory_type lmm_mlp --use_meta_learning 0 --meta_learner_type fixed --des simple_ttt --train_epochs 2 --batch_size 2 --d_model 256 --e_layers 3 --fixed_theta 0.001 --fixed_eta 0.9 --fixed_alpha 0.1
```

**预期**:
- 性能应该优于Baseline
- 如果不如Baseline，说明LMM配置有问题
- 验证了LMM本身的有效性

---

### 模式3: Full Meta-TTT (最复杂，最终目标)

带LMM + 元学习器，自适应策略。

✅ **Meta-Params已集成**: Meta-Learner生成的学习率现在会真正影响LMM更新！

**Bash/Linux/Mac:**
```bash
python titans_main.py \
    --memory_type lmm_mlp \
    --use_meta_learning 1 \
    --meta_learner_type adaptive \
    --des full_meta_ttt \
    --train_epochs 3 \
    --batch_size 2 \
    --d_model 256 \
    --e_layers 3 \
    --meta_learner_hidden_dim 128 \
    --learning_rate 5e-6
```

**PowerShell/Windows:**
```powershell
python titans_main.py --memory_type lmm_mlp --use_meta_learning 1 --meta_learner_type adaptive --des full_meta_ttt --train_epochs 3 --batch_size 2 --d_model 256 --e_layers 3 --meta_learner_hidden_dim 128 --learning_rate 5e-6
```

**预期**:
- 这是终极目标！
- 性能应该是三种模式中最好的
- Meta-Learner学习如何动态调整LMM的更新策略

---

## 📊 查看结果

运行完成后，结果保存在：

- **Checkpoints**: `./checkpoints/continual_forecaster_synthetic_sl64_pl1_{des}/`
- **预测结果**: `./results/continual_forecaster_synthetic_sl64_pl1_{des}_*.csv`
- **可视化图表**: `./figs/continual_forecaster_synthetic_sl64_pl1_{des}_*.jpg`

---

## ⚡ 性能优化说明

**最新优化 (2025-11-18)**：
- ✅ **批量特征提取**：一次性处理整个序列，避免逐个token调用backbone
- ✅ **批量元参数生成**：Meta-Learner一次性生成所有时间步的元参数
- ⚠️ **速度提升情况**：
  - **模式2 (Simple TTT)**: 约1%提升（主要瓶颈在LMM逐步更新）
  - **模式3 (Full Meta-TTT)**: 预期10-50倍加速（消除了64次backbone和Meta-Learner调用）

**注意**：LMM的状态更新必须逐步进行（递归依赖），这是设计上的限制，无法完全并行化。

---

## 🐛 常见问题

### 1. GPU内存不足 (OOM)

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
```bash
# 减小模型和batch size
python titans_main.py \
    --memory_type none \
    --des baseline_small \
    --batch_size 1 \
    --d_model 128 \
    --e_layers 2 \
    --neural_memory_batch_size 1024
```

### 2. 训练出现NaN

**症状**: 输出包含 `nan` 或 `inf`

**解决方案**:
```bash
# 降低学习率，加强梯度裁剪
python titans_main.py \
    --memory_type lmm_mlp \
    --use_meta_learning 0 \
    --des stable_training \
    --learning_rate 1e-6 \
    --clip_grad 0.3 \
    --batch_size 1
```

### 3. 模式2性能不如Baseline

**可能原因**:
1. LMM配置不当
2. 固定元参数不合适
3. 训练不充分

**解决方案**:
```bash
# 尝试不同的固定元参数
python titans_main.py \
    --memory_type lmm_mlp \
    --use_meta_learning 0 \
    --meta_learner_type fixed \
    --fixed_theta 0.0001 \  # 降低学习率
    --fixed_eta 0.95 \      # 增大动量
    --fixed_alpha 0.05 \    # 降低遗忘率
    --train_epochs 5        # 更多epoch
```

### 4. 模式3训练不稳定

**可能原因**:
1. 元学习需要更小的学习率
2. Meta-Learner需要更多训练时间

**解决方案**:
```bash
python titans_main.py \
    --memory_type lmm_mlp \
    --use_meta_learning 1 \
    --meta_learner_type adaptive \
    --learning_rate 1e-6 \      # 更小的学习率
    --train_epochs 10 \          # 更多epoch
    --patience 5                 # 更大的patience
```

---

## 📈 实验建议

### 第一步：快速验证

使用小规模配置快速验证流程：

```bash
# 模式1
python titans_main.py --memory_type none --des test1 \
    --train_epochs 1 --batch_size 2 --d_model 128 --e_layers 2

# 模式2
python titans_main.py --memory_type lmm_mlp --use_meta_learning 0 \
    --des test2 --train_epochs 1 --batch_size 2 --d_model 128 --e_layers 2

# 模式3
python titans_main.py --memory_type lmm_mlp --use_meta_learning 1 \
    --des test3 --train_epochs 1 --batch_size 2 --d_model 128 --e_layers 2
```

### 第二步：完整实验

确认流程正常后，运行完整实验：

```bash
# 依次运行三种模式，使用相同的配置
for mode in baseline simple_ttt full_meta_ttt; do
    echo "Running $mode..."
    # 根据模式调整参数
    # ...
done
```

### 第三步：对比分析

对比三种模式的性能：
- MAE, MSE, RMSE
- 训练时间
- 在线适应能力

---

## 🔍 调试技巧

### 启用详细日志

```bash
python titans_main.py \
    --memory_type lmm_mlp \
    --use_meta_learning 1 \
    --log_interval 10  # 每10步打印一次
```

### 检查模型信息

运行时会自动打印模型架构：
```
模型架构:
  Backbone: TransformerBackbone
  Memory Unit: LMMWrapper
  特征维度: 256
  融合方式: add
  总参数量: 1,234,567
```

### 监控GPU使用

```bash
# 在另一个终端运行
watch -n 1 nvidia-smi
```

---

## 💡 优化建议

### 性能优化

如果想要更好的性能：

```bash
python titans_main.py \
    --memory_type lmm_mlp \
    --use_meta_learning 1 \
    --meta_learner_type adaptive \
    --d_model 384 \              # 更大的模型
    --e_layers 4 \
    --n_heads 12 \
    --train_epochs 10 \           # 更多训练
    --learning_rate 5e-6
```

### 速度优化

如果想要更快的训练：

```bash
python titans_main.py \
    --memory_type lmm_mlp \
    --use_meta_learning 1 \
    --batch_size 4 \              # 更大的batch
    --neural_memory_batch_size 1024 \  # 减少Memory更新频率
    --num_workers 4               # 多线程数据加载
```

---

## 📚 更多信息

- **设计文档**: 参考 `NEW_FRAMEWORK_DESIGN.md`
- **重构报告**: 参考 `REFACTORING_COMPLETE.md`
- **配置文件**: 参考 `configs/README.md`

---

## 🎓 实验流程建议

1. ✅ **验证Baseline** (模式1)
   - 确保基础流程正常
   - 建立性能基准

2. ✅ **验证Simple TTT** (模式2)
   - 验证LMM有效性
   - 调整固定元参数

3. ✅ **验证Full Meta-TTT** (模式3)
   - 验证元学习机制
   - 对比三种模式性能

4. ✅ **深入分析**
   - 可视化元参数变化
   - 分析不同配置的影响
   - 撰写实验报告

---

## ✨ 预期结果

```
模式3 (Full Meta-TTT) > 模式2 (Simple TTT) > 模式1 (Baseline)
```

如果结果符合预期：🎉 恭喜！元学习框架工作正常！

如果结果不符合预期：🔧 参考"常见问题"章节进行调试。

---

**祝实验顺利！** 🚀

如有问题，请检查：
1. 依赖是否完整安装
2. GPU内存是否充足
3. 数据文件是否存在
4. 参数配置是否合理

Good luck! 💪

