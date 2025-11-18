"""
训练器模块 (v2) - 实现内外双循环训练逻辑

这个训练器与models/framework.py中的ContinualForecaster配合使用
实现了设计文档中描述的内外双循环元学习机制。

支持三种实验模式：
- 模式1 (Baseline): 标准在线学习（无LMM）
- 模式2 (Simple TTT): 带LMM，固定更新策略
- 模式3 (Full Meta-TTT): 带LMM，元学习动态策略

训练流程：
1. **内循环 (Inner Loop)**: 在前向传播中，LMM根据惊奇度实时更新自己的参数
2. **外循环 (Outer Loop)**: 通过标准梯度下降优化Backbone和Meta-Learner
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from pathlib import Path
from datetime import datetime

from dataset.data_factory import data_provider, get_data_info
from utils.tools import EarlyStopping, adjust_learning_rate, visual_comprehensive, save_results, get_device
from utils.metrics import metric, print_metrics


class ContinualTrainer:
    """持续学习训练器"""
    
    def __init__(self, args):
        """
        初始化训练器
        
        Args:
            args: 参数配置对象
        """
        self.args = args
        self.device = get_device(args)
        
        # 设置实验标识
        self.setting = f"{args.model_id}_{args.data}_sl{args.seq_len}_pl{args.pred_len}_{args.des}"
        
        # 创建保存路径
        self.path = os.path.join(args.checkpoints, self.setting)
        Path(self.path).mkdir(parents=True, exist_ok=True)
        
        # 构建模型
        self.model = self._build_model()
        
        # 损失函数
        self.criterion = self._get_criterion()
        
        # 优化器
        self.optimizer = None
    
    def _build_model(self):
        """构建模型"""
        print("\n" + "="*70)
        print("构建持续学习预测模型...")
        print("="*70)
        
        # 自动获取数据维度
        input_dim, output_dim = get_data_info(self.args)
        
        # 更新args中的维度信息
        self.args.input_dim = input_dim
        self.args.output_dim = output_dim
        
        print(f"\n数据配置:")
        print(f"  输入维度: {input_dim}")
        print(f"  输出维度: {output_dim}")
        print(f"  序列长度: {self.args.seq_len}")
        print(f"  预测长度: {self.args.pred_len}")
        
        # 使用新的框架构建模型
        from models.framework import build_continual_forecaster
        
        model = build_continual_forecaster(
            backbone_type=self.args.backbone_type,
            memory_type=self.args.memory_type,
            input_dim=input_dim,
            output_dim=output_dim,
            pred_len=self.args.pred_len,
            seq_len=self.args.seq_len,
            backbone_dim=self.args.d_model,
            backbone_depth=self.args.e_layers,
            backbone_heads=self.args.n_heads,
            neural_memory_batch_size=self.args.neural_memory_batch_size,
            memory_chunk_size=self.args.memory_chunk_size,
            memory_model_type=self.args.memory_model_type,
            fusion_type=self.args.fusion_type,
            # 元学习参数 (新增)
            use_meta_learning=getattr(self.args, 'use_meta_learning', 0) == 1,
            meta_learner_type=getattr(self.args, 'meta_learner_type', 'fixed'),
            meta_learner_hidden_dim=getattr(self.args, 'meta_learner_hidden_dim', 128),
        ).to(self.device)
        
        # 打印模型信息
        model_info = model.get_model_info()
        print(f"\n模型架构:")
        print(f"  Backbone: {model_info['backbone']}")
        print(f"  Memory Unit: {model_info['memory_unit']}")
        print(f"  特征维度: {model_info['feature_dim']}")
        print(f"  融合方式: {model_info['fusion_type']}")
        print(f"  总参数量: {model_info['total_params']:,}")
        print(f"  可训练参数: {model_info['trainable_params']:,}")
        
        if 'memory_config' in model_info:
            print(f"\n记忆单元配置:")
            for key, value in model_info['memory_config'].items():
                print(f"  {key}: {value}")
        
        # 测试模型前向传播（检查是否有nan）
        print("\n测试模型前向传播...")
        try:
            test_input = torch.randn(1, self.args.seq_len, input_dim).to(self.device)
            with torch.no_grad():
                test_output, _ = model(test_input, cache=None, return_cache=False)
                if torch.isnan(test_output).any() or torch.isinf(test_output).any():
                    print("⚠ 警告: 模型初始化后测试输出包含nan/inf！")
                    print(f"   测试输出范围: [{test_output.min().item():.6f}, {test_output.max().item():.6f}]")
                else:
                    print(f"✓ 模型测试通过，输出范围: [{test_output.min().item():.6f}, {test_output.max().item():.6f}]")
        except Exception as e:
            print(f"⚠ 警告: 模型测试失败: {e}")
        
        print("="*70)
        
        return model
    
    def _get_criterion(self):
        """获取损失函数"""
        if self.args.loss == 'mse':
            return nn.MSELoss()
        elif self.args.loss == 'mae':
            return nn.L1Loss()
        elif self.args.loss == 'huber':
            return nn.SmoothL1Loss()
        else:
            return nn.MSELoss()
    
    def _get_optimizer(self, params=None):
        """获取优化器"""
        if params is None:
            params = self.model.parameters()
        
        if self.args.optimizer == 'adam':
            optimizer = optim.Adam(
                params, 
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'adamw':
            optimizer = optim.AdamW(
                params,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'sgd':
            optimizer = optim.SGD(
                params,
                lr=self.args.learning_rate,
                momentum=0.9,
                weight_decay=self.args.weight_decay
            )
        else:
            optimizer = optim.Adam(params, lr=self.args.learning_rate)
        
        return optimizer
    
    def _process_one_batch(self, batch_x, batch_y, cache=None):
        """
        处理一个batch
        
        Args:
            batch_x: 输入数据 [batch_size, seq_len, input_dim]
            batch_y: 目标数据 [batch_size, label_len + pred_len, output_dim]
            cache: 记忆状态cache
        
        Returns:
            outputs: 预测结果 [batch_size, pred_len, output_dim]
            batch_y: 真实标签（只取pred_len部分）
            next_cache: 更新后的cache状态
        """
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        
        # 检查输入数据是否有nan/inf
        if torch.isnan(batch_x).any() or torch.isinf(batch_x).any():
            raise ValueError(f"输入数据包含nan/inf值！")
        if torch.isnan(batch_y).any() or torch.isinf(batch_y).any():
            raise ValueError(f"目标数据包含nan/inf值！")
        
        # 根据是否传入cache决定是否维护记忆状态
        try:
            if cache is not None:
                outputs, next_cache = self.model(batch_x, cache=cache, return_cache=True)
            else:
                outputs, _ = self.model(batch_x, cache=None, return_cache=False)
                next_cache = None
        except RuntimeError as e:
            if "nan" in str(e).lower() or "inf" in str(e).lower():
                raise RuntimeError(f"模型前向传播产生nan/inf: {e}")
            raise
        
        # 检查模型输出是否有nan/inf
        if outputs is None:
            raise ValueError("模型输出为None！")
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            # 打印调试信息
            print(f"\n❌ 模型输出包含nan/inf！")
            print(f"   输出形状: {outputs.shape}")
            print(f"   nan数量: {torch.isnan(outputs).sum().item()}")
            print(f"   inf数量: {torch.isinf(outputs).sum().item()}")
            print(f"   输出范围: [{outputs.min().item():.6f}, {outputs.max().item():.6f}]")
            raise ValueError("模型输出包含nan/inf值！")
        
        # 提取预测部分的标签
        if self.args.label_len > 0:
            batch_y = batch_y[:, -self.args.pred_len:, :]
        
        return outputs, batch_y, next_cache
    
    def train(self):
        """预训练阶段：同时训练P和M"""
        print("\n" + "="*70)
        print(f"开始预训练（同时训练Backbone和Memory Unit）")
        print(f"实验设置: {self.setting}")
        print("="*70)
        
        # 获取数据
        train_data, train_loader = data_provider(self.args, flag='train')
        vali_data, vali_loader = data_provider(self.args, flag='val')
        
        # 初始化优化器
        self.optimizer = self._get_optimizer()
        
        # 早停
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        # 训练记录
        train_steps = len(train_loader)
        
        print(f"\n训练配置:")
        print(f"  Epochs: {self.args.train_epochs}")
        print(f"  Batch Size: {self.args.batch_size}")
        print(f"  Learning Rate: {self.args.learning_rate}")
        print(f"  Optimizer: {self.args.optimizer}")
        print(f"  Loss: {self.args.loss}")
        print(f"  Steps per Epoch: {train_steps}")
        
        start_time = time.time()
        
        # 记录训练历史
        train_loss_history = []
        val_loss_history = []
        
        for epoch in range(self.args.train_epochs):
            epoch_time = time.time()
            
            # 训练一个epoch
            train_loss = self._train_epoch(train_loader, epoch)
            
            # 验证
            vali_loss = self._validate(vali_loader)
            
            # 记录loss历史
            train_loss_history.append(train_loss)
            val_loss_history.append(vali_loss)
            
            # 打印信息
            epoch_duration = time.time() - epoch_time
            print(f"\nEpoch {epoch + 1}/{self.args.train_epochs} | "
                  f"Time: {epoch_duration:.2f}s | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {vali_loss:.6f}")
            
            # 早停检查
            early_stopping(vali_loss, self.model, os.path.join(self.path, 'checkpoint.pth'))
            if early_stopping.early_stop:
                print(f"\n早停触发！在第 {epoch + 1} 轮停止训练。")
                break
            
            # 学习率调整
            if epoch > 0:
                adjust_learning_rate(self.optimizer, epoch + 1, self.args, printout=False)
        
        # 加载最佳模型
        best_model_path = os.path.join(self.path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        
        total_time = time.time() - start_time
        print(f"\n预训练完成！总用时: {total_time:.2f}s")
        
        return self.model
    
    def _train_epoch(self, train_loader, epoch):
        """
        训练一个epoch - 实现外循环逻辑
        
        外循环是标准的PyTorch训练循环，但在一次前向传播中
        已经完成了整个序列的内循环（LMM的记忆更新）。
        
        关键点：
        1. 前向传播包含了内循环（LMM自动更新）
        2. 计算任务损失（评估整个内外循环的效果）
        3. 反向传播更新Backbone和Meta-Learner（慢权重）
        4. LMM的参数(M)不直接参与外循环的梯度下降
        """
        self.model.train()
        train_loss = []
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            try:
                # ==== 前向传播（包含内循环）====
                # 训练时不维护cache，每个batch独立
                # 在这个forward中，模型内部会：
                #   1. 对序列的每一步t：
                #      - Backbone提取特征f_t
                #      - Meta-Learner生成元参数(θ_t, η_t, α_t)
                #      - LMM执行内循环更新（快速权重更新）
                #      - 从LMM检索记忆
                #   2. 融合特征并生成最终预测
                outputs, batch_y, _ = self._process_one_batch(batch_x, batch_y)
                
                # ==== 计算最终任务损失 ====
                # 这个损失评估了整个内外循环过程的最终效果
                loss = self.criterion(outputs, batch_y)
                
                # 检查loss是否为nan或inf
                if torch.isnan(loss) or torch.isinf(loss):
                    # 第一个异常batch打印详细信息
                    if i == 0 or (i + 1) % 100 == 0:
                        print(f"\n⚠ 警告: 检测到无效损失值 (nan/inf) 在 Epoch [{epoch + 1}] Step [{i + 1}]")
                        print(f"   输出统计: min={outputs.min().item():.6f}, max={outputs.max().item():.6f}, mean={outputs.mean().item():.6f}")
                        print(f"   目标统计: min={batch_y.min().item():.6f}, max={batch_y.max().item():.6f}, mean={batch_y.mean().item():.6f}")
                    continue
                
                train_loss.append(loss.item())
                
                # ==== 反向传播（外循环）====
                # 梯度会流经：Prediction Head -> Fusion -> Backbone -> Meta-Learner
                # 关键：梯度不会直接更新LMM的参数M，但会更新指导LMM学习的Meta-Learner
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪（更严格的裁剪，防止nan）
                if self.args.clip_grad > 0:
                    try:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                        # 检查梯度是否异常
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            # 减少警告打印频率（每10个异常batch打印一次）
                            if (i + 1) % 10 == 0:
                                print(f"\n⚠ 警告: 检测到异常梯度 (nan/inf) 在 Epoch [{epoch + 1}] Step [{i + 1}]")
                            # 清零梯度并跳过
                            self.optimizer.zero_grad()
                            continue
                    except RuntimeError as e:
                        if "nan" in str(e).lower() or "inf" in str(e).lower():
                            if (i + 1) % 10 == 0:
                                print(f"\n⚠ 警告: 梯度裁剪时出现异常 在 Epoch [{epoch + 1}] Step [{i + 1}]")
                            self.optimizer.zero_grad()
                            continue
                        else:
                            raise
                
                # ==== 更新"慢权重"（外循环）====
                # 更新Backbone和Meta-Learner，让它们在下一次能更好地指导LMM
                # 注意：LMM的权重M不在这里更新，它在内循环中通过惊奇度梯度更新
                self.optimizer.step()
                
                # 定期清理GPU缓存（每50个batch，更频繁）
                if (i + 1) % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                    # 如果内存占用过高，强制同步
                    allocated = torch.cuda.memory_allocated() / 1e9
                    if allocated > 6.0:  # 超过6GB时强制同步
                        torch.cuda.synchronize()
                
                # 打印日志
                if (i + 1) % self.args.log_interval == 0:
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1e9
                        reserved = torch.cuda.memory_reserved() / 1e9
                        print(f"  Epoch [{epoch + 1}] Step [{i + 1}/{len(train_loader)}] | "
                              f"Loss: {loss.item():.6f} | GPU: {allocated:.2f}/{reserved:.2f} GB")
                    else:
                        print(f"  Epoch [{epoch + 1}] Step [{i + 1}/{len(train_loader)}] | Loss: {loss.item():.6f}")
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n⚠ GPU内存不足！尝试清理缓存...")
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print(f"⚠ 建议减小 batch_size 或 neural_memory_batch_size")
                    raise
                else:
                    raise
        
        return np.mean(train_loss)
    
    def _validate(self, vali_loader):
        """验证模型"""
        self.model.eval()
        vali_loss = []
        
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                outputs, batch_y, _ = self._process_one_batch(batch_x, batch_y)
                loss = self.criterion(outputs, batch_y)
                vali_loss.append(loss.item())
        
        return np.mean(vali_loss)
    
    def online_test(self, freeze_backbone=False, load_checkpoint=True):
        """
        在线测试阶段
        
        Args:
            freeze_backbone: True=模式A(仅M学习), False=模式B(M和P都学习)
            load_checkpoint: 是否加载预训练的checkpoint
        """
        print("\n" + "="*70)
        mode_name = "模式A: 仅记忆单元学习" if freeze_backbone else "模式B: 全模型学习"
        print(f"开始在线测试: {mode_name}")
        print(f"实验设置: {self.setting}")
        print("="*70)
        
        # 加载checkpoint
        if load_checkpoint:
            checkpoint_path = os.path.join(self.path, 'checkpoint.pth')
            if os.path.exists(checkpoint_path):
                self.model.load_state_dict(torch.load(checkpoint_path))
                print(f"✓ 已加载预训练模型: {checkpoint_path}")
            else:
                print(f"⚠ 未找到checkpoint: {checkpoint_path}")
        
        # 获取测试数据
        test_data, test_loader = data_provider(self.args, flag='test')
        
        # 根据模式配置模型和优化器
        if freeze_backbone:
            # 模式A: 冻结Backbone，只让Memory Unit学习
            print("\n配置模式A:")
            print("  - 冻结Backbone参数")
            print("  - Memory Unit通过内置机制自动更新")
            print("  - cache跨batch传递，实现持续学习")
            
            self.model.eval()  # eval模式（但不影响NeuralMemory的内部更新）
            
            # 冻结Backbone
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            
            optimizer = None  # 不需要外部optimizer
            
        else:
            # 模式B: P和M都学习
            print("\n配置模式B:")
            print(f"  - Backbone和Memory Unit都参与学习")
            print(f"  - 使用在线学习率: {self.args.online_lr}")
            print("  - cache跨batch传递，实现持续学习")
            
            self.model.train()
            
            # 解冻所有参数
            for param in self.model.parameters():
                param.requires_grad = True
            
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.args.online_lr
            )
        
        print("="*70 + "\n")
        
        # 在线测试循环
        preds = []
        trues = []
        losses = []
        
        # 🔑 关键：初始化cache以维护记忆状态
        cache = None
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            # 🔑 关键：传入cache维持记忆状态
            outputs, batch_y, cache = self._process_one_batch(batch_x, batch_y, cache)
            
            # 计算损失
            loss = self.criterion(outputs, batch_y)
            loss_per_sample = F.mse_loss(outputs, batch_y, reduction='none').mean(dim=(1,2))
            losses.extend(loss_per_sample.detach().cpu().numpy().tolist())
            
            # 在线更新（仅模式B需要）
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                
                if self.args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.model.parameters()), 
                        self.args.clip_grad
                    )
                
                optimizer.step()
            
            # 收集预测和真实值
            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            
            preds.append(pred)
            trues.append(true)
            
            # 打印进度
            if (i + 1) % 10 == 0:
                recent_loss = np.mean(losses[-320:]) if len(losses) >= 320 else np.mean(losses)
                print(f"  进度: Batch [{i+1}/{len(test_loader)}] | Recent Loss: {recent_loss:.6f}")
        
        return self._finalize_test_results(preds, trues, losses, test_data, freeze_backbone)
    
    def _finalize_test_results(self, preds, trues, losses, test_data, freeze_backbone):
        """整理测试结果并进行评估、可视化"""
        
        # 合并结果
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        print(f"\n预测形状: {preds.shape}")
        print(f"真实值形状: {trues.shape}")
        
        # 反标准化
        if hasattr(test_data, 'inverse_transform'):
            preds_orig = test_data.inverse_transform(preds.reshape(-1, preds.shape[-1]))
            trues_orig = test_data.inverse_transform(trues.reshape(-1, trues.shape[-1]))
            preds_orig = preds_orig.reshape(preds.shape)
            trues_orig = trues_orig.reshape(trues.shape)
        else:
            preds_orig = preds
            trues_orig = trues
        
        # 计算指标
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds_orig, trues_orig)
        print_metrics(mae, mse, rmse, mape, mspe, rse, corr)
        
        # 保存结果
        if self.args.save_pred:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            mode_suffix = 'memory_only' if freeze_backbone else 'full_model'
            result_path = os.path.join(self.args.result_path, f'{self.setting}_online_{mode_suffix}_{timestamp}.csv')
            save_results(trues_orig, preds_orig, losses, result_path)
            print(f"✓ 结果已保存到: {result_path}")
        
        # 可视化
        if self.args.save_fig:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            mode_suffix = 'memory_only' if freeze_backbone else 'full_model'
            fig_path = os.path.join(self.args.fig_path, f'{self.setting}_online_{mode_suffix}_{timestamp}.jpg')
            visual_comprehensive(
                trues_orig.flatten(), 
                preds_orig.flatten(), 
                losses if len(losses) > 0 else None,
                fig_path,
                train_size=None
            )
            print(f"✓ 可视化已保存到: {fig_path}")
        
        return mae, mse, rmse, mape, mspe, rse


if __name__ == '__main__':
    print("训练器模块已创建")
    print("请通过titans_main.py运行完整的训练流程")

