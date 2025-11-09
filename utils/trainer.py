"""
训练器模块
处理模型的训练、验证和测试
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

from models.titans_mac import build_model
from dataset.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, visual_loss_curve, visual_comprehensive, save_results, create_experiment_folder, get_device
from utils.metrics import metric, print_metrics


class Trainer:
    """Titans模型训练器"""
    
    def __init__(self, args):
        """
        初始化训练器
        
        Args:
            args: 参数配置对象
        """
        self.args = args
        self.device = get_device(args)
        
        # 设置实验标识
        self.model_id = args.model_id
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
        print("构建Titans MAC模型...")
        print("="*70)
        
        # 自动获取数据维度
        from dataset.data_factory import get_data_info
        input_dim, output_dim = get_data_info(self.args)
        
        # 更新args中的维度信息
        self.args.input_dim = input_dim
        self.args.output_dim = output_dim
        
        # 构建模型
        model = build_model(self.args).to(self.device)
        
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
    
    def _get_optimizer(self):
        """获取优化器"""
        if self.args.optimizer == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.learning_rate,
                momentum=0.9,
                weight_decay=self.args.weight_decay
            )
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        
        return optimizer
    
    def _process_one_batch(self, batch_x, batch_y, neural_mem_state=None):
        """
        处理一个batch
        
        Args:
            batch_x: 输入数据 [batch_size, seq_len, input_dim]
            batch_y: 目标数据 [batch_size, label_len + pred_len, output_dim]
            neural_mem_state: NeuralMemory的cache状态 (seq_index, kv_caches, neural_mem_caches)
        
        Returns:
            outputs: 预测结果 [batch_size, pred_len, output_dim]
            batch_y: 真实标签（只取pred_len部分）
            next_neural_mem_state: 更新后的cache状态（如果传入了cache）
        """
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        
        # 🔑 关键：cache机制控制记忆的累积学习
        if neural_mem_state is not None:
            # 在线学习模式：传入cache并获取更新后的cache
            # cache维护seq_index、kv_caches和neural_mem_caches
            # 这样NeuralMemory可以跨batch累积学习
            outputs, next_neural_mem_state = self.model(
                batch_x, 
                cache=neural_mem_state, 
                return_cache=True
            )
            
            # 处理longterm_mem token的特殊情况
            # 原始库在某些位置会返回None（跳过longterm_mem tokens）
            if outputs is None:
                # 返回空预测，但保留cache供下一个batch使用
                return None, batch_y, next_neural_mem_state
        else:
            # 训练模式或无记忆累积模式：不维护cache
            # 每个batch独立处理
            outputs = self.model(batch_x)
            next_neural_mem_state = None
        
        # 提取预测部分的标签
        if self.args.label_len > 0:
            batch_y = batch_y[:, -self.args.pred_len:, :]
        
        # 调整输出形状以匹配标签
        if outputs.dim() == 2:
            outputs = outputs.unsqueeze(1)  # [B, 1, D]
        
        return outputs, batch_y, next_neural_mem_state
    
    def train(self):
        """训练模型"""
        print("\n" + "="*70)
        print(f"开始训练: {self.setting}")
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
        print(f"\n训练完成！总用时: {total_time:.2f}s")
        
        # 保存训练loss曲线（已禁用，用户不需要）
        # if self.args.save_fig and len(train_loss_history) > 0:
        #     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        #     loss_curve_path = os.path.join(self.args.fig_path, f'{self.setting}_loss_curve_{timestamp}.jpg')
        #     visual_loss_curve(train_loss_history, val_loss_history, loss_curve_path)
        #     print(f"✓ 训练loss曲线已保存到: {loss_curve_path}")
        
        return self.model
    
    def _train_epoch(self, train_loader, epoch):
        """训练一个epoch（支持梯度累积和稀疏标签）"""
        self.model.train()
        train_loss = []
        
        # 梯度累积：只在累积到指定步数时才更新参数
        update_freq = self.args.train_update_freq
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            # 前向传播
            outputs, batch_y, _ = self._process_one_batch(batch_x, batch_y)
            
            # 训练阶段：总是使用所有标签（稀疏标签只影响测试阶段）
            # 计算损失（需要除以累积步数，以保持梯度尺度一致）
            loss = self.criterion(outputs, batch_y) / update_freq
            train_loss.append(loss.item() * update_freq)  # 记录原始损失值
            
            # 反向传播（累积梯度）
            loss.backward()
            
            # 每隔 update_freq 步更新一次参数
            if (i + 1) % update_freq == 0 or (i + 1) == len(train_loader):
                # 梯度裁剪
                if self.args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                
                # 更新参数
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # 打印日志
            if (i + 1) % self.args.log_interval == 0:
                print(f"  Epoch [{epoch + 1}] Step [{i + 1}/{len(train_loader)}] | Loss: {loss.item() * update_freq:.6f}")
        
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
    
    def test(self, load_checkpoint=True):
        """测试模型（三种模式：静态/在线记忆/在线全模型）"""
        print("\n" + "="*70)
        print(f"开始测试: {self.setting}")
        print("="*70)
        
        # 加载checkpoint
        if load_checkpoint:
            checkpoint_path = os.path.join(self.path, 'checkpoint.pth')
            if os.path.exists(checkpoint_path):
                self.model.load_state_dict(torch.load(checkpoint_path))
                print(f"✓ 已加载模型: {checkpoint_path}")
        
        # 获取测试数据
        test_data, test_loader = data_provider(self.args, flag='test')
        
        # 根据是否启用在线学习选择测试方式
        if self.args.online_learning:
            return self._test_with_online_learning(test_data, test_loader)
        else:
            return self._test_no_memory_accumulation(test_data, test_loader)
    
    def _test_no_memory_accumulation(self, test_data, test_loader):
        """
        模式A：无记忆累积模式
        
        行为：
        - 每个batch独立处理，不维护cache
        - NeuralMemory在batch内自动更新（这是原始库的固有机制，无法关闭）
        - batch之间不累积记忆状态
        
        相当于：短期记忆模式，测试预训练模型的即时泛化能力
        """
        print("=" * 70)
        print("测试模式A: 无记忆累积（每个batch独立处理）")
        print("  - NeuralMemory在batch内自动更新（原始库固有机制）")
        print("  - batch之间不传递cache，记忆状态不累积")
        print("  - 相当于'短期记忆'模式")
        print("=" * 70 + "\n")
        
        self.model.eval()
        
        preds = []
        trues = []
        losses = []
        
        # 🔑 关键修改：不使用torch.no_grad()，让NeuralMemory可以正常计算surprise
        # torch.no_grad()会禁用NeuralMemory内部的torch.func.grad计算
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            # 🔑 关键：不传cache，每个batch都是独立的
            outputs, batch_y, _ = self._process_one_batch(batch_x, batch_y, neural_mem_state=None)
            
            # 记录每个样本的loss
            with torch.no_grad():  # 只在计算loss时使用no_grad
                loss_per_sample = F.mse_loss(outputs, batch_y, reduction='none').mean(dim=(1,2))
                losses.extend(loss_per_sample.detach().cpu().numpy().tolist())
            
            # 收集预测和真实值
            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            
            preds.append(pred)
            trues.append(true)
            
            # 打印进度
            if (i + 1) % 10 == 0:  # 改为每10个batch打印一次
                recent_loss = np.mean(losses[-320:]) if len(losses) >= 320 else np.mean(losses)
                print(f"  进度: Batch [{i+1}/{len(test_loader)}] | Recent Loss: {recent_loss:.6f}")
        
        return self._finalize_test_results(preds, trues, losses, test_data)
    
    def _test_with_online_learning(self, test_data, test_loader):
        """
        在线学习测试（Titans持续学习模式）
        
        核心机制：
        - NeuralMemory在forward时自动完成 store + retrieve
        - Store: 计算grad(MSE(M(k), v))，使用自适应lr/动量/遗忘更新记忆权重
        - Retrieve: 返回M(q)作为context
        - 🔑 cache跨batch传递，实现记忆的累积学习
        
        两种子模式：
        模式B: online_update_memory_only=True
          - 只让NeuralMemory自动更新，backbone冻结
          - 轻量级适应，避免灾难性遗忘
        
        模式C: online_update_memory_only=False
          - NeuralMemory自动更新 + 反向传播更新backbone
          - 最大适应能力，但可能过拟合
        """
        mode_name = "模式B: 在线学习 - 仅记忆更新" if self.args.online_update_memory_only else "模式C: 在线学习 - 全模型更新"
        
        print("=" * 70)
        print(mode_name)
        print(f"  - 更新策略: {'仅NeuralMemory自适应' if self.args.online_update_memory_only else 'NeuralMemory + Backbone同时更新'}")
        
        if self.args.online_update_memory_only:
            # === 模式B: 信任NeuralMemory自动更新 ===
            print("  - NeuralMemory在forward时自动更新（自适应lr、动量、遗忘）")
            print("  - Backbone完全冻结")
            print("  - cache跨batch传递，记忆状态累积学习")
            print("=" * 70 + "\n")
            
            self.model.eval()  # 冻结BN/Dropout
            if hasattr(self.model, 'freeze_non_memory_params'):
                self.model.freeze_non_memory_params()
            
            # 不创建optimizer！完全信任NeuralMemory的自包含更新
            online_optimizer = None
        else:
            # === 模式C: NeuralMemory自动更新 + 外部optimizer更新backbone ===
            print(f"  - NeuralMemory自动更新 + Backbone通过反向传播更新（lr={self.args.online_lr}）")
            print("  - cache跨batch传递，记忆状态累积学习")
            print("=" * 70 + "\n")
            
            self.model.train()
            online_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.args.online_lr
            )
        
        preds = []
        trues = []
        losses = []
        
        # 🔑 关键：初始化NeuralMemory cache以维护在线学习状态
        # cache格式: (seq_index, kv_caches, neural_mem_caches)
        # 首次调用时传入None，模型会自动初始化
        # 后续每次forward会返回更新后的cache，持续传入以实现记忆的累积学习
        neural_mem_state = None
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            # 稀疏标签：控制何时使用真实标签
            use_label = True
            if self.args.sparse_label:
                use_label = (i % self.args.sparse_step == 0)
            
            if use_label:
                # === 有标签：NeuralMemory可以学习 ===
                # 🔑 关键：传入neural_mem_state维持记忆状态！
                outputs, batch_y, neural_mem_state = self._process_one_batch(batch_x, batch_y, neural_mem_state)
                
                # 处理longterm_mem token的特殊情况（outputs可能为None）
                if outputs is None:
                    continue
                
                loss = self.criterion(outputs, batch_y)
                # 记录每个样本的loss（reduction='none'然后flatten）
                loss_per_sample = F.mse_loss(outputs, batch_y, reduction='none').mean(dim=(1,2))
                losses.extend(loss_per_sample.detach().cpu().numpy().tolist())
                
                # 只在模式C时才反向传播更新backbone
                if online_optimizer is not None:
                    online_optimizer.zero_grad()
                    loss.backward()
                    
                    if self.args.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(
                            filter(lambda p: p.requires_grad, self.model.parameters()), 
                            self.args.clip_grad
                        )
                    
                    online_optimizer.step()
                # 模式B时：NeuralMemory已在forward中自动更新，state已保存在cache中
            else:
                # === 无标签：只预测，不更新（模拟稀疏标签） ===
                with torch.no_grad():
                    outputs, batch_y, neural_mem_state = self._process_one_batch(batch_x, batch_y, neural_mem_state)
                    
                    # 处理longterm_mem token的特殊情况
                    if outputs is None:
                        continue
                    
                    # 记录每个样本的loss
                    loss_per_sample = F.mse_loss(outputs, batch_y, reduction='none').mean(dim=(1,2))
                    losses.extend(loss_per_sample.detach().cpu().numpy().tolist())
            
            # 保存预测结果
            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            
            preds.append(pred)
            trues.append(true)
            
            # 打印进度
            if (i + 1) % 10 == 0:  # 改为每10个batch打印一次（因为现在batch_size=32）
                label_marker = " [稀疏-无标签]" if self.args.sparse_label and not use_label else ""
                recent_loss = np.mean(losses[-320:]) if len(losses) >= 320 else np.mean(losses)  # 最近320个样本(约10个batch)
                print(f"  进度: Batch [{i+1}/{len(test_loader)}] | Recent Loss: {recent_loss:.6f}{label_marker}")
        
        return self._finalize_test_results(preds, trues, losses, test_data)
    
    def _finalize_test_results(self, preds, trues, losses, test_data):
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
            mode_suffix = 'online' if self.args.online_learning else 'static'
            result_path = os.path.join(self.args.result_path, f'{self.setting}_test_{mode_suffix}_{timestamp}.csv')
            save_results(trues_orig, preds_orig, losses, result_path)
        
        # 可视化（JPG格式）
        if self.args.save_fig:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            mode_suffix = 'online' if self.args.online_learning else 'static'
            
            # 综合可视化（类似experiment_comparison.py的风格）
            comprehensive_fig_path = os.path.join(self.args.fig_path, f'{self.setting}_test_{mode_suffix}_{timestamp}.jpg')
            train_size = None  # 如果是单独测试，无法得知训练集大小
            visual_comprehensive(
                trues_orig.flatten(), 
                preds_orig.flatten(), 
                losses if len(losses) > 0 else None,
                comprehensive_fig_path,
                train_size=train_size
            )
            print(f"✓ 测试结果可视化已保存到: {comprehensive_fig_path}")
        
        return mae, mse, rmse, mape, mspe, rse


if __name__ == '__main__':
    """测试训练器"""
    print("训练器模块已创建")
    print("请通过titans_main.py运行完整的训练流程")

