"""
新的训练器模块 - 适配模块化框架

这个训练器与models/framework.py中的ContinualForecaster配合使用
支持：
- 模块化的Backbone和Memory Unit
- 预训练阶段：同时训练P和M
- 在线测试阶段：
    - 模式A: 仅M学习（P冻结）
    - 模式B: M和P都学习
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
        
        # 根据是否传入cache决定是否维护记忆状态
        if cache is not None:
            outputs, next_cache = self.model(batch_x, cache=cache, return_cache=True)
        else:
            outputs, _ = self.model(batch_x, cache=None, return_cache=False)
            next_cache = None
        
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
        """训练一个epoch"""
        self.model.train()
        train_loss = []
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            # 前向传播（训练时不维护cache，每个batch独立）
            outputs, batch_y, _ = self._process_one_batch(batch_x, batch_y)
            
            # 计算损失
            loss = self.criterion(outputs, batch_y)
            train_loss.append(loss.item())
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            
            # 更新参数
            self.optimizer.step()
            
            # 打印日志
            if (i + 1) % self.args.log_interval == 0:
                print(f"  Epoch [{epoch + 1}] Step [{i + 1}/{len(train_loader)}] | Loss: {loss.item():.6f}")
        
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
    print("新的训练器模块已创建")
    print("请通过titans_main.py运行完整的训练流程")

