"""
时间序列数据加载器
支持多种数据集格式，参考Time-Series-Library设计
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class TimeSeriesDataset(Dataset):
    """
    通用时间序列数据集类
    支持多变量和单变量预测任务
    """
    
    def __init__(self, 
                 root_path, 
                 data_path='data.csv',
                 flag='train', 
                 size=None,
                 features='M', 
                 target='target',
                 scale=True,
                 sparse_label=False,
                 sparse_step=4):
        """
        Args:
            root_path: 数据根目录
            data_path: 数据文件名
            flag: 'train', 'val', 'test'
            size: [seq_len, label_len, pred_len]
            features: 'M'-多变量预测多变量, 'S'-单变量, 'MS'-多变量预测单变量
            target: 目标列名
            scale: 是否标准化
            sparse_label: 是否使用稀疏标签
            sparse_step: 稀疏标签步长
        """
        # 尺寸配置
        if size is None:
            self.seq_len = 96
            self.label_len = 0
            self.pred_len = 1
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # 基础配置
        assert flag in ['train', 'val', 'test']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        self.flag = flag
        
        self.features = features
        self.target = target
        self.scale = scale
        self.sparse_label = sparse_label
        self.sparse_step = sparse_step
        
        self.root_path = root_path
        self.data_path = data_path
        
        # 读取数据
        self.__read_data__()
    
    def __read_data__(self):
        """读取和处理数据"""
        self.scaler = StandardScaler()
        
        # 读取CSV文件
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # 数据集划分边界
        # 默认: 70% train, 10% val, 20% test
        n_total = len(df_raw)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.1)
        
        border1s = [0, n_train - self.seq_len, n_train + n_val - self.seq_len]
        border2s = [n_train, n_train + n_val, n_total]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # 选择特征列
        if self.features == 'M' or self.features == 'MS':
            # 多变量：排除时间列和可能的索引列
            cols_data = df_raw.columns[1:] if 'date' in df_raw.columns[0].lower() else df_raw.columns
            # 过滤数值列
            numeric_cols = df_raw[cols_data].select_dtypes(include=[np.number]).columns
            df_data = df_raw[numeric_cols]
        elif self.features == 'S':
            # 单变量
            if self.target in df_raw.columns:
                df_data = df_raw[[self.target]]
            else:
                # 如果没有指定target列，使用最后一列
                df_data = df_raw.iloc[:, -1:]
        else:
            raise ValueError(f"Unsupported features type: {self.features}")
        
        # 标准化
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        # 处理时间戳（如果有）
        self.data_stamp = None
        if 'date' in df_raw.columns or 'time' in df_raw.columns:
            time_col = 'date' if 'date' in df_raw.columns else 'time'
            df_stamp = df_raw[[time_col]][border1:border2]
            # 简单的时间特征（可以扩展）
            self.data_stamp = np.arange(len(df_stamp)).reshape(-1, 1)
        
        # 提取数据
        self.data_x = data[border1:border2]
        
        if self.features == 'MS':
            # 多变量预测单变量：y只取target列
            target_idx = df_data.columns.get_loc(self.target) if self.target in df_data.columns else -1
            self.data_y = data[border1:border2, target_idx:target_idx+1]
        else:
            self.data_y = data[border1:border2]
        
        # 记录输入输出维度
        self.input_dim = self.data_x.shape[-1]
        self.output_dim = self.data_y.shape[-1]
        
        print(f"✓ {self.flag.upper()} 数据加载完成: "
              f"样本数={len(self.data_x)}, "
              f"输入维度={self.input_dim}, "
              f"输出维度={self.output_dim}")
    
    def __getitem__(self, index):
        """获取一个样本"""
        # 输入序列
        s_begin = index
        s_end = s_begin + self.seq_len
        
        # 输出序列
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        
        # 时间戳
        seq_x_mark = self.data_stamp[s_begin:s_end] if self.data_stamp is not None else np.zeros((self.seq_len, 1))
        seq_y_mark = self.data_stamp[r_begin:r_end] if self.data_stamp is not None else np.zeros((self.label_len + self.pred_len, 1))
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        """反标准化"""
        return self.scaler.inverse_transform(data)


class SyntheticDriftDataset(Dataset):
    """
    合成漂移数据集
    用于测试模型对分布漂移的适应能力
    """
    
    def __init__(self, 
                 root_path,
                 data_path='realistic_drift_data.csv',
                 flag='train',
                 size=None,
                 features='M',
                 target='target',
                 scale=True):
        
        if size is None:
            self.seq_len = 64
            self.label_len = 0
            self.pred_len = 1
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        assert flag in ['train', 'val', 'test']
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        
        self.root_path = root_path
        self.data_path = data_path
        
        self.__read_data__()
    
    def __read_data__(self):
        """读取合成漂移数据"""
        self.scaler = StandardScaler()
        
        # 读取数据
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # 固定划分点（针对漂移数据集）
        train_split = 3500
        test_split = 4000
        
        if self.flag == 'train':
            border1, border2 = 0, train_split
        elif self.flag == 'val':
            border1, border2 = train_split - self.seq_len, test_split
        else:  # test
            border1, border2 = test_split - self.seq_len, len(df_raw)
        
        # 提取特征和目标
        if 'feature1' in df_raw.columns:
            feature_cols = [col for col in df_raw.columns if col.startswith('feature')]
            df_features = df_raw[feature_cols]
        else:
            # 排除日期列，只保留数值列
            cols_data = df_raw.columns[1:] if 'date' in str(df_raw.columns[0]).lower() or 'time' in str(df_raw.columns[0]).lower() else df_raw.columns
            numeric_cols = df_raw[cols_data].select_dtypes(include=[np.number]).columns.tolist()
            # 如果有目标列，从特征中排除
            if self.target in numeric_cols:
                feature_cols = [col for col in numeric_cols if col != self.target]
                df_features = df_raw[feature_cols]
            else:
                df_features = df_raw[numeric_cols[:-1]] if len(numeric_cols) > 1 else df_raw[numeric_cols]
        
        if self.target in df_raw.columns:
            df_target = df_raw[[self.target]]
        else:
            # 使用最后一个数值列作为目标
            numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
            df_target = df_raw[[numeric_cols[-1]]] if numeric_cols else df_raw.iloc[:, -1:]
        
        # 标准化
        if self.scale:
            train_features = df_features[:train_split]
            train_target = df_target[:train_split]
            
            self.scaler_x = StandardScaler()
            self.scaler_y = StandardScaler()
            
            self.scaler_x.fit(train_features.values)
            self.scaler_y.fit(train_target.values)
            
            features = self.scaler_x.transform(df_features.values)
            targets = self.scaler_y.transform(df_target.values)
        else:
            features = df_features.values
            targets = df_target.values
            self.scaler_x = None
            self.scaler_y = None
        
        self.data_x = features[border1:border2]
        self.data_y = targets[border1:border2]
        
        self.input_dim = self.data_x.shape[-1]
        self.output_dim = self.data_y.shape[-1]
        
        print(f"✓ {self.flag.upper()} Drift数据加载: "
              f"样本数={len(self.data_x)}, "
              f"输入维度={self.input_dim}, "
              f"输出维度={self.output_dim}")
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        
        seq_x_mark = np.zeros((self.seq_len, 1))
        seq_y_mark = np.zeros((self.pred_len, 1))
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data, is_target=True):
        """反标准化"""
        if not self.scale:
            return data
        scaler = self.scaler_y if is_target else self.scaler_x
        return scaler.inverse_transform(data)


class ETTDataset(Dataset):
    """
    ETT (Electricity Transformer Temperature) 数据集
    包括ETTh1, ETTh2, ETTm1, ETTm2
    """
    
    def __init__(self,
                 root_path,
                 data_path='ETTh1.csv',
                 flag='train',
                 size=None,
                 features='M',
                 target='OT',
                 scale=True):
        
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        assert flag in ['train', 'val', 'test']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        
        self.__read_data__()
    
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # ETT数据集标准划分
        # 小时数据 or 分钟数据
        is_hourly = 'ETTh' in self.data_path
        
        if is_hourly:
            # 12个月训练，4个月验证，4个月测试
            border1s = [0, 12*30*24 - self.seq_len, 12*30*24 + 4*30*24 - self.seq_len]
            border2s = [12*30*24, 12*30*24 + 4*30*24, 12*30*24 + 8*30*24]
        else:
            # 分钟数据（4倍密度）
            border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4 + 4*30*24*4 - self.seq_len]
            border2s = [12*30*24*4, 12*30*24*4 + 4*30*24*4, 12*30*24*4 + 8*30*24*4]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # 特征选择
        if self.features == 'M' or self.features == 'MS':
            # 排除date列，只保留数值列
            cols_data = df_raw.columns[1:] if 'date' in str(df_raw.columns[0]).lower() else df_raw.columns
            numeric_cols = df_raw[cols_data].select_dtypes(include=[np.number]).columns
            df_data = df_raw[numeric_cols]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        
        # 标准化
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        self.data_x = data[border1:border2]
        
        if self.features == 'MS':
            target_idx = list(df_data.columns).index(self.target)
            self.data_y = data[border1:border2, target_idx:target_idx+1]
        else:
            self.data_y = data[border1:border2]
        
        self.input_dim = self.data_x.shape[-1]
        self.output_dim = self.data_y.shape[-1]
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = np.zeros((self.seq_len, 1))
        seq_y_mark = np.zeros((self.label_len + self.pred_len, 1))
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


if __name__ == '__main__':
    """测试数据加载器"""
    print("测试TimeSeriesDataset...")
    
    # 测试参数
    root_path = '../'
    data_path = 'realistic_drift_data.csv'
    
    # 创建数据集
    dataset = SyntheticDriftDataset(
        root_path=root_path,
        data_path=data_path,
        flag='train',
        size=[64, 0, 1],
        features='M',
        scale=True
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"输入维度: {dataset.input_dim}")
    print(f"输出维度: {dataset.output_dim}")
    
    # 获取一个样本
    x, y, x_mark, y_mark = dataset[0]
    print(f"\n样本形状:")
    print(f"  x: {x.shape}")
    print(f"  y: {y.shape}")
    print(f"  x_mark: {x_mark.shape}")
    print(f"  y_mark: {y_mark.shape}")
    
    print("\n✓ 数据加载器测试通过!")

