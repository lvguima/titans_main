"""
数据工厂
根据配置自动选择和加载相应的数据集
"""

from torch.utils.data import DataLoader
from .data_loader import (
    TimeSeriesDataset,
    SyntheticDriftDataset,
    ETTDataset
)


# 数据集字典
DATA_DICT = {
    'synthetic': SyntheticDriftDataset,
    'custom': TimeSeriesDataset,
    'ETTh1': ETTDataset,
    'ETTh2': ETTDataset,
    'ETTm1': ETTDataset,
    'ETTm2': ETTDataset,
}


def data_provider(args, flag='train'):
    """
    数据提供器工厂函数
    
    Args:
        args: 参数配置对象
        flag: 'train', 'val', 'test'
    
    Returns:
        dataset: 数据集对象
        data_loader: 数据加载器对象
    """
    # 获取数据集类
    Data = DATA_DICT.get(args.data, TimeSeriesDataset)
    
    # 根据flag设置参数
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size  # 测试时使用与训练相同的batch_size，以便观察记忆累积效果
    elif flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:  # train
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
    
    # 创建数据集
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        scale=True
    )
    
    print(f"{flag.upper()} 数据集: {len(data_set)} 样本")
    
    # 创建数据加载器
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=True if args.use_gpu else False
    )
    
    return data_set, data_loader


def get_data_info(args):
    """
    获取数据集信息（输入输出维度）
    
    Args:
        args: 参数配置对象
    
    Returns:
        input_dim: 输入特征维度
        output_dim: 输出特征维度
    """
    # 创建一个临时数据集来获取维度信息
    Data = DATA_DICT.get(args.data, TimeSeriesDataset)
    
    temp_dataset = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag='train',
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        scale=True
    )
    
    input_dim = temp_dataset.input_dim
    output_dim = temp_dataset.output_dim
    
    return input_dim, output_dim


if __name__ == '__main__':
    """测试数据工厂"""
    import argparse
    
    # 创建测试参数
    args = argparse.Namespace()
    args.data = 'synthetic'
    args.root_path = '../'
    args.data_path = 'realistic_drift_data.csv'
    args.seq_len = 64
    args.label_len = 0
    args.pred_len = 1
    args.features = 'M'
    args.target = 'target'
    args.batch_size = 32
    args.num_workers = 0
    args.use_gpu = True
    
    print("测试data_provider...")
    
    # 获取数据信息
    input_dim, output_dim = get_data_info(args)
    print(f"\n数据集信息:")
    print(f"  输入维度: {input_dim}")
    print(f"  输出维度: {output_dim}")
    
    # 创建数据加载器
    train_data, train_loader = data_provider(args, flag='train')
    val_data, val_loader = data_provider(args, flag='val')
    test_data, test_loader = data_provider(args, flag='test')
    
    print(f"\n数据加载器:")
    print(f"  训练集: {len(train_data)} 样本, {len(train_loader)} batches")
    print(f"  验证集: {len(val_data)} 样本, {len(val_loader)} batches")
    print(f"  测试集: {len(test_data)} 样本, {len(test_loader)} batches")
    
    # 测试一个batch
    for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
        print(f"\n样本batch形状:")
        print(f"  batch_x: {batch_x.shape}")
        print(f"  batch_y: {batch_y.shape}")
        break
    
    print("\n✓ 数据工厂测试通过!")

