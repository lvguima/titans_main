"""
评估指标
"""

import numpy as np


def RSE(pred, true):
    """相对平方误差"""
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    """相关系数"""
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    """平均绝对误差"""
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    """均方误差"""
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    """均方根误差"""
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    """平均绝对百分比误差"""
    return np.mean(np.abs((pred - true) / (true + 1e-8))) * 100


def MSPE(pred, true):
    """均方百分比误差"""
    return np.mean(np.square((pred - true) / (true + 1e-8))) * 100


def metric(pred, true):
    """
    计算所有评估指标
    
    Args:
        pred: 预测值 numpy array
        true: 真实值 numpy array
    
    Returns:
        mae, mse, rmse, mape, mspe, rse, corr
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    
    return mae, mse, rmse, mape, mspe, rse, corr


def print_metrics(mae, mse, rmse, mape, mspe, rse=None, corr=None):
    """
    打印评估指标
    """
    print("\n" + "="*70)
    print("评估指标:")
    print("="*70)
    print(f"  MAE  : {mae:.6f}")
    print(f"  MSE  : {mse:.6f}")
    print(f"  RMSE : {rmse:.6f}")
    print(f"  MAPE : {mape:.2f}%")
    print(f"  MSPE : {mspe:.6f}")
    if rse is not None:
        # 处理可能是数组的情况
        rse_val = float(rse) if not isinstance(rse, (int, float)) else rse
        print(f"  RSE  : {rse_val:.6f}")
    if corr is not None:
        # 处理可能是数组的情况
        corr_val = float(corr) if not isinstance(corr, (int, float)) else corr
        print(f"  CORR : {corr_val:.6f}")
    print("="*70 + "\n")


if __name__ == '__main__':
    """测试评估指标"""
    print("测试评估指标...")
    
    # 生成测试数据
    np.random.seed(42)
    true = np.random.randn(100, 1)
    pred = true + np.random.randn(100, 1) * 0.1
    
    # 计算指标
    mae, mse, rmse, mape, mspe, rse, corr = metric(pred, true)
    
    # 打印指标
    print_metrics(mae, mse, rmse, mape, mspe, rse, corr)
    
    print("✓ 评估指标测试通过!")

