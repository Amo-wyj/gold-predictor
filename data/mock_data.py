"""
模拟数据生成器 - 用于离线测试
基于黄金历史特征生成逼真的模拟数据
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_mock_gold_data(days: int = 90) -> pd.DataFrame:
    """生成模拟黄金价格数据"""
    np.random.seed(42)

    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    dates = pd.date_range(end=end_date, periods=days, freq='D')
    
    # 基础价格 + 趋势
    base_price = 2020.0
    trend = np.linspace(0, 50, days)  # 轻微上涨趋势
    
    # 加入随机波动
    noise = np.random.randn(days) * 15
    
    # 加入周期性波动
    cycle = 20 * np.sin(np.linspace(0, 4 * np.pi, days))
    
    # 生成价格
    prices = base_price + trend + noise + cycle
    prices = np.maximum(prices, 1900)  # 设定下限
    
    # 生成 OHLC
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        volatility = abs(np.random.randn()) * 10 + 5
        high = close + abs(np.random.randn()) * volatility
        low = close - abs(np.random.randn()) * volatility
        open_price = low + (high - low) * np.random.random()
        volume = int(np.random.randint(10000, 50000))
        
        data.append({
            'date': date,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    return df


def generate_mock_macro_data(days: int = 250) -> dict:
    """生成模拟宏观数据"""
    np.random.seed(42)
    
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    # macro 数据从与 gold 相同的结束日期向前取 days 天
    dates = pd.date_range(end=end_date, periods=days, freq='D')

    return {
        'dxy': pd.Series(
            104 + np.cumsum(np.random.randn(days) * 0.3),
            index=dates
        ).clip(100, 110),
        'dgs10': pd.Series(
            4.2 + np.cumsum(np.random.randn(days) * 0.05),
            index=dates
        ).clip(3.5, 5.0),
        'dgs5': pd.Series(
            4.0 + np.cumsum(np.random.randn(days) * 0.04),
            index=dates
        ).clip(3.3, 4.8),
        'real_rate': pd.Series(
            1.5 + np.cumsum(np.random.randn(days) * 0.02),
            index=dates
        ).clip(0.5, 2.5),
        'vix': pd.Series(
            18 + np.cumsum(np.random.randn(days) * 0.5),
            index=dates
        ).clip(10, 35),
        'oil': pd.Series(
            75 + np.cumsum(np.random.randn(days) * 0.8),
            index=dates
        ).clip(60, 95),
        'silver': pd.Series(
            23 + np.cumsum(np.random.randn(days) * 0.2),
            index=dates
        ).clip(20, 28),
        'cpi': pd.Series(
            np.linspace(295, 298, days),
            index=dates
        ),
        'fed_rate': pd.Series(
            5.25 * np.ones(days),
            index=dates
        ),
        'unemployment': pd.Series(
            3.8 + np.cumsum(np.random.randn(days) * 0.02),
            index=dates
        ).clip(3.5, 4.5)
    }


def get_latest_mock_price() -> dict:
    """获取最新的模拟价格"""
    gold_df = generate_mock_gold_data(days=2)
    latest = gold_df.iloc[-1]
    
    return {
        'price': latest['close'],
        'open': latest['open'],
        'high': latest['high'],
        'low': latest['low'],
        'change_pct': round(np.random.uniform(-1.5, 1.5), 2),
        'timestamp': datetime.now().isoformat()
    }
