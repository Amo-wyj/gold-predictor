"""
Yahoo Finance 数据采集器
采集：黄金、白银、美元指数、VIX、原油
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

from config import DATA_SOURCES

logger = logging.getLogger(__name__)


class YahooFinanceCollector:
    """Yahoo Finance 市场数据采集"""
    
    def __init__(self):
        self.tickers = DATA_SOURCES["yahoo"]
        self.cache: Dict[str, pd.DataFrame] = {}
    
    def fetch(self, symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
        """获取历史数据"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days}d", auto_adjust=True)
            
            if df.empty:
                logger.warning(f"[Yahoo] {symbol} 无数据")
                return None
            
            df = df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume'
            })
            df.index = pd.to_datetime(df.index).tz_localize(None)
            logger.info(f"[Yahoo] {symbol}: {len(df)} 条记录, 最新 {df.index[-1].date()}")
            return df
            
        except Exception as e:
            logger.error(f"[Yahoo] {symbol} 获取失败: {e}")
            return None
    
    def fetch_all(self, days: int = 365) -> Dict[str, pd.DataFrame]:
        """批量获取所有市场数据"""
        results = {}
        
        for name, symbol in self.tickers.items():
            df = self.fetch(symbol, days)
            if df is not None:
                results[name] = df
        
        return results
    
    def get_latest(self, symbol: str) -> Optional[dict]:
        """获取最新报价"""
        df = self.fetch(symbol, days=5)
        if df is not None and not df.empty:
            latest = df.iloc[-1]
            return {
                "symbol": symbol,
                "price": float(latest['close']),
                "change": float(latest['close'] - df.iloc[-2]['close']) if len(df) > 1 else 0,
                "change_pct": float((latest['close'] / df.iloc[-2]['close'] - 1) * 100) if len(df) > 1 else 0,
                "timestamp": latest.name.isoformat()
            }
        return None


# ===== 便捷函数 =====
def get_gold_price() -> Optional[dict]:
    """获取黄金最新价格"""
    collector = YahooFinanceCollector()
    return collector.get_latest("GC=F")

def get_market_overview() -> Dict[str, dict]:
    """获取市场概览（黄金 + 相关品种）"""
    collector = YahooFinanceCollector()
    symbols = ["GC=F", "SI=F", "DX-Y.NYB", "^VIX", "CL=F"]
    results = {}
    
    for symbol in symbols:
        data = collector.get_latest(symbol)
        if data:
            results[symbol] = data
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    
    collector = YahooFinanceCollector()
    
    print("\n" + "="*60)
    print("📊 市场数据概览")
    print("="*60)
    
    data = collector.fetch_all(days=30)
    for name, df in data.items():
        if not df.empty:
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            change = (latest['close'] / prev['close'] - 1) * 100
            print(f"\n{name}: ${latest['close']:.2f} ({change:+.2f}%)")
