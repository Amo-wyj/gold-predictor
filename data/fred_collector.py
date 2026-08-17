"""
FRED 宏观经济数据采集器
采集：国债收益率、TIPS、美元指数、CPI、PCE、非农等
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging
import os

from config import DATA_SOURCES

logger = logging.getLogger(__name__)


class FREDCollector:
    """FRED (Federal Reserve Economic Data) 数据采集"""
    
    BASE_URL = "https://api.stlouisfed.org/fred"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FRED_API_KEY") or DATA_SOURCES["fred"]["api_key"]
        self.series = DATA_SOURCES["fred"]["series"]
    
    def _make_url(self, endpoint: str, params: dict) -> str:
        """构建 FRED API URL"""
        base = f"{self.BASE_URL}/{endpoint}"
        params["api_key"] = self.api_key
        params["file_type"] = "json"
        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{base}?{query}"
    
    def fetch_series(self, series_id: str, 
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     limit: int = 1000) -> Optional[pd.DataFrame]:
        """获取单个经济指标序列"""
        try:
            import urllib.request
            import json
            
            url = f"{self.BASE_URL}/series/observations"
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "limit": limit,
            }
            if start_date:
                params["observation_start"] = start_date
            if end_date:
                params["observation_end"] = end_date
            
            query = "&".join(f"{k}={v}" for k, v in params.items())
            full_url = f"{url}?{query}"
            
            logger.info(f"[FRED] 请求 {series_id}...")
            
            with urllib.request.urlopen(full_url, timeout=30) as response:
                data = json.loads(response.read().decode())
            
            if "error_code" in data:
                logger.error(f"[FRED] {series_id} 错误: {data.get('error_message', 'Unknown')}")
                return None
            
            observations = data.get("observations", [])
            if not observations:
                logger.warning(f"[FRED] {series_id} 无数据")
                return None
            
            df = pd.DataFrame(observations)
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value'])
            df = df.set_index('date').sort_index()
            df = df[['value']].rename(columns={'value': series_id})
            
            logger.info(f"[FRED] {series_id}: {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.error(f"[FRED] {series_id} 获取失败: {e}")
            return None
    
    def fetch_all(self, start_date: Optional[str] = None, 
                  end_date: Optional[str] = None) -> pd.DataFrame:
        """批量获取所有宏观经济指标"""
        all_data = {}
        
        for name, series_id in self.series.items():
            df = self.fetch_series(series_id, start_date, end_date)
            if df is not None:
                all_data[name] = df[series_id]
        
        if not all_data:
            return pd.DataFrame()
        
        # 合并所有指标
        result = pd.DataFrame(all_data)
        
        # 前向填充缺失值
        result = result.ffill().bfill()
        
        return result
    
    def get_latest(self, series_id: str) -> Optional[dict]:
        """获取最新值"""
        df = self.fetch_series(series_id, limit=10)
        if df is not None and not df.empty:
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            return {
                "series_id": series_id,
                "value": float(latest.values[0]),
                "prev_value": float(prev.values[0]),
                "change": float(latest.values[0] - prev.values[0]),
                "date": latest.name.isoformat()
            }
        return None
    
    def get_macro_overview(self) -> Dict[str, dict]:
        """获取宏观指标概览"""
        overview = {}
        
        key_series = {
            "dgs10": "10年期国债收益率",
            "tips10": "10年期TIPS实际利率",
            "cpi": "CPI同比",
            "fed_rate": "联邦基金利率",
            "nonfarm": "非农就业",
        }
        
        for name, label in key_series.items():
            data = self.get_latest(self.series.get(name, name))
            if data:
                overview[name] = {
                    "label": label,
                    "value": data["value"],
                    "prev": data["prev_value"],
                    "change": data["change"],
                    "date": data["date"]
                }
        
        return overview


def get_real_rate() -> Optional[float]:
    """计算实际利率 = 名义利率 - 通胀率"""
    try:
        collector = FREDCollector()
        tips = collector.get_latest("DFII10")  # TIPS
        cpi = collector.get_latest("CPIAUCSL")  # CPI
        
        if tips and cpi:
            # CPI月率转年率
            cpi_yoy = cpi["value"]
            real_rate = tips["value"] - cpi_yoy
            return real_rate
    except Exception as e:
        logger.error(f"[FRED] 实际利率计算失败: {e}")
    
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    
    # 需要设置 FRED_API_KEY 环境变量
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        print("⚠️ 请设置 FRED_API_KEY 环境变量")
        print("   免费申请: https://fred.stlouisfed.org/docs/api/api_key.html")
        exit(1)
    
    collector = FREDCollector(api_key)
    
    print("\n" + "="*60)
    print("📈 宏观经济指标概览")
    print("="*60)
    
    overview = collector.get_macro_overview()
    for name, data in overview.items():
        print(f"\n{data['label']}: {data['value']:.2f} "
              f"({data['change']:+.2f}, {data['date'][:10]})")
