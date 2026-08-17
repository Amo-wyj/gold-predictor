#!/usr/bin/env python3
"""
📡 MT5 数据桥接服务
在 Windows 云服务器上运行，将 MT5 实时数据推送到云端
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import requests
import time
import logging
import json
from datetime import datetime
from typing import Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


class MT5Bridge:
    """MT5 数据桥接器"""
    
    def __init__(self, api_endpoint: str, api_key: str):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.running = False
        self.last_data = {}
    
    def connect(self) -> bool:
        """连接 MT5 终端"""
        if not mt5.initialize():
            logger.error(f"[MT5] 初始化失败: {mt5.last_error()}")
            return False
        
        logger.info(f"[MT5] 已连接到 {mt5.account_info().server}")
        return True
    
    def disconnect(self):
        """断开 MT5 连接"""
        mt5.shutdown()
        logger.info("[MT5] 已断开连接")
    
    def get_gold_data(self, symbol: str = "XAUUSD") -> Optional[Dict]:
        """获取黄金实时数据"""
        try:
            # 获取Tick数据
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.warning(f"[MT5] 无法获取 {symbol} 数据")
                return None
            
            # 获取OHLC数据
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
            if rates is None:
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            return {
                "symbol": symbol,
                "bid": float(tick.bid),
                "ask": float(tick.ask),
                "spread": float(tick.ask - tick.bid),
                "timestamp": datetime.now().isoformat(),
                "ohlc": {
                    "open": float(df['open'].iloc[-1]),
                    "high": float(df['high'].iloc[-1]),
                    "low": float(df['low'].iloc[-1]),
                    "close": float(df['close'].iloc[-1]),
                    "volume": int(df['tick_volume'].iloc[-1]),
                },
                "history": {
                    "dates": df['time'].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
                    "open": df['open'].tolist(),
                    "high": df['high'].tolist(),
                    "low": df['low'].tolist(),
                    "close": df['close'].tolist(),
                    "volume": df['tick_volume'].tolist(),
                }
            }
            
        except Exception as e:
            logger.error(f"[MT5] 获取数据失败: {e}")
            return None
    
    def push_to_server(self, data: Dict) -> bool:
        """推送数据到云端服务器"""
        try:
            response = requests.post(
                self.api_endpoint,
                json=data,
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": self.api_key
                },
                timeout=10
            )
            
            if response.status_code == 200:
                logger.debug(f"[Bridge] 数据已推送: ${data['bid']:.2f}")
                return True
            else:
                logger.warning(f"[Bridge] 推送失败: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"[Bridge] 推送异常: {e}")
            return False
    
    def run(self, push_interval: int = 60):
        """持续运行数据采集"""
        if not self.connect():
            return
        
        self.running = True
        logger.info(f"[Bridge] 开始采集，间隔 {push_interval} 秒")
        
        try:
            while self.running:
                data = self.get_gold_data()
                
                if data:
                    # 检查数据是否变化
                    if data['bid'] != self.last_data.get('bid'):
                        self.push_to_server(data)
                        self.last_data = data
                
                time.sleep(push_interval)
                
        except KeyboardInterrupt:
            logger.info("[Bridge] 收到停止信号")
        finally:
            self.disconnect()
    
    def stop(self):
        """停止采集"""
        self.running = False


def run_collector():
    """主函数"""
    import os
    
    # 配置
    API_ENDPOINT = os.getenv("MT5_BRIDGE_ENDPOINT", "http://your-server:5000/api/mt5")
    API_KEY = os.getenv("MT5_BRIDGE_KEY", "your-secret-key")
    PUSH_INTERVAL = int(os.getenv("MT5_PUSH_INTERVAL", "60"))
    
    bridge = MT5Bridge(API_ENDPOINT, API_KEY)
    bridge.run(push_interval=PUSH_INTERVAL)


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════╗
    ║         MT5 数据桥接服务                           ║
    ║                                                   ║
    ║  运行环境: Windows 云服务器                        ║
    ║  前置条件: 安装 MT5 终端并登录                     ║
    ║                                                   ║
    ║  环境变量:                                        ║
    ║    MT5_BRIDGE_ENDPOINT  - 云端接收地址             ║
    ║    MT5_BRIDGE_KEY       - API密钥                 ║
    ║    MT5_PUSH_INTERVAL    - 推送间隔(秒)             ║
    ╚═══════════════════════════════════════════════════╝
    """)
    
    run_collector()
