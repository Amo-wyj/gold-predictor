#!/usr/bin/env python3
"""
🥇 黄金预测系统 - 主入口
集成数据采集、特征工程、模型预测、预警推送
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, Optional

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/opt/gold-predictor/logs/main.log', mode='a')
    ] if os.path.exists('/opt/gold-predictor') else [logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


def run_prediction(data_dir: str = None, output_dir: str = None) -> Dict:
    """运行完整预测流程"""
    from data.yahoo_collector import YahooFinanceCollector
    from data.fred_collector import FREDCollector
    from features.feature_engineering import FeatureEngine
    from models.ensemble import EnsemblePredictor
    from bot.telegram_bot import GoldTelegramBot, AlertEngine
    
    logger.info("=" * 60)
    logger.info("🥇 黄金预测系统启动")
    logger.info("=" * 60)
    
    # === 1. 数据采集 ===
    logger.info("\n📥 步骤1: 数据采集")
    
    yahoo = YahooFinanceCollector()
    
    # 获取黄金数据
    gold_df = yahoo.fetch("GC=F", days=730)  # 2年数据
    if gold_df is None:
        logger.error("黄金数据获取失败")
        return {"status": "error", "message": "数据获取失败"}
    
    logger.info(f"黄金数据: {len(gold_df)} 条, 最新 {gold_df.index[-1].date()}")
    
    # 获取相关市场数据
    market_data = {}
    for symbol, name in [("SI=F", "silver"), ("DX-Y.NYB", "dxy"), 
                         ("^VIX", "vix"), ("CL=F", "oil")]:
        df = yahoo.fetch(symbol, days=365)
        if df is not None:
            market_data[name] = df
    
    # 获取宏观数据（可选）
    macro_df = None
    fred_api_key = os.getenv("FRED_API_KEY")
    if fred_api_key and fred_api_key != "YOUR_FRED_API_KEY":
        logger.info("获取宏观经济数据...")
        fred = FREDCollector(fred_api_key)
        macro_df = fred.fetch_all(
            start_date=(datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        )
        logger.info(f"宏观数据: {len(macro_df)} 条")
    
    # === 2. 特征工程 ===
    logger.info("\n🔧 步骤2: 特征工程")
    
    engine = FeatureEngine()
    features = engine.build_features(gold_df, macro_df, market_data)
    logger.info(f"特征数: {len(engine.feature_names)}")
    
    # === 3. 模型预测 ===
    logger.info("\n🤖 步骤3: 模型预测")
    
    predictor = EnsemblePredictor()
    predictor.update_data(gold_df, macro_df)
    
    results = predictor.predict(gold_df['close'])
    
    # 输出报告
    report = predictor.generate_report(results)
    print("\n" + report)
    
    # === 4. 预警检查 ===
    logger.info("\n🔔 步骤4: 预警检查")
    
    current_price = gold_df['close'].iloc[-1]
    alert_bot = GoldTelegramBot()
    alert_engine = AlertEngine(alert_bot)
    
    # 检查并发送预警
    triggered = alert_engine.process_and_alert(
        current_price=current_price,
        prediction=results.get("prediction", {}),
        timestamp=datetime.now()
    )
    
    logger.info(f"触发预警数: {len(triggered)}")
    for alert in triggered:
        logger.info(f"  - {alert['type']}: {alert.get('signal', alert.get('direction', ''))}")
    
    # === 5. 保存结果 ===
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        import json
        import pandas as pd
        
        # 保存预测结果
        with open(f"{output_dir}/prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "current_price": float(current_price),
                "prediction": results.get("prediction", {}),
                "technical": results.get("technical", {}),
            }, f, indent=2, default=str)
        
        # 保存最新数据
        gold_df.to_csv(f"{output_dir}/gold_latest.csv")
        
        logger.info(f"结果已保存至 {output_dir}")
    
    return {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "current_price": float(current_price),
        "prediction": results.get("prediction", {}),
        "technical": results.get("technical", {}),
        "triggered_alerts": len(triggered),
    }


def run_daily_digest():
    """发送每日摘要"""
    from data.yahoo_collector import YahooFinanceCollector
    from data.fred_collector import FREDCollector
    from features.feature_engineering import FeatureEngine
    from models.ensemble import EnsemblePredictor
    from bot.telegram_bot import GoldTelegramBot
    
    logger.info("📊 生成每日摘要...")
    
    yahoo = YahooFinanceCollector()
    gold_df = yahoo.fetch("GC=F", days=365)
    
    if gold_df is None:
        logger.error("无法获取数据")
        return
    
    engine = FeatureEngine()
    features = engine.build_features(gold_df)
    
    predictor = EnsemblePredictor()
    predictor.update_data(gold_df)
    
    results = predictor.predict(gold_df['close'])
    
    # 获取市场概览
    market = yahoo.get_market_overview()
    
    # 发送摘要
    bot = GoldTelegramBot()
    bot.send_daily_digest(
        prediction=results.get("prediction", {}),
        current_price=gold_df['close'].iloc[-1],
        tech=results.get("technical", {}),
        market_overview=market
    )
    
    logger.info("每日摘要已发送")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🥇 黄金预测系统")
    parser.add_argument("--mode", choices=["predict", "digest", "serve"], 
                       default="predict", help="运行模式")
    parser.add_argument("--output", default="/opt/gold-predictor/output", 
                       help="输出目录")
    parser.add_argument("--data-dir", help="数据目录")
    
    args = parser.parse_args()
    
    if args.mode == "predict":
        run_prediction(output_dir=args.output)
    elif args.mode == "digest":
        run_daily_digest()
    elif args.mode == "serve":
        logger.info("启动 Web 服务模式...")
        # 后续实现 API 服务
        logger.info("Web 服务待实现")
