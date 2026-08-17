#!/usr/bin/env python3
"""
🥇 黄金预测系统 - 预测入口
用法:
  python run_predict.py              # 本地预测（模拟数据）
  python run_predict.py --real       # 真实数据（需联网）
  python run_predict.py --dashboard  # 启动 Web Dashboard
  python run_predict.py --full       # 预测 + Dashboard
"""
import sys
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime
from data.mock_data import generate_mock_gold_data, generate_mock_macro_data
from features.feature_engineering import FeatureEngine
from models.arima_model import GoldARIMA
from models.gbm_model import GoldGBM
from models.ensemble import EnsemblePredictor


def predict(use_mock=True, verbose=True):
    """执行预测"""
    print("=" * 60)
    print("🥇 黄金预测系统")
    print("=" * 60)

    # Step 1: 数据
    if use_mock:
        print("\n📥 数据准备: 模拟数据")
        gold_df = generate_mock_gold_data(days=400)
        macro_data = generate_mock_macro_data(days=400)
    else:
        print("\n📥 数据准备: 真实数据")
        from data.yahoo_collector import YahooFinanceCollector
        from data.fred_collector import FREDCollector
        yc = YahooFinanceCollector()
        gold_df = yc.fetch("GC=F", days=400)
        macro_data = FREDCollector().fetch_all()
        if gold_df is None or gold_df.empty:
            print("   ⚠️ Yahoo 数据获取失败，回退到模拟数据")
            gold_df = generate_mock_gold_data(days=400)
            macro_data = generate_mock_macro_data(days=400)
        print(f"   黄金: {len(gold_df)} 天")

    print(f"   黄金数据: {len(gold_df)} 天")
    print(f"   最新价格: ${gold_df['close'].iloc[-1]:.2f}")

    # Step 2: 特征工程
    print("\n🔧 特征工程...")
    fe = FeatureEngine()
    features = fe.build_features(gold_df, macro_data)
    print(f"   特征数: {len(features.columns)}")

    # Step 3: 集成预测
    print("\n🤖 集成预测...")
    ensemble = EnsemblePredictor()
    ensemble.update_data(gold_df, macro_data)
    result = ensemble.predict(gold_df['close'])

    # Step 4: 格式化输出
    if verbose:
        print("\n" + "=" * 60)
        print("📊 预测结果")
        print("=" * 60)
        print(f"⏰ {result.get('timestamp', datetime.now().isoformat())}")
        current_price = result.get('current_price', gold_df['close'].iloc[-1])
        print(f"💰 当前价格: ${current_price:.2f}")

        pred = result.get('prediction', {})
        for h, label in [(1, '明天'), (3, '3天后'), (5, '5天后')]:
            key = f"horizon_{h}d"
            if key in pred:
                p = pred[key]
                signal = p.get('signal', 'NEUTRAL')
                emoji = {
                    'STRONG_BUY': '🟢 强力买入',
                    'BUY': '🟡 买入',
                    'NEUTRAL': '⚪ 中性',
                    'SELL': '🟠 卖出',
                    'STRONG_SELL': '🔴 强力卖出',
                }.get(signal, signal)
                up_pct = p.get('probability_up', 0) * 100
                down_pct = p.get('probability_down', 0) * 100
                price = p.get('predicted_price', 0)
                chg = p.get('price_change_pct', 0)
                conf = p.get('confidence_label', 'LOW')
                print(f"\n  【{label}】")
                print(f"    信号: {emoji}")
                print(f"    上涨概率: {up_pct:.1f}% | 下跌概率: {down_pct:.1f}%")
                print(f"    预测价格: ${price:.2f} ({chg:+.2f}%)")
                print(f"    置信度: {conf}")

        # 技术指标
        ta = result.get('technical_analysis', {})
        if ta:
            print("\n🔧 技术指标:")
            for k, v in ta.items():
                if k == 'technical_score':
                    print(f"    技术评分: {v:.2f} (1=极强看涨, -1=极强看跌)")
                elif isinstance(v, dict) and 'value' in v:
                    print(f"    {k}: {v['value']:.2f} ({v.get('signal','')})")

        # 宏观
        ma = result.get('macro_analysis', {})
        if ma and ma.get('macro_signal') != 'NO_DATA':
            print("\n🌐 宏观环境:")
            for k, v in ma.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.2f}")
                elif isinstance(v, str):
                    print(f"    {k}: {v}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='黄金预测系统')
    parser.add_argument('--real', action='store_true', help='使用真实数据（需联网）')
    parser.add_argument('--dashboard', action='store_true', help='启动 Web Dashboard')
    parser.add_argument('--full', action='store_true', help='预测 + Dashboard')
    args = parser.parse_args()

    if args.dashboard or args.full:
        print("\n🚀 启动 Dashboard...")
        from dashboard.dashboard import run_server
        run_server(host="0.0.0.0", port=5000)
    else:
        result = predict(use_mock=not args.real)
        print("\n✅ 预测完成")
