#!/usr/bin/env python3
"""
完整预测流程测试脚本 - 使用模拟数据
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.filterwarnings('ignore')

from data.mock_data import generate_mock_gold_data, generate_mock_macro_data
from features.feature_engineering import FeatureEngine
from models.arima_model import GoldARIMA
from models.gbm_model import GoldGBM
from models.ensemble import EnsemblePredictor


def main():
    print("=" * 60)
    print("🥇 黄金预测系统 - 完整流程测试")
    print("=" * 60)

    # Step 1: 生成模拟数据
    print("\n📥 Step 1: 数据准备 (模拟数据)")
    gold_df = generate_mock_gold_data(days=400)
    macro_data = generate_mock_macro_data(days=400)
    print(f"   黄金数据: {len(gold_df)} 天")
    print(f"   最新价格: ${gold_df['close'].iloc[-1]:.2f}")

    # Step 2: 特征工程
    print("\n🔧 Step 2: 特征工程")
    fe = FeatureEngine()
    features = fe.build_features(gold_df, macro_data)
    print(f"   生成特征: {len(features.columns)} 个")
    print(f"   最新 RSI(14): {features['rsi_14'].iloc[-1]:.1f}")
    print(f"   最新 MACD: {features['macd'].iloc[-1]:.2f}")

    # Step 3: 训练 ARIMA
    print("\n🤖 Step 3: 模型训练")
    print("   训练 ARIMA...")
    arima = GoldARIMA()
    try:
        arima.fit(gold_df['close'])
        print("   ✅ ARIMA 训练完成")
    except Exception as e:
        print(f"   ⚠️ ARIMA: {e}")
        arima = None

    # Step 4: 训练 GBM（替代 LSTM，无需 TensorFlow）
    print("   训练 GBM...")
    gbm = GoldGBM()
    try:
        gbm.fit(features, target_col='close')
        print("   ✅ GBM 训练完成")
    except Exception as e:
        print(f"   ⚠️ GBM: {e}")

    # Step 5: 集成预测
    print("\n🎯 Step 5: 集成预测")
    ensemble = EnsemblePredictor()
    try:
        # 更新数据
        ensemble.update_data(gold_df, macro_data)

        # 预测
        result = ensemble.predict(gold_df['close'])
    except Exception as e:
        print(f"   ⚠️ 集成预测失败: {e}")
        import traceback; traceback.print_exc()
        return

    # Step 6: 输出报告
    print("\n" + "=" * 60)
    print("📊 预测报告")
    print("=" * 60)

    print(f"\n⏰ 生成时间: {result.get('timestamp', 'N/A')}")
    print(f"💰 当前价格: ${gold_df['close'].iloc[-1]:.2f}")

    pred = result.get('prediction', {})
    for horizon, data in pred.items():
        h_name = {'horizon_1d': '明天', 'horizon_3d': '3天后', 'horizon_5d': '5天后'}.get(horizon, horizon)
        signal = data.get('signal', 'N/A')
        signal_emoji = {
            'STRONG_BUY': '🟢', 'BUY': '🟢',
            'NEUTRAL': '⚪',
            'SELL': '🔴', 'STRONG_SELL': '🔴'
        }.get(signal, '⚪')

        print(f"\n  【{h_name}】")
        print(f"    信号: {signal_emoji} {signal}")
        print(f"    上涨概率: {data.get('probability_up', 0)*100:.1f}%")
        print(f"    下跌概率: {data.get('probability_down', 0)*100:.1f}%")
        print(f"    预测价格: ${data.get('predicted_price', 0):.2f} ({data.get('price_change_pct', 0):+.2f}%)")
        print(f"    置信度: {data.get('confidence_label', 'N/A')}")

    ta = result.get('technical_analysis', {})
    print("\n🔧 技术指标:")
    for k, v in ta.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.2f}")
        else:
            print(f"    {k}: {v}")

    macro = result.get('macro_analysis', {})
    print("\n🌐 宏观环境:")
    for k, v in macro.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.2f}")
        else:
            print(f"    {k}: {v}")

    print("\n" + "=" * 60)
    print("✅ 测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
