#!/usr/bin/env python3
"""
🌐 Web Dashboard - 黄金预测可视化
Flask + Plotly 实时图表
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 全局状态
latest_prediction = {}
latest_price = None
prediction_history = []


def init_dashboard():
    """初始化 Dashboard"""
    global latest_prediction, latest_price
    
    # 尝试加载最新预测结果
    output_dir = "/opt/gold-predictor/output"
    if os.path.exists(output_dir):
        files = [f for f in os.listdir(output_dir) if f.startswith("prediction_")]
        if files:
            latest_file = sorted(files)[-1]
            with open(os.path.join(output_dir, latest_file)) as f:
                data = json.load(f)
                latest_prediction = data.get("prediction", {})
                latest_price = data.get("current_price")


@app.route("/")
def index():
    """主页"""
    return render_template("dashboard.html",
                          price=latest_price,
                          prediction=latest_prediction,
                          timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


@app.route("/api/predict")
def api_predict():
    """获取最新预测"""
    return jsonify({
        "status": "success",
        "price": latest_price,
        "prediction": latest_prediction,
        "timestamp": datetime.now().isoformat()
    })


@app.route("/api/price")
def api_price():
    """获取当前价格（真实数据，限流时回退模拟数据）"""
    try:
        import yfinance as yf
        gold = yf.Ticker("GC=F")
        hist = gold.history(period="5d", auto_adjust=True)
        if not hist.empty:
            current = float(hist['Close'].iloc[-1])
            prev = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current
            change = (current / prev - 1) * 100
            return jsonify({
                "price": current,
                "change_pct": change,
                "source": "yahoo",
                "timestamp": hist.index[-1].isoformat()
            })
    except Exception as e:
        logger.warning(f"Yahoo 获取价格失败，回退到模拟数据: {e}")

    # 回退: 模拟数据
    try:
        from data.mock_data import get_latest_mock_price
        price_data = get_latest_mock_price()
        return jsonify({
            "price": float(price_data['price']),
            "change_pct": float(price_data.get('change_pct', 0.0)),
            "source": "mock",
            "timestamp": datetime.now().isoformat()
        })
    except Exception:
        pass

    return jsonify({"error": "无法获取价格"}), 500


@app.route("/api/technical")
def api_technical():
    """获取技术指标"""
    global latest_prediction
    tech = latest_prediction.get("technical", {})
    return jsonify(tech)


@app.route("/api/history")
def api_history():
    """获取历史预测记录"""
    return jsonify({
        "history": prediction_history[-100:]  # 最近100条
    })


@app.route("/api/chart/price")
def api_chart_price():
    """获取价格图表数据"""
    try:
        import yfinance as yf
        import pandas as pd
        
        gold = yf.Ticker("GC=F")
        df = gold.history(period="6mo", auto_adjust=True)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        
        data = {
            "dates": [d.strftime("%Y-%m-%d") for d in df.index],
            "prices": [float(p) for p in df['Close'].values],
            "volumes": [int(v) for v in df['Volume'].values],
            "highs": [float(h) for h in df['High'].values],
            "lows": [float(l) for l in df['Low'].values],
        }
        
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"图表数据获取失败: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/chart/indicators")
def api_chart_indicators():
    """获取技术指标图表数据"""
    try:
        import yfinance as yf
        import pandas as pd
        import numpy as np
        
        gold = yf.Ticker("GC=F")
        df = gold.history(period="3mo", auto_adjust=True)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        
        # 计算技术指标
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA60'] = df['Close'].rolling(60).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Histogram'] = df['MACD'] - df['Signal']
        
        data = {
            "dates": [d.strftime("%Y-%m-%d") for d in df.index],
            "close": [float(c) for c in df['Close'].values],
            "sma20": [float(s) if pd.notna(s) else None for s in df['SMA20'].values],
            "sma60": [float(s) if pd.notna(s) else None for s in df['SMA60'].values],
            "rsi": [float(r) if pd.notna(r) else None for r in df['RSI'].values],
            "macd": [float(m) if pd.notna(m) else None for m in df['MACD'].values],
            "signal": [float(s) if pd.notna(s) else None for s in df['Signal'].values],
            "histogram": [float(h) if pd.notna(h) else None for h in df['Histogram'].values],
        }
        
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"指标图表数据获取失败: {e}")
        return jsonify({"error": str(e)}), 500


def update_prediction(prediction: Dict, price: float):
    """更新全局预测状态（供主程序调用）"""
    global latest_prediction, latest_price, prediction_history
    
    latest_prediction = prediction
    latest_price = price
    
    prediction_history.append({
        "timestamp": datetime.now().isoformat(),
        "price": price,
        "prediction": prediction
    })
    
    # 保持历史记录不超过1000条
    if len(prediction_history) > 1000:
        prediction_history = prediction_history[-1000:]


def run_server(host: str = "0.0.0.0", port: int = 5000):
    """启动 Web 服务"""
    init_dashboard()
    logger.info(f"启动 Dashboard: http://{host}:{port}")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_server()
