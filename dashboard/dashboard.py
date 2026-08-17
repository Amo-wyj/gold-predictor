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

# 确保能导入项目根目录模块（Render 部署时工作目录在 dashboard/）
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 全局状态
latest_prediction = {}
latest_price = None
prediction_history = []


import threading

_prediction_lock = threading.Lock()
_prediction_initialized = False


def _build_mock_prediction(price_seed: float = 0.0) -> Dict:
    """生成一份纯 mock 预测结果，保证 dashboard 永远有东西显示"""
    import numpy as np
    rng = np.random.default_rng(int(price_seed * 100) % 99991 if price_seed else 42)
    return {
        "horizons": {
            "1d": {"up_probability": float(rng.uniform(0.45, 0.65)),
                   "down_probability": float(rng.uniform(0.35, 0.55)),
                   "confidence": float(rng.uniform(0.55, 0.78))},
            "3d": {"up_probability": float(rng.uniform(0.40, 0.70)),
                   "down_probability": float(rng.uniform(0.30, 0.60)),
                   "confidence": float(rng.uniform(0.50, 0.75))},
            "5d": {"up_probability": float(rng.uniform(0.40, 0.75)),
                   "down_probability": float(rng.uniform(0.25, 0.60)),
                   "confidence": float(rng.uniform(0.48, 0.72))},
        },
        "technical": {
            "RSI": float(rng.uniform(40, 65)),
            "MACD": float(rng.uniform(-3, 3)),
            "BB_position": float(rng.uniform(0.3, 0.7)),
            "ADX": float(rng.uniform(15, 35)),
        },
        "source": "mock",
    }


def _ensure_prediction() -> dict:
    """保证 latest_prediction/latest_price 必有值；冷启动失败也能拼出 mock。返回调试信息。"""
    global latest_prediction, latest_price, _prediction_initialized
    debug = {"steps": [], "errors": []}

    if _prediction_initialized and latest_prediction and latest_price is not None:
        debug["steps"].append("already_initialized")
        return debug

    with _prediction_lock:
        if _prediction_initialized and latest_prediction and latest_price is not None:
            debug["steps"].append("locked_already_initialized")
            return debug
        
        # 1) 尝试加载文件
        output_dir = "/opt/gold-predictor/output"
        if os.path.exists(output_dir):
            try:
                files = [f for f in os.listdir(output_dir) if f.startswith("prediction_")]
                if files:
                    latest_file = sorted(files)[-1]
                    with open(os.path.join(output_dir, latest_file)) as f:
                        data = json.load(f)
                        if data.get("prediction") and data.get("current_price"):
                            latest_prediction = data["prediction"]
                            latest_price = float(data["current_price"])
                            debug["steps"].append(f"loaded_from_file:{latest_file}")
                            _prediction_initialized = True
                            return debug
            except Exception as e:
                debug["errors"].append(f"file_load_error: {e}")

        # 2) 云端实时预测
        try:
            import runpy
            from pathlib import Path
            project_root = Path(__file__).resolve().parent.parent
            spec = runpy.run_path(str(project_root / "run_predict.py"), run_name="__predict_mod")
            predict_fn = spec.get("predict")
            if predict_fn is None:
                raise RuntimeError("run_predict.predict not found")
            
            prediction_result = None
            try:
                prediction_result = predict_fn(use_mock=False)
                debug["steps"].append("predict_real_ok")
            except Exception as e_real:
                debug["errors"].append(f"predict_real_failed: {e_real}")
                try:
                    prediction_result = predict_fn(use_mock=True)
                    debug["steps"].append("predict_mock_ok")
                except Exception as e_mock:
                    debug["errors"].append(f"predict_mock_failed: {e_mock}")
            
            if prediction_result and prediction_result.get("prediction"):
                latest_prediction = prediction_result["prediction"]
                latest_price = float(prediction_result.get("current_price") or 0)
                debug["steps"].append("prediction_set")
                # 写文件
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    with open(f"{output_dir}/prediction_{ts}.json", "w") as f:
                        json.dump({
                            "timestamp": datetime.now().isoformat(),
                            "current_price": latest_price,
                            "prediction": latest_prediction,
                        }, f, indent=2, default=str)
                    debug["steps"].append("prediction_written")
                except Exception as e_write:
                    debug["errors"].append(f"write_error: {e_write}")
        except Exception as e:
            debug["errors"].append(f"predict_pipeline_error: {e}")

        # 3) 兜底 mock
        if not latest_prediction or latest_price is None:
            seed_price = float(latest_price) if latest_price else 2000.0
            latest_prediction = _build_mock_prediction(seed_price)
            if latest_price is None:
                latest_price = seed_price
            debug["steps"].append("fallback_mock")
        
        _prediction_initialized = True
        return debug


def init_dashboard():
    """初始化 Dashboard"""
    global latest_prediction, latest_price

    # 1) 尝试加载最新预测结果文件
    output_dir = "/opt/gold-predictor/output"
    loaded_from_file = False
    if os.path.exists(output_dir):
        files = [f for f in os.listdir(output_dir) if f.startswith("prediction_")]
        if files:
            latest_file = sorted(files)[-1]
            try:
                with open(os.path.join(output_dir, latest_file)) as f:
                    data = json.load(f)
                    latest_prediction = data.get("prediction", {})
                    latest_price = data.get("current_price")
                    loaded_from_file = True
                    logger.info(f"从 {latest_file} 加载预测成功")
            except Exception as e:
                logger.warning(f"加载预测文件失败: {e}")

    # 2) 云端 / Render 部署：跑一次实时预测（yfinance 可用就用真实数据，不可用自动回退 mock）
    if not loaded_from_file or not latest_prediction:
        try:
            import runpy
            from pathlib import Path
            project_root = Path(__file__).resolve().parent.parent
            # 导入 run_predict 模块并调用 predict()
            spec = runpy.run_path(str(project_root / "run_predict.py"), run_name="__predict_mod")
            result = spec.get("predict")
            if result is None:
                raise RuntimeError("run_predict.predict 函数未找到")

            # 先试真实数据，失败回退 mock
            prediction_result = None
            try:
                prediction_result = result(use_mock=False)
            except Exception as e_real:
                logger.warning(f"真实数据预测失败，回退 mock: {e_real}")
                prediction_result = result(use_mock=True)

            if prediction_result and "prediction" in prediction_result:
                latest_prediction = prediction_result["prediction"]
                latest_price = float(prediction_result.get("current_price", 0) or 0)
                # 顺手写入 output 文件，供下次重启加载
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    with open(f"{output_dir}/prediction_{ts}.json", "w") as f:
                        json.dump({
                            "timestamp": datetime.now().isoformat(),
                            "current_price": latest_price,
                            "prediction": latest_prediction,
                        }, f, indent=2, default=str)
                    logger.info(f"云端预测完成并已写入 {ts}.json")
                except Exception as e_write:
                    logger.warning(f"写入 prediction 文件失败: {e_write}")
        except Exception as e:
            logger.warning(f"云端自启预测失败，使用 mock 兜底: {e}")

    # 3) 兜底：如果仍然没有数据，用 mock 生成一份展示
    if not latest_prediction or latest_price is None:
        try:
            from data.mock_data import get_latest_mock_price, generate_mock_gold_data
            import numpy as np
            price_data = get_latest_mock_price()
            latest_price = float(price_data['price'])
            # 生成轻量 mock 预测结果
            rng = np.random.default_rng(int(latest_price) % 10000)
            latest_prediction = {
                "horizons": {
                    "1d": {
                        "up_probability": float(rng.uniform(0.45, 0.65)),
                        "down_probability": float(rng.uniform(0.35, 0.55)),
                        "confidence": float(rng.uniform(0.55, 0.78))
                    },
                    "3d": {
                        "up_probability": float(rng.uniform(0.40, 0.70)),
                        "down_probability": float(rng.uniform(0.30, 0.60)),
                        "confidence": float(rng.uniform(0.50, 0.75))
                    },
                    "5d": {
                        "up_probability": float(rng.uniform(0.40, 0.75)),
                        "down_probability": float(rng.uniform(0.25, 0.60)),
                        "confidence": float(rng.uniform(0.48, 0.72))
                    }
                },
                "technical": {
                    "RSI": float(rng.uniform(40, 65)),
                    "MACD": float(rng.uniform(-3, 3)),
                    "BB_position": float(rng.uniform(0.3, 0.7)),
                    "ADX": float(rng.uniform(15, 35))
                },
                "source": "mock"
            }
            logger.info("使用 mock 数据初始化 dashboard")
        except Exception as e:
            logger.error(f"Mock 兜底也失败: {e}")


@app.route("/")
def index():
    """主页"""
    _ensure_prediction()
    return render_template("dashboard.html",
                          price=latest_price,
                          prediction=latest_prediction,
                          timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


@app.route("/api/predict")
def api_predict():
    """获取最新预测"""
    debug = _ensure_prediction()
    return jsonify({
        "status": "success",
        "price": latest_price,
        "prediction": latest_prediction,
        "timestamp": datetime.now().isoformat(),
        "_debug": debug  # 临时调试信息，上线后删除
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

        if df.empty:
            raise ValueError("Yahoo 返回空数据")

        data = {
            "dates": [d.strftime("%Y-%m-%d") for d in df.index],
            "prices": [float(p) for p in df['Close'].values],
            "volumes": [int(v) for v in df['Volume'].values],
            "highs": [float(h) for h in df['High'].values],
            "lows": [float(l) for l in df['Low'].values],
        }

        return jsonify(data)

    except Exception as e:
        logger.warning(f"图表价格获取失败，回退 mock: {e}")
        try:
            from data.mock_data import generate_mock_gold_data
            df = generate_mock_gold_data(days=120)
            return jsonify({
                "dates": [d.strftime("%Y-%m-%d") for d in df['date']],
                "prices": [float(p) for p in df['close']],
                "volumes": [int(v) for v in df['volume']],
                "highs": [float(h) for h in df['high']],
                "lows": [float(l) for l in df['low']],
            })
        except Exception as e2:
            logger.error(f"mock 图表也失败: {e2}")
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

        if df.empty:
            raise ValueError("Yahoo 返回空数据")

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
        logger.warning(f"指标图表获取失败，回退 mock: {e}")
        try:
            from data.mock_data import generate_mock_gold_data
            import pandas as pd
            import numpy as np

            df = generate_mock_gold_data(days=90)
            df = df.sort_values('date').reset_index(drop=True)
            close = df['close'].astype(float)

            sma20 = close.rolling(20).mean()
            sma60 = close.rolling(60).mean()
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            hist = macd - signal

            return jsonify({
                "dates": [d.strftime("%Y-%m-%d") for d in df['date']],
                "close": [float(c) for c in close],
                "sma20": [float(s) if pd.notna(s) else None for s in sma20],
                "sma60": [float(s) if pd.notna(s) else None for s in sma60],
                "rsi": [float(r) if pd.notna(r) else None for r in rsi],
                "macd": [float(m) if pd.notna(m) else None for m in macd],
                "signal": [float(s) if pd.notna(s) else None for s in signal],
                "histogram": [float(h) if pd.notna(h) else None for h in hist],
            })
        except Exception as e2:
            logger.error(f"mock 指标图表也失败: {e2}")
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
