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
latest_technical = {}  # 技术指标独立存储，供 /api/technical 使用
latest_price = None
prediction_history = []
_latest_xgb_passes: bool = False   # P1: ML AUC 验证标记（白名单方案是否生效）
_latest_ml_auc: Dict = {}          # P1: 各 horizon CV AUC（LightGBM/XGBoost）
_latest_ml_model: str = "XGBoost"  # P1: 实际 ML 模型名

import threading

_prediction_lock = threading.Lock()
_prediction_initialized = False


# ================================================================
# P1 Phase ⑤：5年历史数据自动补全（Render 启动时执行）
# 本地 IP 不通 yfinance/FRED/Stooq，Render 实例 IP 可用
# ================================================================
def _ensure_5yr_history(project_root):
    """确保 data/gold_5yr.parquet 存在，不存在则从 yfinance 下载"""
    import pandas as pd
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    fpath = os.path.join(data_dir, "gold_5yr.parquet")

    if os.path.exists(fpath):
        try:
            df = pd.read_parquet(fpath)
            if len(df) >= 1000:  # 约 5 年交易日约 1260 天
                logger.info(f"[5yr] 本地已有 {len(df)} 条历史数据，跳过下载")
                return
        except Exception as e:
            logger.warning(f"[5yr] 本地文件读取失败，将重新下载: {e}")

    logger.info("[5yr] 正在下载 5 年黄金历史数据（Render 云端 IP）...")
    try:
        import yfinance as yf
        end = pd.Timestamp.today()
        start = end - pd.DateOffset(years=5)
        ticker = yf.Ticker("GC=F")
        df = ticker.history(start=start, end=end)
        if df.empty:
            raise RuntimeError("yfinance 返回空")
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        df = df[["open", "high", "low", "close", "volume"]]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.sort_index().dropna(subset=["close"])
        df.to_parquet(fpath)
        logger.info(f"[5yr] ✅ 已保存 {len(df)} 条数据 → {fpath} ({os.path.getsize(fpath)//1024} KB)")
    except Exception as e:
        logger.warning(f"[5yr] ⚠️ 下载失败（不影响启动，继续用实时数据训练）: {e}")


# 技术指标英文信号 → 中文（idempotent：已是中文则原样返回）
TECH_SIGNAL_ZH = {
    'OVERSOLD': '超卖', 'OVERBOUGHT': '超买', 'NEUTRAL': '中性',
    'BULLISH': '看涨', 'BEARISH': '看跌',
    'AT_UPPER_BAND': '触及上轨', 'AT_LOWER_BAND': '触及下轨',
    'STRONG_TREND': '强趋势', 'WEAK_TREND': '弱趋势', 'RANGING': '震荡',
    '中轨': '中轨', '上轨': '上轨', '下轨': '下轨',
    '强势': '强势', '偏弱': '偏弱', '看涨': '看涨', '看跌': '看跌',
}


def _translate_signal(sig):
    """英文技术指标信号 → 中文"""
    if not sig or not isinstance(sig, str):
        return sig
    return TECH_SIGNAL_ZH.get(sig, sig)


def _normalize_technical(raw: dict) -> dict:
    """统一技术指标格式：提取 RSI / MACD / Bollinger / ADX 数值与信号。
    MACD value 已在上游归一化为占价百分比并钳制 [-10,10]，此处做防御性钳制。
    """
    out = {}
    # RSI
    rsi = raw.get('rsi', raw.get('RSI'))
    if isinstance(rsi, dict):
        val = rsi.get('value', rsi.get('rsi_value', 50))
        sig = rsi.get('signal', '中性')
        if isinstance(val, (int, float)):
            sig = '超买' if val > 70 else ('超卖' if val < 30 else '中性')
        out['RSI'] = {'value': float(val), 'signal': _translate_signal(sig)}
    elif isinstance(rsi, (int, float)):
        sig = '超买' if rsi > 70 else ('超卖' if rsi < 30 else '中性')
        out['RSI'] = {'value': float(rsi), 'signal': sig}
    # MACD（value 为占价百分比，钳制 -10~+10）
    macd = raw.get('macd', raw.get('MACD'))
    if isinstance(macd, dict):
        val = macd.get('value', macd.get('macd', 0))
        sig = macd.get('signal', '中性')
        if isinstance(val, (int, float)) and not sig:
            sig = '看涨' if val > 0 else '看跌'
        out['MACD'] = {'value': max(-10.0, min(10.0, float(val))), 'signal': _translate_signal(sig)}
    elif isinstance(macd, (int, float)):
        sig = '看涨' if macd > 0 else '看跌'
        out['MACD'] = {'value': max(-10.0, min(10.0, float(macd))), 'signal': sig}
    # Bollinger Bands
    bb = raw.get('bollinger', raw.get('BB', raw.get('bollinger_bands')))
    if isinstance(bb, dict):
        pos = bb.get('position', bb.get('bb_position', bb.get('value', 0.5)))
        sig = bb.get('signal', '中性')
        if isinstance(pos, (int, float)) and not sig:
            sig = '上轨' if pos > 0.8 else ('下轨' if pos < 0.2 else '中轨')
        out['Bollinger'] = {'value': float(pos), 'signal': _translate_signal(sig)}
    elif isinstance(bb, (int, float)):
        sig = '上轨' if bb > 0.8 else ('下轨' if bb < 0.2 else '中轨')
        out['Bollinger'] = {'value': float(bb), 'signal': sig}
    # ADX
    adx = raw.get('adx', raw.get('ADX'))
    if isinstance(adx, dict):
        val = adx.get('value', adx.get('adx_value', 20))
        sig = adx.get('signal', '偏弱')
        if isinstance(val, (int, float)) and not sig:
            sig = '强势' if val > 25 else '偏弱'
        out['ADX'] = {'value': float(val), 'signal': _translate_signal(sig)}
    elif isinstance(adx, (int, float)):
        sig = '强势' if adx > 25 else '偏弱'
        out['ADX'] = {'value': float(adx), 'signal': sig}
    return out


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
    """保证 latest_prediction/latest_price/latest_technical 必有值；冷启动失败也能拼出 mock。"""
    global latest_prediction, latest_technical, latest_price, _prediction_initialized, _latest_xgb_passes, _latest_ml_auc, _latest_ml_model
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
                            raw_tech = data.get("technical_analysis", {})
                            latest_technical = _normalize_technical(raw_tech)
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
                raw_tech = prediction_result.get("technical_analysis", {})
                latest_technical = _normalize_technical(raw_tech)
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
                            "technical_analysis": raw_tech,
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
            # mock 也填充 technical
            import numpy as np
            rng = np.random.default_rng(int(seed_price) % 99991)
            latest_technical = {
                "RSI":  {"value": float(rng.uniform(40, 65)), "signal": "中性"},
                "MACD": {"value": float(rng.uniform(-3, 3)),  "signal": "看跌" if rng.uniform() < 0.5 else "看涨"},
                "Bollinger": {"value": float(rng.uniform(0.3, 0.7)), "signal": "中轨"},
                "ADX":  {"value": float(rng.uniform(15, 35)),  "signal": "偏弱"},
            }
            debug["steps"].append("fallback_mock")

        _prediction_initialized = True
        return debug


def init_dashboard():
    """初始化 Dashboard"""
    global latest_prediction, latest_technical, latest_price

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
                    raw_tech = data.get("technical_analysis", {})
                    latest_technical = _normalize_technical(raw_tech)
                    latest_price = data.get("current_price")
                    loaded_from_file = True
                    logger.info(f"从 {latest_file} 加载预测成功，技术指标: {list(latest_technical.keys())}")
            except Exception as e:
                logger.warning(f"加载预测文件失败: {e}")

    # 2) P1 Phase ⑤：确保 5 年历史数据存在
    _ensure_5yr_history(project_root)

    # 3) 云端 / Render 部署：跑一次实时预测
    if not loaded_from_file or not latest_prediction:
        try:
            import runpy
            from pathlib import Path
            project_root = Path(__file__).resolve().parent.parent
            spec = runpy.run_path(str(project_root / "run_predict.py"), run_name="__predict_mod")
            result = spec.get("predict")
            if result is None:
                raise RuntimeError("run_predict.predict 函数未找到")

            prediction_result = None
            try:
                prediction_result = result(use_mock=False)
            except Exception as e_real:
                logger.warning(f"真实数据预测失败，回退 mock: {e_real}")
                prediction_result = result(use_mock=True)

            if prediction_result and "prediction" in prediction_result:
                latest_prediction = prediction_result["prediction"]
                raw_tech = prediction_result.get("technical_analysis", {})
                latest_technical = _normalize_technical(raw_tech)
                latest_price = float(prediction_result.get("current_price", 0) or 0)
                logger.info(f"云端预测完成，技术指标: {list(latest_technical.keys())}")
                # P1 Fix: 保存 XGBoost AUC 验证结果（白名单方案）
                global _latest_xgb_passes
                _latest_xgb_passes = prediction_result.get("_xgb_passes_threshold", False)
                _latest_ml_auc = prediction_result.get("_ml_cv_auc", {})
                _latest_ml_model = prediction_result.get("_ml_model", "XGBoost")
                logger.info(f"[P1] {_latest_ml_model} AUC 验证: {_latest_xgb_passes} | CV AUC: {_latest_ml_auc}")
                # 写入 output 文件
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    with open(f"{output_dir}/prediction_{ts}.json", "w") as f:
                        json.dump({
                            "timestamp": datetime.now().isoformat(),
                            "current_price": latest_price,
                            "prediction": latest_prediction,
                            "technical_analysis": raw_tech,
                        }, f, indent=2, default=str)
                    logger.info(f"预测文件已写入 {ts}.json")
                except Exception as e_write:
                    logger.warning(f"写入 prediction 文件失败: {e_write}")
        except Exception as e:
            logger.warning(f"云端自启预测失败，使用 mock 兜底: {e}")

    # 3) 兜底：如果仍然没有数据，用 mock
    if not latest_prediction or latest_price is None:
        try:
            from data.mock_data import get_latest_mock_price
            import numpy as np
            price_data = get_latest_mock_price()
            latest_price = float(price_data['price'])
            rng = np.random.default_rng(int(latest_price) % 99991)
            latest_prediction = {
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
                "source": "mock",
            }
            latest_technical = {
                "RSI":       {"value": float(rng.uniform(40, 65)),  "signal": "中性"},
                "MACD":      {"value": float(rng.uniform(-3, 3)),   "signal": "看跌"},
                "Bollinger": {"value": float(rng.uniform(0.3, 0.7)),"signal": "中轨"},
                "ADX":       {"value": float(rng.uniform(15, 35)),  "signal": "偏弱"},
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
    """获取最新预测（含标准化技术指标）"""
    global latest_technical, _latest_xgb_passes, _latest_ml_auc, _latest_ml_model
    debug = _ensure_prediction()
    resp = {
        "status": "success",
        "price": latest_price,
        "prediction": latest_prediction,
        "technical": latest_technical,
        "timestamp": datetime.now().isoformat(),
        "_xgb_passes_threshold": _latest_xgb_passes,   # P1: ML是否通过AUC验证
        "_ml_model": _latest_ml_model,                  # P1: 实际ML模型（LightGBM/XGBoost）
        "_ml_cv_auc": _latest_ml_auc,                   # P1: 各horizon CV AUC
    }
    # 仅在有错误时保留调试信息
    if debug.get("errors"):
        resp["_debug"] = debug
    return jsonify(resp)


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
    """获取标准化技术指标（RSI / MACD / Bollinger / ADX）"""
    global latest_technical
    if not latest_technical:
        _ensure_prediction()
    return jsonify(latest_technical)


@app.route("/api/history")
def api_history():
    """获取历史预测记录"""
    return jsonify({"history": prediction_history[-100:]})


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

        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA60'] = df['Close'].rolling(60).mean()
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
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
    global latest_prediction, latest_technical, latest_price, prediction_history
    latest_prediction = prediction
    latest_price = price
    prediction_history.append({
        "timestamp": datetime.now().isoformat(),
        "price": price,
        "prediction": prediction
    })
    if len(prediction_history) > 1000:
        prediction_history = prediction_history[-1000:]


def run_server(host: str = "0.0.0.0", port: int = 5000):
    """启动 Web 服务"""
    init_dashboard()
    logger.info(f"启动 Dashboard: http://{host}:{port}")
    app.run(host=host, port=port, debug=False)


@app.route("/api/debug/features")
def api_debug_features():
    """
    🎯 特征选择分析端点
    返回 Top 20 核心特征 + 完整排名
    用途：①特征选择阶段，识别130个指标中哪些真正有用
    访问：GET https://gold-predictor-wu2p.onrender.com/api/debug/features
    """
    import yfinance as yf
    import numpy as np
    import pandas as pd

    try:
        import xgboost as xgb
        HAS_XGB = True
    except ImportError:
        HAS_XGB = False

    try:
        # ── 1. 获取数据 ──────────────────────────────────────────
        gold = yf.Ticker("GC=F").history(period="2y", auto_adjust=True)
        gold.index = pd.to_datetime(gold.index).tz_localize(None)

        if len(gold) < 200:
            return jsonify({"error": f"数据不足: {len(gold)} 天"}), 400

        close = gold["Close"]
        high = gold["High"]
        low = gold["Low"]
        volume = gold["Volume"]

        # ── 2. 构建特征（与 feature_engineering.py 完全一致）──────
        features = {}

        # 移动平均线
        for p in [5, 10, 20, 60, 120, 200]:
            features[f"sma_{p}"] = close.rolling(p).mean()
            features[f"ema_{p}"] = close.ewm(span=p, adjust=False).mean()

        # RSI
        for p in [7, 14, 21]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(p).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(p).mean()
            rs = gain / (loss + 1e-10)
            features[f"rsi_{p}"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        features["macd"] = macd_line
        features["macd_signal"] = macd_line.ewm(span=9, adjust=False).mean()
        features["macd_hist"] = macd_line - macd_line.ewm(span=9, adjust=False).mean()

        # KDJ
        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        rsv = 100 * (close - low14) / (high14 - low14 + 1e-10)
        features["kdj_k"] = rsv.ewm(com=2, adjust=False).mean()
        features["kdj_d"] = features["kdj_k"].ewm(com=2, adjust=False).mean()
        features["kdj_j"] = 3 * features["kdj_k"] - 2 * features["kdj_d"]

        # ATR
        tr1 = high - low
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        features["atr_14"] = tr.rolling(14).mean()
        features["atr_28"] = tr.rolling(28).mean()

        # 布林带
        for p in [20, 60]:
            sma = close.rolling(p).mean()
            std = close.rolling(p).std()
            upper = sma + 2 * std
            lower = sma - 2 * std
            features[f"bb_width_{p}"] = (upper - lower) / (sma + 1e-10)
            features[f"bb_position_{p}"] = (close - lower) / (upper - lower + 1e-10)

        # ADX
        for p in [14, 28]:
            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            tr_p = features["atr_14"] if p == 14 else features["atr_28"]
            plus_di = 100 * plus_dm.rolling(p).mean() / (tr_p + 1e-10)
            minus_di = 100 * minus_dm.rolling(p).mean() / (tr_p + 1e-10)
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            features[f"adx_{p}"] = dx.rolling(p).mean()
            features[f"plus_di_{p}"] = plus_di
            features[f"minus_di_{p}"] = minus_di

        # 成交量
        features["volume_sma_20"] = volume.rolling(20).mean()
        features["volume_ratio"] = volume / (features["volume_sma_20"] + 1)
        obv = [0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i-1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])
        features["obv"] = pd.Series(obv, index=close.index)
        features["obv_sma_10"] = features["obv"].rolling(10).mean()

        # 收益率
        for p in [1, 2, 3, 5, 10, 20]:
            features[f"return_{p}d"] = close.pct_change(p)
            features[f"volatility_{p}d"] = close.pct_change().rolling(p).std()

        # 黄金特有
        for p in [20, 60]:
            features[f"high_{p}d"] = high.rolling(p).max()
            features[f"low_{p}d"] = low.rolling(p).min()
            features[f"position_in_range_{p}d"] = (close - features[f"low_{p}d"]) / (
                features[f"high_{p}d"] - features[f"low_{p}d"] + 1e-10)

        features["intraday_range"] = (high - low) / (close + 1e-10)
        features["close_position"] = (close - low) / (high - low + 1e-10)

        # 跨资产
        cross_ok = False
        for name, sym in [("vix", "^VIX"), ("dxy", "UUP"), ("tlt", "TLT"), ("spy", "SPY")]:
            try:
                ex = yf.Ticker(sym).history(start=gold.index[0], end=gold.index[-1], auto_adjust=True)
                ex.index = pd.to_datetime(ex.index).tz_localize(None)
                ex_c = ex["Close"].reindex(gold.index).ffill()
                features[f"corr_{name}_close"] = close.rolling(20).corr(ex_c)
                features[f"corr_{name}_return"] = close.pct_change().rolling(10).corr(ex_c.pct_change())
                cross_ok = True
            except Exception:
                pass

        feat_df = pd.DataFrame(features, index=gold.index)
        feat_df["close"] = close
        feat_df = feat_df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

        # 特征列
        exclude = {"close", "volume", "open", "high", "low"}
        feature_cols = [c for c in feat_df.columns if c.lower() not in exclude]
        n_features = len(feature_cols)

        if n_features < 10:
            return jsonify({"error": "特征数不足"}), 400

        # ── 3. 打分：相关性轨道 ─────────────────────────────────
        horizons = [1, 3, 5]
        scores = {col: 0.0 for col in feature_cols}

        for h in horizons:
            future_ret = close.shift(-h) / close - 1
            direction = (future_ret > 0).astype(int)
            for col in feature_cols:
                corr = feat_df[col].corr(direction)
                if not np.isnan(corr):
                    scores[col] += abs(corr) / len(horizons)

        # ── 4. 打分：XGBoost permutation importance（如果可用）──
        base_auc = None
        if HAS_XGB:
            try:
                X_raw = feat_df[feature_cols].values
                X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
                std = np.std(X_raw, axis=0)
                std[std == 0] = 1.0
                X_scaled = (X_raw - np.mean(X_raw, axis=0)) / std

                future_ret = close.shift(-3) / close - 1
                labels = (future_ret > 0).astype(int).values
                valid = ~np.isnan(future_ret.values)
                X_v, y_v = X_scaled[valid], labels[valid]
                split = int(len(X_v) * 0.8)
                X_tr, X_va = X_v[:split], X_v[split:]
                y_tr, y_va = y_v[:split], y_v[split:]

                from sklearn.metrics import roc_auc_score
                model = xgb.XGBClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, n_jobs=-1, verbosity=0, use_label_encoder=False,
                )
                model.fit(X_tr, y_tr)
                va_proba = model.predict_proba(X_va)[:, 1]
                base_auc = roc_auc_score(y_va, va_proba)

                # ── 白名单对比实验：直接用 all-col 训练数据切片 ─────
                whitelist_auc_3d = None
                whitelist_auc_1d = None
                try:
                    from config import FEATURE_WHITELIST
                    wl_cols = [c for c in FEATURE_WHITELIST if c in feature_cols]
                    logger.info(f"[FeatureExp] 白名单命中 {len(wl_cols)}/{len(FEATURE_WHITELIST)} 个: {wl_cols}")
                    if len(wl_cols) < 5:
                        logger.warning("[FeatureExp] 白名单列不足5个，跳过")
                    else:
                        # feat_df 中有 "close" 列吗？没有就用 outer scope 的 close Series
                        price_series = feat_df["close"] if "close" in feat_df.columns else close
                        logger.info(f"[FeatureExp] feat_df shape: {feat_df.shape}, close type: {type(price_series).__name__}")
                        # 3天标签
                        future_ret_3d = price_series.shift(-3) / price_series - 1
                        labels_3d = (future_ret_3d > 0).astype(int).values
                        valid_3d = ~np.isnan(future_ret_3d.values)
                        logger.info(f"[FeatureExp] 3d标签 有效: {valid_3d.sum()}/{len(valid_3d)}, split={split}")
                        # 白名单特征
                        X_wl = feat_df[wl_cols].values[valid_3d]
                        y_wl = labels_3d[valid_3d]
                        X_wl = np.nan_to_num(X_wl, nan=0.0, posinf=0.0, neginf=0.0)
                        X_wl_split = int(len(X_wl) * 0.8)
                        std_wl = np.std(X_wl, axis=0)
                        std_wl[std_wl == 0] = 1.0
                        X_wl_scaled = (X_wl - np.mean(X_wl, axis=0)) / std_wl
                        model_wl = xgb.XGBClassifier(
                            n_estimators=200, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=42, n_jobs=-1, verbosity=0,
                        )
                        model_wl.fit(X_wl_scaled[:X_wl_split], y_wl[:X_wl_split])
                        wl_proba_3d = model_wl.predict_proba(X_wl_scaled[X_wl_split:])[:, 1]
                        whitelist_auc_3d = roc_auc_score(y_wl[X_wl_split:], wl_proba_3d)
                        # 1天标签
                        future_ret_1d = price_series.shift(-1) / price_series - 1
                        labels_1d = (future_ret_1d > 0).astype(int).values
                        valid_1d = ~np.isnan(future_ret_1d.values)
                        X_wl_1d = feat_df[wl_cols].values[valid_1d]
                        y_wl_1d = labels_1d[valid_1d]
                        X_wl_1d = np.nan_to_num(X_wl_1d, nan=0.0, posinf=0.0, neginf=0.0)
                        X_wl_1d_split = int(len(X_wl_1d) * 0.8)
                        std_wl_1d = np.std(X_wl_1d, axis=0)
                        std_wl_1d[std_wl_1d == 0] = 1.0
                        X_wl_1d_scaled = (X_wl_1d - np.mean(X_wl_1d, axis=0)) / std_wl_1d
                        model_wl_1d = xgb.XGBClassifier(
                            n_estimators=200, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=42, n_jobs=-1, verbosity=0,
                        )
                        model_wl_1d.fit(X_wl_1d_scaled[:X_wl_1d_split], y_wl_1d[:X_wl_1d_split])
                        wl_proba_1d = model_wl_1d.predict_proba(X_wl_1d_scaled[X_wl_1d_split:])[:, 1]
                        whitelist_auc_1d = roc_auc_score(y_wl_1d[X_wl_1d_split:], wl_proba_1d)
                        logger.info(f"[FeatureExp] 白名单AUC 3d={whitelist_auc_3d:.4f} 1d={whitelist_auc_1d:.4f} vs 全特征AUC 3d={base_auc:.4f}")
                except Exception as e_wl:
                    import traceback as _tb
                    logger.warning(f"[FeatureExp] 白名单对比失败: {e_wl}")
                    logger.warning(f"[FeatureExp] TRACE: {_tb.format_exc()[-400:]}")

                # permutation importance
                for i, col in enumerate(feature_cols):
                    X_perm = X_va.copy()
                    np.random.seed(i)
                    X_perm[:, i] = X_perm[np.random.permutation(len(X_perm)), i]
                    perm_auc = roc_auc_score(y_va, model.predict_proba(X_perm)[:, 1])
                    drop = max(0.0, base_auc - perm_auc)
                    scores[col] += (drop / 2.0)

            except Exception as e_xgb:
                logger.warning(f"XGBoost 打分失败: {e_xgb}")

        # ── 5. 归一化并排序 ──────────────────────────────────────
        max_s = max(scores.values()) if max(scores.values()) > 0 else 1
        ranked = sorted([(col, scores[col] / max_s) for col in feature_cols],
                        key=lambda x: x[1], reverse=True)

        top20 = ranked[:20]

        # ── 6. 分层建议 ──────────────────────────────────────────
        tier1 = [(c, s) for c, s in ranked[:8] if s >= 0.4]
        tier2 = [(c, s) for c, s in ranked[8:15] if s >= 0.25]
        tier3 = [(c, s) for c, s in ranked[15:25]]

        return jsonify({
            "status": "ok",
            "n_total_features": n_features,
            "n_samples": len(gold),
            "xgb_available": HAS_XGB,
            "base_auc_3d": round(base_auc, 4) if base_auc else None,
            "whitelist_auc_3d": round(whitelist_auc_3d, 4) if whitelist_auc_3d else None,
            "whitelist_auc_1d": round(whitelist_auc_1d, 4) if whitelist_auc_1d else None,
            "auc_comparison": {
                "all_features_auc_3d": round(base_auc, 4) if base_auc else None,
                "whitelist_auc_3d": round(whitelist_auc_3d, 4) if whitelist_auc_3d else None,
                "whitelist_auc_1d": round(whitelist_auc_1d, 4) if whitelist_auc_1d else None,
                "improvement": round((whitelist_auc_3d - base_auc) * 100, 1) if (whitelist_auc_3d and base_auc) else None,
                "verdict": "白名单胜出 ✅" if (whitelist_auc_3d and base_auc and whitelist_auc_3d > base_auc) else ("全特征胜出 ⚠️" if (whitelist_auc_3d and base_auc) else "待验证")
            },
            "top20": [{"rank": i+1, "feature": c, "score": round(s, 4)} for i, (c, s) in enumerate(top20)],
            "tier1_required": [{"rank": i+1, "feature": c, "score": round(s, 4)} for i, (c, s) in enumerate(tier1)],
            "tier2_optional": [{"rank": 9+i, "feature": c, "score": round(s, 4)} for i, (c, s) in enumerate(tier2)],
            "tier3_experimental": [{"rank": 16+i, "feature": c, "score": round(s, 4)} for i, (c, s) in enumerate(tier3)],
            "recommendation": {
                "whitelist": [c for c, _ in tier1] + [c for c, _ in tier2[:4]],
                "description": f"Tier-1({len(tier1)}) 必须保留 + Tier-2({min(len(tier2),4)}) 建议保留 = 共{len(tier1)+min(len(tier2),4)}个特征"
            },
            "generated_at": datetime.now().isoformat(),
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()[-500:]}), 500


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_server()
