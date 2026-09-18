"""
Phase 3：多时间框架（日线 vs 4h）
- yfinance 拉 1h，重采样为 4h
- 与日线方向对比 → timeframe_agreement
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _four_hour_bias(df_1h: pd.DataFrame) -> Dict:
    """基于 4h K 计算短期偏向 score∈[-1,1]。"""
    if df_1h is None or df_1h.empty:
        return {"score": 0.0, "label": "NO_DATA", "bars": 0}

    df = df_1h.copy()
    # 列名兼容
    cols = {c.lower(): c for c in df.columns}
    close_col = cols.get("close") or cols.get("adj close")
    if close_col is None:
        return {"score": 0.0, "label": "NO_DATA", "bars": 0}

    # 确保 DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    ohlc = pd.DataFrame({
        "open": df[cols.get("open", close_col)],
        "high": df[cols.get("high", close_col)],
        "low": df[cols.get("low", close_col)],
        "close": df[close_col],
    })
    h4 = ohlc.resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }).dropna()

    if len(h4) < 6:
        return {"score": 0.0, "label": "TOO_FEW", "bars": len(h4)}

    last = h4.iloc[-1]
    prev = h4.iloc[-2]
    ret1 = float(last["close"] / prev["close"] - 1)
    sma = float(h4["close"].tail(12).mean())
    above_sma = float(last["close"] > sma)
    candle = 1.0 if last["close"] >= last["open"] else -1.0

    # 近 6 根 4h 动量
    mom = float(h4["close"].iloc[-1] / h4["close"].iloc[-6] - 1)
    score = np.clip(mom * 40.0 + ret1 * 20.0 + 0.15 * candle + 0.1 * (1 if above_sma else -1), -1, 1)

    if score >= 0.2:
        label = "H4_BULLISH"
    elif score <= -0.2:
        label = "H4_BEARISH"
    else:
        label = "H4_NEUTRAL"

    return {
        "score": round(float(score), 4),
        "label": label,
        "bars": int(len(h4)),
        "last_close": round(float(last["close"]), 2),
        "mom_6bars": round(mom, 5),
        "above_sma12": bool(above_sma),
    }


def analyze_timeframes(daily_signal: Optional[str] = None) -> Dict:
    """
    daily_signal: 如 STRONG_BUY / BUY / SELL / STRONG_SELL / NEUTRAL
    """
    from config import OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    result = {
        "timeframe_agreement": "UNKNOWN",
        "h4": {},
        "daily_signal": daily_signal,
        "agreement_score": 0.0,
        "updated_at": datetime.now().isoformat(),
        "source": "yfinance_1h_resample_4h",
    }

    try:
        import yfinance as yf
        ticker = yf.Ticker("GC=F")
        hist = ticker.history(period="60d", interval="1h", auto_adjust=True)
        if hist is None or hist.empty:
            result["timeframe_agreement"] = "NO_DATA"
            result["error"] = "empty_1h"
            return result
        h4 = _four_hour_bias(hist)
        result["h4"] = h4
    except Exception as e:
        logger.warning(f"[TF] 4h 分析失败: {e}")
        result["timeframe_agreement"] = "ERROR"
        result["error"] = str(e)[:120]
        return result

    h4_score = float(h4.get("score") or 0.0)
    daily = (daily_signal or "NEUTRAL").upper()
    daily_bull = daily in ("BUY", "STRONG_BUY")
    daily_bear = daily in ("SELL", "STRONG_SELL")
    h4_bull = h4_score >= 0.2
    h4_bear = h4_score <= -0.2

    if daily_bull and h4_bull:
        agree = "AGREE_BULL"
        a_score = 0.6
    elif daily_bear and h4_bear:
        agree = "AGREE_BEAR"
        a_score = -0.6
    elif (daily_bull and h4_bear) or (daily_bear and h4_bull):
        agree = "DIVERGE"
        a_score = 0.0
    else:
        agree = "MIXED"
        a_score = float(np.clip(h4_score * 0.5, -0.3, 0.3))

    result["timeframe_agreement"] = agree
    result["agreement_score"] = round(a_score, 4)

    try:
        path = os.path.join(OUTPUT_DIR, "timeframe_daily.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(analyze_timeframes("BUY"), ensure_ascii=False, indent=2))
