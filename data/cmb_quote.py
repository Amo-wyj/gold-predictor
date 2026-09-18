"""
招行积存金参考价：国际现货（美元/盎司）→ 人民币/克。

只做换算，不猜招行买卖点差。成交价以招行 App 为准。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

TROY_OZ_GRAMS = 31.1034768
FX_TICKER = "CNY=X"
CACHE_MAX_AGE_SEC = 3600


def _fx_cache_path() -> str:
    from config import OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, "usd_cny.json")


def _read_fx_cache() -> Optional[float]:
    path = _fx_cache_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            cached = json.load(f)
        ts = cached.get("updated_at")
        rate = cached.get("usd_cny")
        if not ts or rate is None:
            return None
        age = (datetime.now() - datetime.fromisoformat(ts)).total_seconds()
        if age > CACHE_MAX_AGE_SEC:
            return None
        rate = float(rate)
        if rate < 5 or rate > 10:
            return None
        return rate
    except Exception:
        return None


def fetch_usd_cny(force_refresh: bool = False) -> float:
    if not force_refresh:
        cached = _read_fx_cache()
        if cached is not None:
            return cached

    import yfinance as yf
    hist = yf.Ticker(FX_TICKER).history(period="5d", interval="1d", auto_adjust=True)
    if hist is None or hist.empty:
        raise RuntimeError("无法获取美元兑人民币汇率")
    rate = float(hist["Close"].iloc[-1])
    if rate < 5 or rate > 10:
        raise RuntimeError(f"汇率异常: {rate}")

    try:
        with open(_fx_cache_path(), "w", encoding="utf-8") as f:
            json.dump({
                "usd_cny": rate,
                "updated_at": datetime.now().isoformat(),
                "source": FX_TICKER,
            }, f)
    except Exception as e:
        logger.warning(f"[CMB] 汇率缓存写入失败: {e}")
    return rate


def to_cny_per_gram(usd_per_oz: float, usd_cny: Optional[float] = None) -> Dict:
    """国际金价（美元/盎司）折人民币/克，不含招行点差。"""
    price = float(usd_per_oz)
    if price < 1000:
        raise ValueError(f"金价过低，拒绝换算: {price}")
    rate = float(usd_cny) if usd_cny is not None else fetch_usd_cny()
    cny = price / TROY_OZ_GRAMS * rate
    return {
        "cny_per_gram": round(cny, 2),
        "usd_per_oz": round(price, 2),
        "usd_cny": round(rate, 4),
        "troy_oz_grams": TROY_OZ_GRAMS,
        "spread_included": False,
        "note": "国际金折人民币参考价，未扣招行买卖点差。买卖时机可参考国际金，成交价以招行 App 为准。",
        "source": "GC=F / CNY=X",
    }
