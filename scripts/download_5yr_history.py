#!/usr/bin/env python3
"""
下载 5 年黄金历史数据 + 宏观数据并保存为 Parquet。
优先用 FRED（走 pandas_datareader）+ 本地 yfinance。

用法：python3 scripts/download_5yr_history.py
作者：gold-predictor P1 Phase ⑤
日期：2026-08-20
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("download_5yr")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH  = PROJECT_ROOT / "data" / "gold_5yr.parquet"
MACRO_PATH   = PROJECT_ROOT / "data" / "macro_5yr.parquet"


# ================================================================
# 方案 A：Stooq 免费历史数据（全球黄金，HTTP 直接下载）
# ================================================================
def download_gold_from_stooq() -> pd.DataFrame:
    """用 Stooq 免费 API 下载 GC=F 黄金历史数据（5年日频）"""
    import urllib.request, io
    log.info("正在从 Stooq 下载 GC=F 黄金历史数据（5年）...")
    # Stooq 格式：symbol_us.csv?d1=start&d2=end
    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=5)
    url = (f"https://stooq.com/q/d/l/?"
           f"s=gc%3Df&"
           f"d1={start.strftime('%Y%m%d')}&"
           f"d2={end.strftime('%Y%m%d')}&"
           f"i=d")  # d=日频
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                raw = resp.read().decode()
            if "404" in raw or "Not Found" in raw:
                raise RuntimeError(f"Stooq 返回 404: {url}")
            df = pd.read_csv(io.StringIO(raw), parse_dates=[0], index_col=0)
            break
        except Exception as e:
            log.warning(f"Stooq 下载失败 attempt {attempt+1}: {e}")
            if attempt < 2: time.sleep(10)
            else: raise

    if df.empty:
        raise RuntimeError("Stooq 返回空数据")

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # Stooq 列名：Date, Open, High, Low, Close, Volume
    rename = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    df.rename(columns=rename, inplace=True)
    if "close" not in df.columns and "Close" in df.columns:
        df.columns = [c.title() for c in df.columns]
        df.rename(columns={"Close": "close", "Open": "open", "High": "high", "Low": "low", "Volume": "volume"}, inplace=True)
    log.info(f"  Stooq 下载：{len(df)} 条，{df.index[0].date()} ~ {df.index[-1].date()}，最后价格 ${df['close'].iloc[-1]:.2f}")
    return df


# ================================================================
# 方案 B：yfinance 补全近 1 年日频数据（接在 FRED 后面）
# ================================================================
def supplement_daily_from_yfinance(fred_df: pd.DataFrame) -> pd.DataFrame:
    """用 yfinance 下载近 1 年日频数据，与 FRED 对接"""
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance 未安装，跳过日频补充"); return fred_df

    log.info("正在从 yfinance 下载近 1 年日频数据（接续 FRED）...")
    ticker = yf.Ticker("GC=F")
    # 截取 FRED 最新日期之后的数据
    start = fred_df.index[-1] + pd.Timedelta(days=1)
    end = pd.Timestamp.today()
    if start >= end:
        log.info("  FRED 数据已是最新，跳过 yfinance"); return fred_df

    for attempt in range(3):
        try:
            df_yf = ticker.history(start=start, end=end)
            if df_yf.empty:
                raise RuntimeError("yfinance 返回空")
            df_yf = df_yf.rename(columns={"Close": "close"})[["close"]]
            df_yf.index = pd.to_datetime(df_yf.index).tz_localize(None)
            log.info(f"  yfinance 补充：{len(df_yf)} 条，{df_yf.index[0].date()} ~ {df_yf.index[-1].date()}")
            combined = pd.concat([fred_df, df_yf]).resample("B").ffill()
            return combined
        except Exception as e:
            log.warning(f"yfinance 下载失败 attempt {attempt+1}: {e}")
            if attempt < 2: time.sleep(15)
    log.warning("yfinance 全部失败，返回 FRED 数据")
    return fred_df


# ================================================================
# 方案 C：FRED 宏观数据（月度5年）
# ================================================================
def download_macro_from_fred() -> pd.DataFrame:
    """用 pandas_datareader 从 FRED 下载 5 年宏观数据"""
    try:
        import pandas_datareader as pdr
    except ImportError:
        log.warning("pandas_datareader 未安装，跳过宏观数据"); return pd.DataFrame()

    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=5)
    series_map = {
        "dxy_fred": "DTWEXBGS",
        "dgs10":    "DGS10",
        "cpi":      "CPIAUCSL",
        "vix":      "VIXCLS",
        "m2":       "M2SL",
        "unrate":   "UNRATE",
    }
    dfs = {}
    for name, fid in series_map.items():
        try:
            s = pdr.get_data_fred(fid, start, end)
            if not s.empty:
                s.columns = [name]
                dfs[name] = s.resample("B").ffill()
                log.info(f"  {name}: {len(s)} 条 ✅")
        except Exception as e:
            log.warning(f"  {name} 失败: {e}")
    if not dfs: return pd.DataFrame()
    macro = pd.concat(dfs.values(), axis=1)
    macro.index = pd.to_datetime(macro.index).tz_localize(None)
    return macro


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MACRO_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. Stooq 黄金日频（5年，直接日频，不需要 yfinance 补充）
    gold_df = download_gold_from_stooq()

    # 2. 标准化列名（已是日频，无需 resample）
    gold_df = gold_df.dropna(subset=["close"])
    gold_df.index.name = "Date"
    if "open" not in gold_df.columns:
        gold_df["open"] = gold_df["close"]
        gold_df["high"] = gold_df["close"]
        gold_df["low"]  = gold_df["close"]
        gold_df["volume"] = 0.0
    gold_df = gold_df[["open", "high", "low", "close", "volume"]]

    # 4. 保存
    gold_df.to_parquet(OUTPUT_PATH)
    sz = OUTPUT_PATH.stat().st_size / 1024 / 1024
    log.info(f"✅ 黄金历史数据已保存 → {OUTPUT_PATH} ({sz:.1f} MB)")

    # 5. 宏观数据
    try:
        macro_df = download_macro_from_fred()
        if not macro_df.empty:
            macro_df.to_parquet(MACRO_PATH)
            log.info(f"✅ 宏观数据已保存 → {MACRO_PATH}")
    except Exception as e:
        log.warning(f"宏观数据失败（不阻断）: {e}")

    # 6. 验证
    df_check = pd.read_parquet(OUTPUT_PATH)
    log.info(f"验证：{len(df_check)} 条，{df_check.index[0].date()} ~ {df_check.index[-1].date()}，${df_check['close'].iloc[-1]:.2f}/oz")

    print(f"\n🎉 完成！共 {len(df_check)} 条黄金历史数据")
    print(f"   {df_check.index[0].date()} ~ {df_check.index[-1].date()}")
    print(f"   最近价格：${df_check['close'].iloc[-1]:.2f}/oz")


if __name__ == "__main__":
    main()
