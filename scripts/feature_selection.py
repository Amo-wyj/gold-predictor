#!/usr/bin/env python3
"""
特征选择脚本
目标：从 130 个技术指标中找出预测能力最强的 Top 20
方法：相关性分析 + XGBoost 特征重要性 双轨验证
"""

import os
import sys
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

# ── Setup ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("feature_selection")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ── Data Fetch (try FRED first for long history) ──────────────────────────
def fetch_gold_data(days=2000):
    """获取黄金期货历史数据（尝试 FRED 5年，fallback yfinance 2年）"""
    end = datetime.today()
    start = end - timedelta(days=days)

    # FRED: 直接用 fredapi 拉日线
    try:
        from fredapi import Fred
        from config import FRED_API_KEY
        fred = Fred(api_key=FRED_API_KEY)
        gold_series = fred.get_series("GOLDAMGBD228NLBM", start, end)
        df = gold_series.to_frame(name="close")
        df.index = pd.to_datetime(df.index)
        df = df.resample("B").last().dropna()  # 工作日重采样
        df["open"] = df["high"] = df["low"] = df["close"]
        df["volume"] = 0
        df = df[["open", "high", "low", "close", "volume"]]
        logger.info(f"[Data] FRED 成功: {len(df)} 个工作日, {df.index[0].date()} → {df.index[-1].date()}")
        return df
    except Exception as e:
        logger.warning(f"[Data] FRED 失败 ({e})，用 yfinance...")

    # yfinance fallback
    ticker = yf.Ticker("GC=F")
    df = ticker.history(start=start, end=end, auto_adjust=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["open", "high", "low", "close", "volume"]
    logger.info(f"[Data] yfinance 成功: {len(df)} 天, {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ── Feature Engineering ────────────────────────────────────────────────────
def build_features(df):
    """从 OHLCV 构建所有技术指标（与 FeatureEngine 逻辑一致）"""
    d = df.copy()
    features = {}

    # 价格列
    close = d["close"]
    high = d["high"]
    low = d["low"]
    volume = d["volume"]

    # === 移动平均线 ===
    for p in [5, 10, 20, 60, 120, 200]:
        features[f"sma_{p}"] = close.rolling(p).mean()
        features[f"ema_{p}"] = close.ewm(span=p, adjust=False).mean()

    # === RSI ===
    for p in [7, 14, 21]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(p).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(p).mean()
        rs = gain / loss
        features[f"rsi_{p}"] = 100 - (100 / (1 + rs))

    # === MACD ===
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    features["macd"] = macd_line
    features["macd_signal"] = macd_signal
    features["macd_hist"] = macd_line - macd_signal

    # === KDJ ===
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    rsv = 100 * (close - low14) / (high14 - low14 + 1e-10)
    k = rsv.ewm(com=2, adjust=False).mean()
    d_ = k.ewm(com=2, adjust=False).mean()
    j = 3 * k - 2 * d_
    features["kdj_k"] = k
    features["kdj_d"] = d_
    features["kdj_j"] = j

    # === ATR ===
    tr1 = high - low
    tr2 = np.abs(high - close.shift())
    tr3 = np.abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    features["atr_14"] = tr.rolling(14).mean()
    features["atr_28"] = tr.rolling(28).mean()

    # === 布林带 ===
    for p in [20, 60]:
        sma = close.rolling(p).mean()
        std = close.rolling(p).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        features[f"bb_width_{p}"] = (upper - lower) / sma
        features[f"bb_position_{p}"] = (close - lower) / (upper - lower + 1e-10)

    # === ADX ===
    for p in [14, 28]:
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr_p = features["atr_14"] if p == 14 else features["atr_28"]
        plus_di = 100 * plus_dm.rolling(p).mean() / tr_p
        minus_di = 100 * minus_dm.rolling(p).mean() / tr_p
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        features[f"adx_{p}"] = dx.rolling(p).mean()
        features[f"plus_di_{p}"] = plus_di
        features[f"minus_di_{p}"] = minus_di

    # === 成交量 ===
    features["volume_sma_20"] = volume.rolling(20).mean()
    features["volume_ratio"] = volume / (features["volume_sma_20"] + 1)

    # OBV
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

    # === 收益率特征 ===
    for p in [1, 2, 3, 5, 10, 20]:
        features[f"return_{p}d"] = close.pct_change(p)
        features[f"volatility_{p}d"] = close.pct_change().rolling(p).std()

    # === 黄金特有 ===
    for p in [20, 60]:
        features[f"high_{p}d"] = high.rolling(p).max()
        features[f"low_{p}d"] = low.rolling(p).min()
        features[f"position_in_range_{p}d"] = (close - features[f"low_{p}d"]) / (
            features[f"high_{p}d"] - features[f"low_{p}d"] + 1e-10)

    # 日内波动
    features["intraday_range"] = (high - low) / close
    features["close_position"] = (close - low) / (high - low + 1e-10)

    # === 跨资产（用 yfinance 额外拉）===
    try:
        symbols = {
            "dxy": "UUP",      # 美元指数 ETF
            "vix": "^VIX",    # VIX 恐慌指数
            "tlt": "TLT",     # 20年债券 ETF
            "spy": "SPY",     # 标普 500
        }
        for name, sym in symbols.items():
            extra = yf.Ticker(sym).history(start=d.index[0], end=d.index[-1], auto_adjust=True)
            extra.index = pd.to_datetime(extra.index).tz_localize(None)
            reindexed = extra["Close"].reindex(d.index).ffill()
            features[f"corr_{name}_close"] = close.rolling(20).corr(reindexed)
            features[f"corr_{name}_return"] = close.pct_change().rolling(10).corr(reindexed.pct_change())
    except Exception as e:
        logger.warning(f"[Cross-Asset] 跳过: {e}")

    # 合并
    feat_df = pd.DataFrame(features, index=d.index)
    feat_df["close"] = close
    feat_df["volume"] = volume

    # 清理 inf / NaN
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
    feat_df = feat_df.ffill().fillna(0)

    return feat_df


# ── Feature Importance Scoring ──────────────────────────────────────────────
def score_features(feat_df, horizons=[1, 3, 5]):
    """双轨打分：① 相关性 ② XGBoost 重要性"""

    # 特征列（排除 OHLCV）
    feature_cols = [c for c in feat_df.columns
                    if c not in ("close", "volume", "open", "high", "low")]

    scores = {col: 0.0 for col in feature_cols}
    reasons = {col: "" for col in feature_cols}

    # ── ① 相关性打分 ──
    logger.info("[Scoring] 轨道1: 相关性分析...")
    corr_scores = []
    for h in horizons:
        future_ret = feat_df["close"].shift(-h) / feat_df["close"] - 1
        direction = (future_ret > 0).astype(int)  # 涨跌二分类

        for col in feature_cols:
            # 用 abs 相关性（正负都算预测能力）
            corr = feat_df[col].corr(direction)
            if not np.isnan(corr):
                scores[col] += abs(corr) / len(horizons)
                corr_scores.append(abs(corr))

    # ── ② XGBoost 重要性打分 ──
    logger.info("[Scoring] 轨道2: XGBoost 特征重要性...")
    try:
        import xgboost as xgb
        from sklearn.metrics import roc_auc_score

        # 准备数据
        X = feat_df[feature_cols].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        std = np.std(X, axis=0)
        std[std == 0] = 1.0
        X = (X - np.mean(X, axis=0)) / std

        # 用 horizon=3 的方向标签做单次训练取重要性
        future_ret = feat_df["close"].shift(-3) / feat_df["close"] - 1
        labels = (future_ret > 0).astype(int)
        labels = labels.values
        valid = ~np.isnan(future_ret.values)
        X_valid = X[valid]
        y_valid = labels[valid]

        # 简单 train/test split（最后 20% 当验证）
        split = int(len(X_valid) * 0.8)
        X_tr, X_va = X_valid[:split], X_valid[split:]
        y_tr, y_va = y_valid[:split], y_valid[split:]

        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbosity=0, use_label_encoder=False,
        )
        model.fit(X_tr, y_tr)

        # permutation importance（打乱每个特征，看 AUC 下降多少）
        va_proba = model.predict_proba(X_va)[:, 1]
        base_auc = roc_auc_score(y_va, va_proba)

        importances = []
        for i, col in enumerate(feature_cols):
            X_perm = X_va.copy()
            np.random.seed(i)
            perm_idx = np.random.permutation(len(X_perm))
            X_perm[:, i] = X_perm[perm_idx, i]
            perm_proba = model.predict_proba(X_perm)[:, 1]
            perm_auc = roc_auc_score(y_va, perm_proba)
            drop = base_auc - perm_auc
            importances.append(drop)
            # XGBoost 原生重要性也加进来
            scores[col] += (drop + model.feature_importances_[i] * base_auc) / 2

        logger.info(f"[XGBoost] 基础 AUC={base_auc:.3f}")

    except ImportError:
        logger.warning("[Scoring] xgboost 未安装，仅用相关性打分")
    except Exception as e:
        logger.warning(f"[Scoring] XGBoost 失败: {e}，仅用相关性打分")

    # 归一化到 [0, 1]
    max_score = max(scores.values()) if max(scores.values()) > 0 else 1
    for col in scores:
        scores[col] /= max_score

    # 排序
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return ranked, scores


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("🎯 黄金预测 — 特征选择分析")
    logger.info("=" * 60)

    # 1. 获取数据
    df = fetch_gold_data(days=2000)  # ~5年 FRED 或 2年 yfinance
    logger.info(f"数据量: {len(df)} 天, 最新价格: ${df['close'].iloc[-1]:.2f}")

    # 2. 构建特征
    logger.info("构建技术指标特征...")
    feat_df = build_features(df)
    n_feats = len([c for c in feat_df.columns if c not in ('close','volume','open','high','low')])
    logger.info(f"特征总数: {n_feats} 个")

    # 3. 打分
    ranked, scores = score_features(feat_df)

    # 4. 输出 Top 20
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"🏆 TOP 20 核心特征（按综合重要性排序）")
    logger.info("=" * 60)
    logger.info(f"{'排名':<4} {'特征名':<30} {'得分':<8}")
    logger.info("-" * 44)
    for i, (col, score) in enumerate(ranked[:20], 1):
        bar = "█" * int(score * 10)
        logger.info(f"{i:<4} {col:<30} {score:.4f} {bar}")

    # 5. 输出建议保留的特征列表（Top 20）
    top20 = [col for col, _ in ranked[:20]]
    logger.info("")
    logger.info("📋 建议保留的 Top 20 特征:")
    logger.info("FEATURE_LIST = " + repr(top20))

    # 6. Tier 分层
    logger.info("")
    logger.info("📊 特征分层建议:")
    tier1 = [col for col, s in ranked[:8] if s >= 0.5]
    tier2 = [col for col, s in ranked[8:15] if s >= 0.3]
    tier3 = [col for col, s in ranked[15:20]]
    logger.info(f"  Tier-1 (必须保留, {len(tier1)}个): {tier1}")
    logger.info(f"  Tier-2 (建议保留, {len(tier2)}个): {tier2}")
    logger.info(f"  Tier-3 (可尝试, {len(tier3)}个): {tier3}")

    logger.info("")
    logger.info("✅ 特征选择完成！")
    logger.info("的下一步: 把 Top 20 特征填入 config.py 的 FEATURE_WHITELIST，模型只喂这些特征")
    logger.info("=" * 60)

    return top20


if __name__ == "__main__":
    top20 = main()
