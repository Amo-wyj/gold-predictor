"""
特征工程模块
生成用于LSTM模型的技术指标 + 宏观因子特征
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FeatureEngine:
    """特征工程引擎"""
    
    def __init__(self):
        self.feature_names: List[str] = []
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标"""
        df = df.copy()
        
        # === 移动平均线 ===
        for period in [5, 10, 20, 60, 120, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # === 动量指标 ===
        # RSI (Relative Strength Index)
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 随机指标 KDJ
        low14 = df['low'].rolling(window=14).min()
        high14 = df['high'].rolling(window=14).max()
        df['rsv'] = 100 * (df['close'] - low14) / (high14 - low14)
        df['k'] = df['rsv'].ewm(com=2, adjust=False).mean()
        df['d'] = df['k'].ewm(com=2, adjust=False).mean()
        df['j'] = 3 * df['k'] - 2 * df['d']
        
        # === 波动率指标 ===
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(window=14).mean()
        df['atr_28'] = tr.rolling(window=28).mean()
        
        # 布林带
        for period in [20, 60]:
            sma = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            df[f'bb_upper_{period}'] = sma + 2 * std
            df[f'bb_lower_{period}'] = sma - 2 * std
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
        
        # === 趋势指标 ===
        # ADX (Average Directional Index)
        for period in [14, 28]:
            plus_dm = df['high'].diff()
            minus_dm = -df['low'].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            tr14 = df['atr_14'] if period == 14 else df['atr_28']
            
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr14)
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr14)
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            df[f'adx_{period}'] = dx.rolling(window=period).mean()
            df[f'plus_di_{period}'] = plus_di
            df[f'minus_di_{period}'] = minus_di
        
        # === 成交量指标 ===
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # OBV (On-Balance Volume)
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['obv'] = obv
        df['obv_sma_10'] = df['obv'].rolling(window=10).mean()
        
        # === 价格变化特征 ===
        for period in [2, 3, 5, 10, 20]:
            df[f'return_{period}d'] = df['close'].pct_change(period)
            df[f'volatility_{period}d'] = df['close'].pct_change().rolling(window=period).std()
        
        # === 黄金特有指标 ===
        # 金价相对高低点
        for period in [20, 60]:
            df[f'high_{period}d'] = df['high'].rolling(window=period).max()
            df[f'low_{period}d'] = df['low'].rolling(window=period).min()
            df[f'position_in_range_{period}d'] = (df['close'] - df[f'low_{period}d']) / (df[f'high_{period}d'] - df[f'low_{period}d'])
        
        # 日内波动率
        df['intraday_range'] = (df['high'] - df['low']) / df['close']
        
        # 收盘位置
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])

        # ============================================================
        # P1 Phase ③：特征工程升级（2026-08-20）
        # 基于 Top20 分析：均线族占 13/20，新增互补特征补盲区
        # ============================================================

        # 1) 动量特征：纯价格变化率（独立于均线，互补均线族）
        for period in [3, 7, 14]:
            df[f'momentum_{period}d'] = df['close'].pct_change(period)

        # 2) 收益率分布特征（偏度/峰度）：捕捉尾部风险和尖峰分布
        for period in [10, 20]:
            ret = df['close'].pct_change()
            roll = ret.rolling(window=period)
            n = roll.count()
            mean = roll.mean()
            std = roll.std()
            # 偏度：衡量收益率分布的不对称性
            m2 = roll.apply(lambda x: ((x - x.mean()) ** 2).mean(), raw=False)
            m3 = roll.apply(lambda x: ((x - x.mean()) ** 3).mean(), raw=False)
            with np.errstate(divide='ignore', invalid='ignore'):
                df[f'return_skew_{period}d'] = m3 / (m2 ** 1.5 + 1e-10)
            # 峰度：衡量收益率分布的尖峰程度
            m4 = roll.apply(lambda x: ((x - x.mean()) ** 4).mean(), raw=False)
            with np.errstate(divide='ignore', invalid='ignore'):
                df[f'return_kurt_{period}d'] = m4 / (m2 ** 2 + 1e-10) - 3  # 超额峰度

        # 3) ATR 归一化版本：消除价格规模影响，跨周期可比
        for period in [14, 28]:
            if f'atr_{period}' in df.columns:
                df[f'atr_pct_{period}'] = df[f'atr_{period}'] / df['close']

        # 4) 布林带偏离度（绝对值）：非相对位置，是价格偏离均线的百分比
        for period in [20, 60]:
            if f'bb_upper_{period}' in df.columns:
                df[f'bb_deviation_{period}d'] = (df['close'] - (df[f'bb_upper_{period}'] + df[f'bb_lower_{period}']) / 2) / df['close']

        # 5) 成交量异常比率：近期成交量超过 2 倍均量的天数比例（10日内）
        vol_ma = df['volume'].rolling(window=20).mean()
        df['vol_anomaly_ratio_10d'] = df['volume'].rolling(10).apply(
            lambda x: (x > 2 * x.mean()).sum() / len(x) if len(x) > 0 and x.mean() > 0 else 0,
            raw=False
        )

        # 6) 金银比时序（独立拉取）：金银比对黄金有宏观领先性
        #    注意：白银数据依赖 yfinance，这里仅计算当日已知窗口
        #    若历史窗口不足，回退为常数（不崩溃）
        #    真实金银比由 _add_cross_asset_features 在实时数据获取时补充
        df['gold_silver_proxy'] = np.nan
        if 'close' in df.columns and len(df) > 20:
            # 用最近 20 日波动率比例做白银代理（波动率与价格成反向）
            vol_gold = df['close'].pct_change().rolling(20).std()
            vol_proxy = vol_gold * 0.5  # 白银波动率约为黄金 2x，取 0.5x 作为保守代理
            with np.errstate(divide='ignore', invalid='ignore'):
                df['gold_silver_proxy'] = np.where(vol_proxy > 0, 1 / (vol_proxy + 1e-10), np.nan)

        logger.info(f"[Features] Phase ③ 特征升级完成，新增约 20 个特征，当前总特征数约 {len([c for c in df.columns if c not in ['open','high','low','close','volume']])}")

        return df

    def add_macro_features(self, gold_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
        """融合宏观经济因子"""
        gold_df = gold_df.copy()
        
        if macro_df.empty:
            logger.warning("[Features] 宏观数据为空，跳过宏观特征融合")
            return gold_df
        
        # 对齐日期
        macro_df = macro_df.reindex(gold_df.index).ffill()
        
        # 添加宏观特征
        for col in macro_df.columns:
            gold_df[f'macro_{col}'] = macro_df[col]
        
        # === 衍生宏观特征 ===
        # 实际利率 = TIPS收益率 - CPI年率（简化）
        if 'macro_dgs10' in gold_df.columns and 'macro_cpi' in gold_df.columns:
            gold_df['real_rate_proxy'] = gold_df['macro_dgs10'] - gold_df['macro_cpi'] / 12
        
        # 美元强弱变化
        if 'macro_dxy_fred' in gold_df.columns:
            gold_df['dxy_change_1d'] = gold_df['macro_dxy_fred'].pct_change()
            gold_df['dxy_change_5d'] = gold_df['macro_dxy_fred'].pct_change(5)
        
        # 利率变化
        if 'macro_dgs10' in gold_df.columns:
            gold_df['yield_change_1d'] = gold_df['macro_dgs10'].diff()
            gold_df['yield_change_5d'] = gold_df['macro_dgs10'].diff(5)
        
        return gold_df
    
    def add_cross_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加交叉特征"""
        df = df.copy()
        
        # 黄金与相关品种的比率
        if 'close' in df.columns:
            # 金银比
            if 'macro_silver' in df.columns:
                df['gold_silver_ratio'] = df['close'] / df['macro_silver']
            
            # 金油比
            if 'macro_oil' in df.columns:
                df['gold_oil_ratio'] = df['close'] / df['macro_oil']
        
        # 技术指标交叉
        if 'rsi_14' in df.columns and 'rsi_28' in df.columns:
            df['rsi_divergence'] = df['rsi_14'] - df['rsi_28']
        
        if 'adx_14' in df.columns:
            df['trend_strength'] = df['adx_14']
        
        # MACD 交叉信号
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            df['macd_cross'] = df['macd'] - df['macd_signal']
        
        return df
    
    def _add_cross_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """拉取 VIX / 美元 / 美债 / 美股数据，生成与黄金的相关性特征"""
        try:
            import yfinance as yf
            from config import CROSS_ASSET_SYMBOLS
        except Exception:
            return df

        gold_close = df['close'] if 'close' in df.columns else None
        if gold_close is None or len(df) < 20:
            return df

        symbols = CROSS_ASSET_SYMBOLS
        for name, sym in symbols.items():
            try:
                end = df.index[-1]
                start = df.index[0]
                ex = yf.Ticker(sym).history(start=start, end=end, auto_adjust=True)
                if ex.empty:
                    continue
                ex.index = pd.to_datetime(ex.index).tz_localize(None)
                ex_c = ex['Close'].reindex(df.index).ffill().dropna()
                if len(ex_c) < 10:
                    continue

                # 收益率相关性
                gold_ret = gold_close.pct_change().dropna()
                ex_ret = ex_c.pct_change().dropna()
                min_len = min(len(gold_ret), len(ex_ret))
                if min_len >= 10:
                    corr = gold_ret.iloc[-min_len:].corr(ex_ret.iloc[-min_len:])
                    if not np.isnan(corr):
                        df.loc[ex_c.index, f'corr_{name}_return'] = corr

                # 价格相关性（20日滚动）
                df[f'corr_{name}_price'] = gold_close.rolling(20).corr(ex_c.reindex(gold_close.index).ffill())

                logger.info(f"[Cross-Asset] {name}({sym}) 相关性={corr:.3f}" if not np.isnan(corr) else f"[Cross-Asset] {name} skip")
            except Exception as e:
                logger.warning(f"[Cross-Asset] {name} failed: {e}")
        return df

    def filter_whitelist(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        只保留白名单特征（130 → 12 大幅降噪）
        外部调用：model.fit 之前用此方法过滤
        """
        from config import FEATURE_WHITELIST
        keep = [c for c in FEATURE_WHITELIST if c in df.columns]
        drop = [c for c in df.columns if c not in keep
                and c not in ('close', 'open', 'high', 'low', 'volume')]
        if drop:
            logger.info(f"[Whitelist] 过滤掉 {len(drop)} 列，保留 {len(keep)} 列")
            df = df.drop(columns=drop)
        return df

    def build_features(self, gold_df: pd.DataFrame, 
                       macro_df: Optional[pd.DataFrame] = None,
                       additional_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """完整特征构建流程"""
        
        logger.info(f"[Features] 原始数据: {len(gold_df)} 条")
        
        # 1. 技术指标
        df = self.add_technical_indicators(gold_df)
        logger.info(f"[Features] 技术指标完成")
        
        # 2. 宏观因子
        if macro_df is not None and macro_df:  # 防止 {} 进入 pd.concat
            # 支持 dict 输入（每个 key 对应一个 series）
            if isinstance(macro_df, dict):
                # 重新索引 macro 数据以匹配 gold_df 的日期，列名用 key
                macro_df = pd.concat(
                    {key: series.reindex(gold_df.index).ffill()
                     for key, series in macro_df.items()},
                    axis=1
                )

            df = self.add_macro_features(df, macro_df)
            logger.info(f"[Features] 宏观因子融合完成")
        
        # 3. 额外数据（白银、VIX等）
        if additional_data:
            for name, data in additional_data.items():
                if not data.empty:
                    merged = data.reindex(df.index).ffill()
                    for col in merged.columns:
                        df[f'ext_{name}_{col}'] = merged[col]

        # 3.5 跨资产特征（VIX / 美元 / 美债 / 美股，用 yfinance 实时拉）
        df = self._add_cross_asset_features(df)

        # 4. 交叉特征
        df = self.add_cross_features(df)
        
        # 5. 清理：跳过预热期，保留有足够数据的行
        # 先替换 inf
        df = df.replace([np.inf, -np.inf], np.nan)
        # 找到所有技术指标列（非 OHLCV）
        indicator_cols = [c for c in df.columns
                         if c not in ['open', 'high', 'low', 'close', 'volume']]
        # 找"最后一个指标列的第一个非NaN索引"——这才是真正的预热期结束点
        # 比如 MA200 需要200天预热，所有指标中最长的那个决定起点
        warmup_end = None
        for col in indicator_cols:
            first = df[col].first_valid_index()
            if first is not None:
                if warmup_end is None or first > warmup_end:
                    warmup_end = first
        if warmup_end is not None:
            df = df.loc[df.index >= warmup_end]
            logger.info(f"[Features] 预热期结束: {warmup_end.date()}，剩余 {len(df)} 行")
        # 对剩余 NaN 用前向填充，再均值填充
        df = df.ffill().fillna(df.mean(numeric_only=True))
        # 删除仍有 NaN 的行（极少）
        df = df.dropna(subset=['close'])
        # 全 NaN 的列删掉
        all_nan_cols = df.columns[df.isna().all()].tolist()
        if all_nan_cols:
            df = df.drop(columns=all_nan_cols)
            logger.warning(f"[Features] 删除全NaN列: {all_nan_cols}")
        
        self.feature_names = [col for col in df.columns 
                             if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        logger.info(f"[Features] 最终特征数: {len(self.feature_names)}")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        return self.feature_names
    
    def normalize_features(self, df: pd.DataFrame, 
                          scaler=None) -> Tuple[pd.DataFrame, object]:
        """标准化特征"""
        from sklearn.preprocessing import StandardScaler
        
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        if scaler is None:
            scaler = StandardScaler()
            df[feature_cols] = scaler.fit_transform(df[feature_cols])
        else:
            df[feature_cols] = scaler.transform(df[feature_cols])
        
        return df, scaler
    
    def create_sequences(self, df: pd.DataFrame, 
                        target_col: str = 'close',
                        seq_length: int = 60,
                        horizons: List[int] = [1, 3, 5]) -> Tuple[np.ndarray, np.ndarray, dict]:
        """创建时间序列样本"""
        
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        # 特征矩阵
        X = df[feature_cols].values
        
        # 目标：未来收益率
        future_returns = {}
        for h in horizons:
            future_returns[h] = df[target_col].shift(-h) / df[target_col] - 1
        
        sequences_X = []
        sequences_y = {}
        for h in horizons:
            sequences_y[h] = []
        
        for i in range(len(df) - max(horizons) - seq_length + 1):
            seq_x = X[i:i+seq_length]
            sequences_X.append(seq_x)
            for h in horizons:
                y_val = future_returns[h].iloc[i + seq_length - 1]
                if pd.notna(y_val):
                    sequences_y[h].append(y_val)
                else:
                    sequences_y[h].append(np.nan)
        
        X = np.array(sequences_X)
        
        # Padding 填充（如果有NaN）
        for h in horizons:
            y = np.array(sequences_y[h])
            # 用0填充NaN（极少数情况）
            y = np.nan_to_num(y, nan=0.0)
            sequences_y[h] = y
        
        info = {
            "feature_cols": feature_cols,
            "seq_length": seq_length,
            "horizons": horizons,
            "n_samples": len(X),
            "feature_dim": X.shape[2],
        }
        
        return X, sequences_y, info


if __name__ == "__main__":
    import yfinance as yf
    
    logging.basicConfig(level=logging.INFO)
    
    # 测试
    gold = yf.Ticker("GC=F").history(period="2y", auto_adjust=True)
    gold.index = pd.to_datetime(gold.index).tz_localize(None)
    
    engine = FeatureEngine()
    features = engine.add_technical_indicators(gold)
    features = features.dropna()
    
    print(f"\n✅ 特征工程测试成功")
    print(f"   样本数: {len(features)}")
    print(f"   特征数: {len([c for c in features.columns if c not in ['open','high','low','close','volume']])}")
    print(f"\n特征列表:")
    for col in features.columns[-20:]:
        print(f"   - {col}")
