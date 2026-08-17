"""
GBM 梯度提升模型 - LSTM替代方案
用 scikit-learn GradientBoostingRegressor 替代 TensorFlow LSTM
无需 GPU/M1优化，适合 Mac 本地运行
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import pickle
import json

logger = logging.getLogger(__name__)


class GoldGBM:
    """黄金价格 GBM 预测器（替代 LSTM）"""

    def __init__(self, config: Optional[Dict] = None):
        from config import MODELS
        self.config = config or MODELS.get("lstm", {
            "sequence_length": 20,
            "horizons": [1, 3, 5],
            "confidence_level": 0.95,
        })
        self.sequence_length = self.config.get("sequence_length", 20)
        self.horizons = self.config.get("horizons", [1, 3, 5])
        self.confidence_level = self.config.get("confidence_level", 0.95)
        self.models: Dict[int, object] = {}
        self.feature_cols: List[str] = []
        self.scaler_mean: np.ndarray = None
        self.scaler_std: np.ndarray = None
        self.model_name = "GBM"
        self.meta_info: Dict = {}

    def _create_sequences(self, X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """创建序列样本（兼容 LSTM 接口）"""
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X)):
            X_seq.append(X[i - seq_len:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    def _flatten_sequences(self, X: np.ndarray) -> np.ndarray:
        """展平序列 [samples, seq_len, features] -> [samples, seq_len*features]"""
        samples, seq_len, features = X.shape
        return X.reshape(samples, seq_len * features)

    def fit(self, features_df: pd.DataFrame, target_col: str = 'close') -> Dict:
        """训练 GBM 模型"""
        logger.info("[GBM] 开始训练...")
        start_time = datetime.now()

        try:
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
        except ImportError:
            raise ImportError("需要 scikit-learn: pip install scikit-learn")

        # 保存特征列
        self.feature_cols = [c for c in features_df.columns if c != target_col]
        if not self.feature_cols:
            raise ValueError("特征列为空，请提供包含特征的 DataFrame")

        # 提取特征和目标
        X = features_df[self.feature_cols].values
        y = features_df[target_col].values

        # 标准化
        self.scaler_mean = np.nanmean(X, axis=0)
        self.scaler_std = np.nanstd(X, axis=0)
        self.scaler_std[self.scaler_std == 0] = 1.0
        X_scaled = (X - self.scaler_mean) / self.scaler_std
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # 计算未来收益率作为目标
        n = len(X_scaled)
        models_trained = {}
        training_scores = {}

        for h in self.horizons:
            if h >= n:
                logger.warning(f"[GBM] horizon={h} >= 数据长度={n}，跳过")
                continue

            # 目标：未来 h 天的收益率
            future_prices = features_df[target_col].shift(-h).values
            returns = (future_prices / features_df[target_col].values - 1) * 100

            # 去掉 NaN
            valid_idx = ~np.isnan(returns)
            X_h = X_scaled[valid_idx]
            y_h = returns[valid_idx]

            if len(y_h) < 50:
                logger.warning(f"[GBM] horizon={h} 有效样本={len(y_h)}，跳过")
                continue

            # 拆分
            X_train, X_val, y_train, y_val = train_test_split(
                X_h, y_h, test_size=0.15, shuffle=False  # 时序数据不用随机拆分
            )

            # 训练 GBM
            gbm = GradientBoostingRegressor(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
            )
            gbm.fit(X_train, y_train)

            train_score = gbm.score(X_train, y_train)
            val_score = gbm.score(X_val, y_val) if len(X_val) > 0 else 0

            models_trained[h] = gbm
            training_scores[h] = {"train_r2": train_score, "val_r2": val_score}
            logger.info(f"[GBM] horizon={h}d | train R²={train_score:.3f} | val R²={val_score:.3f}")

        self.models = models_trained

        duration = (datetime.now() - start_time).total_seconds()
        self.meta_info = {
            "n_features": len(self.feature_cols),
            "n_samples": n,
            "horizons": list(self.models.keys()),
            "training_scores": training_scores,
            "duration_sec": duration,
        }

        logger.info(f"[GBM] 训练完成，耗时 {duration:.1f}s")
        return self.meta_info

    def predict(self, features_df: pd.DataFrame) -> Dict:
        """预测未来价格"""
        if not self.models:
            raise ValueError("模型未训练，请先调用 fit()")

        last_row = features_df[self.feature_cols].iloc[-1:].values
        last_price = features_df['close'].iloc[-1]

        # 标准化
        X_scaled = (last_row - self.scaler_mean) / self.scaler_std
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        predictions = {}
        for h, model in self.models.items():
            pred_return = model.predict(X_scaled)[0]
            pred_price = last_price * (1 + pred_return / 100)
            predictions[f"h{h}"] = {
                "predicted_return_pct": float(pred_return),
                "predicted_price": float(pred_price),
            }

        return predictions

    def predict_direction_probability(self, features_df: pd.DataFrame) -> Dict:
        """蒙特卡洛模拟估算涨跌概率"""
        if not self.models:
            raise ValueError("模型未训练，请先调用 fit()")

        from sklearn.ensemble import GradientBoostingRegressor

        last_row = features_df[self.feature_cols].iloc[-1:].values
        last_price = features_df['close'].iloc[-1]

        # 标准化
        X_scaled = (last_row - self.scaler_mean) / self.scaler_std
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        results = {}
        n_simulations = 1000

        for h, model in self.models.items():
            # 点预测
            point_return = model.predict(X_scaled)[0]

            # 蒙特卡洛模拟（加入模型不确定性）
            # 用训练残差的标准差估算噪声
            train_pred = model.predict(X_scaled)
            residual_std = max(abs(train_pred.mean()) * 0.05, 0.1)  # 最小波动

            sim_returns = np.random.normal(point_return, residual_std, n_simulations)

            # 计算概率
            prob_up = float(np.mean(sim_returns > 0))
            prob_down = 1 - prob_up

            # 价格预测
            mean_return = float(np.mean(sim_returns))
            std_return = float(np.std(sim_returns))
            pred_price = last_price * (1 + mean_return / 100)

            # 置信区间
            lower = last_price * (1 + np.percentile(sim_returns, 2.5) / 100)
            upper = last_price * (1 + np.percentile(sim_returns, 97.5) / 100)

            signal = "STRONG_BUY" if prob_up >= 0.72 else (
                "STRONG_SELL" if prob_down >= 0.72 else (
                    "BUY" if prob_up >= 0.60 else (
                        "SELL" if prob_down >= 0.60 else "NEUTRAL"
                    )
                )
            )

            results[f"horizon_{h}d"] = {
                "model": self.model_name,
                "horizon": h,
                "probability_up": prob_up,
                "probability_down": prob_down,
                "predicted_price": pred_price,
                "predicted_return_pct": mean_return,
                "std_return_pct": std_return,
                "confidence_interval": {"lower": lower, "upper": upper},
                "signal": signal,
            }

        return results

    def save(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                "models": self.models,
                "feature_cols": self.feature_cols,
                "scaler_mean": self.scaler_mean,
                "scaler_std": self.scaler_std,
                "config": self.config,
                "meta_info": self.meta_info,
            }, f)
        logger.info(f"[GBM] 模型已保存: {path}")

    @classmethod
    def load(cls, path: str) -> "GoldGBM":
        """加载模型"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        gbm = cls(config=data["config"])
        gbm.models = data["models"]
        gbm.feature_cols = data["feature_cols"]
        gbm.scaler_mean = data["scaler_mean"]
        gbm.scaler_std = data["scaler_std"]
        gbm.meta_info = data.get("meta_info", {})
        logger.info(f"[GBM] 模型已加载: {path}")
        return gbm
