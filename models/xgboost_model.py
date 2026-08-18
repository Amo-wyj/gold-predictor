"""
XGBoost 黄金预测模型
替代 sklearn GradientBoosting，用原生概率输出 + 严格 AUC 验证

为什么不用现有 GoldGBM：
1. GBM 用蒙特卡洛模拟估算概率（噪声大）；XGBoost 原生输出 probability
2. GBM 训练集包含验证集（过拟合）；XGBoost 用时序 5-fold 交叉验证
3. XGBoost 处理缺失值、稀疏特征更鲁棒（130 个技术指标中有 NaN）
4. 训练速度更快（~2-3x），精度更高（梯度二阶导优化）

集成定位：
  ensemble.py 中 MODEL_WEIGHTS["gbm"]: 0.45 → XGBoost 0.45（替换）
  新增 MODEL_WEIGHTS["xgb_ensemble"]: 0.20（与 ARIMA 并行）
  最终 ensemble 权重：ARIMA 0.25 + GBM(XGBoost) 0.45 + technical 0.20 + macro 0.10
"""

import os
import logging
import pickle
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GoldXGBoost:
    """
    黄金价格 XGBoost 预测器
    - 用 XGBClassifier（分类）而非回归+模拟，直接输出涨跌概率
    - 时序 5-fold 交叉验证，验证集永远在训练集之后（防数据泄露）
    - AUC > 0.65 才上线，否则回滚
    """

    def __init__(self, config: Optional[Dict] = None):
        from config import MODELS
        self.config = config or MODELS.get("xgboost", {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "gamma": 0.1,
            "horizons": [1, 3, 5],
            "n_folds": 5,
            "auc_threshold": 0.65,
        })
        self.n_estimators = self.config.get("n_estimators", 200)
        self.max_depth = self.config.get("max_depth", 4)
        self.learning_rate = self.config.get("learning_rate", 0.05)
        self.subsample = self.config.get("subsample", 0.8)
        self.colsample_bytree = self.config.get("colsample_bytree", 0.8)
        self.min_child_weight = self.config.get("min_child_weight", 5)
        self.gamma = self.config.get("gamma", 0.1)
        self.horizons = self.config.get("horizons", [1, 3, 5])
        self.n_folds = self.config.get("n_folds", 5)
        self.auc_threshold = self.config.get("auc_threshold", 0.65)

        self.models: Dict[int, object] = {}
        self.feature_cols: List[str] = []
        self.scaler_mean: np.ndarray = None
        self.scaler_std: np.ndarray = None
        self.model_name = "XGBoost"
        self.meta_info: Dict = {}
        self._is_trained = False
        self._passes_threshold = False

    def _get_feature_cols(self, features_df: pd.DataFrame) -> List[str]:
        """提取特征列（排除 OHLCV + target）"""
        exclude = {'open', 'high', 'low', 'close', 'volume',
                   'date', 'timestamp', 'index', 'target'}
        return [c for c in features_df.columns if c.lower() not in exclude]

    def _build_label(self, prices: pd.Series, horizon: int) -> pd.Series:
        """
        构建二分类标签：未来 horizon 天上涨=1，下跌/平=0
        用 prices（原始价格序列）而非 features_df，因为 features_df 已经 dropna 过了
        """
        future = prices.shift(-horizon)
        labels = (future > prices).astype(int)  # 上涨=1，下跌或平=0
        return labels

    def _time_series_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fold_size: int,
    ) -> List[tuple]:
        """
        时序交叉验证划分（不用 random shuffle，保持时间顺序）
        返回 [(train_idx, val_idx), ...] 列表，n_folds 组
        """
        n = len(X)
        splits = []
        for i in range(self.n_folds):
            # val 在最近端，train 在更早
            val_end = n - i * fold_size
            val_start = max(0, val_end - fold_size)
            train_end = val_start
            train_start = max(0, train_end - (n - fold_size * self.n_folds))
            if train_end - train_start < 50 or val_end - val_start < 10:
                break
            splits.append((np.arange(train_start, train_end), np.arange(val_start, val_end)))
        return splits

    def fit(self, features_df: pd.DataFrame, prices: Optional[pd.Series] = None) -> Dict:
        """
        训练 XGBoost 分类模型

        Args:
            features_df: 特征 DataFrame（来自 FeatureEngine.build_features，130+ 列）
            prices: 可选，原始价格序列（用于构建标签）；若 None 则用 features_df['close']
        """
        logger.info("[XGBoost] 开始训练...")
        start_time = datetime.now()

        try:
            import xgboost as xgb
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError("需要 xgboost: pip install xgboost")

        # 特征列：优先使用白名单（12核的特征），没有则用全部
        all_cols = self._get_feature_cols(features_df)
        if not all_cols:
            raise ValueError("特征列为空，请检查 FeatureEngine 输出")

        # 白名单过滤（130 → 12）
        try:
            from config import FEATURE_WHITELIST
            self.feature_cols = [c for c in FEATURE_WHITELIST if c in all_cols]
            if len(self.feature_cols) >= 5:
                logger.info(f"[XGBoost] 使用特征白名单 {len(self.feature_cols)} 个（降噪后）: {self.feature_cols}")
            else:
                self.feature_cols = all_cols
                logger.info(f"[XGBoost] 白名单未生效，回退全部 {len(all_cols)} 个特征")
        except Exception:
            self.feature_cols = all_cols
            logger.info(f"[XGBoost] 回退全部 {len(all_cols)} 个特征（config 未定义）")

        # 价格序列（用于构建标签）
        if prices is None:
            prices = features_df['close']

        # 标准化（用于训练，不改原始数据）
        X_raw = features_df[self.feature_cols].values
        self.scaler_mean = np.nanmean(X_raw, axis=0)
        self.scaler_std = np.nanstd(X_raw, axis=0)
        self.scaler_std[self.scaler_std == 0] = 1.0
        X_scaled = (X_raw - self.scaler_mean) / self.scaler_std
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        n = len(X_scaled)
        fold_size = max(n // (self.n_folds + 1), 30)
        cv_splits = self._time_series_cv(X_scaled, None, fold_size)

        logger.info(f"[XGBoost] 数据量={n}，fold_size={fold_size}，CV folds={len(cv_splits)}")

        cv_results = {}
        models_trained = {}

        for h in self.horizons:
            if h >= n - fold_size:
                logger.warning(f"[XGBoost] horizon={h}d 数据不足，跳过")
                continue

            # 构建标签（对齐到 features_df 的索引）
            labels = self._build_label(prices, h).reindex(features_df.index).fillna(0).values

            # 标签分布
            pos_rate = labels.mean()
            logger.info(f"[XGBoost] horizon={h}d | 上涨比例={pos_rate:.1%}")

            # 每折验证
            fold_aucs = []
            fold_models = []

            for fold_i, (train_idx, val_idx) in enumerate(cv_splits):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = labels[train_idx], labels[val_idx]

                if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                    continue

                # 计算 scale_pos_weight（处理类别不平衡）
                n_pos = y_train.sum()
                n_neg = len(y_train) - n_pos
                scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

                model = xgb.XGBClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
                    min_child_weight=self.min_child_weight,
                    gamma=self.gamma,
                    scale_pos_weight=scale_pos_weight,
                    use_label_encoder=False,
                    eval_metric='auc',
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                )

                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )

                val_proba = model.predict_proba(X_val)[:, 1]
                from sklearn.metrics import roc_auc_score
                try:
                    auc = roc_auc_score(y_val, val_proba)
                    fold_aucs.append(auc)
                except ValueError:
                    pass

                fold_models.append(model)

            if fold_aucs:
                mean_auc = float(np.mean(fold_aucs))
                std_auc = float(np.std(fold_aucs))
                logger.info(
                    f"[XGBoost] horizon={h}d | CV AUC={mean_auc:.3f}±{std_auc:.3f} "
                    f"(folds={len(fold_aucs)}, threshold={self.auc_threshold})"
                )
                cv_results[h] = {
                    "mean_auc": mean_auc,
                    "std_auc": std_auc,
                    "n_folds": len(fold_aucs),
                    "passes_threshold": mean_auc >= self.auc_threshold,
                }

                # 用全量数据训练最终模型（CV 评估后）
                final_model = xgb.XGBClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
                    min_child_weight=self.min_child_weight,
                    gamma=self.gamma,
                    scale_pos_weight=scale_pos_weight,
                    use_label_encoder=False,
                    eval_metric='auc',
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                )
                final_model.fit(X_scaled, labels, verbose=False)
                models_trained[h] = final_model
            else:
                logger.warning(f"[XGBoost] horizon={h}d CV 失败，跳过")

        self.models = models_trained
        self._is_trained = True

        # 判断是否全部通过阈值
        passed = [h for h, r in cv_results.items() if r["passes_threshold"]]
        self._passes_threshold = len(passed) >= len(self.horizons)

        duration = (datetime.now() - start_time).total_seconds()
        self.meta_info = {
            "n_features": len(self.feature_cols),
            "n_samples": n,
            "horizons": list(self.models.keys()),
            "cv_results": cv_results,
            "passes_threshold": self._passes_threshold,
            "auc_threshold": self.auc_threshold,
            "duration_sec": round(duration, 1),
        }

        status = "✅ PASS" if self._passes_threshold else "⚠️ BELOW THRESHOLD"
        logger.info(f"[XGBoost] 训练完成 {status}，耗时 {duration:.1f}s")
        return self.meta_info

    def predict_direction_probability(
        self,
        features_df: pd.DataFrame,
    ) -> Dict:
        """
        预测各 horizon 涨跌概率（XGBoost 原生概率，非模拟）

        Returns:
            {
                "horizon_1d": {
                    "model": "XGBoost",
                    "horizon": 1,
                    "probability_up": float,
                    "probability_down": float,
                    "predicted_price": float,
                    "signal": "STRONG_BUY"|"BUY"|"NEUTRAL"|"SELL"|"STRONG_SELL",
                    "auc_cv": float,
                },
                ...
            }
        """
        if not self._is_trained or not self.models:
            raise ValueError("模型未训练，请先调用 fit()")

        # 最后一行特征
        last_row = features_df[self.feature_cols].iloc[-1:].values
        last_price = features_df['close'].iloc[-1]

        # 标准化
        X_scaled = (last_row - self.scaler_mean) / self.scaler_std
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        results = {}

        for h, model in self.models.items():
            prob_up = float(model.predict_proba(X_scaled)[0, 1])
            prob_down = 1.0 - prob_up

            # 信号
            if prob_up >= 0.72:
                signal = "STRONG_BUY"
            elif prob_up >= 0.60:
                signal = "BUY"
            elif prob_down >= 0.72:
                signal = "STRONG_SELL"
            elif prob_down >= 0.60:
                signal = "SELL"
            else:
                signal = "NEUTRAL"

            # 价格预测（用概率加权期望收益率）
            # 简单估计：假设上涨平均 +1.5%/天，下跌平均 -1.5%/天
            mean_return = prob_up * 1.5 - prob_down * 1.5
            pred_price = last_price * (1 + mean_return * h / 100)

            results[f"horizon_{h}d"] = {
                "model": self.model_name,
                "horizon": h,
                "probability_up": prob_up,
                "probability_down": prob_down,
                "predicted_price": float(pred_price),
                "predicted_return_pct": float(mean_return * h),
                "signal": signal,
                "auc_cv": self.meta_info.get("cv_results", {}).get(h, {}).get("mean_auc", None),
            }

        return results

    def get_validation_report(self) -> Dict:
        """获取交叉验证报告（供 ensemble 决策是否接入）"""
        return {
            "is_trained": self._is_trained,
            "passes_threshold": self._passes_threshold,
            "cv_results": self.meta_info.get("cv_results", {}),
            "auc_threshold": self.auc_threshold,
            "horizons": list(self.models.keys()),
            "meta_info": self.meta_info,
        }

    def save(self, path: str):
        """保存模型到磁盘"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                "models": self.models,
                "feature_cols": self.feature_cols,
                "scaler_mean": self.scaler_mean,
                "scaler_std": self.scaler_std,
                "config": self.config,
                "meta_info": self.meta_info,
                "_is_trained": self._is_trained,
                "_passes_threshold": self._passes_threshold,
            }, f)
        logger.info(f"[XGBoost] 模型已保存: {path}")

    @classmethod
    def load(cls, path: str) -> "GoldXGBoost":
        """从磁盘加载模型"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        xgb_model = cls(config=data["config"])
        xgb_model.models = data["models"]
        xgb_model.feature_cols = data["feature_cols"]
        xgb_model.scaler_mean = data["scaler_mean"]
        xgb_model.scaler_std = data["scaler_std"]
        xgb_model.meta_info = data.get("meta_info", {})
        xgb_model._is_trained = data.get("_is_trained", True)
        xgb_model._passes_threshold = data.get("_passes_threshold", False)
        logger.info(f"[XGBoost] 模型已加载: {path}")
        return xgb_model


# ===== 独立测试 =====
if __name__ == "__main__":
    import yfinance as yf

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # 获取数据
    gold = yf.Ticker("GC=F").history(period="2y", auto_adjust=True)
    gold.index = pd.to_datetime(gold.index).tz_localize(None)
    logger.info(f"数据: {len(gold)} 天, 最新价格: ${gold['close'].iloc[-1]:.2f}")

    # 特征工程
    from features.feature_engineering import FeatureEngine
    fe = FeatureEngine()
    features = fe.build_features(gold, None)
    logger.info(f"特征: {len(features.columns)} 列, {len(features)} 行")

    # 训练
    xgb_model = GoldXGBoost()
    meta = xgb_model.fit(features, prices=gold['close'])

    # 验证报告
    report = xgb_model.get_validation_report()
    print("\n" + "=" * 50)
    print("📊 XGBoost 交叉验证报告")
    print("=" * 50)
    for h, r in report["cv_results"].items():
        mark = "✅" if r["passes_threshold"] else "⚠️"
        print(f"  horizon={h}d | AUC={r['mean_auc']:.3f}±{r['std_auc']:.3f} {mark}")
    print(f"  阈值: AUC > {report['auc_threshold']}")
    print(f"  整体通过: {'✅ YES' if report['passes_threshold'] else '⚠️ NO — 不接入 ensemble'}")

    # 预测
    if report["passes_threshold"]:
        results = xgb_model.predict_direction_probability(features)
        print("\n📈 XGBoost 预测结果")
        for k, v in results.items():
            print(f"  {k}: P(up)={v['probability_up']:.1%} signal={v['signal']}")
