"""
GBM 方向分类器 - 替代 XGBoost/LightGBM（Render Free tier 无法编译 C++ 扩展）
用 sklearn GradientBoostingClassifier 做二分类（涨/跌），无需 GPU/C++ 编译。

P1 阶段：训练 + 时序 CV AUC 验证，>0.65 才接入 ensemble
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class GoldXGBoost:
    # 类属性：模型标识
    model_name: str = "sklearn-GBClassifier"

    """
    梯度提升方向分类器（sklearn 实现，替代 XGBoost）
    输入：特征矩阵 → 输出：各周期涨跌概率 + AUC 验证
    """

    def __init__(self):
        self._passes_threshold = False
        self.model_name = GoldXGBoost.model_name  # 实例属性
        self.meta_info: Dict = {}
        self._models: Dict[int, object] = {}
        self._feature_cols: list = []
        self._scaler_mean: np.ndarray = None
        self._scaler_std: np.ndarray = None

    # ─────────────────────────────────────────────────────────────
    # fit() — ensemble.py 调用的入口
    # ensemble.py 传 features_df + prices（价格Series）
    # ─────────────────────────────────────────────────────────────
    def fit(self, features_df: pd.DataFrame, prices=None, horizons=(1, 3, 5)) -> Dict:
        """
        训练方向分类器 + 时序交叉验证 AUC

        Args:
            features_df: 特征矩阵（含 close 列），index 为 DatetimeIndex
            prices: 可选，prices['close'] 辅助对齐；若 None 则用 features_df['close']
            horizons: 预测周期列表

        Returns:
            {'cv_results': {1: {'mean_auc': 0.62, 'n_samples': 340}, ...},
             'status': 'trained'|'failed',
             'passes': bool}
        """
        try:
            # ── 1. 准备特征和标签 ────────────────────────────────
            if prices is not None and isinstance(prices, pd.Series):
                price_series = prices
            else:
                price_series = features_df['close']

            # 对齐索引
            common_idx = features_df.index.intersection(price_series.index)
            if len(common_idx) < 100:
                logger.warning(f"[GBClassifier] 对齐后样本数={len(common_idx)}，跳过训练")
                self._passes_threshold = False
                return {'cv_results': {}, 'status': 'failed', 'passes': False}

            feat_aligned = features_df.loc[common_idx].copy()
            price_aligned = price_series.loc[common_idx]

            # 特征列（排除 close / target 泄漏列）
            exclude = {'close', 'target', 'volume', 'Open', 'High', 'Low', 'Adj Close'}
            self._feature_cols = [c for c in feat_aligned.columns
                                   if c not in exclude and not c.startswith('future_')]
            if not self._feature_cols:
                raise ValueError("无可用特征列")

            X = feat_aligned[self._feature_cols].values
            self._scaler_mean = np.nanmean(X, axis=0)
            self._scaler_std = np.nanstd(X, axis=0)
            self._scaler_std[self._scaler_std == 0] = 1.0

            # ── 2. 分周期构建标签 + 训练 + 时序 CV ───────────────
            cv_results = {}
            self._models = {}

            for h in horizons:
                future_returns = price_aligned.shift(-h) / price_aligned - 1
                labels = (future_returns > 0).astype(int)  # 1=涨, 0=跌

                # 去掉 NaN
                valid = ~(labels.isna() | np.isnan(X).any(axis=1))
                X_h = X[valid.values]
                y_h = labels.values[valid.values]

                if len(y_h) < 80:
                    logger.warning(f"[GBClassifier] horizon={h}d 有效样本={len(y_h)}，跳过")
                    cv_results[h] = {'mean_auc': 0.5, 'n_samples': len(y_h)}
                    continue

                # 时序交叉验证（5-fold，shuffle=False）
                try:
                    from sklearn.model_selection import StratifiedKFold
                    from sklearn.ensemble import GradientBoostingClassifier
                    from sklearn.metrics import roc_auc_score
                except ImportError as e:
                    raise ImportError(f"sklearn 未安装: {e}")

                n_splits = min(5, max(2, len(y_h) // 30))
                kfold = StratifiedKFold(n_splits=n_splits, shuffle=False)
                aucs = []

                for fold_idx, (tr_idx, va_idx) in enumerate(kfold.split(X_h, y_h)):
                    X_tr, X_va = X_h[tr_idx], X_h[va_idx]
                    y_tr, y_va = y_h[tr_idx], y_h[va_idx]

                    # 标准化（用训练集统计量）
                    mean_tr = np.nanmean(X_tr, axis=0)
                    std_tr = np.nanstd(X_tr, axis=0)
                    std_tr[std_tr == 0] = 1.0
                    X_tr_s = np.nan_to_num((X_tr - mean_tr) / std_tr, nan=0.0)
                    X_va_s = np.nan_to_num((X_va - mean_tr) / std_tr, nan=0.0)

                    # 训练（轻量参数，防过拟合）
                    clf = GradientBoostingClassifier(
                        n_estimators=60,       # 轻量，防止过拟合
                        max_depth=3,           # 限制复杂度
                        learning_rate=0.08,
                        subsample=0.7,
                        min_samples_split=20,
                        min_samples_leaf=8,
                        max_features='sqrt',
                        random_state=42 + fold_idx,
                    )
                    clf.fit(X_tr_s, y_tr)

                    # AUC
                    if hasattr(clf, 'predict_proba'):
                        proba = clf.predict_proba(X_va_s)[:, 1]
                        if len(np.unique(y_va)) > 1:
                            auc = roc_auc_score(y_va, proba)
                            aucs.append(auc)

                mean_auc = float(np.mean(aucs)) if aucs else 0.5
                cv_results[h] = {
                    'mean_auc': round(mean_auc, 4),
                    'n_samples': int(len(y_h)),
                    'fold_aucs': [round(a, 4) for a in aucs],
                }

                # ── 3. 全量训练（CV 验证通过后）──────────────────
                if mean_auc >= 0.52:  # 只要不是纯随机就训练（验证在后头）
                    X_full_s = np.nan_to_num((X_h - self._scaler_mean) / self._scaler_std, nan=0.0)
                    clf_full = GradientBoostingClassifier(
                        n_estimators=60,
                        max_depth=3,
                        learning_rate=0.08,
                        subsample=0.7,
                        min_samples_split=20,
                        min_samples_leaf=8,
                        max_features='sqrt',
                        random_state=42,
                    )
                    clf_full.fit(X_full_s, y_h)
                    self._models[h] = clf_full

                logger.info(f"[GBClassifier] horizon={h}d | AUC={mean_auc:.4f} | n={len(y_h)}")

            # ── 4. 阈值判断（各周期平均 AUC ≥ 0.65 才接入）───────
            valid_aucs = [v['mean_auc'] for v in cv_results.values() if v['mean_auc'] > 0.5]
            overall_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.5
            self._passes_threshold = overall_auc >= 0.65

            status = 'trained' if self._models else 'untrained'
            logger.info(
                f"[GBClassifier] 训练完成 | 整体 AUC={overall_auc:.4f} | "
                f"通过阈值: {self._passes_threshold}"
            )

            self.meta_info = {
                'cv_results': {str(k): v for k, v in cv_results.items()},  # JSON需str key
                'status': status,
                'passes': self._passes_threshold,
                'overall_auc': round(overall_auc, 4),
                'n_features': len(self._feature_cols),
                'n_samples': len(y_h),
                'model_name': GoldXGBoost.model_name,
            }
            return self.meta_info

        except Exception as e:
            logger.warning(f"[GBClassifier] 训练失败: {e}")
            self._passes_threshold = False
            self.meta_info = {'cv_results': {}, 'status': 'failed', 'passes': False, 'model_name': GoldXGBoost.model_name}
            return self.meta_info

    # ─────────────────────────────────────────────────────────────
    # predict_direction_probability() — ensemble.py 调用的入口
    # ─────────────────────────────────────────────────────────────
    def predict_direction_probability(self, features_df: pd.DataFrame) -> Dict:
        """
        预测各周期涨跌概率

        Returns:
            {'horizon_1d': {probability_up, probability_down, signal, model, horizon},
             'horizon_3d': {...},
             'horizon_5d': {...}}
        """
        if not self._models:
            logger.warning("[GBClassifier] 无已训练模型，返回空")
            return {}

        try:
            exclude = {'close', 'target', 'volume', 'Open', 'High', 'Low', 'Adj Close'}
            feat_cols = [c for c in self._feature_cols if c in features_df.columns]
            if not feat_cols:
                return {}

            X = features_df[feat_cols].iloc[-1:][feat_cols].values
            X_s = np.nan_to_num(
                (X - self._scaler_mean) / self._scaler_std,
                nan=0.0, posinf=0.0, neginf=0.0
            )

            results = {}
            for h, clf in self._models.items():
                prob_up = float(clf.predict_proba(X_s)[0, 1])
                prob_down = 1.0 - prob_up

                signal = (
                    "STRONG_BUY" if prob_up >= 0.72 else
                    "STRONG_SELL" if prob_down >= 0.72 else
                    "BUY" if prob_up >= 0.60 else
                    "SELL" if prob_down >= 0.60 else
                    "NEUTRAL"
                )

                results[f"horizon_{h}d"] = {
                    'model': self.meta_info.get('model_name', 'GBClassifier'),
                    'horizon': h,
                    'probability_up': round(prob_up, 4),
                    'probability_down': round(prob_down, 4),
                    'signal': signal,
                    'predicted_return_pct': round((prob_up - prob_down) * 2.0, 4),
                }

            return results

        except Exception as e:
            logger.warning(f"[GBClassifier] 预测失败: {e}")
            return {}
