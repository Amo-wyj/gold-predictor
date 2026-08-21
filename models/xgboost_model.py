"""
XGBoost 模型 stub - Render Free tier 无法编译 xgboost，临时替换为占位模块。
实际预测降级为 ARIMA + Technical。

P1 阶段：跳过 ML 模型，等待升级到付费实例后恢复。
"""

import numpy as np
import logging
logger = logging.getLogger(__name__)


class GoldXGBoost:
    """XGBoost 预测器（stub 版本）"""
    model_name = "XGBoost-stub"

    def __init__(self):
        # xgboost 真实包在此处会被导入；stub 里什么都不做
        self._passes_threshold = False
        self.meta_info = {"cv_results": {}}

    def fit(self, features, prices=None, horizons=(1, 3, 5)):
        """Stub: 直接返回失败"""
        self._passes_threshold = False
        logger.warning("[XGBoost-stub] Render Free tier 无 xgboost，跳过 ML 训练")
        return {"status": "stub", "passes": False}

    def predict_direction_probability(self, features, horizon=1):
        """Stub: 返回 None"""
        return None
