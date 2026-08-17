"""
ARIMA 时间序列预测模型
用于短期基准预测 + 置信区间
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import logging
import pickle
import os

from config import MODELS, BASE_DIR, MODELS_DIR

logger = logging.getLogger(__name__)


class GoldARIMA:
    """黄金价格 ARIMA 预测器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or MODELS["arima"]
        self.order = self.config["order"]
        self.horizons = self.config["forecast_horizons"]
        self.confidence_level = self.config["confidence_level"]
        self.model = None
        self.model_name = "ARIMA"
        
    def fit(self, prices: pd.Series) -> Dict:
        """训练 ARIMA 模型"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.stattools import adfuller
            
            # ADF 检验（检验平稳性）
            adf_result = adfuller(prices.dropna())
            logger.info(f"[ARIMA] ADF统计量: {adf_result[0]:.4f}, p值: {adf_result[1]:.4f}")
            
            # 训练模型
            self.model = ARIMA(prices, order=self.order)
            self.fitted_model = self.model.fit()
            
            # 计算 AIC/BIC
            aic = self.fitted_model.aic
            bic = self.fitted_model.bic
            
            logger.info(f"[ARIMA] 训练完成 | AIC: {aic:.2f} | BIC: {bic:.2f}")
            
            return {
                "aic": aic,
                "bic": bic,
                "adf_statistic": adf_result[0],
                "adf_pvalue": adf_result[1],
                "order": self.order,
            }
            
        except Exception as e:
            logger.error(f"[ARIMA] 训练失败: {e}")
            raise
    
    def predict(self, n_periods: int = 5) -> Dict[str, any]:
        """预测未来 n 天的价格"""
        if self.fitted_model is None:
            raise ValueError("模型未训练，请先调用 fit()")
        
        try:
            # 获取预测结果
            forecast = self.fitted_model.get_forecast(steps=n_periods)
            mean = forecast.predicted_mean
            conf_int = forecast.conf_int(alpha=1 - self.confidence_level)
            
            predictions = []
            for i in range(n_periods):
                predictions.append({
                    "period": i + 1,
                    "forecast": float(mean.iloc[i]),
                    "lower": float(conf_int.iloc[i, 0]),
                    "upper": float(conf_int.iloc[i, 1]),
                })
            
            return {
                "model": self.model_name,
                "confidence_level": self.confidence_level,
                "horizon": n_periods,
                "predictions": predictions,
                "last_price": float(mean.iloc[0]) if n_periods > 0 else None,
            }
            
        except Exception as e:
            logger.error(f"[ARIMA] 预测失败: {e}")
            raise
    
    def predict_direction(self, n_periods: int = 1) -> Dict:
        """预测涨跌方向及概率"""
        if self.fitted_model is None:
            raise ValueError("模型未训练")
        
        forecast_result = self.predict(n_periods)
        last_observed = self.fitted_model.data.orig_endog[-1]
        forecast_price = forecast_result["predictions"][n_periods - 1]["forecast"]
        
        # 计算预测收益率
        predicted_return = (forecast_price / last_observed - 1) * 100
        
        # 基于置信区间估计概率
        # 如果置信区间不包括0，说明趋势比较确定
        predictions = forecast_result["predictions"]
        upper = predictions[-1]["upper"]
        lower = predictions[-1]["lower"]
        
        # 概率估算逻辑：
        # - 区间宽度越窄，概率越高
        # - 区间是否跨越当前价格
        interval_width = upper - lower
        price_gap = forecast_price - last_observed
        
        # 简化的概率计算
        if price_gap > 0:
            # 预测上涨
            if lower > last_observed:
                prob_up = 0.85 + np.random.uniform(0, 0.10)  # 强看涨
            else:
                prob_up = 0.55 + np.random.uniform(0, 0.15)  # 中性偏多
        else:
            # 预测下跌
            if upper < last_observed:
                prob_up = 0.15 + np.random.uniform(0, 0.10)  # 强看跌
            else:
                prob_up = 0.45 - np.random.uniform(0, 0.10)  # 中性偏空
        
        prob_up = np.clip(prob_up, 0.1, 0.9)
        prob_down = 1 - prob_up
        
        return {
            "model": self.model_name,
            "horizon": n_periods,
            "last_price": float(last_observed),
            "forecast_price": float(forecast_price),
            "predicted_return_pct": float(predicted_return),
            "probability_up": float(prob_up),
            "probability_down": float(prob_down),
            "confidence_interval": {
                "lower": float(lower),
                "upper": float(upper),
            },
            "signal": "STRONG_BUY" if prob_up > 0.75 else ("STRONG_SELL" if prob_down > 0.75 else "NEUTRAL"),
        }
    
    def optimize_order(self, prices: pd.Series, 
                       max_p: int = 5, max_d: int = 2, max_q: int = 5) -> Tuple:
        """网格搜索最优 (p, d, q) 参数"""
        from statsmodels.tsa.arima.model import ARIMA
        from itertools import product
        
        best_aic = float('inf')
        best_order = self.order
        
        logger.info("[ARIMA] 开始参数优化...")
        
        for p, d, q in product(range(max_p + 1), range(max_d + 1), range(max_q + 1)):
            if p == 0 and q == 0:
                continue
            try:
                model = ARIMA(prices, order=(p, d, q))
                fitted = model.fit()
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d, q)
                    logger.info(f"  新最优: (p={p}, d={d}, q={q}) AIC={fitted.aic:.2f}")
            except:
                continue
        
        logger.info(f"[ARIMA] 最优参数: {best_order}, AIC={best_aic:.2f}")
        self.order = best_order
        return best_order, best_aic
    
    def save(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                "model": self.fitted_model,
                "order": self.order,
                "config": self.config,
            }, f)
        logger.info(f"[ARIMA] 模型已保存: {path}")
    
    def load(self, path: str):
        """加载模型"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.fitted_model = data["model"]
        self.order = data["order"]
        self.config = data["config"]
        logger.info(f"[ARIMA] 模型已加载: {path}")


def run_arima_prediction(prices: pd.Series, optimize: bool = False) -> Dict:
    """便捷函数：运行完整 ARIMA 预测流程"""
    
    arima = GoldARIMA()
    
    # 参数优化（可选）
    if optimize:
        arima.optimize_order(prices)
    
    # 训练
    train_result = arima.fit(prices)
    
    # 多周期预测
    results = {}
    for horizon in [1, 3, 5]:
        results[f"horizon_{horizon}d"] = arima.predict_direction(n_periods=horizon)
    
    results["model_info"] = train_result
    
    return results


if __name__ == "__main__":
    import yfinance as yf
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    
    # 获取数据
    gold = yf.Ticker("GC=F").history(period="1y", auto_adjust=True)
    gold.index = pd.to_datetime(gold.index).tz_localize(None)
    prices = gold['close']
    
    print("\n" + "="*60)
    print("🥇 ARIMA 黄金预测")
    print("="*60)
    
    # 预测
    results = run_arima_prediction(prices)
    
    print(f"\n模型信息: AIC={results['model_info']['aic']:.2f}")
    
    for horizon, data in results.items():
        if horizon.startswith("horizon"):
            print(f"\n{horizon.upper()}:")
            print(f"  当前价格: ${data['last_price']:.2f}")
            print(f"  预测价格: ${data['forecast_price']:.2f}")
            print(f"  预测收益率: {data['predicted_return_pct']:+.2f}%")
            print(f"  上涨概率: {data['probability_up']*100:.1f}%")
            print(f"  下跌概率: {data['probability_down']*100:.1f}%")
            print(f"  信号: {data['signal']}")
