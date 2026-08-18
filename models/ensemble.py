"""
集成预测模块
融合 ARIMA + GBM + 规则引擎，输出最终预测信号
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime

from models.arima_model import GoldARIMA
from models.gbm_model import GoldGBM
from models.xgboost_model import GoldXGBoost
from features.feature_engineering import FeatureEngine

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """集成预测器：融合多模型输出"""
    
    # 模型权重（可调）
    # P1 变更：XGBoost 替换 GBM(GradientBoosting)，AUC > 0.65 才接入
    # XGBoost 验证失败时自动降级回 GBM
    MODEL_WEIGHTS = {
        "arima": 0.25,
        "xgboost": 0.45,   # P1: XGBoost 原生概率替代 GBM 蒙特卡洛模拟
        "gbm": 0.00,        # GBM 保留学名（仅作降级备选，不参与默认 ensemble）
        "technical": 0.20,
        "macro": 0.10,
    }
    
    # 技术指标阈值
    TECHNICAL_THRESHOLDS = {
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "macd_bullish_cross": 0,
        "bb_lower_touch": 0.05,     # 收盘价接近布林下轨5%以内
        "bb_upper_touch": 0.95,     # 收盘价接近布林上轨95%以内
    }
    
    def __init__(self):
        self.arima = GoldARIMA()
        self.gbm = GoldGBM()
        self.xgboost_model: Optional[GoldXGBoost] = None  # P1: 延迟初始化
        self.xgboost_passes_threshold = False              # P1: AUC 验证标记
        self.feature_engine = FeatureEngine()
        self.current_features = None
        self.current_price = None
        self.current_macro = None
        self.last_prediction_time = None
    
    def update_data(self, gold_df: pd.DataFrame, 
                    macro_df: Optional[pd.DataFrame] = None) -> bool:
        """更新数据并重新计算特征"""
        try:
            # 构建特征
            self.current_macro = macro_df
            self.current_features = self.feature_engine.build_features(
                gold_df, macro_df
            )
            self.current_price = gold_df['close'].iloc[-1]
            self.last_prediction_time = datetime.now()
            
            logger.info(f"[Ensemble] 数据更新完成 | 最新价格: ${self.current_price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"[Ensemble] 数据更新失败: {e}")
            return False
    
    def analyze_technical(self) -> Dict:
        """技术分析"""
        if self.current_features is None:
            return {}
        
        df = self.current_features
        
        signals = {}
        
        # RSI 分析
        rsi_14 = df['rsi_14'].iloc[-1]
        if rsi_14 < 30:
            signals['rsi'] = {"value": rsi_14, "signal": "OVERSOLD", "weight": 0.3}
        elif rsi_14 > 70:
            signals['rsi'] = {"value": rsi_14, "signal": "OVERBOUGHT", "weight": 0.3}
        else:
            signals['rsi'] = {"value": rsi_14, "signal": "NEUTRAL", "weight": 0.1}
        
        # MACD 分析（value 归一化为占价百分比，钳制 [-10,10]，避免异常 tick 放大）
        macd = df['macd'].iloc[-1]
        macd_signal = df['macd_signal'].iloc[-1]
        macd_hist = df['macd_hist'].iloc[-1]
        close_now = df['close'].iloc[-1]
        macd_pct = (macd / close_now * 100.0) if close_now else float(macd)
        macd_display = float(np.clip(macd_pct, -10.0, 10.0))
        
        if macd > macd_signal and macd_hist > 0:
            signals['macd'] = {"value": macd_display, "signal": "BULLISH", "weight": 0.25}
        elif macd < macd_signal and macd_hist < 0:
            signals['macd'] = {"value": macd_display, "signal": "BEARISH", "weight": 0.25}
        else:
            signals['macd'] = {"value": macd_display, "signal": "NEUTRAL", "weight": 0.1}
        
        # 布林带分析
        bb_pos = df['bb_position_20'].iloc[-1]
        if bb_pos < 0.1:
            signals['bollinger'] = {"value": bb_pos, "signal": "AT_LOWER_BAND", "weight": 0.2}
        elif bb_pos > 0.9:
            signals['bollinger'] = {"value": bb_pos, "signal": "AT_UPPER_BAND", "weight": 0.2}
        else:
            signals['bollinger'] = {"value": bb_pos, "signal": "NEUTRAL", "weight": 0.1}
        
        # ADX 趋势强度
        adx = df['adx_14'].iloc[-1]
        if adx > 25:
            trend = "STRONG_TREND"
        elif adx > 15:
            trend = "WEAK_TREND"
        else:
            trend = "RANGING"
        signals['adx'] = {"value": adx, "signal": trend, "weight": 0.15}
        
        # KDJ 超买超卖
        k = df['k'].iloc[-1]
        d = df['d'].iloc[-1]
        j = df['j'].iloc[-1]
        
        if j < 20:
            signals['kdj'] = {"value": j, "signal": "OVERSOLD", "weight": 0.15}
        elif j > 80:
            signals['kdj'] = {"value": j, "signal": "OVERBOUGHT", "weight": 0.15}
        else:
            signals['kdj'] = {"value": j, "signal": "NEUTRAL", "weight": 0.1}
        
        # 计算技术综合信号
        tech_score = 0
        for name, data in signals.items():
            if "BULLISH" in data['signal'] or "OVERSOLD" in data['signal']:
                tech_score += data['weight']
            elif "BEARISH" in data['signal'] or "OVERBOUGHT" in data['signal']:
                tech_score -= data['weight']
        
        signals['technical_score'] = tech_score
        
        return signals
    
    def analyze_macro(self, macro_df: Optional[pd.DataFrame] = None) -> Dict:
        """宏观分析"""
        def _is_empty(m):
            if m is None: return True
            if isinstance(m, dict): return len(m) == 0
            if hasattr(m, 'empty'): return m.empty
            return True

        if _is_empty(macro_df):
            macro_df = self.current_macro

        if _is_empty(macro_df):
            return {"macro_signal": "NO_DATA"}

        # dict -> DataFrame
        if isinstance(macro_df, dict):
            import pandas as pd
            macro_df = pd.DataFrame(macro_df)
        
        signals = {}
        latest = macro_df.iloc[-1]

        # 美元指数影响（列名: dxy）
        if 'dxy' in macro_df.columns:
            dxy_change = macro_df['dxy'].pct_change().iloc[-1]
            signals['dxy'] = latest['dxy']
            signals['dxy_change'] = float(dxy_change) if pd.notna(dxy_change) else 0.0
            if dxy_change < -0.01:
                signals['dxy_signal'] = "WEAKER_USD_BULLISH"
            elif dxy_change > 0.01:
                signals['dxy_signal'] = "STRONGER_USD_BEARISH"
            else:
                signals['dxy_signal'] = "NEUTRAL"

        # 实际利率（列名: real_rate）
        if 'real_rate' in macro_df.columns:
            signals['real_rate'] = float(latest['real_rate'])
            if latest['real_rate'] < 0:
                signals['real_rate_signal'] = "NEGATIVE_REAL_RATE_BULLISH"
            else:
                signals['real_rate_signal'] = "POSITIVE_REAL_RATE_BEARISH"

        # 国债收益率变化（列名: dgs10）
        if 'dgs10' in macro_df.columns:
            signals['dgs10'] = float(latest['dgs10'])
            yield_change = macro_df['dgs10'].diff().iloc[-1]
            if yield_change < -0.05:
                signals['yield_signal'] = "YIELD_DROP_BULLISH"
            elif yield_change > 0.05:
                signals['yield_signal'] = "YIELD_RISE_BEARISH"
            else:
                signals['yield_signal'] = "NEUTRAL"

        # VIX（列名: vix）
        if 'vix' in macro_df.columns:
            signals['vix'] = float(latest['vix'])
            signals['vix_signal'] = "HIGH_RISK_BULLISH" if latest['vix'] > 25 else "NORMAL"

        # 综合宏观信号
        bullish_count = sum(1 for v in signals.values() if isinstance(v, str) and 'BULLISH' in v)
        bearish_count = sum(1 for v in signals.values() if isinstance(v, str) and 'BEARISH' in v)
        if bullish_count > bearish_count:
            signals['macro_signal'] = "MACRO_BULLISH"
        elif bearish_count > bullish_count:
            signals['macro_signal'] = "MACRO_BEARISH"
        else:
            signals['macro_signal'] = "MACRO_NEUTRAL"
        
        return signals
    
    def predict(self, prices: pd.Series) -> Dict:
        """综合预测（包含 P1 XGBoost 集成）"""
        logger.info("[Ensemble] 开始综合预测...")

        # 1. ARIMA 预测
        try:
            arima_results = {}
            arima = GoldARIMA()
            arima.fit(prices)
            for h in [1, 3, 5]:
                arima_results[f"h{h}"] = arima.predict_direction(h)
        except Exception as e:
            logger.warning(f"[Ensemble] ARIMA 预测失败: {e}")
            arima_results = {}

        # 2. XGBoost 预测（P1 新增：延迟训练，优先使用）
        xgb_results = {}
        if self.current_features is not None:
            try:
                self.xgboost_model = GoldXGBoost()
                meta = self.xgboost_model.fit(self.current_features, prices=prices)
                self.xgboost_passes_threshold = self.xgboost_model._passes_threshold
                if self.xgboost_passes_threshold:
                    xgb_results = self.xgboost_model.predict_direction_probability(
                        self.current_features
                    )
                    aucs = [round(r['mean_auc'], 3) for r in meta.get('cv_results', {}).values()]
                    logger.info(
                        f"[Ensemble] XGBoost AUC 验证通过 {aucs}，接入 ensemble"
                    )
                else:
                    aucs = {str(h) + 'd': round(r['mean_auc'], 3)
                            for h, r in meta.get('cv_results', {}).items()}
                    logger.warning(
                        f"[Ensemble] XGBoost AUC 未达阈值 0.65，"
                        f"不接入 ensemble，降级至 GBM: {aucs}"
                    )
            except ImportError as e:
                logger.warning(f"[Ensemble] XGBoost 未安装: {e}，降级至 GBM")
                self.xgboost_passes_threshold = False
            except Exception as e:
                logger.warning(f"[Ensemble] XGBoost 预测失败: {e}，降级至 GBM")
                self.xgboost_passes_threshold = False

        # 3. GBM 降级预测（XGBoost 验证失败时兜底）
        gbm_results = {}
        if not xgb_results and self.current_features is not None:
            try:
                gbm = GoldGBM()
                gbm.fit(self.current_features)
                gbm_results = gbm.predict_direction_probability(self.current_features)
                logger.info("[Ensemble] GBM 降级预测正常（XGBoost 不可用）")
            except Exception as e:
                logger.warning(f"[Ensemble] GBM 降级预测也失败: {e}")

        # 4. 技术分析
        tech_analysis = self.analyze_technical()

        # 5. 集成输出（XGBoost 优先，GBM 降级兜底）
        active_model = "xgb" if xgb_results else ("gbm" if gbm_results else None)
        final_prediction = self._ensemble_output(
            arima_results,
            xgb_results if xgb_results else gbm_results,
            tech_analysis,
            active_model=active_model,
        )

        return {
            "prediction": final_prediction,
            "arima": arima_results,
            "xgboost": xgb_results,
            "gbm": gbm_results,
            "technical_analysis": tech_analysis,
            "macro_analysis": self.analyze_macro(),
            "current_price": self.current_price,
            "timestamp": datetime.now().isoformat(),
            "_xgb_passes_threshold": self.xgboost_passes_threshold,  # P1 debug
        }

        # 3. 技术分析
        tech_analysis = self.analyze_technical()

        # 4. 集成输出
        final_prediction = self._ensemble_output(
            arima_results, gbm_results, tech_analysis
        )


    
    def _ensemble_output(
        self,
        arima: Dict,
        ml_model: Dict,  # XGBoost 或 GBM（统一接口）
        technical: Dict,
        active_model: str = "xgb",  # "xgb" | "gbm"
    ) -> Dict:
        """集成各模型输出（XGBoost 优先，GBM 降级兜底）"""

        horizons = [1, 3, 5]
        ensemble_results = {}
        ml_weight = self.MODEL_WEIGHTS["xgboost"] if active_model == "xgb" else self.MODEL_WEIGHTS["gbm"]

        for h in horizons:
            horizon_key = f"h{h}" if f"h{h}" in arima else f"horizon_{h}d"

            # 收集各模型预测概率
            prob_up_list = []
            weights = []

            # ARIMA
            if arima and horizon_key in arima:
                prob_up_list.append(arima[horizon_key]['probability_up'])
                weights.append(self.MODEL_WEIGHTS["arima"])

            # XGBoost / GBM（统一入口）
            if ml_model and f"horizon_{h}d" in ml_model:
                prob_up_list.append(ml_model[f"horizon_{h}d"]['probability_up'])
                weights.append(ml_weight)

            # 技术分析（规则引擎）
            if technical:
                tech_score = technical.get('technical_score', 0)
                tech_prob = (tech_score + 1) / 2  # [-1, 1] -> [0, 1]
                tech_prob = np.clip(tech_prob, 0.2, 0.8)
                prob_up_list.append(tech_prob)
                weights.append(self.MODEL_WEIGHTS["technical"])
            
            # 加权平均
            if prob_up_list:
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]
                
                ensemble_prob_up = sum(p * w for p, w in zip(prob_up_list, weights))
            else:
                ensemble_prob_up = 0.5
            
            ensemble_prob_down = 1 - ensemble_prob_up
            
            # 信号判定
            if ensemble_prob_up >= 0.72:
                signal = "STRONG_BUY"
            elif ensemble_prob_up >= 0.60:
                signal = "BUY"
            elif ensemble_prob_down >= 0.72:
                signal = "STRONG_SELL"
            elif ensemble_prob_down >= 0.60:
                signal = "SELL"
            else:
                signal = "NEUTRAL"
            
            # 置信度
            confidence = max(ensemble_prob_up, ensemble_prob_down)
            
            # ARIMA 价格预测
            arima_price = None
            if arima and horizon_key in arima:
                arima_price = arima[horizon_key].get('forecast_price')
            
            current = self.current_price or 2000.0
            predicted = arima_price or (current * (1 + (ensemble_prob_up - 0.5) * 0.02))
            
            ensemble_results[f"horizon_{h}d"] = {
                "probability_up": float(ensemble_prob_up),
                "probability_down": float(ensemble_prob_down),
                "confidence": float(confidence),
                "signal": signal,
                "confidence_label": self._confidence_label(confidence),
                "predicted_price": float(predicted),
                "price_change_pct": float((predicted / current - 1) * 100),
            }
        
        return ensemble_results
    
    def _confidence_label(self, confidence: float) -> str:
        """置信度等级"""
        if confidence >= 0.80:
            return "HIGH"
        elif confidence >= 0.65:
            return "MEDIUM"
        else:
            return "LOW"
    
    def generate_report(self, prediction_result: Dict) -> str:
        """生成人类可读的分析报告"""
        current_price = self.current_price or 0
        prediction = prediction_result["prediction"]
        tech = prediction_result["technical"]
        
        report = []
        report.append("=" * 55)
        report.append("🥇 黄金综合分析报告")
        report.append("=" * 55)
        report.append(f"\n⏰ 生成时间: {prediction_result['timestamp'][:19]}")
        report.append(f"💰 当前价格: ${current_price:.2f}")
        
        report.append("\n\n📊 各周期预测:")
        for h in [1, 3, 5]:
            key = f"horizon_{h}d"
            if key in prediction:
                p = prediction[key]
                horizon_label = "明天" if h == 1 else f"{h}天后"
                report.append(f"\n  【{horizon_label}】")
                report.append(f"    信号: {p['signal']}")
                report.append(f"    上涨概率: {p['probability_up']*100:.1f}%")
                report.append(f"    下跌概率: {p['probability_down']*100:.1f}%")
                report.append(f"    置信度: {p['confidence_label']} ({p['confidence']*100:.1f}%)")
        
        report.append("\n\n📈 技术指标:")
        if 'rsi' in tech:
            report.append(f"  RSI(14): {tech['rsi']['value']:.1f} - {tech['rsi']['signal']}")
        if 'macd' in tech:
            report.append(f"  MACD: {tech['macd']['signal']}")
        if 'bollinger' in tech:
            report.append(f"  布林带: {tech['bollinger']['signal']}")
        if 'adx' in tech:
            report.append(f"  ADX: {tech['adx']['value']:.1f} - {tech['adx']['signal']}")
        
        # 模型对比
        report.append("\n\n🤖 模型预测对比:")
        for h in [1, 3, 5]:
            key_h = f"h{h}" if f"h{h}" in prediction_result.get('arima', {}) else f"horizon_{h}d"
            arima_prob = prediction_result.get('arima', {}).get(key_h, {}).get('probability_up', 0)
            lstm_prob = prediction_result.get('lstm', {}).get(f"horizon_{h}d", {}).get('probability_up', 0)
            horizon_label = "明天" if h == 1 else f"{h}天后"
            report.append(f"  {horizon_label}: ARIMA {arima_prob*100:.0f}% | LSTM {lstm_prob*100:.0f}%")
        
        report.append("\n" + "=" * 55)
        
        return "\n".join(report)


if __name__ == "__main__":
    import yfinance as yf
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    
    # 获取数据
    gold = yf.Ticker("GC=F").history(period="2y", auto_adjust=True)
    gold.index = pd.to_datetime(gold.index).tz_localize(None)
    
    # 初始化预测器
    predictor = EnsemblePredictor()
    predictor.update_data(gold)
    
    # 预测
    results = predictor.predict(gold['close'])
    
    # 输出报告
    report = predictor.generate_report(results)
    print(report)
