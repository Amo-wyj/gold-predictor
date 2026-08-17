"""
Telegram Bot - 预警推送 + 交互查询
"""

import logging
import os
from typing import Dict, List, Optional
from datetime import datetime
import json

from config import DELIVERY, ALERTS

logger = logging.getLogger(__name__)


class GoldTelegramBot:
    """黄金预警 Telegram Bot"""
    
    def __init__(self, bot_token: str = None, chat_ids: List[str] = None):
        self.bot_token = bot_token or DELIVERY["telegram"]["bot_token"]
        self.chat_ids = chat_ids or DELIVERY["telegram"]["chat_ids"]
        self.api_base = f"https://api.telegram.org/bot{self.bot_token}"
    
    def _send_request(self, method: str, params: dict = None) -> Optional[dict]:
        """发送 Telegram API 请求"""
        import urllib.request
        import urllib.parse
        
        try:
            url = f"{self.api_base}/{method}"
            data = json.dumps(params).encode() if params else None
            
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"}
            )
            
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode())
                
            if not result.get("ok"):
                logger.error(f"[Telegram] API错误: {result}")
                return None
            
            return result.get("result")
            
        except Exception as e:
            logger.error(f"[Telegram] 请求失败: {e}")
            return None
    
    def send_message(self, text: str, chat_id: str = None,
                     parse_mode: str = "Markdown",
                     disable_notification: bool = False) -> bool:
        """发送消息"""
        if not self.bot_token or self.bot_token == "YOUR_TELEGRAM_BOT_TOKEN":
            logger.warning("[Telegram] Bot Token 未配置，跳过发送")
            return False
        
        target_ids = [chat_id] if chat_id else self.chat_ids
        success = True
        
        for cid in target_ids:
            result = self._send_request("sendMessage", {
                "chat_id": cid,
                "text": text,
                "parse_mode": parse_mode,
                "disable_notification": disable_notification,
            })
            
            if result is None:
                success = False
            else:
                logger.info(f"[Telegram] 消息已发送至 {cid[:10]}...")
        
        return success
    
    def send_photo(self, photo_url: str, caption: str = None, chat_id: str = None) -> bool:
        """发送图片"""
        if not self.bot_token or self.bot_token == "YOUR_TELEGRAM_BOT_TOKEN":
            return False
        
        target_ids = [chat_id] if chat_id else self.chat_ids
        
        for cid in target_ids:
            params = {"chat_id": cid, "photo": photo_url}
            if caption:
                params["caption"] = caption
            
            result = self._send_request("sendPhoto", params)
            if result is None:
                return False
        
        return True
    
    # ===== 格式化消息 =====
    
    def format_prediction_alert(self, prediction: Dict, current_price: float) -> str:
        """格式化模型信号预警"""
        lines = [
            "🔔 *模型信号预警*",
            "─" * 30,
            f"💰 当前价格: *${current_price:.2f}*",
            ""
        ]
        
        for h in [1, 3, 5]:
            key = f"horizon_{h}d"
            if key in prediction:
                p = prediction[key]
                horizon_label = "📍 明日" if h == 1 else f"📍 {h}日后"
                
                emoji = "🟢" if p['probability_up'] > 0.65 else ("🔴" if p['probability_down'] > 0.65 else "⚪")
                
                lines.append(f"{horizon_label} {emoji} {p['signal']}")
                lines.append(f"   上涨 {p['probability_up']*100:.1f}% | 下跌 {p['probability_down']*100:.1f}%")
                lines.append(f"   置信度: {p['confidence_label']}")
                lines.append("")
        
        lines.append("─" * 30)
        lines.append(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
        
        return "\n".join(lines)
    
    def format_price_alert(self, current_price: float, change_pct: float,
                          threshold: float, direction: str) -> str:
        """格式化价格突破预警"""
        direction_emoji = "📈" if direction == "up" else "📉"
        direction_text = "突破上涨" if direction == "up" else "突破下跌"
        
        lines = [
            f"{direction_emoji} *价格{direction_text}预警*",
            "─" * 30,
            f"💰 当前价格: *${current_price:.2f}*",
            f"📊 变动幅度: *{change_pct:+.2f}%*",
            f"⚡ 触发阈值: ±{threshold*100:.1f}%",
            "",
            "⚠️ 请关注市场动态",
            "─" * 30,
        ]
        
        return "\n".join(lines)
    
    def format_daily_digest(self, prediction: Dict, current_price: float,
                           tech: Dict, market_overview: Dict) -> str:
        """格式化每日摘要"""
        lines = [
            "🌅 *每日黄金分析摘要*",
            "─" * 30,
            f"📅 {datetime.now().strftime('%Y-%m-%d %A')}",
            f"💰 当前价格: *${current_price:.2f}*",
            ""
        ]
        
        # 核心预测
        lines.append("📊 *核心预测*")
        h1 = prediction.get("horizon_1d", {})
        if h1:
            signal_emoji = "🟢" if h1['probability_up'] > 0.6 else ("🔴" if h1['probability_down'] > 0.6 else "⚪")
            lines.append(f"  {signal_emoji} 明日信号: {h1.get('signal', 'N/A')}")
            lines.append(f"  📈 上涨概率: {h1['probability_up']*100:.1f}%")
            lines.append(f"  📉 下跌概率: {h1['probability_down']*100:.1f}%")
        lines.append("")
        
        # 技术指标摘要
        lines.append("📈 *技术指标*")
        if 'rsi' in tech:
            rsi = tech['rsi']
            rsi_status = "超卖" if "OVERSOLD" in rsi['signal'] else ("超买" if "OVERBOUGHT" in rsi['signal'] else "中性")
            lines.append(f"  RSI(14): {rsi['value']:.1f} - {rsi_status}")
        if 'macd' in tech:
            lines.append(f"  MACD: {tech['macd']['signal']}")
        if 'adx' in tech:
            lines.append(f"  ADX: {tech['adx']['value']:.1f} - {tech['adx']['signal']}")
        lines.append("")
        
        # 市场概览
        if market_overview:
            lines.append("🌐 *相关市场*")
            for name, data in market_overview.items():
                if data:
                    change = data.get('change_pct', 0)
                    emoji = "📈" if change > 0 else "📉"
                    lines.append(f"  {emoji} {name}: ${data.get('price', 0):.2f} ({change:+.2f}%)")
        
        lines.append("")
        lines.append("─" * 30)
        lines.append("🔔 详情请查看 Dashboard")
        
        return "\n".join(lines)
    
    def format_macro_event_alert(self, event_name: str, event_date: str,
                                 impact: str, preparation: str) -> str:
        """格式化财经事件预警"""
        lines = [
            f"🏛️ *财经事件预警*",
            "─" * 30,
            f"📌 事件: *{event_name}*",
            f"📅 日期: {event_date}",
            f"⚡ 影响级别: {impact}",
            "",
            f"💡 建议: {preparation}",
            "─" * 30,
        ]
        
        return "\n".join(lines)
    
    def format_error_alert(self, error_msg: str) -> str:
        """格式化错误通知"""
        lines = [
            "❌ *系统错误通知*",
            "─" * 30,
            f"错误信息: {error_msg}",
            f"⏰ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "─" * 30,
        ]
        
        return "\n".join(lines)
    
    # ===== 便捷发送函数 =====
    
    def send_prediction_alert(self, prediction: Dict, current_price: float) -> bool:
        """发送模型信号预警"""
        msg = self.format_prediction_alert(prediction, current_price)
        return self.send_message(msg)
    
    def send_price_alert(self, current_price: float, change_pct: float,
                        threshold: float, direction: str) -> bool:
        """发送价格突破预警"""
        msg = self.format_price_alert(current_price, change_pct, threshold, direction)
        return self.send_message(msg)
    
    def send_daily_digest(self, prediction: Dict, current_price: float,
                         tech: Dict, market_overview: Dict) -> bool:
        """发送每日摘要"""
        msg = self.format_daily_digest(prediction, current_price, tech, market_overview)
        return self.send_message(msg)
    
    def send_macro_alert(self, event_name: str, event_date: str,
                        impact: str, preparation: str) -> bool:
        """发送财经事件预警"""
        msg = self.format_macro_event_alert(event_name, event_date, impact, preparation)
        return self.send_message(msg)


# ===== 预警引擎 =====
class AlertEngine:
    """预警引擎"""
    
    def __init__(self, bot: GoldTelegramBot):
        self.bot = bot
        self.config = ALERTS
        self.last_price = None
        self.price_history = []
        self.alert_history = {}  # 防止重复预警
    
    def check_price_alert(self, current_price: float, timestamp: datetime = None) -> List[Dict]:
        """检查价格突破"""
        if current_price is None:
            return []
        
        alerts = []
        ts = timestamp or datetime.now()
        ts_key = ts.strftime("%Y%m%d%H")
        
        self.price_history.append(current_price)
        if len(self.price_history) > 100:
            self.price_history.pop(0)
        
        if len(self.price_history) < 20:
            return []
        
        # 计算近期高低点
        lookback = self.config["price_breakout"]["lookback_days"]
        recent_prices = self.price_history[-lookback:]
        
        high = max(recent_prices)
        low = min(recent_prices)
        mid = (high + low) / 2
        
        thresholds = self.config["price_breakout"]["thresholds"]
        
        for name, threshold in thresholds.items():
            # 上破
            if self.price_history[-1] > high * (1 + threshold):
                alert_key = f"up_{name}_{ts_key}"
                if alert_key not in self.alert_history.get(ts_key, []):
                    alerts.append({
                        "type": "price_breakout",
                        "direction": "up",
                        "threshold_name": name,
                        "price": current_price,
                        "change_pct": (current_price - high) / high * 100,
                        "threshold": threshold,
                    })
                    self.alert_history.setdefault(ts_key, []).append(alert_key)
            
            # 下破
            if self.price_history[-1] < low * (1 - threshold):
                alert_key = f"down_{name}_{ts_key}"
                if alert_key not in self.alert_history.get(ts_key, []):
                    alerts.append({
                        "type": "price_breakout",
                        "direction": "down",
                        "threshold_name": name,
                        "price": current_price,
                        "change_pct": (current_price - low) / low * 100,
                        "threshold": threshold,
                    })
                    self.alert_history.setdefault(ts_key, []).append(alert_key)
        
        return alerts
    
    def check_model_signal(self, prediction: Dict, current_price: float) -> List[Dict]:
        """检查模型信号"""
        alerts = []
        
        for h in [1, 3, 5]:
            key = f"horizon_{h}d"
            if key in prediction:
                p = prediction[key]
                prob_up = p['probability_up']
                prob_down = p['probability_down']
                confidence = p['confidence']
                
                # 强信号检查
                if prob_up > self.config["model_signal"]["strong_threshold"] and confidence > 0.70:
                    alerts.append({
                        "type": "model_signal",
                        "signal": "strong_buy",
                        "horizon": h,
                        "probability": prob_up,
                        "confidence": confidence,
                        "price": current_price,
                    })
                elif prob_down > self.config["model_signal"]["strong_threshold"] and confidence > 0.70:
                    alerts.append({
                        "type": "model_signal",
                        "signal": "strong_sell",
                        "horizon": h,
                        "probability": prob_down,
                        "confidence": confidence,
                        "price": current_price,
                    })
        
        return alerts
    
    def check_macro_events(self, days_ahead: int = 3) -> List[Dict]:
        """检查重大财经事件"""
        # 预定义财经日历（可扩展为 API 获取）
        events = {
            "fomc_meeting": {
                "name": "美联储FOMC会议",
                "impact": "🔴 高",
                "preparation": "会议前1-2天黄金可能震荡，会后若维持利率或降息，金价通常上涨"
            },
            "nonfarm_payroll": {
                "name": "非农就业报告",
                "impact": "🔴 高",
                "preparation": "数据好于预期→美元涨→金价跌；差于预期→反向"
            },
            "cpi_release": {
                "name": "美国CPI数据",
                "impact": "🟠 中高",
                "preparation": "通胀高于预期→实际利率下降→金价上涨"
            },
            "pce_release": {
                "name": "美国PCE物价指数",
                "impact": "🟠 中",
                "preparation": "美联储最关注的通胀指标，影响降息预期"
            },
            "gdp_release": {
                "name": "美国GDP数据",
                "impact": "🟡 中",
                "preparation": "GDP强→可能收紧政策→金价承压"
            },
            "speech_powell": {
                "name": "鲍威尔讲话",
                "impact": "🟠 中高",
                "preparation": "关注货币政策指引，鸽派发言→金价上涨"
            },
        }
        
        # 简化版：按固定周期模拟（实际应接财经日历API）
        alerts = []
        return alerts  # 暂时返回空，后续接入真实日历API
    
    def process_and_alert(self, current_price: float, prediction: Dict = None,
                         timestamp: datetime = None) -> List[Dict]:
        """处理所有预警检查并发送通知"""
        triggered = []
        
        # 价格突破检查
        if self.config["price_breakout"]["enabled"]:
            price_alerts = self.check_price_alert(current_price, timestamp)
            for alert in price_alerts:
                self.bot.send_price_alert(
                    alert['price'], alert['change_pct'],
                    alert['threshold'], alert['direction']
                )
                triggered.append(alert)
        
        # 模型信号检查
        if prediction and self.config["model_signal"]["enabled"]:
            model_alerts = self.check_model_signal(prediction, current_price)
            for alert in model_alerts:
                self.bot.send_prediction_alert(prediction, current_price)
                triggered.append(alert)
        
        # 财经事件检查
        if self.config["macro_events"]["enabled"]:
            macro_alerts = self.check_macro_events()
            for alert in macro_alerts:
                self.bot.send_macro_alert(
                    alert['event_name'], alert['event_date'],
                    alert['impact'], alert['preparation']
                )
                triggered.append(alert)
        
        return triggered


if __name__ == "__main__":
    # 测试
    logging.basicConfig(level=logging.INFO)
    
    bot = GoldTelegramBot()
    
    # 模拟发送测试消息
    test_msg = """
🔔 *黄金预测系统测试*

✅ 系统运行正常
⏰ 时间: {time}

💡 稍后开始正式推送分析
""".format(time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # 只有配置了真实Token才发送
    if bot.bot_token and bot.bot_token != "YOUR_TELEGRAM_BOT_TOKEN":
        bot.send_message(test_msg)
    else:
        print("⚠️ 请配置 TELEGRAM_BOT_TOKEN 后再测试发送功能")
