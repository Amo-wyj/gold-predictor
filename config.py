"""
黄金预测系统 - 配置文件
"""

# === 路径配置 ===
BASE_DIR = "/opt/gold-predictor"  # Render 部署路径
DATA_DIR = f"{BASE_DIR}/data"
MODELS_DIR = f"{BASE_DIR}/models"
FEATURES_DIR = f"{BASE_DIR}/features"

# === 数据源配置 ===
DATA_SOURCES = {
    # MT5 数据（Windows云服务器采集后推送）
    "mt5": {
        "endpoint": "http://YOUR_MT5_BRIDGE_IP:5000/api/gold",
        "api_key": "YOUR_MT5_BRIDGE_KEY",
    },
    
    # Yahoo Finance - 黄金/白银/其他商品
    "yahoo": {
        "gold": "GC=F",           # Gold Futures
        "silver": "SI=F",         # Silver Futures
        "oil": "CL=F",            # Crude Oil
        "dxy": "DX-Y.NYB",        # US Dollar Index
        "vix": "^VIX",            # VIX Index
    },
    
    # FRED API - 宏观经济数据（免费API Key）
    "fred": {
        "api_key": "b174af24d93f1ca58902ee7b4b4a1935",  # 免费申请: https://fred.stlouisfed.org/docs/api/api_key.html
        "series": {
            "dgs10": "DGS10",           # 10年期国债收益率
            "dgs5": "DGS5",             # 5年期国债收益率
            "tips10": "DFII10",         # 10年期TIPS（实际利率）
            "dxy_fred": "DTWEXBGS",     # 美元指数
            "cpi": "CPIAUCSL",         # CPI同比
            "pce": "PCEPI",            # PCE通胀
            "fed_rate": "DFF",         # 联邦基金利率
            "nonfarm": "PAYEMS",       # 非农就业
            "unemployment": "UNRATE",  # 失业率
        }
    }
}

# === 模型配置 ===
MODELS = {
    "arima": {
        "order": (5, 1, 2),         # (p, d, q) - 后续根据AIC优化
        "forecast_horizons": [1, 3, 5],  # 预测1天/3天/5天
        "confidence_level": 0.95,
    },
    "lstm": {
        "sequence_length": 60,      # 用60天历史数据预测
        "horizons": [1, 3, 5],
        "epochs": 100,
        "batch_size": 32,
        "hidden_units": 64,
        "dropout": 0.2,
        "early_stopping_patience": 10,
    }
}

# === 预警配置 ===
ALERTS = {
    # 价格突破预警
    "price_breakout": {
        "enabled": True,
        "thresholds": {
            "strong": 0.015,   # 1.5% 突破（强信号）
            "moderate": 0.008, # 0.8% 突破（中等信号）
            "weak": 0.003,     # 0.3% 突破（弱信号）
        },
        "lookback_days": 20,       # 用过去20天的高低点作为参考
    },
    
    # 模型信号预警
    "model_signal": {
        "enabled": True,
        "strong_threshold": 0.75,  # 概率 > 75% 触发
        "moderate_threshold": 0.65,
    },
    
    # 财经事件预警
    "macro_events": {
        "enabled": True,
        "event_days_before_alert": 3,
        "event_types": [
            "fomc_meeting",      # 美联储会议
            "nonfarm_payroll",   # 非农就业
            "cpi_release",       # CPI发布
            "pce_release",       # PCE发布
            "gdp_release",       # GDP发布
            "speech_powell",     # 鲍威尔讲话
        ]
    }
}

# === 推送配置 ===
DELIVERY = {
    "telegram": {
        "enabled": True,
        "bot_token": "YOUR_TELEGRAM_BOT_TOKEN",
        "chat_ids": ["YOUR_CHAT_ID"],  # 支持多个ID
    },
    
    # 每日定时推送时间（上海时区）
    "daily_digest": {
        "enabled": True,
        "times": ["08:00", "20:00"],  # 早晚报
        "include": ["prediction", "macro_calendar", "key_levels"],
    },
    
    # 模型更新频率
    "model_update": {
        "frequency": "6h",  # 每6小时重新训练/更新预测
        "retrain_threshold_days": 7,  # 每7天完全重训练
    }
}

# === 数据库配置 ===
DATABASE = {
    "url": "postgresql://user:password@host:5432/gold_predictor",
    "pool_size": 5,
}

# === 日志配置 ===
LOGGING = {
    "level": "INFO",
    "format": "%(asctime)s | %(levelname)-8s | %(message)s",
    "file": f"{BASE_DIR}/logs/gold_predictor.log",
}

# === 特征白名单（来自特征选择分析，130→12）===
# Top 12 核心特征（2026-08-18 分析结果）
# 覆盖：EMA均线族(6个) + ADX空头力度 + ATR波动率 + OBV量能 + SMA均线族(3个)
FEATURE_WHITELIST = [
    "ema_200",      # #1 长期趋势
    "ema_120",      # #2
    "ema_60",       # #3
    "ema_5",        # #4 短期动量
    "ema_20",       # #5
    "ema_10",       # #6
    "minus_di_28",  # #7 ADX空头力度
    "atr_28",       # #8 波动率
    "sma_5",        # #9
    "obv_sma_10",  # #10 量能
    "sma_120",     # #11
    "sma_20",       # #12
]

# 跨资产特征（额外增强，与 whitelist 配合使用）
CROSS_ASSET_SYMBOLS = {
    "vix": "^VIX",     # VIX 恐慌指数
    "dxy": "UUP",       # 美元指数 ETF
    "tlt": "TLT",       # 20年美债 ETF
    "spy": "SPY",       # 标普 500 ETF
}

