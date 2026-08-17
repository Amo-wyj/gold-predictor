# 🥇 黄金价格预测系统

> 专业级黄金涨跌预测工具，集成 ARIMA + LSTM 模型，提供概率预测与实时预警

## 功能特性

- **多模型融合**: ARIMA 线性预测 + LSTM 深度学习 + 技术分析规则引擎
- **多周期预测**: 日内（当天）、短期（3-5天）涨跌方向与概率
- **宏观因子融合**: 美元指数、国债收益率、通胀数据、VIX 等
- **实时预警**: 价格突破关键位、模型强信号、重大财经事件
- **每日推送**: 早晚两次分析摘要，直推 Telegram

## 系统架构

```
数据采集层 (Yahoo Finance + FRED + MT5)
       ↓
特征工程层 (技术指标 + 宏观因子 + 交叉特征)
       ↓
模型推理层 (ARIMA + LSTM + 集成输出)
       ↓
预警推送层 (Telegram Bot + 定时任务)
```

## 快速开始

### 1. 环境要求

- Python 3.9+
- 推荐 4GB+ RAM（LSTM 训练）
- 网络环境可访问 Yahoo Finance

### 2. 安装依赖

```bash
cd ~/qclaw/projects/gold-predictor
pip install -r requirements.txt
```

### 3. 配置 API Key

```bash
# Telegram Bot Token（必填）
export TELEGRAM_BOT_TOKEN="你的BotToken"

# FRED API Key（可选，免费申请）
export FRED_API_KEY="你的FREDKey"
```

### 4. 运行预测

```bash
# 单次预测
python main.py --mode predict

# 发送每日摘要
python main.py --mode digest
```

## 部署到云端（Render.com）

### 1. Fork 本项目到 GitHub

### 2. 在 Render.com 创建

- **New → Blueprint**
- 连接 GitHub 仓库
- 填入环境变量：
  - `TELEGRAM_BOT_TOKEN`
  - `FRED_API_KEY`
  - `DATABASE_URL`（PostgreSQL 自动创建）

### 3. 定时任务

在 Render 添加 Cron Job：
- `0 8,20 * * *` → 每日早晚推送
- `0 */6 * * *` → 每6小时更新模型

## API 接口

```
GET /api/predict     - 获取最新预测
GET /api/price       - 获取当前价格
GET /api/technical   - 获取技术指标
```

## 模型说明

### ARIMA 模型
- 擅长捕捉线性趋势
- 提供置信区间
- 参数自动优化

### LSTM 模型
- 擅长捕捉非线性模式
- 融合宏观因子
- 蒙特卡洛概率估算

### 集成策略
- ARIMA: 25%
- LSTM: 45%
- 技术分析: 20%
- 宏观分析: 10%

## 预警规则

| 类型 | 触发条件 | 推送方式 |
|------|---------|---------|
| 价格突破 | 突破20日高低点 0.3%-1.5% | 立即推送 |
| 模型强信号 | 概率 > 75%，置信度 > 70% | 立即推送 |
| 财经事件 | FOMC/非农/CPI前3天 | 提前预警 |
| 每日摘要 | 08:00 / 20:00 | 定时推送 |

## 文件结构

```
gold-predictor/
├── config.py              # 全局配置
├── main.py                # 主入口
├── requirements.txt       # 依赖列表
├── README.md
├── data/                  # 数据采集
│   ├── yahoo_collector.py
│   └── fred_collector.py
├── features/              # 特征工程
│   └── feature_engineering.py
├── models/                # 预测模型
│   ├── arima_model.py
│   ├── lstm_model.py
│   └── ensemble.py
├── bot/                   # 预警推送
│   └── telegram_bot.py
├── dashboard/             # Web 界面（待开发）
└── scripts/               # 工具脚本
```

## 注意事项

⚠️ **风险提示**: 本系统仅供辅助决策，不构成投资建议。预测有风险，投资需谨慎。

⚠️ **数据延迟**: Yahoo Finance 数据有15分钟延迟，实时数据需 MT5 或专业数据源。

⚠️ **模型局限**: 市场受突发事件影响时，模型预测可能失效。
