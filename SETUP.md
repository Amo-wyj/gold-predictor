# 🥇 黄金预测系统 - 部署指南

## 快速预览

系统已完整构建，包含：

```
gold-predictor/
├── config.py              # 全局配置
├── main.py                # 主入口
├── requirements.txt       # Python依赖
├── README.md              # 项目说明
├── render.yaml            # 云端部署配置
│
├── data/
│   ├── yahoo_collector.py # Yahoo Finance 数据采集
│   └── fred_collector.py  # FRED 宏观经济数据采集
│
├── features/
│   └── feature_engineering.py  # 特征工程（60+技术指标）
│
├── models/
│   ├── arima_model.py     # ARIMA 时间序列预测
│   ├── lstm_model.py      # LSTM 深度学习预测
│   └── ensemble.py        # 多模型集成
│
├── bot/
│   └── telegram_bot.py    # Telegram 预警推送 + Alert Engine
│
├── dashboard/
│   ├── dashboard.py       # Web Dashboard API
│   └── templates/
│       └── dashboard.html # 可视化界面
│
└── scripts/
    ├── mt5_bridge.py      # MT5 数据桥接（Windows端）
    └── deploy_render.sh   # 部署脚本
```

---

## 第一步：申请 API Key（必填）

### 1. Telegram Bot Token

1. 打开 Telegram，搜索 **@BotFather**
2. 发送 `/newbot`
3. 按提示设置机器人名称
4. 获得 Token，格式如：`123456789:ABCdefGHIjklMNOpqrsTUVwxyz`

> 💡 记住这个 Token，后续配置要用

### 2. FRED API Key（推荐，免费）

1. 打开 https://fred.stlouisfed.org/docs/api/api_key.html
2. 用邮箱注册并申请 API Key
3. 几秒钟后收到 Key，免费、无额度限制

---

## 第二步：本地测试

### 1. 安装依赖

```bash
cd ~/qclaw/projects/gold-predictor

# 如果你用的是 QClaw 内置 Python：
/Users/wangyingjie/Library/Application\ Support/QClaw/python/bin/python3 -m pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
export TELEGRAM_BOT_TOKEN="你的BotToken"
export FRED_API_KEY="你的FREDKey"
```

### 3. 运行测试

```bash
cd ~/qclaw/projects/gold-predictor
python main.py --mode predict
```

成功的话，你会看到类似输出：

```
====================================================
🥇 黄金综合分析报告
====================================================

⏰ 生成时间: 2025-01-15 20:30:00
💰 当前价格: $2045.30

📊 各周期预测:

  【明天】
    信号: 🟢 STRONG_BUY
    上涨概率: 72.5%
    下跌概率: 27.5%
    置信度: HIGH (72.5%)

  【3天后】
    信号: 🟡 BUY
    上涨概率: 65.2%
    下跌概率: 34.8%
    置信度: MEDIUM (65.2%)

  【5天后】
    信号: ⚪ NEUTRAL
    上涨概率: 52.1%
    下跌概率: 47.9%
    置信度: LOW (52.1%)
```

---

## 第三步：部署到云端

### 方案A：Render.com（推荐，完全免费）

**1. Fork 项目到 GitHub**
```bash
# 如果你会 Git，直接推送
cd ~/qclaw/projects/gold-predictor
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/你的用户名/gold-predictor.git
git push -u origin main
```

**2. 在 Render.com 创建服务**

1. 访问 https://render.com → Sign Up（用 GitHub 登录）
2. **New → Blueprint**
3. 连接你的 GitHub 仓库
4. Render 会自动读取 `render.yaml` 创建所有服务
5. 填入环境变量：
   - `TELEGRAM_BOT_TOKEN`
   - `FRED_API_KEY`
6. 点击 **Apply Blueprint**

**3. 完成！**

- Web Dashboard: `https://gold-predictor.onrender.com`
- Telegram 每日推送: 自动在 08:00 和 20:00 发送
- 每6小时自动更新模型预测

### 方案B：其他云平台

如果你有其他云服务器（阿里云/腾讯云/Vultr），基本步骤：

```bash
# 1. 上传代码
scp -r ~/qclaw/projects/gold-predictor user@your-server:/opt/

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
export TELEGRAM_BOT_TOKEN="xxx"
export FRED_API_KEY="xxx"

# 4. 设置定时任务（crontab）
crontab -e
# 添加：
# 0 8,20 * * * /opt/gold-predictor/venv/bin/python /opt/gold-predictor/main.py --mode digest
# 0 */6 * * * /opt/gold-predictor/venv/bin/python /opt/gold-predictor/main.py --mode predict

# 5. 启动 Dashboard
nohup python dashboard/dashboard.py &
```

---

## 第四步：配置 MT5 数据源（可选）

目前系统使用 Yahoo Finance 数据（免费，有延迟）。如果你需要**实时数据**：

### 1. 租一台 Windows 云服务器

推荐：
- 阿里云 Windows Server 2019，2核4G，约 ¥50/月
- 腾讯云 Windows Server，2核4G，约 ¥45/月

### 2. 在 Windows 服务器上配置

1. 安装 MT5 终端
2. 在 MT5 中登录你的交易账户
3. 安装 Python 和依赖：
   ```powershell
   pip install MetaTrader5 pandas requests
   ```
4. 下载并运行 `scripts/mt5_bridge.py`

### 3. 配置云端接收

在云端环境变量添加：
```bash
export MT5_BRIDGE_ENDPOINT="http://你的Windows服务器IP:5000/api/mt5"
export MT5_BRIDGE_KEY="你的安全密钥"
```

---

## 使用 Telegram Bot

### 基本命令

向你的 Bot 发送以下命令：

| 命令 | 功能 |
|------|------|
| `/start` | 开始使用 |
| `/predict` | 获取最新预测 |
| `/price` | 获取当前价格 |
| `/help` | 帮助信息 |

### 自动推送

配置好后，以下内容会自动推送到你的 Telegram：

1. **每日早报** (08:00) - 当日分析摘要
2. **每日晚报** (20:00) - 晚间复盘
3. **价格预警** - 突破关键位时立即推送
4. **模型强信号** - 概率 > 75% 时立即推送
5. **财经事件预警** - FOMC/非农前3天提醒

---

## 故障排除

### Q: 运行报错 "ModuleNotFoundError"
A: 依赖未安装，执行 `pip install -r requirements.txt`

### Q: Telegram 消息没收到
A: 检查：
1. Bot Token 是否正确
2. 你的 Chat ID 是否在配置中
3. 用 `/start` 命令先和 Bot 互动

### Q: 预测结果不准确
A: 正常！模型仅供参考。检查：
1. 数据源是否正常
2. 是否有重大突发事件影响市场

### Q: 云端部署失败
A: 检查：
1. GitHub 仓库是否公开
2. Render 日志中的具体错误
3. 环境变量是否配置正确

---

## 下一步优化建议

1. **加入更多数据源**：原油、比特币、人民币汇率
2. **优化 LSTM 模型**：加入注意力机制
3. **添加回测模块**：验证历史表现
4. **开发微信推送**：除了 Telegram，增加微信通知

---

> ⚠️ **风险提示**：本系统仅供辅助决策，不构成投资建议。预测有风险，投资需谨慎。
