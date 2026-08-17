# 黄金预测项目进度追踪 (2026-08-17)

## 当前状态：✅ 核心功能全部跑通

### 测试命令
```bash
cd ~/qclaw/projects/gold-predictor
/usr/bin/python3 run_predict.py                    # 本地预测（模拟数据）
/usr/bin/python3 test_pipeline.py                  # 详细流程测试
/usr/bin/python3 --real run_predict.py             # 真实数据（需 Yahoo/FRED API）
```

### 最新测试输出 (2026-08-17)
```
🥇 黄金预测系统
📥 数据: 400天 | 最新价格: $2088.57
🔧 特征: 82个 (技术+宏观)
🤖 ARIMA ✅ | GBM ✅

📊 预测结果:
  【明天】🔴 STRONG_SELL  涨23.8% | 跌76.2% | $2071.86 (-0.80%)
  【3天后】🔴 STRONG_SELL 涨24.5% | 跌75.5% | $2069.60 (-0.91%)
  【5天后】🔴 STRONG_SELL 涨24.1% | 跌75.9% | $2070.74 (-0.85%)
  置信度: MEDIUM
🔧 技术指标: RSI=54.4 | MACD=5.6(BULLISH) | 技术评分=0.25
🌐 宏观: DXY=106.71 | 实际利率=1.85(BEARISH) | VIX=31.25(HIGH)
```

## 已完成文件清单
```
gold-predictor/
├── config.py                  ✅ 配置（数据源/模型/预警/Telegram）
├── run_predict.py             ✅ 预测入口脚本（新增）
├── test_pipeline.py           ✅ 完整流程测试脚本
├── data/
│   ├── mock_data.py           ✅ 模拟数据生成（离线测试用）
│   ├── yahoo_collector.py     ✅ Yahoo Finance 采集
│   └── fred_collector.py      ✅ FRED 宏观数据采集
├── features/
│   └── feature_engineering.py ✅ 特征工程（67→82个特征）
├── models/
│   ├── arima_model.py         ✅ ARIMA 模型（statsmodels）
│   ├── gbm_model.py           ✅ GBM 模型（sklearn，替代TensorFlow LSTM）
│   └── ensemble.py            ✅ 集成预测（ARIMA 25% + GBM 45% + 技术分析 20% + 宏观 10%）
├── bot/
│   └── telegram_bot.py        ✅ Telegram Bot + AlertEngine
├── dashboard/
│   ├── dashboard.py           ✅ Flask Web Dashboard（含完整API）
│   └── templates/
│       └── dashboard.html     ✅ 可视化界面（Tailwind+Plotly）
├── scripts/
│   ├── mt5_bridge.py          ✅ MT5 数据桥接（Windows端推送）
│   └── deploy_render.sh       ✅ Render 部署脚本
├── render.yaml                 ✅ Render Blueprint
├── SETUP.md                   ✅ 部署指南
├── README.md                  ✅ 项目文档
└── requirements.txt           ✅ 依赖列表
```

## 修复记录（2026-08-17）
| 日期 | 问题 | 修复 |
|------|------|------|
| 08-17 | GBM: np.random.Normal→normal | 修复大小写 |
| 08-17 | GBM: horizon>=数据长度跳过 | 数据量250→400天 |
| 08-17 | 宏观全NaN（日期不对齐） | mock_data统一datetime.now()去掉时分秒 |
| 08-17 | analyze_macro找不到列名 | 改用原始列名(dxy/dgs10/real_rate/vix) |
| 08-17 | current_macro是dict非DataFrame | 增加dict→DataFrame转换 |
| 08-17 | LSTM TensorFlow太大下载超时 | 替换为sklearn GradientBoostingRegressor |

## 待完成（需用户提供信息）
- [ ] Telegram Bot Token（@BotFather申请）
- [ ] FRED API Key（免费注册：fred.stlouisfed.org）
- [ ] GitHub 账号（用于 Render 部署）
- [ ] 租 Windows 云服务器跑 MT5（约¥30-50/月）
- [ ] 接入真实财经日历 API（替代空白的 check_macro_events）

## 运行环境
- 系统 Python: /usr/bin/python3（3.9.6，macOS内置）
- 所有依赖已安装（清华源）
- QClaw 内置 Python 3.11.10 缺包且 pip 网络超时，改用系统 Python

## 注意事项
- Yahoo Finance 当前限流中（Too Many Requests），模拟数据可用
- DXY=106.71 是 mock 数据，非真实值
- 模拟数据基于随机种子42，每次生成相同
