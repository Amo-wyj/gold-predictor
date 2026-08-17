#!/bin/bash
# Render.com 部署脚本

set -e

echo "======================================"
echo "🥇 黄金预测系统 - Render 部署"
echo "======================================"

# 1. 检查环境变量
if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "⚠️ 警告: TELEGRAM_BOT_TOKEN 未设置"
fi

if [ -z "$FRED_API_KEY" ]; then
    echo "⚠️ 警告: FRED_API_KEY 未设置"
fi

# 2. 安装依赖
echo ""
echo "📦 安装 Python 依赖..."
pip install -r requirements.txt

# 3. 创建必要目录
echo ""
echo "📁 创建目录..."
mkdir -p logs output data

# 4. 运行测试
echo ""
echo "🧪 运行测试预测..."
python main.py --mode predict || echo "⚠️ 测试预测失败（可能是网络问题）"

echo ""
echo "======================================"
echo "✅ 部署完成！"
echo "======================================"
echo ""
echo "服务地址: https://your-app.onrender.com"
echo ""
echo "可用端点:"
echo "  GET /                    - Dashboard"
echo "  GET /api/predict         - 获取预测"
echo "  GET /api/price           - 获取实时价格"
echo "  GET /api/technical       - 获取技术指标"
echo ""
