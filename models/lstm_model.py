"""
LSTM 深度学习预测模型
用于非线性趋势预测 + 宏观因子融合
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Tuple, Optional, List
from datetime import datetime

import pickle
import json
from config import MODELS

logger = logging.getLogger(__name__)


class GoldLSTM:
    """黄金价格 LSTM 预测器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or MODELS["lstm"]
        self.sequence_length = self.config["sequence_length"]
        self.horizons = self.config["horizons"]
        self.epochs = self.config["epochs"]
        self.batch_size = self.config["batch_size"]
        self.hidden_units = self.config["hidden_units"]
        self.dropout = self.config["dropout"]
        self.patience = self.config["early_stopping_patience"]
        
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.feature_cols = None
        self.model_name = "LSTM"
        
        # 延迟导入 TensorFlow
        self._tf = None
        self._layers = None
        self._opt = None
    
    @property
    def tf(self):
        """延迟加载 TensorFlow"""
        if self._tf is None:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            self._tf = __import__('tensorflow', fromlist=['tensorflow'])
            self._layers = __import__('tensorflow.keras.layers', fromlist=['layers'])
            self._opt = __import__('tensorflow.keras.optimizers', fromlist=['Adam'])
        return self._tf
    
    @property
    def layers(self):
        self.tf  # 确保已加载
        return self._layers
    
    @property
    def opt(self):
        self.tf  # 确保已加载
        return self._opt
    
    def _build_model(self, input_shape: Tuple[int, int]) -> object:
        """构建 LSTM 模型架构"""
        model = self.tf.keras.Sequential([
            # LSTM 层 1
            self.layers.LSTM(self.hidden_units, return_sequences=True, 
                           input_shape=input_shape),
            self.layers.Dropout(self.dropout),
            
            # LSTM 层 2
            self.layers.LSTM(self.hidden_units // 2, return_sequences=False),
            self.layers.Dropout(self.dropout),
            
            # 全连接层
            self.layers.Dense(32, activation='relu'),
            self.layers.Dropout(self.dropout / 2),
            
            # 输出层（每个 horizon 一个输出）
            self.layers.Dense(len(self.horizons), activation='linear', name='output')
        ])
        
        model.compile(
            optimizer=self._opt.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"[LSTM] 模型架构构建完成 | 输入形状: {input_shape}")
        return model
    
    def _create_sequences(self, X: np.ndarray, y: Dict[int, np.ndarray], 
                         seq_length: int) -> Tuple[np.ndarray, Dict]:
        """创建时间序列样本"""
        X_seq = []
        y_seq = {h: [] for h in self.horizons}
        
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i + seq_length])
            for h in self.horizons:
                idx = i + seq_length + h - 1
                if idx < len(y[h]):
                    y_seq[h].append(y[h][idx])
                else:
                    y_seq[h].append(0.0)
        
        X_seq = np.array(X_seq)
        for h in self.horizons:
            y_seq[h] = np.array(y_seq[h])
        
        return X_seq, y_seq
    
    def fit(self, features_df: pd.DataFrame, target_col: str = 'close') -> Dict:
        """训练 LSTM 模型"""
        from sklearn.preprocessing import StandardScaler
        
        logger.info("[LSTM] 开始训练...")
        
        # 准备数据
        self.feature_cols = [c for c in features_df.columns 
                           if c not in ['open', 'high', 'low', 'close', 'volume']]
        X_data = features_df[self.feature_cols].values
        y_data = features_df[target_col].values
        
        # 标准化
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_scaled = self.scaler_X.fit_transform(X_data)
        
        # 目标：未来收益率
        y_returns = {}
        for h in self.horizons:
            future_price = features_df[target_col].shift(-h).values
            y_returns[h] = (future_price / features_df[target_col].values - 1)
        
        # 创建序列
        X_seq, y_seq = self._create_sequences(X_scaled, y_returns, self.sequence_length)
        
        # 分割训练/验证集
        split_idx = int(len(X_seq) * 0.85)
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        
        y_train = {h: y_seq[h][:split_idx] for h in self.horizons}
        y_val = {h: y_seq[h][split_idx:] for h in self.horizons}
        
        logger.info(f"[LSTM] 训练集: {len(X_train)}, 验证集: {len(X_val)}")
        
        # 构建模型
        self.model = self._build_model((X_train.shape[1], X_train.shape[2]))
        
        # 早停回调
        early_stop = self.tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            restore_best_weights=True,
            verbose=1
        )
        
        # 训练（多输出）
        history = self.model.fit(
            X_train, 
            {f'output_{i}': y_train[h] for i, h in enumerate(self.horizons)},
            validation_data=(
                X_val,
                {f'output_{i}': y_val[h] for i, h in enumerate(self.horizons)}
            ),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        # 评估
        val_loss = min(history.history['val_loss'])
        train_loss = min(history.history['loss'])
        
        logger.info(f"[LSTM] 训练完成 | 训练损失: {train_loss:.6f} | 验证损失: {val_loss:.6f}")
        
        return {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epochs_trained": len(history.history['loss']),
            "n_features": len(self.feature_cols),
            "sequence_length": self.sequence_length,
        }
    
    def predict(self, features_df: pd.DataFrame) -> Dict:
        """预测未来收益率"""
        if self.model is None:
            raise ValueError("模型未训练，请先调用 fit()")
        
        # 标准化
        X_data = features_df[self.feature_cols].values[-self.sequence_length:]
        X_scaled = self.scaler_X.transform(X_data)
        X_input = X_scaled.reshape(1, self.sequence_length, -1)
        
        # 预测
        predictions = self.model.predict(X_input, verbose=0)
        
        current_price = features_df['close'].iloc[-1]
        
        results = {}
        for i, h in enumerate(self.horizons):
            predicted_return = float(predictions[0, i])
            predicted_price = current_price * (1 + predicted_return)
            
            # 基于预测误差历史估算置信区间
            # 简化版：使用固定的波动率估算
            volatility = features_df['close'].pct_change().std() * np.sqrt(h)
            conf_width = volatility * current_price * 1.96  # 95% 置信区间
            
            results[f"horizon_{h}d"] = {
                "predicted_return": predicted_return * 100,  # 转为百分比
                "predicted_price": predicted_price,
                "lower_bound": predicted_price - conf_width,
                "upper_bound": predicted_price + conf_width,
            }
        
        return results
    
    def predict_direction_probability(self, features_df: pd.DataFrame) -> Dict:
        """预测涨跌方向概率（使用蒙特卡洛模拟）"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        current_price = features_df['close'].iloc[-1]
        results = self.predict(features_df)
        
        # 蒙特卡洛模拟估算概率
        n_simulations = 1000
        all_returns = []
        
        # 基于历史波动率进行模拟
        returns_std = features_df['close'].pct_change().std()
        
        for h in self.horizons:
            predicted_return = results[f"horizon_{h}d"]["predicted_return"] / 100
            
            # 模拟未来价格
            simulated_returns = np.random.normal(
                loc=predicted_return,
                scale=returns_std * np.sqrt(h),
                size=n_simulations
            )
            
            # 计算概率
            prob_up = np.mean(simulated_returns > 0)
            prob_down = 1 - prob_up
            
            # 信号强度
            if prob_up >= 0.75:
                signal = "STRONG_BUY"
            elif prob_up >= 0.65:
                signal = "BUY"
            elif prob_down >= 0.75:
                signal = "STRONG_SELL"
            elif prob_down >= 0.65:
                signal = "SELL"
            else:
                signal = "NEUTRAL"
            
            results[f"horizon_{h}d"].update({
                "probability_up": float(prob_up),
                "probability_down": float(prob_down),
                "signal": signal,
                "confidence": float(max(prob_up, prob_down)),
            })
        
        return results
    
    def save(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path.replace('.pkl', '.h5'))
        
        with open(path, 'wb') as f:
            pickle.dump({
                "scaler_X": self.scaler_X,
                "scaler_y": self.scaler_y,
                "feature_cols": self.feature_cols,
                "config": self.config,
                "horizons": self.horizons,
            }, f)
        logger.info(f"[LSTM] 模型已保存: {path}")
    
    def load(self, path: str):
        """加载模型"""
        self.model = self.tf.keras.models.load_model(path.replace('.pkl', '.h5'))
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.scaler_X = data["scaler_X"]
        self.scaler_y = data["scaler_y"]
        self.feature_cols = data["feature_cols"]
        self.config = data["config"]
        self.horizons = data["horizons"]
        
        logger.info(f"[LSTM] 模型已加载: {path}")


def run_lstm_prediction(features_df: pd.DataFrame) -> Dict:
    """便捷函数：运行完整 LSTM 预测流程"""
    
    lstm = GoldLSTM()
    
    # 训练
    train_result = lstm.fit(features_df)
    
    # 预测
    predictions = lstm.predict_direction_probability(features_df)
    
    predictions["model_info"] = train_result
    
    return predictions


if __name__ == "__main__":
    import yfinance as yf
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from features.feature_engineering import FeatureEngine
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    
    print("\n" + "="*60)
    print("🥇 LSTM 黄金预测")
    print("="*60)
    
    # 获取数据
    gold = yf.Ticker("GC=F").history(period="2y", auto_adjust=True)
    gold.index = pd.to_datetime(gold.index).tz_localize(None)
    
    # 特征工程
    engine = FeatureEngine()
    features = engine.add_technical_indicators(gold)
    features = features.dropna()
    
    # 训练 & 预测
    results = run_lstm_prediction(features)
    
    print(f"\n模型信息: 验证损失={results['model_info']['val_loss']:.6f}")
    
    for key, data in results.items():
        if key.startswith("horizon"):
            print(f"\n{key.upper()}:")
            print(f"  预测收益率: {data['predicted_return']:+.2f}%")
            print(f"  预测价格: ${data['predicted_price']:.2f}")
            print(f"  上涨概率: {data['probability_up']*100:.1f}%")
            print(f"  下跌概率: {data['probability_down']*100:.1f}%")
            print(f"  信号: {data['signal']}")
