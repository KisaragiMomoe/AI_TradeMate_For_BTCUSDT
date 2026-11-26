import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.exchange import BinanceDataFetcher
from core.feature_engine import build_advanced_dataset

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

#获取数据
print("获取数据中...")
fetcher = BinanceDataFetcher(is_testnet = False)
df = fetcher.fetch_ohlcv("BTC/USDT:USDT", "1h", 5000) #可改币种

df_rate = fetcher.fetch_funding_rate_history("BTC/USDT:USDT", limit = 1000)
if df_rate is not None:
    # 重新采样对齐到 1h，并向前填充
    df = df.join(df_rate, how = "left")
    df['funding_rate'] = df['funding_rate'].ffill().fillna(0.0001)
    print("资金费率合并成功！")
else:
    df['funding_rate'] = 0.0001
    print("警告：资金费率获取失败，使用默认值。")

print("正在计算 17 个技术指标与因子...")
df = build_advanced_dataset(df)
features = [
    'MA5', 'MA20', 'EMA10', 'Price_MA20_Ratio',
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'ATR', 'ATR_Ratio', 'BB_Std', 'BB_Width',
    'Vol_Change', 'Ret_3h', 'Ret_6h',
    'ai_sentiment', 'funding_rate',
    
    'Price_SMA200_Ratio',
    'RSI_24',
    'Vol_Ratio_24',
    'Upper_Shadow', 'Lower_Shadow'
]
target = 'target'

split = int(len(df) * 0.8)
train_df = df.iloc[:split].copy()
test_df = df.iloc[split:].copy()

print(f"训练集样本数: {len(train_df)} | 测试集样本数: {len(test_df)}")

train_data = lgb.Dataset(train_df[features], label = train_df[target])
test_data = lgb.Dataset(test_df[features], label = test_df[target], reference = train_data)

params = {
    'objective': 'regression',  # 回归树。
    'metric': 'rmse',        # 评估指标用 均方
    'learning_rate': 0.03,
    'num_leaves': 31,
    'max_depth': 4,
    'min_data_in_leaf': 100,
    'lambda_l2': 1.0,
    'verbose': -1
}

print("开始训练模型...")
bst = lgb.train(
    params, 
    train_data,
    num_boost_round = 500,
    valid_sets = [test_data],
    callbacks = [lgb.early_stopping(stopping_rounds = 50)] #50轮loss不降直接停
)

#预测与交易信号生成
test_df['pred_return'] = bst.predict(test_df[features])

print("\n=== 模型预测值统计 ===")
print(test_df['pred_return'].describe())
# print("预测大于 0.0015 的次数:", (test_df['pred_return'] > 0.0015).sum())
# print("预测小于 -0.0015 的次数:", (test_df['pred_return'] < -0.0015).sum())

# 设定交易阈值（扣除手续费后的最小可接受收益）
# 假设双边手续费万分之八，滑点万分之二，总成本 0.1%
threshold = 0.0005

test_df['signal'] = 0
test_df.loc[test_df['pred_return'] > threshold, 'signal'] = 1
test_df.loc[test_df['pred_return'] < -threshold, 'signal'] = -1

#回测
fee_rate = 0.0004
test_df['strategy_return'] = 0.0
position = 0
entry_price = 0.0
cooldown = 0

for i in range(1, len(test_df)):
    current_price = test_df['close'].iloc[i]
    prev_price = test_df['close'].iloc[i - 1]
    sma_200 = test_df['SMA_200'].iloc[i]
    hour_ret = current_price / prev_price - 1
    strategy_ret = 0.0
    # 1.有持仓，检查盈亏
    if position != 0:
        if position == 1: pnl_pct = (current_price - entry_price) / entry_price
        else: pnl_pct = (entry_price - current_price) / entry_price

        if pnl_pct >= 0.03 or pnl_pct <= -0.015:
            strategy_ret = position * hour_ret - fee_rate
            position = 0
            cooldown = 0
        else:
            strategy_ret = position * hour_ret
            cooldown -= 1

            # 2. 触发移动保本止损 (如果浮盈曾超过 1%，就把止损线设在开仓价 + 0.1%)
            if pnl_pct >= 0.01:
                if position == 1 and current_price <= entry_price * 1.001:
                    # 回落跌破保本线，平仓
                    strategy_ret -= fee_rate
                    position = 0
                elif position == -1 and current_price >= entry_price * 0.999:
                    strategy_ret -= fee_rate
                    position = 0

            # 3. 如果冷却期结束，强制平仓
            if position != 0 and cooldown <= 0:
                strategy_ret -= fee_rate
                position = 0
    # 2.无持仓，准备开仓
    if position == 0:
        signal = test_df['signal'].iloc[i - 1]
        if signal != 0:
            if (signal == 1 and current_price > sma_200) or (signal == -1 and current_price < sma_200):
                position = signal
                entry_price = current_price
                strategy_ret -= fee_rate
                cooldown = 4
    test_df.iloc[i, test_df.columns.get_loc('strategy_return')] = strategy_ret

# ================= 5. 计算净值与评估 =================
test_df['cum_strategy'] = (1 + test_df['strategy_return']).cumprod() - 1
test_df['cum_hold'] = test_df['close'] / test_df['close'].iloc[0] - 1

total_return = test_df['cum_strategy'].iloc[-1] * 100
hold_return = test_df['cum_hold'].iloc[-1] * 100

# 计算胜率
trades = test_df[test_df['strategy_return'] != 0]
win_trades = trades[trades['strategy_return'] > 0]
loss_trades = trades[trades['strategy_return'] < 0]

win_rate = (win_trades.shape[0] / trades.shape[0]) * 100 if trades.shape[0] > 0 else 0
avg_win = win_trades['strategy_return'].mean() * 100 if win_trades.shape[0] > 0 else 0
avg_loss = loss_trades['strategy_return'].mean() * 100 if loss_trades.shape[0] > 0 else 0

print("\n=== 终极进化版策略回测结果 ===")
print(f"策略总收益率: {total_return:.2f}%")
print(f"买入持有收益率 (基准): {hold_return:.2f}%")
print(f"交易胜率: {win_rate:.2f}%")
print(f"单笔平均盈利: {avg_win:.3f}% | 单笔平均亏损: {avg_loss:.3f}%")

# 保存模型到本地文件
bst.save_model("lgb_trading_model.txt")
print("✅ 模型已保存为 lgb_trading_model.txt，可供实盘引擎加载！")