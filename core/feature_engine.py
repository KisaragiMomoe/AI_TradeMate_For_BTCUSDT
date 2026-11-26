import pandas as pd
import numpy as np

def build_advanced_dataset(df, is_live = False):
    """
    构建工业级 LightGBM 训练集（包含大趋势、多周期动量、量价与形态特征）
    """
    df = df.copy()

    df['MA5'] = df['close'].rolling(5).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['EMA10'] = df['close'].ewm(span = 10, adjust = False).mean()
    df['Price_MA20_Ratio'] = df['close'] / df['MA20'] # 价格偏离均线度

    df['SMA_200'] = df['close'].rolling(200).mean()
    df['Price_SMA200_Ratio'] = df['close'] / df['SMA_200'] - 1

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window = 14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window = 14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))

    gain_24 = (delta.where(delta > 0, 0)).rolling(window = 24).mean()
    loss_24 = (-delta.where(delta < 0, 0)).rolling(window = 24).mean()
    df['RSI_24'] = 100 - (100 / (1 + gain_24 / loss_24))

    exp12 = df['close'].ewm(span = 12, adjust = False).mean()
    exp26 = df['close'].ewm(span = 26, adjust = False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_Signal'] = df['MACD'].ewm(span = 9, adjust = False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis = 1).max(axis = 1)
    df['ATR'] = tr.rolling(14).mean()
    df['ATR_Ratio'] = df['ATR'] / df['close']

    df['BB_Std'] = df['close'].rolling(20).std()
    df['BB_Width'] = (df['BB_Std'] * 4) / df['MA20'] # 布林带开口宽度

    df['Vol_Change'] = df['volume'].pct_change(3) # 过去3小时成交量变化
    df['Ret_3h'] = df['close'].pct_change(3)      # 过去3小时收益率
    df['Ret_6h'] = df['close'].pct_change(6)      # 过去6小时收益率

     # 👈 新增：量比特征（当前成交量是过去 24 小时平均的几倍）
    df['Vol_MA_24'] = df['volume'].rolling(24).mean()
    df['Vol_Ratio_24'] = df['volume'] / (df['Vol_MA_24'] + 1e-8)

    # 👈 新增：K 线微观形态特征（多空博弈痕迹）
    high_low_diff = df['high'] - df['low'] + 1e-8
    df['Body_Ratio'] = abs(df['close'] - df['open']) / high_low_diff
    df['Upper_Shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / high_low_diff
    df['Lower_Shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / high_low_diff

    df['ai_sentiment'] = 50.0
    df.loc[df['RSI'] > 70, 'ai_sentiment'] = 85.0
    df.loc[df['RSI'] < 30, 'ai_sentiment'] = 15.0

    df['future_return'] = df['close'].shift(-4) / df['close'] - 1
    # df['target'] = np.where(df['future_return'] > 0.001, 1, 0)

    # df['funding_rate'] = funding_rate # 把资金费率作为一个特征 砍掉了因为要抓取真实情况

    if not is_live:
        # 离线训练模式：计算未来收益率，并删掉包含 NaN 的所有行
        df['target'] = df['close'].shift(-1) / df['close'] - 1
        df = df.dropna()
    else:
        # 实盘模式：必须等所有指标预热完毕，把新特征也加进 subset
        df = df.dropna(subset=[
            'SMA_200', 'RSI_24', 'Vol_Ratio_24', 
            'Upper_Shadow', 'Lower_Shadow', 'RSI', 'MACD_Hist'
        ])
    
    return df