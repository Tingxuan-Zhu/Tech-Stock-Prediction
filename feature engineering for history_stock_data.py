import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import ta
import numpy as np
import traceback

def calculate_features(group):
    # 将 'Date' 设置为当前这个“小数据块”的索引，确保时间序列计算正确
    group = group.set_index('Date').sort_index()
    #计算每日收益率
    group['Daily_Return'] = group['Close'].pct_change()

    #计算不同时间窗口的移动平均线（MA）
    group['MA_5'] = group['Close'].rolling(window=5).mean()
    group['MA_20'] = group['Close'].rolling(window=20).mean()
    group['MA_50'] = group['Close'].rolling(window=50).mean()

    #计算波动率
    group['Volatility_20d'] = group['Daily_Return'].rolling(window=20).std()

    ### 技术指标

    # 相对强弱指数（RSI）
    group['RSI'] = ta.momentum.rsi(group['Close'], window=14)

    # 移动平均线收敛 / 发散指标 （MACD）
    group['MACD'] = ta.trend.macd_diff(group['Close'])

    # 布林带 (Bollinger Bands)
    try:
        bollinger = ta.volatility.BollingerBands(group['Close'], window=20)
        group['BB_Upper'] = bollinger.bollinger_hband()
        group['BB_Lower'] = bollinger.bollinger_lband()
        group['BB_Width'] = group['BB_Upper'] - group['BB_Lower']
        # 计算价格相对于布林带的位置 (Price position relative to Bollinger Bands)
        group['BB_Position'] = (group['Close'] - group['BB_Lower']) / (group['BB_Upper'] - group['BB_Lower'])
    except:
        group['BB_Upper'] = group['BB_Lower'] = group['BB_Width'] = group['BB_Position'] = np.nan

    # 创建滞后特征
    group['Lag_1d_Close'] = group['Close'].shift(1)
    group['Lag_5d_Close'] = group['Close'].shift(5)

    # 价格动量特征 (Price Momentum Features)
    group['Price_Change_5d'] = (group['Close'] - group['Lag_5d_Close']) / group['Lag_5d_Close']
    group['Price_Change_20d'] = (group['Close'] - group['Close'].shift(20)) / group['Close'].shift(20)

    # 从日期中提取特征 (Extract Date Features)
    group['DayOfWeek'] = group.index.dayofweek
    group['Month'] = group.index.month
    group['Quarter'] = group.index.quarter

    return group.reset_index()


try:
    df = pd.read_csv('historical_stock_data.csv', parse_dates=['Date'])
    print("Get data successfully")             #这一步非常重要，尤其是后面的index_col 和 parse_dates，
                                                                 # 这两个让这个csv文件进入pd中的索引是日期，而不是pandas默认的数字

    df_featured = df.groupby('Ticker', group_keys=False).apply(calculate_features).reset_index(drop=True)      # 这会为每个'Ticker'分组，然后将该组数据传入 calculate_features 函数
    print("Calculate features successfully"     )


    ### 数据标准化和最终处理
    df_featured.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 更智能的缺失值处理 (Smarter missing value handling)
    print("缺失值统计 (Missing values count):")
    print(df_featured.isnull().sum())

    # 删除缺失值过多的行 (Remove rows with too many missing values)
    # 如果一行中有超过50%的特征列为空，则删除该行
    feature_columns = [col for col in df_featured.columns if col not in ['Date', 'Ticker']]
    missing_threshold = len(feature_columns) * 0.5
    df_featured = df_featured.dropna(thresh=len(df_featured.columns) - missing_threshold)

    columns_to_drop = []
    for col in df.columns:
        if any(keyword in str(col).lower() for keyword in ['unnamed', 'index', 'level_0']):
            columns_to_drop.append(col)

    if columns_to_drop:
        print(f"\n Found and removing unexpected columns): {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)

    # 定义需要标准化的特征列
    numerical_features = [
        'Daily_Return', 'MA_5', 'MA_20', 'MA_50', 'Volatility_20d',
        'RSI', 'MACD', 'BB_Width', 'BB_Position',
        'Lag_1d_Close', 'Lag_5d_Close', 'Price_Change_5d', 'Price_Change_20d'
    ]

    # StandardScaler
    existing_features = [col for col in numerical_features if col in df_featured.columns]
    scaler = StandardScaler()
    if existing_features:
        # StandardScaler
        scaler = StandardScaler()

        # Standardize selected features
        df_featured[existing_features] = scaler.fit_transform(df_featured[existing_features])

        print("Data standardization completed")


    # 预览内容
    print("Data standardiation successfully")
    print(df_featured.head())

    # 清洗数据
    print(df_featured.isnull().sum())

    df_featured.to_csv('feature_engineering_for_history_stock_data.csv', index=False)

    print("Successfully saved feature_engineering_for_history_stock_data.csv")

except FileNotFoundError:
    print('File not found')
except Exception as e:
    print(f"There is an error: {e}")
    traceback.print_exc()




