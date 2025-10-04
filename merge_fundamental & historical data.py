import pandas as pd
import numpy as np
import yfinance as yf
import traceback

historical_file = 'feature_engineering_for_history_stock_data.csv'
fundamental_file = 'featured_fundamental_data.csv'

try:
    df_hist = pd.read_csv(historical_file, parse_dates=['Date'])
    df_fundamental = pd.read_csv(fundamental_file)

    # 按股票分组创建明日收盘价 (Create tomorrow's close price by ticker group)
    df_hist = df_hist.sort_values(['Ticker', 'Date'])
    df_hist['Tomorrow_Close'] = df_hist.groupby('Ticker')['Close'].shift(-1)

    # 创建二分类目标：明日涨跌 (Create binary target: up/down tomorrow)
    df_hist['Target'] = (df_hist['Tomorrow_Close'] > df_hist['Close']).astype(int)

    # 创建连续目标：明日收益率 (Create continuous target: tomorrow's return)
    df_hist['Tomorrow_Return'] = (df_hist['Tomorrow_Close'] - df_hist['Close']) / df_hist['Close']

    # 删除无法预测的最后一天数据 (Remove last day data that cannot be predicted)
    df_hist = df_hist.dropna(subset=['Tomorrow_Close'])


    # Merge the data
    df_merged = pd.merge(df_hist, df_fundamental, how='left', on='Ticker')
    df_merged = df_merged.dropna(subset=[col for col in df_merged.columns if col != 'Ticker'])
    # 检查合并后的缺失值 (Check missing values after merge)
    missing_after_merge = df_merged.isnull().sum()
    print("Missing values after merge:")
    missing_cols = missing_after_merge[missing_after_merge > 0]
    if len(missing_cols) > 0:
        for col, count in missing_cols.items():
            print(f"{col}: {count} ({count / len(df_merged) * 100:.1f}%)")
    else:
        print("There is no missing values")

    # One-Hot Encoding
    df_merged = pd.get_dummies(df_merged, columns=['Ticker'], drop_first=False)

    # Time feature engineering

    df_merged['Year'] = df_merged['Date'].dt.year
    df_merged['Month'] = df_merged['Date'].dt.month
    df_merged['DayOfYear'] = df_merged['Date'].dt.dayofyear
    df_merged['WeekOfYear'] = df_merged['Date'].dt.isocalendar().week

    # Data cleaning and validation

    # Handle infinite values
    df_merged.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Remove missing values
    initial_rows = len(df_merged)
    df_merged.dropna(subset=[col for col in df_merged.columns if col != 'Ticker'], inplace=True)
    final_rows = len(df_merged)

    # 数据排序 (Sort data)
    df_merged = df_merged.sort_values('Date').reset_index(drop=True)

    # Create train / validation / test split markers

    # Time-based split
    df_merged = df_merged.sort_values('Date')
    total_days = len(df_merged['Date'].unique())

    # 60% 训练，20% 验证，20% 测试 (60% train, 20% validation, 20% test)
    train_end_date = df_merged['Date'].unique()[int(total_days * 0.6)]
    val_end_date = df_merged['Date'].unique()[int(total_days * 0.8)]

    df_merged['Split'] = 'test'
    df_merged.loc[df_merged['Date'] <= val_end_date, 'Split'] = 'valid'
    df_merged.loc[df_merged['Date'] <= train_end_date, 'Split'] = 'train'

    # Cleaning data here
    # 1. 先处理历史数据的技术指标null（整行删除）
    df_merged = df_merged.dropna(subset=['RSI', 'MACD', 'BB_Upper'], how='any')

    # 2. 对fundamental数据的异常0值处理
    # 检查哪些fundamental字段不应该为0
    financial_cols = ['Market_Cap', 'Revenue', 'EPS']  # 这些字段为0是异常的
    df_merged = df_merged[df_merged[financial_cols].ne(0).all(axis=1)]

    # 3. 对剩余的偶发null用前向填充
    df_merged = df_merged.fillna(method='ffill')
    df_merged = df_merged.fillna(method='bfill')

    # Store data
    df_merged.to_csv('final_data.csv', index=False)
    print("Data saved successfully")

except FileNotFoundError:
    print("File not found")
except Exception as e:
    print(f"There is an error: {e}")
    traceback.print_exc()
