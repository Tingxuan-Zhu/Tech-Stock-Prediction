import yfinance as yf
import pandas as pd
import numpy as np

# 这里我读取之前的csv文件

file_path = 'historical_stock_data.csv'
file_path_2 = 'fundamental_data.csv'
try:
    df = pd.read_csv(file_path)
    df2 = pd.read_csv(file_path_2)

    df['Date'] = pd.to_datetime(df['Date'])
except FileNotFoundError as e:
    print(f"The file doesn't exist: {e}")
    # 开始清洗数据

# 这里统计数据缺失,诊断是否有问题
print(df.isnull().sum())
print(df2.isnull().sum())

    # 价格数据用前向填充 (Forward fill for price data)
price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
for col in price_columns:
    if col in df.columns:
        df[col] = df.groupby('Ticker')[col].fillna(method='ffill')

# 成交量用0填充是合理的 (Volume filled with 0 is reasonable)
if 'Volume' in df.columns:
    df['Volume'].fillna(0, inplace=True)

    financial_columns = ['Market_Cap', 'Revenue', 'Operating_Cash_Flow', 'Free_Cash_Flow']
    for col in financial_columns:
        if col in df2.columns:
            df2[col].fillna(0, inplace=True)

ratio_columns = ['EPS', 'PE_Ratio', 'Dividend_Yield']
for col in ratio_columns:
    if col in df2.columns:
        # 按行业计算中位数填充 (Fill with sector median)
        df2[col] = df2.groupby('Sector')[col].transform(lambda x: x.fillna(x.median()))
        # 如果某个行业没有数据，用整体中位数填充 (If sector has no data, use overall median)
        df2[col].fillna(df2[col].median(), inplace=True)

df2['Sector'].fillna('Unknown', inplace=True)

    #再次诊断是否成功全部替换
    # print(df.isnull().sum())
    # print(df)

    # print(df2.isnull().sum())
    # print(df2)

    # 覆盖到原数据库中
df.to_csv('historical_stock_data.csv', index = False)
df2.to_csv('fundamental_data.csv', index = False)


