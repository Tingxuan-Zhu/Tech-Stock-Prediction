import yfinance as yf
import pandas as pd
import numpy as np
import traceback

# Upload the csv. database
fundamental_file = 'fundamental_data.csv'

try:
    # Read the data
    df = pd.read_csv(fundamental_file)
    print("Read fundamental data successfully")
    print(df.head())

    # 特征工程第一部分：计算新的财务比率

    # 计算市销率 (Price-to-Sales, P/S Ratio) = 市值 / 收入
    df['PS_Ratio'] = np.where(df['Revenue'] > 0, df['Market_Cap'] / df['Revenue'], np.nan)

    # 计算市现率 (Price-to-Cash-Flow, P/CF Ratio) = 市值 / 现金流
    df['PCF_Ratio'] = np.where(df['Operating_Cash_Flow'] > 0, df['Market_Cap'] / df['Operating_Cash_Flow'], np.nan)

    # 计算自由现金流收益率 (Free Cash Flow Yield)
    df['FCF_Yield'] = np.where(df['Market_Cap'] > 0, df['Free_Cash_Flow'] / df['Market_Cap'], np.nan)

    # 计算盈利收益率 (Earnings Yield) = 每股收益 / 股价 ≈ 1 / 市盈率     这个指标可以看作是市盈率的倒数，对于比较不同公司的盈利能力很有用
    df['Earnings_Yield'] = np.where(df['PE_Ratio'] > 0, 1 / df['PE_Ratio'], np.nan)

    # 计算ROE代理指标 (ROE Proxy) = 净收益 / 市值 (Net Income / Market Cap)
    df['ROE_Proxy'] = np.where(df['Market_Cap'] > 0, (df['EPS'] * df['Market_Cap'] / df['PE_Ratio']) / df['Market_Cap'], np.nan)

    # 资产负债比率的代理 (Asset Efficiency Proxy) = 收入 / 市值
    df['Asset_Efficiency'] = np.where(df['Market_Cap'] > 0, df['Revenue'] / df['Market_Cap'], np.nan)

    # 特征工程第二部分：进行同业对比分析 (Feature Engineering Part 2: Sector Comparison)

    # 定义我们要按行业分析的指标列 (Define metrics for sector analysis)
    stats_to_compare = ['PE_Ratio', 'PS_Ratio', 'PCF_Ratio', 'Dividend_Yield', 'Earnings_Yield', 'FCF_Yield',
                        'Asset_Efficiency']

    # 过滤掉'Unknown'行业进行更准确的行业对比 (Filter out 'Unknown' sector for more accurate comparison)
    df_known_sector = df[df['Sector'] != 'Unknown'].copy()

    # 使用 groupby('Sector') 计算每个行业的各项指标的统计值 (Calculate sector statistics)
    sector_stats = df_known_sector.groupby('Sector')[stats_to_compare].agg(['mean', 'median', 'std']).fillna(0)

    # 展平多层列名 (Flatten multi-level column names)
    sector_stats.columns = ['_'.join(col).strip() for col in sector_stats.columns.values]

    # 重命名列以便理解 (Rename columns for clarity)
    sector_stats = sector_stats.add_suffix('_Sector')

    print("行业统计计算完成 (Sector statistics calculated)")

    # 将行业统计值合并回原始数据框 (Merge sector statistics back to original dataframe)
    df = pd.merge(df, sector_stats, left_on='Sector', right_index=True, how='left')

    # 计算每个公司指标相对于其行业的偏离度 (Calculate deviation from sector average)
    print("计算行业偏离度 (Calculating sector deviations)...")

    for stat in stats_to_compare:
        # 相对于行业均值的偏离 (Deviation from sector mean)
        mean_col = f'{stat}_mean_Sector'
        if mean_col in df.columns:
            df[f'{stat}_Vs_Sector_Mean'] = np.where(
                df[mean_col] != 0,
                (df[stat] - df[mean_col]) / df[mean_col],
                0
            )

        # 相对于行业中位数的偏离 (Deviation from sector median) - 更稳健的指标
        median_col = f'{stat}_median_Sector'
        if median_col in df.columns:
            df[f'{stat}_Vs_Sector_Median'] = np.where(
                df[median_col] != 0,
                (df[stat] - df[median_col]) / df[median_col],
                0
            )

        # Z-Score (标准化分数) - 表示公司指标在行业中的标准差位置
        std_col = f'{stat}_std_Sector'
        if std_col in df.columns and mean_col in df.columns:
            df[f'{stat}_ZScore'] = np.where(
                df[std_col] > 0,
                (df[stat] - df[mean_col]) / df[std_col],
                0
            )

    # 特征工程第三部分：创建复合指标 (Feature Engineering Part 3: Composite Indicators)

    # 价值投资评分 (Value Investment Score) - 综合多个价值指标
    value_metrics = ['PE_Ratio', 'PS_Ratio', 'PCF_Ratio']

    # 标准化价值指标 (Normalize value metrics) - 越低越好
    for metric in value_metrics:
        if metric in df.columns:
            # 取倒数然后标准化，这样高分表示更好的价值
            df[f'{metric}_Inverted'] = np.where(df[metric] > 0, 1 / df[metric], 0)

    # 计算价值投资综合评分 (Calculate composite value score)
    value_cols = [f'{metric}_Inverted' for metric in value_metrics if f'{metric}_Inverted' in df.columns]
    if value_cols:
        df['Value_Score'] = df[value_cols].mean(axis=1)

    # 成长性指标 (Growth Indicators) - 基于现金流和收益
    if 'Free_Cash_Flow' in df.columns and 'Operating_Cash_Flow' in df.columns:
        df['Cash_Flow_Quality'] = np.where(
            df['Operating_Cash_Flow'] > 0,
            df['Free_Cash_Flow'] / df['Operating_Cash_Flow'],
            0
        )

    print("Composite indicators created")

    # 数据清理和验证 (Data Cleaning and Validation)


    # 显示缺失值情况 (Show missing values)
    print("Missing values statistics:")
    missing_counts = df.isnull().sum()
    print(missing_counts[missing_counts > 0])



    # 最终验证 (Final validation)
    print(f"\nFinal data shape: {df.shape}")
    print(f"Total remaining missing values: {df.isnull().sum().sum()}")

    # 选择重要列进行预览 (Select important columns for preview)
    important_cols = ['Ticker', 'Sector', 'PE_Ratio', 'PS_Ratio', 'Earnings_Yield', 'Value_Score',
                      'PE_Ratio_Vs_Sector_Mean']
    preview_cols = [col for col in important_cols if col in df.columns]

    print(f"\n Important features preview):")
    print(df[preview_cols].head())

    # 保存处理后的数据 (Save processed data)
    df.to_csv('featured_fundamental_data.csv', index=False)
    print("Feature engineered fundamental data saved successfully")


except FileNotFoundError:
    print("文件未找到 (File not found)")
except Exception as e:
    print(f"发生错误 (Error occurred): {e}")
    import traceback

    traceback.print_exc()