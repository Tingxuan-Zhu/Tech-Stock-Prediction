import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.style.use('default')


def load_and_inspect_data():
    try:
        df = pd.read_csv('final_data.csv', parse_dates=['Date'])
        print(f"Data loaded: {df.shape}")
        return df
    except FileNotFoundError:
        print("Error: final_data.csv not found")
        return None


def analyze_data_quality(df):
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]

    if len(missing_data) > 0:
        print("Missing values found:")
        for col, count in missing_data.items():
            print(f"  {col}: {count} ({count / len(df) * 100:.2f}%)")
    else:
        print("Data quality check: No missing values")


def analyze_price_distributions(df):
    price_cols = ['Open', 'High', 'Low', 'Close']
    available_price_cols = [col for col in price_cols if col in df.columns]

    if available_price_cols:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Stock Price Distribution Analysis', fontsize=16, fontweight='bold')

        if 'Close' in available_price_cols:
            axes[0, 0].hist(df['Close'], bins=50, alpha=0.7, color='blue', edgecolor='black')
            axes[0, 0].axvline(df['Close'].mean(), color='red', linestyle='--', label=f'Mean: {df["Close"].mean():.2f}')
            axes[0, 0].set_title('Close Price Distribution')
            axes[0, 0].set_xlabel('Close Price ($)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        if 'Daily_Return' in df.columns:
            df_clean = df.dropna(subset=['Daily_Return'])
            if not df_clean.empty:
                axes[0, 1].hist(df_clean['Daily_Return'], bins=50, alpha=0.7, color='green', edgecolor='black')
                axes[0, 1].axvline(0, color='red', linestyle='--', label='Zero return')
                axes[0, 1].set_title('Daily Returns Distribution')
                axes[0, 1].set_xlabel('Daily Return')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)

        if 'Volume' in df.columns:
            axes[1, 0].hist(df['Volume'], bins=50, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 0].set_title('Volume Distribution')
            axes[1, 0].set_xlabel('Volume')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)

            axes[1, 1].hist(np.log(df['Volume'] + 1), bins=50, alpha=0.7, color='purple', edgecolor='black')
            axes[1, 1].set_title('Volume Distribution (Log Scale)')
            axes[1, 1].set_xlabel('log(Volume)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('price_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Price distribution analysis completed")


def analyze_technical_indicators(df):
    technical_indicators = ['RSI', 'MACD', 'MA_5', 'MA_20', 'MA_50', 'Volatility_20d',
                            'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Position']

    available_indicators = [col for col in technical_indicators if col in df.columns]

    if available_indicators:
        n_indicators = len(available_indicators)
        cols = 3
        rows = (n_indicators + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
        fig.suptitle('Technical Indicators Distribution', fontsize=16, fontweight='bold')

        if rows == 1:
            axes = [axes] if n_indicators == 1 else axes
        else:
            axes = axes.flatten()

        for i, indicator in enumerate(available_indicators):
            if i < len(axes):
                data = df[indicator].dropna()
                if not data.empty:
                    axes[i].hist(data, bins=30, alpha=0.7, edgecolor='black')
                    axes[i].axvline(data.mean(), color='red', linestyle='--',
                                    label=f'Mean: {data.mean():.4f}')
                    axes[i].set_title(f'{indicator} Distribution')
                    axes[i].set_xlabel(indicator)
                    axes[i].set_ylabel('Frequency')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig('technical_indicators_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Technical indicators analysis completed")


def analyze_correlations(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    analysis_cols = [col for col in numeric_cols if not col.startswith('Ticker_')
                     and col not in ['Year', 'Month', 'DayOfYear', 'WeekOfYear', 'DayOfWeek', 'Quarter']]

    if len(analysis_cols) > 1:
        corr_matrix = df[analysis_cols].corr()

        plt.figure(figsize=(16, 14))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                    square=True, fmt='.2f', cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))

        if high_corr_pairs:
            print("High correlation pairs found:")
            for feat1, feat2, corr_val in high_corr_pairs:
                print(f"  {feat1} ↔ {feat2}: {corr_val:.3f}")

        if 'Target' in analysis_cols:
            target_corr = corr_matrix['Target'].drop('Target').abs().sort_values(ascending=False)
            print("Top features correlated with target:")
            for feature, corr_val in target_corr.head(10).items():
                original_corr = corr_matrix.loc[feature, 'Target']
                print(f"  {feature}: {original_corr:.4f}")

        print("Correlation analysis completed")


def analyze_time_series_patterns(df):
    if 'Date' not in df.columns or 'Close' not in df.columns:
        print("Missing Date or Close price data")
        return

    ticker_cols = [col for col in df.columns if col.startswith('Ticker_')]

    if ticker_cols:
        df_analysis = df.copy()
        df_analysis['Stock'] = 'Unknown'

        for ticker_col in ticker_cols:
            stock_name = ticker_col.replace('Ticker_', '')
            df_analysis.loc[df_analysis[ticker_col] == 1, 'Stock'] = stock_name

        plt.figure(figsize=(18, 12))

        plt.subplot(2, 2, 1)
        for stock in df_analysis['Stock'].unique():
            if stock != 'Unknown':
                stock_data = df_analysis[df_analysis['Stock'] == stock]
                plt.plot(stock_data['Date'], stock_data['Close'], label=stock, alpha=0.8)

        plt.title('Stock Close Price Time Series')
        plt.xlabel('Date')
        plt.ylabel('Close Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        if 'Daily_Return' in df.columns:
            plt.subplot(2, 2, 2)
            for stock in df_analysis['Stock'].unique():
                if stock != 'Unknown':
                    stock_data = df_analysis[df_analysis['Stock'] == stock]
                    stock_data_clean = stock_data.dropna(subset=['Daily_Return'])
                    if not stock_data_clean.empty:
                        plt.plot(stock_data_clean['Date'], stock_data_clean['Daily_Return'],
                                 label=stock, alpha=0.7)

            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            plt.title('Daily Returns Time Series')
            plt.xlabel('Date')
            plt.ylabel('Daily Return')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)

        if 'RSI' in df.columns:
            plt.subplot(2, 2, 3)
            for stock in df_analysis['Stock'].unique():
                if stock != 'Unknown':
                    stock_data = df_analysis[df_analysis['Stock'] == stock]
                    stock_data_clean = stock_data.dropna(subset=['RSI'])
                    if not stock_data_clean.empty:
                        plt.plot(stock_data_clean['Date'], stock_data_clean['RSI'],
                                 label=stock, alpha=0.7)

            plt.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought')
            plt.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold')
            plt.title('RSI Time Series')
            plt.xlabel('Date')
            plt.ylabel('RSI')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)

        if 'Volume' in df.columns:
            plt.subplot(2, 2, 4)
            for stock in df_analysis['Stock'].unique():
                if stock != 'Unknown':
                    stock_data = df_analysis[df_analysis['Stock'] == stock]
                    plt.plot(stock_data['Date'], stock_data['Volume'], label=stock, alpha=0.7)

            plt.title('Volume Time Series')
            plt.xlabel('Date')
            plt.ylabel('Volume')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig('time_series_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Time series analysis completed")


def analyze_fundamental_indicators(df):
    fundamental_cols = ['EPS', 'PE_Ratio', 'Market_Cap', 'Revenue', 'Operating_Cash_Flow',
                        'Free_Cash_Flow', 'Dividend_Yield']

    available_fundamental = [col for col in fundamental_cols if col in df.columns]

    if available_fundamental:
        cols = 3
        rows = (len(available_fundamental) + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
        fig.suptitle('Fundamental Indicators Distribution', fontsize=16, fontweight='bold')

        if rows == 1:
            axes = [axes] if len(available_fundamental) == 1 else axes
        else:
            axes = axes.flatten()

        for i, indicator in enumerate(available_fundamental):
            if i < len(axes):
                data = df[indicator].dropna()
                if not data.empty and data.std() > 0:
                    axes[i].hist(data, bins=30, alpha=0.7, edgecolor='black')
                    axes[i].axvline(data.mean(), color='red', linestyle='--',
                                    label=f'Mean: {data.mean():.2f}')
                    axes[i].set_title(f'{indicator} Distribution')
                    axes[i].set_xlabel(indicator)
                    axes[i].set_ylabel('Frequency')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig('fundamental_indicators_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Fundamental indicators analysis completed")


def generate_summary(df):
    print("\nEDA Summary Report:")
    print(f"Total records: {len(df):,}")
    print(f"Features: {df.shape[1]}")
    print(f"Time span: {(df['Date'].max() - df['Date'].min()).days} days")

    missing_pct = (df.isnull().sum().sum() / (len(df) * df.shape[1])) * 100
    print(f"Data completeness: {100 - missing_pct:.2f}%")

    if 'Target' in df.columns:
        target_balance = df['Target'].mean()
        print(f"Target balance (up ratio): {target_balance:.3f}")

    numeric_features = len(df.select_dtypes(include=[np.number]).columns)
    print(f"Numeric features: {numeric_features}")

    print("EDA analysis ready for Week 3 modeling")


def main():
    df = load_and_inspect_data()
    if df is None:
        return

    analyze_data_quality(df)
    analyze_price_distributions(df)
    analyze_technical_indicators(df)
    analyze_correlations(df)
    analyze_time_series_patterns(df)
    analyze_fundamental_indicators(df)
    generate_summary(df)

    print("\nGenerated files:")
    print("- price_distributions.png")
    print("- technical_indicators_distribution.png")
    print("- correlation_matrix.png")
    print("- time_series_analysis.png")
    print("- fundamental_indicators_distribution.png")


if __name__ == "__main__":
    main()