import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import traceback

class TradingBacktest:       #Initial_captial
    def __init__(self, initial_capital = 100000):
        self.initial_capital = initial_capital
        self.results = {}

    def load_data_and_predictions(self):        # Load data and model predictions
        try:
            df = pd.read_csv('final_data.csv', parse_dates=['Date'])
            print("Data loaded")

            test_data = df[df['Split'] == 'test'].copy()
            return test_data
        except FileNotFoundError:
            print("File not found")
            return None

    def generate_predictions(self, test_data, model, scaler):   # Generate predictions using trad=ined model
        exclude_cols = ['Date', 'Target', 'Tomorrow_Close', 'Tomorrow_Return', 'Split', 'Sector']
        ticker_cols = [col for col in test_data.columns if col.startswith('Ticker_')]
        exclude_cols.extend(ticker_cols)

        feature_cols = [col for col in test_data.columns if col not in exclude_cols]
        X_test = test_data[feature_cols]

        # Standarlized the feature
        X_test_scaled = scaler.transform(X_test)

        # Get the prediction
        predicitons = model.predict(X_test_scaled)

        # Get probability
        if hasattr(model, 'predict_proba'):
            pred_proba = model.predict_proba(X_test_scaled)
            test_data['Prediction_Proba'] = pred_proba[:, 1]

        test_data['Prediction'] = predicitons

        return test_data

    def strategy_ml_signals(self,data):     # Strat 1: ML-based trading signals
        signals = data.copy()

        signals['Position'] = signals['Prediction']

        signals['Strategy_Return'] = signals['Position'].shift(1) * signals['Tomorrow_Return']

        return signals

    def strategy_buy_and_hold(self, data):      # Buy and hold baseline
        signals = data.copy()
        signals['Position'] = 1
        signals['Strategy_Return'] = signals['Tomorrow_Return']

        return signals

    def strategy_threshold_based(self, data, threshold = 0.6):      # Threshold-based strategy
        signals = data.copy()

        if 'Prediction_Proba' in signals.columns:
            signals['Position'] = (signals['Prediction_Proba'] > threshold).astype(int)
        else:
            signals['Position'] = signals['Prediction']

        signals['Strategy_Return'] = signals['Position'].shift(1) * signals['Tomorrow_Return']

        return signals

    def calculate_performance_metrics(self, signals, strategy_name):    # Calculate strategy performance metrics
        signals = signals.dropna(subset=['Strategy_Return'])

        if len(signals) == 0:
            print("NO SIGNALS FOUND")
            return None

        # Cumulative Returns
        signals["Cumulative_Return"] = (1 + signals['Strategy_Return']).cumprod()
        total_return = signals['Cumulative_Return'].iloc[-1]-1

        # Annualized Return
        n_days = len(signals)
        n_years = n_days / 252
        annualized_return = (1 + total_return) ** (1 / n_years) -1

        # Volatility
        volatility = signals["Strategy_Return"].std() * np.sqrt(252)

        # Sharpe Ratio
        risk_free_rate = 0.0453
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0

        # Maximum Drawndown
        cumulative  = signals["Cumulative_Return"]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        max_drawdown = drawdown.min()

        # Win Rate
        win_rate = (signals['Strategy_Return'] > 0).sum() / len(signals)

        # Number of Trades
        if 'Position' in signals.columns:
            n_trades = (signals['Position'].diff() != 0).sum()
        else:
            n_trades = 0

        metrics = {
            'Strategy': strategy_name,
            'Total_Return': total_return,
            'Annualized_Return': annualized_return,
            'Volatility': volatility,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown,
            'Win_Rate': win_rate,
            'Number_of_Trades': n_trades,
            'Final_Portfolio_Value': self.initial_capital * (1 + total_return)
        }
        return metrics, signals

    def run_backtest(self, test_data, model = None, scaler = None):     # Run complete backtest
        print("Running Backtest")

        if model is not None and scaler is not None:
            test_data = self.generate_predictions(test_data, model, scaler)

        if 'Prediction' not in test_data.columns:
            print("No prediction found")
            return

        all_results = []
        all_signals = {}

        # Strat1: ML signal
        print("Strategy 1: Machine Learning signal")
        signal_ml = self.strategy_ml_signals(test_data)
        metrics_ml, signals_ml = self.calculate_performance_metrics(signal_ml, "ML Strategy")
        all_results.append(metrics_ml)
        all_signals['ML Strategy'] = signal_ml

        # Strat 2: Buy and hold
        print("Strategy 2: Buy and Hold")
        signals_bh = self.strategy_buy_and_hold(test_data)
        metrics_bh, signals_bh = self.calculate_performance_metrics(signals_bh, "Buy & Hold")
        all_results.append(metrics_bh)
        all_signals['Buy & Hold'] = signals_bh

        # Strat 3: Threshold and probability
        print("Strategy 3: Threshold Based")
        signals_th = self.strategy_threshold_based(test_data, threshold = 0.6)
        metrics_th, signals_th = self.calculate_performance_metrics(signals_th, "Threshold Strategy")
        all_results.append(metrics_th)
        all_signals['Threshold Strategy'] = signals_th

        self.results = pd.DataFrame(all_results)
        self.signals = all_signals

        return self.results, all_signals

    def plot_results(self):     # Visualize backtest results
        if not hasattr(self, 'signals'):
            print("No results found")
            return

        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)        #Cumulative Returns Comparison
        for strategy_name, signals in self.signals.items():
            if 'Cumulative_Return' in signals.columns:
                plt.plot(signals['Date'], signals['Cumulative_Return'], label = strategy_name, linewidth = 2)

        plt.title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')

        plt.subplot(2, 2, 2)        #Drawdown Analysis
        for strategy_name, signals in self.signals.items():
            if 'Cumulative_Return' in signals.columns:
                cumulative = signals['Cumulative_Return']
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                plt.plot(signals['Date'], drawdown, label = strategy_name, linewidth = 2)

        plt.title('Drawdown Analysis', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plt.subplot(2, 2, 3)        #Daily Returns Distribution
        for strategy_name, signals in self.signals.items():
            if 'Strategy_Return' in signals.columns:
                plt.hist(signals['Strategy_Return'].dropna(), bins = 50, alpha = 0.5, label = strategy_name)

        plt.title('Daily Returns Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Daily Return')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)


        plt.subplot(2, 2, 4)        # Key Metrics Comparison
        metrics_to_plot = ['Total_Return', 'Sharpe_Ratio', 'Win_Rate']
        x = np.arange(len(self.results))
        width = 0.25

        for i, metric in enumerate(metrics_to_plot):
            plt.bar(x + i * width, self.results[metric], width, label=metric)

        plt.xlabel('Strategy')
        plt.ylabel('Metric Value')
        plt.title('Key Metrics Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x + width, self.results['Strategy'], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("backtest_results.png has been saved")

    def generate_report(self):
        print("BACKTESTING RESULTS REPORT")

        print(f"\nInitial Capital: ${self.initial_capital:,.2f}")
        print(f"Backtest Period: {len(self.signals['Buy & Hold'])} days")

        print("\nStrategy Performance Comparison:")
        print("-" * 60)

        for _, row in self.results.iterrows():
            print(f"\nStrategy: {row['Strategy']}")
            print(f"  Total Return: {row['Total_Return'] * 100:.2f}%")
            print(f"  Annualized Return: {row['Annualized_Return'] * 100:.2f}%")
            print(f"  Volatility: {row['Volatility'] * 100:.2f}%")
            print(f"  Sharpe Ratio: {row['Sharpe_Ratio']:.3f}")
            print(f"  Max Drawdown: {row['Max_Drawdown'] * 100:.2f}%")
            print(f"  Win Rate: {row['Win_Rate'] * 100:.2f}%")
            print(f"  Number of Trades: {int(row['Number_of_Trades'])}")
            print(f"  Final Portfolio: ${row['Final_Portfolio_Value']:,.2f}")

        # 找出最佳策略
        best_strategy = self.results.loc[self.results['Sharpe_Ratio'].idxmax()]
        print("\n" + "=" * 60)
        print(f"Best Strategy: {best_strategy['Strategy']}")
        print(f"Sharpe Ratio: {best_strategy['Sharpe_Ratio']:.3f}")
        print("=" * 60)

        # 保存结果到CSV
        self.results.to_csv('backtest_performance_metrics.csv', index=False)
        print("\n backtest_performance_metrics.csv has been saved")

def main():     # Main Fuction: Run complete backtesting workflow
    backtest = TradingBacktest(initial_capital = 100000)
    test_data = backtest.load_data_and_predictions()

    if test_data is None:
        return

    if 'Target' in test_data.columns and 'Prediction' not in test_data.columns:
        test_data['Prediction'] = test_data['Target']

    results, signals = backtest.run_backtest(test_data)

    backtest.plot_results()
    backtest.generate_report()

    print("Backtest Complete")

if __name__ == "__main__":
    main()
