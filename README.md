# Tech-Stock-Prediction

AI-Driven Stock Price Prediction & Trading Strategy using Machine Learning

## Project Overview

This project applies machine learning to predict stock price movements and implement automated trading strategies. We analyzed 11 major technology stocks over one year, achieving **73.4% prediction accuracy** and **40% returns** with significantly lower risk than buy-and-hold investing.

**Stocks Analyzed:** AAPL, MSFT, GOOGL, AMZN, META, NVDA, AMD, INTC, IBM, ORCL, CSCO  
**Period:** October 2024 - September 2025  
**Total Data Points:** ~2,750 daily observations

---

## Key Results

| Metric | ML Strategy | Buy & Hold | Improvement |
|--------|-------------|------------|-------------|
| **Total Return** | 40.0% | 28.0% | +12.0% |
| **Sharpe Ratio** | 2.63 | 1.35 | +94.8% |
| **Max Drawdown** | -9.2% | -31.9% | -71.2% |
| **Prediction Accuracy** | 73.4% | - | - |

---

## Installation

```bash
git clone https://github.com/Tingxuan-Zhu/Tech-Stock-Prediction.git
cd Tech-Stock-Prediction
pip install -r requirements.txt
```

---

## Usage

Run the pipeline in order:

```bash
# Step 1: Data Collection
python "historical data.py"
python "fundamental info for stock.py"

# Step 2: Data Preprocessing
python "Clean and preprocess the data.py"

# Step 3: Feature Engineering
python "feature engineering for history_stock_data.py"
python "feature_engineering_for_fundamental_data.py"
python "merge_fundamental & historical data.py"

# Step 4: Analysis & Modeling
python “EDA_anaysis.py”
python "Machine Learning.py"

# Step 5: Backtesting
python "Backtesting.py"
```

---

## Project Structure

```
Tech-Stock-Prediction/
│
├── historical data.py                          # Yahoo Finance API data extraction
├── fundamental info for stock.py               # Company fundamentals collection
├── Clean and preprocess the data.py            # Data cleaning pipeline
├── feature engineering for history_stock_data.py   # Technical indicators
├── feature_engineering_for_fundamental_data.py     # Fundamental features
├── merge_fundamental & historical data.py      # Data integration
├── EDA_anaysis.py                              # Exploratory data analysis
├── Machine Learning.py                         # Model training & evaluation
├── Backtesting.py                              # Trading strategy backtesting
├── requirements.txt                            # Python dependencies
└── README.md                                   # This file
```

---

## Features

### Technical Indicators (25+ features)

- **Trend:** MA_5, MA_20, MA_50, MACD
- **Momentum:** RSI, Price_Change_5d, Price_Change_20d
- **Volatility:** Volatility_20d, Bollinger Bands (Upper, Lower, Width, Position)
- **Lag Features:** Lag_1d_Close, Lag_5d_Close

### Fundamental Features (40+ features)

- **Valuation Ratios:** PE_Ratio, PS_Ratio, PCF_Ratio
- **Financial Health:** FCF_Yield, Cash_Flow_Quality, ROE_Proxy
- **Sector Comparison:** Z-scores, sector mean/median deviations
- **Composite Metrics:** Value_Score

---

## Machine Learning Models

### Models Tested

**Classification (Up/Down Prediction):**
- Logistic Regression
- Decision Tree
- Random Forest
- **Gradient Boosting** (Best performer)
- Neural Network (MLP)

**Regression (Return Prediction):**
- Linear/Ridge/Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Neural Network (MLP Regressor)

### Best Model Performance

```
Gradient Boosting Classifier (Test Set):
├── Accuracy:  73.4%
├── Precision: 80.6%
├── Recall:    80.6%
└── F1-Score:  0.806
```

---

## Trading Strategies

### Strategy 1: ML Strategy
Buy when the model predicts up, hold cash when predicts down.

**Results:** 40.0% return | Sharpe 2.63 | Max Drawdown -9.2%

### Strategy 2: Buy & Hold (Baseline)
Simple passive strategy - always hold.

**Results:** 28.0% return | Sharpe 1.35 | Max Drawdown -31.9%

### Strategy 3: Threshold Strategy
Only trade when prediction confidence > 60%.

**Results:** 41.0% return | Sharpe 2.65 | Max Drawdown -8.8%

---

## Sample Visualizations

The pipeline generates:
- Price distribution analysis
- Technical indicator distributions
- Feature correlation heatmap
- Time series patterns
- Confusion matrix
- Feature importance ranking
- Backtest performance comparison

---

## Key Findings

1. **DayOfYear** is the most important feature (0.25 importance) - strong seasonal patterns
2. Technical indicators significantly outperform fundamentals for daily prediction
3. Selective trading (high-confidence only) beats constant trading
4. ML strategies achieve 2.6x better Sharpe Ratio than buy-and-hold
5. 71% reduction in maximum drawdown compared to passive investing

---

## Limitations

- **Data:** Only 1 year of historical data
- **Market Cycle:** Test period was bullish; bear market performance unknown
- **Costs:** Transaction costs and slippage are not modeled
- **Features:** Missing sentiment analysis, macroeconomic indicators, options data

---

## Future Improvements

- Extend to 5-10 years, covering multiple market cycles
- Add sentiment analysis from news and social media
- Include macroeconomic indicators (GDP, inflation, Fed rates)
- Implement LSTM/Transformer architectures for time series
- Build a real-time trading pipeline with automated execution
- Add a comprehensive risk management system

---

## Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
yfinance>=0.1.70
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
ta>=0.10.0
```


---

## Disclaimer

**This project is for display purposes only.** This is not financial advice. Trading involves substantial risk of loss. Past performance does not guarantee future results. The strategies presented should not be used for actual trading without proper risk assessment. Always consult a qualified financial advisor before making investment decisions.

---

## Author

**Tingxuan Zhu**

---

 **Star this repository if you find it helpful!**

