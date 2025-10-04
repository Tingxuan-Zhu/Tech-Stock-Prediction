import yfinance as yf
import pandas as pd

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "INTC", "IBM", "ORCL", "CSCO" ]

fundamental_data_list = []

for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        data = {
            "Ticker": ticker,
            'Sector': info.get('sector'),
            "EPS": info.get("trailingEps"),
            'PE_Ratio': info.get('trailingPE'),
            'Market_Cap': info.get('marketCap'),
            "Revenue": info.get("totalRevenue"),
            'Operating_Cash_Flow': info.get('operatingCashflow'),  # 经营现金流 (Operating Cash Flow)
            "Free_Cash_Flow": info.get("freeCashflow"),            # 自由现金流 (Free Cash Flow)
            'Dividend_Yield': info.get('dividendYield')
        }

        fundamental_data_list.append(data)
        print(f"successfully fetched {ticker}")
    except Exception as e:
        print(f"error fetching {ticker}")

fundamental_data = pd.DataFrame(fundamental_data_list)

print(fundamental_data.head())

fundamental_data.to_csv('fundamental_data.csv' , index=False)

print("print successfully fundamental data")




