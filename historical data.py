import yfinance as yf
import pandas as pd

tickers = ["AAPL",  "MSFT",   "GOOGL",  "AMZN", "META", "NVDA", "AMD", "INTC", "IBM",  "ORCL",  "CSCO" ]

hist_data_wide = yf.download(tickers , period = "1y", interval="1d")

print("Data downloaded")
print(hist_data_wide.head())

# This form is not great. I need to change it.
hist_data = hist_data_wide.stack(level = 1).reset_index()
hist_data = hist_data.rename(columns = {"level_1":"Ticker"})
hist_data = hist_data.sort_values(by = ["Ticker", "Date"])

hist_data = hist_data.reset_index(drop=True)
hist_data['Date'] = pd.to_datetime(hist_data['Date'])

print("Form changed successfully")
print(hist_data.head())
print(f"Total rows: {len(hist_data)}")

print("Form changed successfully")
print(hist_data.head())
hist_data.to_csv('historical_stock_data.csv', index=False)
