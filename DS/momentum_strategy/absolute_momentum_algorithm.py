import pandas as pd 
import yfinance as yf

# set parameters
tickers = ["AAPL"]
lookback = 200
threshold = 0.05

# download data from Yahoo Finance API
data = yf.download(tickers, period="max")

# calculate returns and rolling standard deviation
returns = data["Adj Close"].pct_change()
std = returns.rolling(lookback).std()

# calculate the absolute momentum signal
signal = returns > threshold * std

# calculate positions based on the signal
positions = signal.astype(int).diff()

# calculate strategy returns
strategy_returns = positions.shift(1) * returns

# calculate cumulative returns
cumulative_returns = (1 + strategy_returns) .cumprod ()

# plot the cumulative returns
cumulative_returns.plot
