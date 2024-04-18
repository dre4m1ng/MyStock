import yfinance as yf
import backtrader as bt
import pandas as pd
import quantstats as qs
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

# Ensure matplotlib uses the right backend for Streamlit
plt.switch_backend('Agg')

# Download historical data for a stock
def download_data(stock, start, end):
    data = yf.download(stock, start=start, end=end)
    if data.empty:
        raise ValueError(f"No data returned for {stock}. Check the ticker symbol and date range.")
    data.columns = [col.lower().replace(' ', '_') for col in data.columns]
    return data

# Define the SMA Cross strategy
class SmaCross(bt.Strategy):
    params = (('sma1', 50), ('sma2', 200),)
    def __init__(self):
        sma1 = bt.ind.SMA(period=self.params.sma1)
        sma2 = bt.ind.SMA(period=self.params.sma2)
        self.crossover = bt.ind.CrossOver(sma1, sma2)
    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            self.sell()

# Define the EMA Trend strategy
class EmaTrend(bt.Strategy):
    params = (('ema_period', 50),)
    def __init__(self):
        ema = bt.ind.EMA(period=self.params.ema_period)
        self.trend = bt.ind.CrossOver(self.data.close, ema)
    def next(self):
        if not self.position:
            if self.trend > 0:
                self.buy()
        elif self.trend < 0:
            self.sell()

# Define the SMA Mean Reversion strategy
class SmaMeanReversion(bt.Strategy):
    params = (('sma_period', 30), ('devfactor', 2.0),)
    def __init__(self):
        sma = bt.ind.SMA(period=self.params.sma_period)
        deviation = bt.ind.StdDev(self.data.close, period=self.params.sma_period) * self.params.devfactor
        self.upper_band = sma + deviation
        self.lower_band = sma - deviation
    def next(self):
        if not self.position:
            if self.data.close[0] < self.lower_band[0]:
                self.buy()
        elif self.data.close[0] > self.upper_band[0]:
            self.sell()

# Define new short-term trading strategies
class ScalpingStrategy(bt.Strategy):
    params = (('sma_period', 10), ('distance', 0.0025),)
    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(period=self.params.sma_period)
    def next(self):
        if not self.position:
            if self.data.close[0] > self.sma[0] + (self.sma[0] * self.params.distance):
                self.buy()
        elif self.data.close[0] < self.sma[0] - (self.sma[0] * self.params.distance):
            self.sell()

class NewsImpactMomentumStrategy(bt.Strategy):
    params = (('momentum_period', 5), ('volume_multiplier', 1.5),)
    def __init__(self):
        self.momentum = bt.indicators.RateOfChange(period=self.params.momentum_period)
        self.average_volume = bt.indicators.SimpleMovingAverage(self.data.volume, period=self.params.momentum_period)
    def next(self):
        if self.data.volume[0] > self.average_volume[0] * self.params.volume_multiplier:
            if self.momentum[0] > 0:
                self.buy()
            elif self.momentum[0] < 0:
                self.sell()

# Backtest function with enhanced error handling
def backtest(data, strategy):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy)
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)
    cerebro.broker.set_cash(10000)
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
    
    try:
        results = cerebro.run()
        return results, cerebro, strategy.__name__
    except Exception as e:
        return None, f"Error during backtesting: {str(e)}"

# Streamlit app for strategy selection and analysis
def portfolio_page():
    st.title("Stock Portfolio and Testing")

    strategies = {
        "SMA Crossover": SmaCross,
        "EMA Trend Following": EmaTrend,
        "SMA Mean Reversion": SmaMeanReversion,
        "Scalping Strategy": ScalpingStrategy,
        "News Impact Momentum": NewsImpactMomentumStrategy
    }

    stock = st.text_input("Enter Stock Ticker", "AAPL")
    start_date = st.date_input("Start Date", pd.Timestamp("2020-01-01"))
    end_date = st.date_input("End Date", pd.Timestamp("2022-01-01"))
    selected_strategy = st.selectbox("Select Strategy", list(strategies.keys()))

    if st.button("Run Analysis"):
        try:
            data = download_data(stock, start_date, end_date)
            results, cerebro, strategy_name = backtest(data, strategies[selected_strategy])
            strat = results[0]
            portfolio_stats = strat.analyzers.getbyname('pyfolio')
            returns, positions, transactions, gross_lev = portfolio_stats.get_pf_items()

            if returns.empty:
                st.error("No returns data to analyze. Check if any trades were made.")
                return

            st.header("QuantStats Analysis")
            qs.reports.html(returns, output='report.html')
            HtmlFile = open("report.html", 'r', encoding='utf-8')
            source_code = HtmlFile.read() 
            components.html(source_code, height = 800)

            st.header("Equity Curve")

            try:
                # Conditional plotting based on strategy
                if strategy_name != "NewsImpactMomentumStrategy":
                    fig = cerebro.plot()[0][0].get_figure()
                    st.pyplot(fig)
                else:
                    st.write("Plotting skipped for News Impact Momentum Strategy due to complexity.")
                
                # fig = cerebro.plot()[0][0].get_figure()
                # # fig = cerebro.plot(style='bar', volume=False, subplot=False)[0][0].get_figure()
                # st.pyplot(fig)
            except Exception as e:
                st.error(f"Failed to plot due to: {str(e)}")            
            # fig = cerebro.plot()[0][0].get_figure()
            # st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            