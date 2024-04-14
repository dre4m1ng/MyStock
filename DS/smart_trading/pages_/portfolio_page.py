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

# Define a simple moving average strategy
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

# Backtest the strategy
def backtest(data):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaCross)
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)
    cerebro.broker.set_cash(10000)
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
    results = cerebro.run()
    return results, cerebro

# Streamlit app for portfolio analysis
def portfolio_page():
    st.title("Stock Portfolio and Testing")

    stock = st.text_input("Enter Stock Ticker", "AAPL")
    start_date = st.date_input("Start Date", pd.Timestamp("2020-01-01"))
    end_date = st.date_input("End Date", pd.Timestamp("2022-01-01"))

    if st.button("Run Analysis"):
        try:
            data = download_data(stock, start_date, end_date)
            results, cerebro = backtest(data)
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
            components.html(source_code, height=800)

            st.header("Equity Curve")
            fig = cerebro.plot()[0][0].get_figure()
            st.pyplot(fig)
        
        except Exception as e:
            st.error(f"An error occurred: {e}")