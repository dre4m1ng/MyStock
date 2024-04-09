# pages/dashboard.py

import streamlit as st
import yfinance as yf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import numpy as np
from datetime import date

def calculate_historical_volatility(data):
    log_returns = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    return log_returns.std() * np.sqrt(252)

def calculate_rsi(data, window=14):
    delta = data['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    RS = gain / loss
    return 100 - (100 / (1 + RS))

def perform_seasonal_decomposition(data, model='multiplicative'):
    """
    Dynamically perform seasonal decomposition on stock data.

    Args:
    - data: The time series data with 'Date' as the index and 'Adj Close' as the value.
    - model: The type of decomposition ('additive' or 'multiplicative').

    Returns:
    - Decomposition plot.
    """
    # Adjust period based on the dataset size
    period = max(2, len(data) // 2)  # Ensures at least 2 observations per cycle

    decomposition = seasonal_decompose(data['Adj Close'], model=model, period=period, extrapolate_trend='freq')
    fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    ax[0].plot(decomposition.observed, label='Observed')
    ax[0].set_ylabel('Observed')
    ax[0].legend()

    ax[1].plot(decomposition.trend, label='Trend')
    ax[1].set_ylabel('Trend')
    ax[1].legend()

    ax[2].plot(decomposition.seasonal, label='Seasonal')
    ax[2].set_ylabel('Seasonal')
    ax[2].legend()

    ax[3].plot(decomposition.resid, label='Residual')
    ax[3].set_ylabel('Residual')
    ax[3].legend()

    plt.tight_layout()
    return fig

def show_dashboard():
    st.title('Stock Analysis Dashboard')

    # Example for fetching and displaying stock data
    ticker = st.sidebar.text_input('Enter Stock Ticker', value='AAPL').upper()
    # User inputs for selecting the date range
    start_date = st.sidebar.date_input('Start Date', value=date(2021, 1, 1))
    end_date = st.sidebar.date_input('End Date', value=date.today())

    # Fetching data based on user input
    data = yf.download(ticker, start=start_date, end=end_date)

    if not data.empty:
        data.reset_index(inplace=True)  # Ensure that 'Date' is a column, not the index
        # Display some data
        st.write(f"Displaying data for: {ticker}")
        st.line_chart(data['Close'])

        # Display Historical Volatility
        hist_vol = calculate_historical_volatility(data)
        st.write(f"Historical Volatility: {hist_vol:.2%}")

         # Calculate RSI
        data['RSI'] = calculate_rsi(data)

        # Plot RSI with overbought and oversold lines
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data['Date'], data['RSI'], label='RSI')
        ax.axhline(70, linestyle='--', color='red', label='Overbought (70)')
        ax.axhline(30, linestyle='--', color='green', label='Oversold (30)')
        ax.set_ylabel('RSI')
        ax.set_title(f"RSI for {ticker}")
        ax.legend()
        st.pyplot(fig)

        # Plot ACF and PACF as an example
        daily_returns = data.set_index('Date')['Adj Close'].pct_change().dropna()

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_acf(daily_returns, ax=ax)
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_pacf(daily_returns, ax=ax)
        st.pyplot(fig)

        # Seasonal Decomposition
        st.write("Seasonal Decomposition Analysis")
        # Perform and display the dynamic seasonal decomposition
        fig = perform_seasonal_decomposition(data.set_index('Date'), model='multiplicative')
        st.pyplot(fig)

    else:
        st.write("No data available for this ticker.")
        