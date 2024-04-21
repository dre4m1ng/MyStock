# pages/dashboard.py

import streamlit as st
import yfinance as yf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
import plotly.graph_objects as go
import pandas as pd

# def plot_candlestick_chart(data):
#     """
#     Plots a candlestick chart for the given data.

#     Args:
#     - data (DataFrame): The stock price data.
#     """
#     fig = go.Figure(data=[go.Candlestick(x=data.index,
#                                          open=data['Open'],
#                                          high=data['High'],
#                                          low=data['Low'],
#                                          close=data['Close'])])
    
#     fig.update_layout(title='Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
#     fig.update_xaxes(type='category')
#     fig.update_layout(xaxis_rangeslider_visible=False)  # Hide the range slider
#     return fig


def calculate_support_resistance(data, window=20):
    """
    Calculate rolling maximum (resistance) and minimum (support) of the stock's price.
    
    Args:
    - data (DataFrame): The stock price data.
    - window (int): The rolling window size to calculate support and resistance levels.
    
    Returns:
    - data (DataFrame): The modified DataFrame with new columns for support and resistance.
    """
    data['Resistance'] = data['High'].rolling(window=window).max()
    data['Support'] = data['Low'].rolling(window=window).min()
    return data


def plot_candlestick_chart(data):
    """
    Plots a candlestick chart with support and resistance lines for the given data.

    Args:
    - data (DataFrame): The stock price data.
    """
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])

    # Add resistance line
    fig.add_trace(go.Scatter(x=data.index, y=data['Resistance'], name='Resistance', line=dict(color='red', width=1.5)))

    # Add support line
    fig.add_trace(go.Scatter(x=data.index, y=data['Support'], name='Support', line=dict(color='green', width=1.5)))

    fig.update_layout(title='Candlestick Chart with Support & Resistance', xaxis_title='Date', yaxis_title='Price')
    fig.update_xaxes(type='category')
    fig.update_layout(xaxis_rangeslider_visible=False)  # Hide the range slider
    return fig


def calculate_historical_volatility(data):
    log_returns = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    return log_returns.std() * np.sqrt(252)

def calculate_rsi(data, window=14):
    delta = data['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    RS = gain / loss
    return 100 - (100 / (1 + RS))

def plot_rsi_and_price(data, ticker):
    # Create subplots with shared x-axis for better comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot stock price on the first subplot
    ax1.plot(data['Date'], data['Adj Close'], label='Adj Close', color='blue')
    ax1.set_title(f"{ticker} Stock Price")
    ax1.set_ylabel('Price')
    ax1.legend(loc="upper left")
    
    # Plot RSI on the second subplot
    ax2.plot(data['Date'], data['RSI'], label='RSI', color='purple')
    ax2.axhline(70, linestyle='--', color='red', label='Overbought (70)')
    ax2.axhline(30, linestyle='--', color='green', label='Oversold (30)')
    ax2.set_title('Relative Strength Index (RSI)')
    ax2.set_ylabel('RSI')
    ax2.set_xlabel('Date')
    ax2.legend(loc="upper left")
    
    plt.tight_layout()  # Adjust layout to make room for the shared x-label
    st.pyplot(fig)


def calculate_bollinger_bands(data, window=20, num_of_std=2):
    """
    Calculate Bollinger Bands for the given data.

    Args:
    - data (DataFrame): The stock price data.
    - window (int): The moving average window size.
    - num_of_std (int): The number of standard deviations to use for the bands.

    Returns:
    - A DataFrame with Bollinger Band columns added.
    """
    # Calculate the moving average (middle band)
    data['middle_band'] = data['Adj Close'].rolling(window=window).mean()
    
    # Calculate the standard deviation
    std_dev = data['Adj Close'].rolling(window=window).std()
    
    # Calculate the upper and lower bands
    data['upper_band'] = data['middle_band'] + (std_dev * num_of_std)
    data['lower_band'] = data['middle_band'] - (std_dev * num_of_std)
    
    return data

def plot_bollinger_bands(data):
    """
    Plot the Bollinger Bands along with the stock's adjusted closing price.

    Args:
    - data (DataFrame): The stock price data with Bollinger Bands calculated.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['Date'], data['Adj Close'], label='Adj Close', color='blue')
    ax.plot(data['Date'], data['middle_band'], label='Middle Band (SMA)', color='orange')
    ax.plot(data['Date'], data['upper_band'], label='Upper Band', linestyle='--', color='green')
    ax.plot(data['Date'], data['lower_band'], label='Lower Band', linestyle='--', color='red')
    
    ax.fill_between(data['Date'], data['upper_band'], data['lower_band'], color='grey', alpha=0.1)
    
    ax.set_title('Bollinger Bands')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    
    st.pyplot(fig)


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

def get_watchlist_data(watchlist, start_date, end_date):
    """Fetch and store closing prices for each ticker in the watchlist."""
    watchlist_data = pd.DataFrame()
    
    for ticker in watchlist:
        ticker_data = yf.download(ticker, start=start_date, end=end_date)
        if not ticker_data.empty:
            # Use 'Close' for actual closing prices; adjust if needed
            watchlist_data[ticker] = ticker_data['Close']
            
    return watchlist_data


def plot_watchlist_comparison(watchlist_data):
    """Plot an overlaid line chart for the watchlist stocks."""
    fig = go.Figure()
    
    for ticker in watchlist_data.columns:
        fig.add_trace(go.Scatter(x=watchlist_data.index, y=watchlist_data[ticker], mode='lines', name=ticker))
        
    fig.update_layout(title='Watchlist Comparison', xaxis_title='Date', yaxis_title='Price', legend_title='Ticker')
    return fig

def show_dashboard():
    st.title('Stock Analysis Dashboard')

    ticker = st.sidebar.text_input('Enter Stock Ticker', value='AAPL').upper()
    start_date = st.sidebar.date_input('Start Date', value=date(2021, 1, 1))
    end_date = st.sidebar.date_input('End Date', value=date.today())

    data = yf.download(ticker, start=start_date, end=end_date)

    if not data.empty:
        data.reset_index(inplace=True)  # Ensure that 'Date' is a column

        # Tabs for different analyses including the watchlist
        tab1, tab2, tab3, tab4, tab5, watchlist_tab = st.tabs([
        "Overview", 
        "Candlestick Chart", 
        "Bollinger Bands", 
        "Price & RSI Analysis", 
        "Seasonal Decomposition", 
        "Watchlist"
        ])

        with tab1:
            st.write(f"Displaying data for: {ticker}")
            st.line_chart(data['Close'])

        with tab2:
            # Calculate Support and Resistance
            data_with_levels = calculate_support_resistance(data)
            # Display Candlestick Chart with Support and Resistance
            st.plotly_chart(plot_candlestick_chart(data_with_levels.set_index('Date')), use_container_width=True)

        with tab3:
            # Calculate and Plot Bollinger Bands
            data_with_bands = calculate_bollinger_bands(data)
            plot_bollinger_bands(data_with_bands)
        
        with tab4:
            # Display Historical Volatility
            hist_vol = calculate_historical_volatility(data)
            st.write(f"Historical Volatility: {hist_vol:.2%}")

            # Calculate RSI
            data['RSI'] = calculate_rsi(data)
            
            # Plot RSI and stock price for comparison
            plot_rsi_and_price(data, ticker)  # Assuming you've implemented a function like the one suggested earlier

        with tab5:
            # Seasonal Decomposition
            st.write("Seasonal Decomposition Analysis")
            # Ensure data is prepared with 'Date' as index if needed for seasonal_decompose
            data_for_decomp = data.set_index('Date') if 'Date' in data else data
            if 'Adj Close' in data_for_decomp:
                # Perform and display the dynamic seasonal decomposition
                fig = perform_seasonal_decomposition(data_for_decomp, model='multiplicative')
                st.pyplot(fig)
            else:
                st.write("Seasonal decomposition requires 'Adj Close' in the dataset.")            
        # Consider placing additional or less frequently accessed analyses in expanders or additional tabs
        with watchlist_tab:
            watchlist_input = st.text_area("Enter stock tickers separated by commas (e.g., AAPL,MSFT,GOOGL)", value="AAPL,MSFT")
            watchlist = [ticker.strip().upper() for ticker in watchlist_input.split(',')]
            
            # Fetch and plot data for the watchlist
            watchlist_data = get_watchlist_data(watchlist, start_date, end_date)
            if not watchlist_data.empty:
                st.plotly_chart(plot_watchlist_comparison(watchlist_data), use_container_width=True)
            else:
                st.error("No data available for the specified tickers or date range.")

    else:
        st.write("No data available for this ticker.")
