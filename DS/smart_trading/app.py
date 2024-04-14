import streamlit as st
# from streamlit import experimental_pages as pages

# Import pages
from pages_.forecast import show_forecast_page
from pages_.dashboard import show_dashboard
from pages_.causal_inference import show_causal_inference
from pages_.portfolio_page import portfolio_page
# from pages.trading import show_trading

# Page configuration
st.set_page_config(page_title="Trading Automation", layout="wide")

pages = {
    "Dashboard": show_dashboard,
    "Stock Forecast": show_forecast_page,
    "Causal Inference": show_causal_inference,
    "Trading": portfolio_page,
    # "Trading": show_trading_page
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))

page = pages[selection]
page()