import streamlit as st
# from streamlit import experimental_pages as pages

# Import pages
from pages.forecast import show_forecast_page
# from pages.dashboard import show_dashboard
from pages.causal_inference import show_causal_inference
# from pages.trading import show_trading

# Page configuration
st.set_page_config(page_title="Trading Automation", layout="wide")

pages_ = {
    "Stock Forecast": show_forecast_page,
    # "Dashboard": show_dashboard_page,
    "Causal Inference": show_causal_inference,
    # "Trading": show_trading_page
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(pages_.keys()))

page = pages_[selection]
page()