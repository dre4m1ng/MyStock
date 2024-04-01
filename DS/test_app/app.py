import streamlit as st
import pandas as pd
import yaml
from your_data_module import get_current_price  # This assumes you've refactored your code

# Load configuration
def load_config():
    with open('./config/config.yaml', encoding='UTF-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

cfg = load_config()

# Streamlit UI
st.title('Trading Bot Interface')

code = st.text_input('Enter Stock Code', '')

if st.button('Fetch Current Price'):
    access_token = get_access_token(cfg)  # Assuming you've adapted this function to accept config
    current_price = get_current_price(code, cfg, access_token)  # Adapt this function as well
    st.write(current_price)
