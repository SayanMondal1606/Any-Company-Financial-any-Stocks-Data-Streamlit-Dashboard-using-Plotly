import Streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

base_url = 'https://financialmodelingprep.com/api/v3'
API_KEY = 'r55UkdaTRIcKzrByVEaAkPJb6lsCXFWh'

st.title('Financial Modeling Prep Stock Screener Statements')
st.markdown('Financial Modeling Prep Stock Screener Statements')

symbol = st.sidebar.text_input('Ticker', value='MSFT')
financial_data = st.sidebar.selectbox('Financial Data Type', options=(
    'income-statement', 'balance-sheet-statement', 'cash-flow-statement', 'income-statement-growth',
    'balance-sheet-statement-growth', 'cash-flow-statement-growth', 'ratio-ttm', 'ratio', 'financial-growth', 'quote', 'rating',
    'enterprise-value', 'key-metrics-ttm', 'key-metrics', 'historical-rating', 'discounted-cash-flow',
    'historical-discounted-cash-flow-statement', 'historical-price-fall', 'Historical Price smaller intervals'
))

if financial_data == 'Historical Price smaller intervals':
    intervals = st.sidebar.selectbox('Interval', options=('1min', '5min', '15min', '30min', '1hour', '4hour'))
    financial_data = f'historical-chart/{intervals}'

transpose = st.sidebar.selectbox('Transpose', options=('yes', 'no'))
url = f'{base_url}/{financial_data}/{symbol}?apikey={API_KEY}'

response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    if transpose == 'yes':
        df = pd.DataFrame(data).transpose()
    else:
        df = pd.DataFrame(data)
    st.write(df)
else:
    st.error(f"Failed to fetch data: {response.status_code} - {response.text}")
