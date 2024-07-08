import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from ta.trend import MACD
from ta.momentum import RSIIndicator

# FMP API setup
base_url = 'https://financialmodelingprep.com/api/v3'
API_KEY = 'u0RL9NZ8J79VOyjijLJDfezpmaDqG1AD'  # Replace with your actual API key

st.title('Financial Analysis Dashboard with Stock Search')

# Search functionality
search_term = st.text_input('Search for a stock (e.g. AAPL, MSFT):')

@st.cache_data
def search_stocks(query):
    url = f"{base_url}/search?query={query}&limit=10&apikey={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        st.error(f"Failed to fetch search results: {response.status_code} - {response.text}")
        return pd.DataFrame()

if search_term:
    search_results = search_stocks(search_term)
    if not search_results.empty:
        st.write("Search Results:")
        selected_stock = st.selectbox("Select a stock:", search_results['symbol'].tolist())
    else:
        st.write("No results found.")
        selected_stock = None
else:
    selected_stock = None

# Main dashboard functionality
if selected_stock:
    st.header(f"Analysis for {selected_stock}")

    # Fetch historical data
    @st.cache_data
    def fetch_historical_data(symbol):
        url = f"{base_url}/historical-price-full/{symbol}?apikey={API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()['historical']
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            return df
        else:
            st.error(f"Failed to fetch data for {symbol}: {response.status_code} - {response.text}")
            return None

    df = fetch_historical_data(selected_stock)

    if df is not None:
        # Add technical indicators
        macd = MACD(close=df['close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        rsi = RSIIndicator(close=df['close'])
        df['RSI'] = rsi.rsi()

        # Create price chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df['date'],
                                     open=df['open'],
                                     high=df['high'],
                                     low=df['low'],
                                     close=df['close'],
                                     name='Price'))
        fig.add_trace(go.Scatter(x=df['date'], y=df['MACD'], name='MACD'))
        fig.add_trace(go.Scatter(x=df['date'], y=df['MACD_signal'], name='MACD Signal'))
        fig.add_trace(go.Scatter(x=df['date'], y=df['RSI'], name='RSI', yaxis='y2'))

        fig.update_layout(
            title=f'{selected_stock} Stock Price and Indicators',
            yaxis_title='Price',
            yaxis2=dict(title='RSI', overlaying='y', side='right', range=[0, 100]),
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(fig)

        # Display recent data
        st.subheader("Recent Data")
        st.dataframe(df.tail().style.highlight_max(axis=0))

        # Basic statistics
        st.subheader("Basic Statistics")
        st.write(df['close'].describe())

        # Additional analysis can be added here

else:
    st.write("Enter a stock symbol in the search box to begin analysis.")

# You can add more sections and analyses here as needed