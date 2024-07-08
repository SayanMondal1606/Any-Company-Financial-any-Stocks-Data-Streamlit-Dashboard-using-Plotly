import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import numpy as np

base_url = 'https://financialmodelingprep.com/api/v3'
API_KEY = 'r55UkdaTRIcKzrByVEaAkPJb6lsCXFWh'

st.title('Advanced Financial Analysis Dashboard')
st.markdown('Comprehensive Financial Data Visualization and Analysis')

# Main sidebar for data selection
with st.sidebar:
    st.header("Data Selection")
    symbol = st.text_input('Ticker', value='MSFT', key='ticker_input')
    financial_data = st.selectbox('Financial Data Type', 
        options=(
            'income-statement', 'balance-sheet-statement', 'cash-flow-statement', 'income-statement-growth',
            'balance-sheet-statement-growth', 'cash-flow-statement-growth', 'ratio-ttm', 'ratio', 'financial-growth', 'quote', 'rating',
            'enterprise-value', 'key-metrics-ttm', 'key-metrics', 'historical-rating', 'discounted-cash-flow',
            'historical-discounted-cash-flow-statement', 'historical-price-fall', 'Historical Price smaller intervals'
        ),
        key='financial_data_select'
    )

    if financial_data == 'Historical Price smaller intervals':
        intervals = st.selectbox('Interval', 
            options=('1min', '5min', '15min', '30min', '1hour', '4hour'),
            key='interval_select'
        )
        financial_data = f'historical-chart/{intervals}'

# Fetch data
url = f'{base_url}/{financial_data}/{symbol}?apikey={API_KEY}'
response = requests.get(url)

def technical_analysis(df, column):
    """Perform technical analysis on the given column."""
    analysis = []
    values = df[column].astype(float)
    
    # Trend analysis
    slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(values)), values)
    trend = "upward" if slope > 0 else "downward"
    analysis.append(f"The overall trend is {trend} with a slope of {slope:.2f}.")
    
    # Volatility
    volatility = values.pct_change().std()
    analysis.append(f"Volatility: {volatility:.2%}")
    
    # Moving averages
    ma_short = values.rolling(window=5).mean().iloc[-1]
    ma_long = values.rolling(window=20).mean().iloc[-1]
    if ma_short > ma_long:
        analysis.append("Short-term average is above long-term average, suggesting a potential bullish trend.")
    else:
        analysis.append("Short-term average is below long-term average, suggesting a potential bearish trend.")
    
    # Recent performance
    recent_change = (values.iloc[-1] - values.iloc[0]) / values.iloc[0]
    analysis.append(f"Recent performance change: {recent_change:.2%}")
    
    return "\n".join(analysis)

if response.status_code == 200:
    data = response.json()
    df = pd.DataFrame(data)
    
    # Display data table
    st.subheader("Data Table")
    st.dataframe(df)
    
    # Function to create graphs with technical analysis
    def create_graph_with_analysis(df, column):
        if pd.api.types.is_numeric_dtype(df[column]) and not df[column].isnull().all():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', name=column))
            
            # Add moving averages
            fig.add_trace(go.Scatter(x=df.index, y=df[column].rolling(window=5).mean(), mode='lines', name='5-period MA'))
            fig.add_trace(go.Scatter(x=df.index, y=df[column].rolling(window=20).mean(), mode='lines', name='20-period MA'))
            
            fig.update_layout(title=f"{column} Over Time", xaxis_title="Period", yaxis_title="Value")
            
            analysis = technical_analysis(df, column)
            
            return fig, analysis
        return None, None

    # Prepare list of numeric columns for graphing
    numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and not df[col].isnull().all()]

    # Display graphs and analysis for each numeric column
    st.subheader("Financial Data Visualization and Analysis")
    for column in numeric_columns:
        fig, analysis = create_graph_with_analysis(df, column)
        if fig and analysis:
            st.plotly_chart(fig)
            st.markdown(f"**Technical Analysis for {column}:**")
            st.write(analysis)
            st.markdown("---")  # Add a separator between graphs

else:
    st.error(f"Failed to fetch data: {response.status_code} - {response.text}")