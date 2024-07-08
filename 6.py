import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from ta.trend import MACD
from ta.momentum import RSIIndicator

base_url = 'https://financialmodelingprep.com/api/v3'
API_KEY = 'r55UkdaTRIcKzrByVEaAkPJb6lsCXFWh'

st.title('Advanced Financial Analysis Dashboard')
st.markdown('Comprehensive Financial Data Visualization and Analysis')

# Main sidebar for data selection
with st.sidebar:
    st.header("Data Selection")
    symbols = st.text_input('Tickers (comma-separated for multiple)', value='MSFT,AAPL', key='ticker_input')
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

def fetch_data(symbol, financial_data):
    url = f'{base_url}/{financial_data}/{symbol}?apikey={API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        st.error(f"Failed to fetch data for {symbol}: {response.status_code} - {response.text}")
        return None

def add_technical_indicators(df):
    if 'close' in df.columns:
        macd = MACD(close=df['close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        rsi = RSIIndicator(close=df['close'])
        df['RSI'] = rsi.rsi()
    return df

def technical_analysis(df, column):
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
    
    # MACD and RSI analysis (if applicable)
    if 'MACD' in df.columns and 'RSI' in df.columns:
        last_macd = df['MACD'].iloc[-1]
        last_macd_signal = df['MACD_signal'].iloc[-1]
        last_rsi = df['RSI'].iloc[-1]
        
        if last_macd > last_macd_signal:
            analysis.append("MACD is above its signal line, suggesting a bullish trend.")
        else:
            analysis.append("MACD is below its signal line, suggesting a bearish trend.")
        
        if last_rsi > 70:
            analysis.append("RSI is above 70, indicating the stock may be overbought.")
        elif last_rsi < 30:
            analysis.append("RSI is below 30, indicating the stock may be oversold.")
        else:
            analysis.append(f"RSI is at {last_rsi:.2f}, indicating neutral momentum.")
    
    return "\n".join(analysis)

def create_graph_with_analysis(df, column, symbol):
    if pd.api.types.is_numeric_dtype(df[column]) and not df[column].isnull().all():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', name=f'{symbol} - {column}'))
        
        # Add moving averages
        fig.add_trace(go.Scatter(x=df.index, y=df[column].rolling(window=5).mean(), mode='lines', name='5-period MA'))
        fig.add_trace(go.Scatter(x=df.index, y=df[column].rolling(window=20).mean(), mode='lines', name='20-period MA'))
        
        # Add MACD and RSI if available
        if 'MACD' in df.columns and 'RSI' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'))
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], mode='lines', name='MACD Signal'))
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', yaxis='y2'))
            
            fig.update_layout(
                yaxis2=dict(title='RSI', overlaying='y', side='right', range=[0, 100]),
                showlegend=True
            )
        
        fig.update_layout(title=f"{symbol} - {column} Over Time", xaxis_title="Period", yaxis_title="Value")
        
        analysis = technical_analysis(df, column)
        
        return fig, analysis
    return None, None

def cross_metric_analysis(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_columns].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu',
        zmin=-1, zmax=1
    ))
    fig.update_layout(title='Cross-Metric Correlation Heatmap', height=800, width=800)
    
    return fig

def predictive_analysis(df, column):
    if len(df) < 30:  # Need sufficient data for prediction
        return "Insufficient data for predictive analysis"
    
    df['timestamp'] = range(len(df))
    X = df[['timestamp']]
    y = df[column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    future_timestamps = np.array(range(len(df), len(df) + 30)).reshape(-1, 1)  # Predict next 30 periods
    future_predictions = model.predict(future_timestamps)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=pd.date_range(start=df.index[-1], periods=31, freq='D')[1:], 
                             y=future_predictions, mode='lines', name='Predicted'))
    fig.update_layout(title=f'Predictive Analysis for {column}', xaxis_title='Date', yaxis_title='Value')
    
    return fig

# Fetch and process data for each symbol
symbols_list = [symbol.strip() for symbol in symbols.split(',')]
dataframes = {}

for symbol in symbols_list:
    df = fetch_data(symbol, financial_data)
    if df is not None:
        df = add_technical_indicators(df)
        dataframes[symbol] = df

# Display data and analysis for each symbol
for symbol, df in dataframes.items():
    st.header(f"Analysis for {symbol}")
    
    # Display data table
    st.subheader("Data Table")
    st.dataframe(df)
    
    # Display graphs and analysis for each numeric column
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        fig, analysis = create_graph_with_analysis(df, column, symbol)
        if fig and analysis:
            st.plotly_chart(fig)
            st.markdown(f"**Technical Analysis for {column}:**")
            st.write(analysis)
            st.markdown("---")
    
    # Cross-metric analysis
    st.subheader("Cross-Metric Analysis")
    correlation_fig = cross_metric_analysis(df)
    st.plotly_chart(correlation_fig)
    
    # Predictive analysis
    st.subheader("Predictive Analysis")
    for column in numeric_columns[:3]:  # Limit to first 3 columns for brevity
        prediction_fig = predictive_analysis(df, column)
        if isinstance(prediction_fig, go.Figure):
            st.plotly_chart(prediction_fig)
        else:
            st.write(prediction_fig)
    
    st.markdown("---")

# Comparison between stocks
if len(dataframes) > 1:
    st.header("Stock Comparison")
    comparison_fig = go.Figure()
    for symbol, df in dataframes.items():
        if 'close' in df.columns:
            comparison_fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name=symbol))
    comparison_fig.update_layout(title='Stock Price Comparison', xaxis_title='Date', yaxis_title='Close Price')
    st.plotly_chart(comparison_fig)