# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('AAPL', 'NVDA', 'META', 'TSM', 'QQQ', 'AVGO')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Done loading data!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open", line=dict(color='lightblue')))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close", line=dict(color='blue')))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Fit the Prophet model
m = Prophet()
m.fit(df_train)

# Make future dataframe
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Display forecast data
st.subheader('Forecast data')
st.write(forecast.tail())

# Create custom forecast plot with colored lines
def plot_forecast(m, forecast):
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=forecast['ds'], 
        y=forecast['yhat'], 
        name='Forecast',
        line=dict(color='cyan')
    ))
    
    # Upper and lower bounds
    fig.add_trace(go.Scatter(
        x=forecast['ds'], 
        y=forecast['yhat_upper'], 
        name='Upper Bound',
        line=dict(color='lightblue'),
        fill=None
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'], 
        y=forecast['yhat_lower'], 
        name='Lower Bound',
        line=dict(color='navy'),
        fill='tonexty'
    ))

    fig.layout.update(title_text=f'Forecast plot for {n_years} years', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Plot forecast
plot_forecast(m, forecast)

# Display forecast components
st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)