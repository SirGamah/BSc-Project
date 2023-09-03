# Import various libraries
from datetime import date
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from plotly import graph_objs as go

# App Design
st.title("Stock Price Prophet")
        # Set Stocks under consideration
st.header("Available Stocks")
st.write("1. AngloGold Ashanti Limited - AU")
st.write("2. MTN Group Limited - MTNOY")
st.write("3. Vodafone Group Public Limited Company - VOD")
st.write("4. Unilever PLC - UL")
st.write("5. Airtel Africa PLC - AAF.L")

# Getting user inputs
stocks = ("AAPL","AU", "MTNOY", "VOD", "UL", "AAF.L") # Set stock symbols
selected_stock = st.selectbox("Select stock for prediction", stocks) # Set user input for stock symbols

if selected_stock:
    min_date = date(2000, 1, 1) # Set minimum date to be selected by user
    max_date = date.today() # Set maximum date to today
    start_date = date(2015, 1, 1) # Set a defualt date
    start = st.date_input("Pick a Start Date", value=start_date, min_value=min_date, max_value=max_date) # Set user date input
    start = start.strftime("%Y-%m-%d") # Change date format to required format for yfinance library

    end = date.today().strftime("%Y-%m-%d") # Set end date for data retrieval to today

    # Set years for prediction input
    n_years = st.slider("Pick number of year(s) for prediction:", 1, 5)
    period = n_years * 365

    if start:
        # Data retrieval
        @st.cache_data
        def load_data(ticker):
            data = yf.download(ticker, start, end)
            data.reset_index(inplace = True)
            return data

        data = load_data(selected_stock)

        # Show data
        st.subheader("Raw Stock Historical Data")
        st.write(data.tail())

        # Visualization
        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x = data['Date'], y = data['Open'], name = "stock_open", line=dict(color='green')))
            fig.add_trace(go.Scatter(x = data['Date'], y = data['Close'], name = "stock_close", line=dict(color='blue')))
            fig.layout.update(title_text = "Time Series Data", xaxis_rangeslider_visible = True)
            st.plotly_chart(fig)

        st.subheader('Time Series Analysis')
        plot_raw_data()

        # Plot Candle Stick Analysis
        def plot_candle():
            last_30_days_data = data.iloc[-30:]
            trace1 = {
                'x': last_30_days_data.Date,
                'open': last_30_days_data.Open,
                'close': last_30_days_data.Close,
                'high': last_30_days_data.High,
                'low': last_30_days_data.Low,
                'type': 'candlestick',
                'name': selected_stock,
                'showlegend': False

            }

            df = [trace1]
            # Configure graph layout
            layout = go.Layout({
                'title': {
                    'text': selected_stock,
                    'font': {
                        'size': 15
                    }
                }
            })

            # Plot fig
            fig = go.Figure(data = df, layout = layout)
            st.plotly_chart(fig)

        st.subheader("Last 30 Days Candle Stick Analysis")
        plot_candle() # Plot clandle Analysis

        # Stock Forecasting
        df_train = data[['Date', 'Close']] # Select columns from raw data for prediction
        df_train = df_train.rename(columns = {"Date": "ds", "Close": "y"}) # Convert columns for Prophet to understand

        # Make prediction
        model = Prophet() # Initialize model
        model.fit(df_train) # Fit model on training data
        future = model.make_future_dataframe(periods = period) # Make future prediction
        forecast = model.predict(future)

        # Show forecast data
        st.subheader("Forecast data")
        st.write(forecast.tail())

        #  visualize Forecast Data
        st.subheader("Forecast Analysis")
        fig1 = plot_plotly(model, forecast)
        st.plotly_chart(fig1)

        # Plot Forecast Trends
        st.subheader("Forecast Trend Analysis")
        fig2 = model.plot_components(forecast)
        st.write(fig2)
    else:
        st.warning("Please select a start date.")
else:
    st.warning("Please select a stock for prediction.")