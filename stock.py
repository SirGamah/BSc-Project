# Import various libraries
from datetime import date # for manipulating dates
import streamlit as st # Python library for building web application
from streamlit_option_menu import option_menu # for setting up menu bar
import pandas as pd # for data manipulation
import matplotlib.pyplot as plt # for data analysis and visualization

import yfinance as yf # for pulling historical stock data from Yahoo! Finance
from prophet import Prophet # ML algorithm for data forecating
from prophet.plot import plot_plotly, plot_components_plotly # for creating interactive visualizations
from plotly import graph_objs as go # for creating interactive visualizations

#-----------Web page setting-------------------#
page_title = "Stock Price Forecast Web App"
page_icon = "📈"
viz_icon = "📊"
pred_icon = "📉"
date_picker_icon = "📆"
title_icon = "💹"
stock_icon = "📋"
picker_icon = "👇"
data_icon = "📑"
layout = "centered"

#--------------------Page configuration------------------#
st.set_page_config(page_title = page_title, page_icon = page_icon, layout = layout)

#--------------------Web App Design----------------------#

selected = option_menu(
    menu_title = page_title + " " + page_icon,
    options = ['Home', 'Analysis', 'Forecast', 'About'],
    icons = ["house-fill", "book-half", "robot", "envelope-fill"],
    default_index = 0,
    orientation = "horizontal"
)



# Home page
if selected == "Home":
    st.subheader("These are the available stock exchange companies to choose for the forecast.")
    st.write("1. AngloGold Ashanti Limited - AU")
    st.write("2. MTN Group Limited - MTNOY")
    st.write("3. Vodafone Group Public Limited Company - VOD")
    st.write("4. Unilever PLC - UL")
    st.write("5. Airtel Africa PLC - AAF.L")

# Analysis page
if selected == "Analysis":
    # Getting user inputs
    stocks = ("AAPL","AU", "MTNOY", "VOD", "UL", "AAF.L") # Set stock symbols
    selected_stock = st.selectbox("Select stock symbol " + picker_icon, stocks) # Set user input for stock symbols

    if selected_stock:
        min_date = date(2000, 1, 1) # Set minimum date to be selected by user
        max_date = date.today() # Set maximum date to today
        start_date = date(2015, 1, 1) # Set a defualt date
        start = st.date_input("Pick a Start Date " + picker_icon, value=start_date, min_value=min_date, max_value=max_date) # Set user date input
        start = start.strftime("%Y-%m-%d") # Change date format to required format for yfinance library

        end = date.today().strftime("%Y-%m-%d") # Set end date for data retrieval to today
        
        if start:
            # Data retrieval
            @st.cache_data # Cache the retrieved data to enhance app speed and performance
            
            # Define a load_data function to retreive historical stock data
            def load_data(ticker, start):
                data = yf.download(ticker, start, end) # get data from Yahoo! Fincane using selected stock symbol, and picked date
                data.reset_index(inplace = True) # reset the index of the retreived data 
                return data

            data = load_data(selected_stock, start) # Load the retreived data

            # Show data
            st.subheader("Raw Stock Historical Data " + data_icon)
            st.write(data.head()) # Show first 5 days of the stock data
            st.write(data.tail()) # Show last 5 days of the stock data

            # Visualization
            # Define a plot_raw_data function to plot a Time Series Analysis of the Opening and Closing stock price
            def plot_raw_data():
                fig = go.Figure()
                fig.add_trace(go.Scatter(x = data['Date'], y = data['Open'], name = "stock_open", line=dict(color='green')))
                fig.add_trace(go.Scatter(x = data['Date'], y = data['Close'], name = "stock_close", line=dict(color='blue')))
                fig.layout.update(title_text = "Time Series Data", xaxis_rangeslider_visible = True)
                st.plotly_chart(fig)

            st.subheader("Time Series Analysis " + pred_icon)
            plot_raw_data() # Show plot

            # Plot Candle Stick Analysis
            # Define a plot_candle function to show candle sticks analysis of the last 30 days of the stock data
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

            st.subheader("Last 30 Days Candle Stick Analysis " + viz_icon)
            plot_candle() # Plot clandle Analysis

# Forecast/Prediction page
if selected == "Forecast":
    
    # Set years for prediction input
    n_years = st.slider("Pick number of year(s) for forecast " + picker_icon, 1, 5) # Set date picker slider between 1 to 5 years
    period = n_years * 365 # Convert selected years to days
    stocks = ("AAPL","AU", "MTNOY", "VOD", "UL", "AAF.L") # Set stock symbols
    selected_stock = st.selectbox("Select stock symbol " + picker_icon, stocks) # Set user input for stock symbols

    if selected_stock:
        min_date = date(2000, 1, 1) # Set minimum date to be selected by user
        max_date = date.today() # Set maximum date to today
        start_date = date(2015, 1, 1) # Set a defualt date
        start = st.date_input("Pick a Start Date " + picker_icon, value=start_date, min_value=min_date, max_value=max_date) # Set user date input
        start = start.strftime("%Y-%m-%d") # Change date format to required format for yfinance library

        end = date.today().strftime("%Y-%m-%d") # Set end date for data retrieval to today

        @st.cache_data # Cache the retrieved data to enhance app speed and performance
        # Define a load_data function to retreive historical stock data
        def load_data(ticker, start):
            data = yf.download(ticker, start, end) # get data from Yahoo! Fincane using selected stock symbol, and picked date
            data.reset_index(inplace = True) # reset the index of the retreived data 
            return data
        
        
        data = load_data(selected_stock, start) # Load the retreived data
        # Stock Forecasting
        df_train = data[['Date', 'Close']] # Select columns from raw data for prediction
        df_train = df_train.rename(columns = {"Date": "ds", "Close": "y"}) # Convert columns for Prophet to understand

        # Make prediction
        model = Prophet() # Initialize model
        model.fit(df_train) # Fit model on training data
        future = model.make_future_dataframe(periods = period) # Make future prediction
        forecast = model.predict(future)

        # Show forecast data
        st.subheader("Forecast data " + data_icon)
        st.write(forecast.tail()) # Show last 5 days of forcasted data

        # Visualize Forecast Data
        st.subheader("Forecast Analysis " + pred_icon)
        fig1 = plot_plotly(model, forecast) # Use plotly library to plot forecast data
        st.plotly_chart(fig1) # Show plot

        # Plot Forecast Trends
        st.subheader("Forecast Trend Analysis " + page_icon)
        fig2 = model.plot_components(forecast) # Retreive trend components from the model
        st.write(fig2) # Plot trend components
    
# About page
if selected == "About":
    # About Web App
    st.write("""#### Stock Price Forecast Web App is an online system that forecasts stock closing prices of 5 selected African stock exchange companies!""")
    st.write("""### Get in touch""")
    st.write("""###### Email: gamahrichard5@gmail.com""")
    st.write("""###### GitHub: https://github.com/SirGamah/BSc-Project""")
    st.write("""###### WhatsApp: https://wa.me/233542124371""")
