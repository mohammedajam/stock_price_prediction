# streamlit_app.py

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("Stock Price Predictor")

# 1. User input
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):")

if ticker:
    # 2. Fetch historical data
    data = yf.download(ticker, start="2015-01-01")
    st.subheader("Historical Closing Prices")
    st.line_chart(data['Close'])

    # 3. Scale data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

    # 4. Prepare last 60 days for prediction
    last_60_days = scaled_data[-60:]
    X_input = last_60_days.reshape((1,60,1))

    # 5. Load trained LSTM model
    model = load_model("stock_lstm_model.h5")

    # 6. Predict tomorrow's price
    predicted_price = model.predict(X_input)
    predicted_price_value = scaler.inverse_transform(predicted_price)[0][0]
    last_close = float(data['Close'].iloc[-1])
    direction = "Up " if predicted_price_value > last_close else "Down "

    # 7. Show results
    st.subheader("Prediction")
    st.write(f"Last Close Price: ${last_close:.2f}")
    st.write(f"Tomorrow's Predicted Price: ${predicted_price_value:.2f}")
    st.write(f"Predicted Direction: {direction}")

    # Assume RMSE is known from training

    rmse_value = 7.06
    st.subheader("Prediction Error Info")
    st.write(f"Estimated prediction error : ±${rmse_value:.2f}")
    st.write("The Prices may vary within this range.")


    # 8. Plot historical + predicted
    st.subheader("Historical Closing Prices with Prediction")
    plt.figure(figsize=(12,5))
    plt.plot(data['Close'], label='Historical Close')
    plt.scatter(data.index[-1] + pd.Timedelta(days=1), predicted_price_value, color='red', label='Predicted Price')
    plt.title(f"{ticker} Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(plt)
