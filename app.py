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
    # Basic validations for fetched data
    if data is None or data.empty:
        st.error(f"No data found for ticker: {ticker}. Check the symbol or your network.")
    elif 'Close' not in data.columns or data['Close'].dropna().empty:
        st.error(f"No closing price data available for {ticker}.")
    else:
        st.subheader("Historical Closing Prices")
        st.line_chart(data['Close'])

        # 3. Scale data
        close_series = data['Close'].dropna()
        if len(close_series) < 60:
            st.warning("Not enough historical data (need at least 60 closing prices) to make a prediction.")
        else:
            scaler = MinMaxScaler(feature_range=(0,1))
            scaled_data = scaler.fit_transform(close_series.values.reshape(-1,1))

            # 4. Prepare last 60 days for prediction
            last_60_days = scaled_data[-60:]
            X_input = last_60_days.reshape((1,60,1))

            # 5. Load trained LSTM model
            try:
                model = load_model("stock_lstm_model.h5")
            except Exception as e:
                st.error(f"Failed to load model: {e}")
            else:
                # 6. Predict tomorrow's price
                try:
                    predicted_price = model.predict(X_input)
                    predicted_price_value = scaler.inverse_transform(predicted_price)[0][0]
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                else:
                    # Use last valid close value
                    last_close_val = float(close_series.iloc[-1])
                    direction = "Up" if predicted_price_value > last_close_val else "Down"

                    # 7. Show results
                    st.subheader("Prediction")
                    st.write(f"Last Close Price: ${last_close_val:.2f}")
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
                    plt.plot(close_series, label='Historical Close')
                    plt.scatter(close_series.index[-1] + pd.Timedelta(days=1), predicted_price_value, color='red', label='Predicted Price')
                    plt.title(f"{ticker} Stock Price Prediction")
                    plt.xlabel("Date")
                    plt.ylabel("Price")
                    plt.legend()
                    st.pyplot(plt)
