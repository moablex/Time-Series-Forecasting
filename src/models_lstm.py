# src/models_lstm.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -------------------
# Data preparation
# -------------------
def prepare_lstm_data(series, window_size):
    """
    Convert a 1D series into LSTM input/output samples.
    """
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    X = np.array(X)
    y = np.array(y)
    # Reshape for LSTM: [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

# -------------------
# Model building
# -------------------
def build_lstm(input_shape):
    """
    Build and compile a simple LSTM model.
    """
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# -------------------
# Model training
# -------------------
def fit_lstm(series, window_size=10, epochs=20, batch_size=16, verbose=1):
    """
    Train an LSTM model on the given series.
    Returns: trained model, window_size, and the training data (for scaling/forecasting).
    """
    X, y = prepare_lstm_data(series, window_size)
    model = build_lstm((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model, window_size, series

# -------------------
# Forecasting
# -------------------
def forecast_lstm(model, history_series, window_size, steps=10):
    """
    Forecast future values using the trained LSTM model.
    history_series: original 1D numpy array or list
    steps: number of future predictions
    """
    history = list(history_series)
    predictions = []

    for _ in range(steps):
        x_input = np.array(history[-window_size:])
        x_input = x_input.reshape((1, window_size, 1))
        yhat = model.predict(x_input, verbose=0)
        predictions.append(yhat[0, 0])
        history.append(yhat[0, 0])

    return np.array(predictions)
