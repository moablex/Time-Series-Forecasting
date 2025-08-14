import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

def fit_auto_arima(train_series, seasonal=False, m=1):
    model = auto_arima(train_series, seasonal=seasonal, m=m,
                       trace=True, error_action='ignore', suppress_warnings=True)
    return model

def forecast_arima(model, steps):
    forecast = model.predict(n_periods=steps)
    return forecast
