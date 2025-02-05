import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import yfinance as yf
import keras

vix_data = yf.download('^VIX', start='2024-01-01', end='2024-11-01', interval='1d')
vix=vix_data['Adj Close']


#print(vix)

plt.figure(figsize=(10, 6))
plt.plot(vix)
plt.title('VIX')
plt.xlabel('Date')
plt.ylabel('VIX')
plt.show()


print("nowhere?")
model = arch_model(vix, vol='GARCH', p=1, q=1, dist='normal')
print("there")
# Fit the model
model_fit = model.fit(disp='off')
print("here")
print(model_fit.summary())



forecast_horizon = 3
forecasts = model_fit.forecast(horizon=forecast_horizon)


variance_forecasts = forecasts.variance[-1:]
print("Forecasted Variances:\n", variance_forecasts)


#cond_vol = model_fit.conditional_volatility

forecast_var = forecasts.variance.iloc[-1]
forecast_vol = np.sqrt(forecast_var) * np.sqrt(252)  # annualize forecast



plt.figure(figsize=(10,6))
plt.plot(forecast_vol)
plt.title('Conditional Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.show()



def run_garch_11(returns, forecast_horizon):
    """
    Fits a standard GARCH(1,1) model and returns its one-step-ahead forecast,
    annualized.
    """
    print("this si the forecast horizon{}".format(forecast_horizon))
    model = arch_model(returns, vol='GARCH', p=1, q=1, dist='normal')
    res = model.fit(disp='off')
    forecast = res.forecast(horizon=forecast_horizon)
    forecast_var = forecast.variance.iloc[-1]
    forecast_vol = np.sqrt(forecast_var) * np.sqrt(252)  # annualize forecast
    return forecast_vol, res

def compute_returns(price_series):
    """
    Computes log returns from the price series.
    Returns the returns in percentage terms.
    """
    returns = np.log(price_series).diff().dropna() * 100
    return returns

def compute_historical_volatility(returns, window=30):
    """
    Computes the rolling historical volatility (annualized) using a specified window.
    (Calculated as the rolling standard deviation multiplied by √252.)
    """
    hist_vol = returns.rolling(window=window).std() * np.sqrt(252)
    return hist_vol.dropna()

model_name, model_func = MODELS[model_choice]
forecast_series = compute_rolling_forecast(model_func, returns, initial_window)
plot_volatility(model_name, hist_vol, forecast_series, ticker, years_input, forecast_horizon)
metrics = evaluate_forecast(forecast_series, hist_vol)

def plot_volatility(model_name, hist_vol, forecast_series, ticker, years_span, forecast_horizon):
    """
    Plots the historical volatility and the rolling forecasted volatility.
    The title is augmented with the ticker, time span, forecast horizon, and a timestamp.
    The plot is saved in the "outputs" folder with the specified file name format.
    """
    plt.figure(figsize=(10, 6))
    # Limit historical volatility to start at the first forecast date for clarity
    hist_subset = hist_vol.loc[forecast_series.index[0]:]
    plt.plot(hist_subset.index, hist_subset, label='Historical Volatility', color='black', linewidth=1.5)
    plt.plot(forecast_series.index, forecast_series, label=f'{model_name} Rolling Forecast', linestyle='-')
    
  def compute_rolling_forecast(model_func, returns, initial_window=60):
    """
    Computes a rolling one-step-ahead forecast for each day from initial_window to the end.
    
    For each day t (starting at index=initial_window), the model is fitted to returns[:t]
    and a one-day-ahead forecast is computed. The function returns a Series of annualized
    forecasted volatilities indexed by the date corresponding to t.
    """
    forecast_values = []
    forecast_dates = []
    for t in range(initial_window, len(returns)):
        sub_returns = returns.iloc[:t]
        try:
            forecast_series, _ = model_func(sub_returns, forecast_horizon=1)
            forecast_values.append(forecast_series.iloc[0])
            forecast_dates.append(returns.index[t])
        except Exception as e:
            forecast_values.append(np.nan)
            forecast_dates.append(returns.index[t])
    return pd.Series(data=forecast_values, index=forecast_dates)