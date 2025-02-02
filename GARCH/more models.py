#!/usr/bin/env python3
"""
Volatility Forecasting using Various GARCH-type Models with Rolling Forecasts

This script downloads price data for a user-specified ticker over a specified
time span, computes returns and historical volatility (via a rolling window),
and fits various GARCH-type models (using the arch library). Instead of computing
a static forecast (only a few days beyond the sample), the script computes a
rolling one-step-ahead forecast for each day (after an initial estimation window).
The resulting rolling forecast series (annualized) is then plotted along with the
historical (annualized) volatility for comparison.

The title of the saved graph includes the ticker, time frame, forecast horizon, and a timestamp.
The file is saved in a folder called "outputs" with a file name format:
    For a single model: "GARCH-AAPL-6years-3days-timestamp.png"
    For multiple models: "GARCH-EGARCH-GJR-GARCH-AAPL-6years-3days-timestamp.png"

Models included (by index):
    0: GARCH(1,1)
    1: EGARCH
    2: IGARCH (approximated using GARCH(1,1) with forced constraint)
    3: FIGARCH      (Not implemented)
    4: APARCH       (Not implemented)
    5: GJR-GARCH
    6: CGARCH       (Not implemented)
    7: NGARCH       (Not implemented)
    8: HYGARCH      (Not implemented)
    9: Multivariate GARCH (BEKK/DCC) (Not implemented)
    10: PGARCH      (Not implemented)
    11: QGARCH      (Not implemented)

For models that are not implemented, a NotImplementedError is raised.
"""

# Global imports
import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import datetime

# Global variable for the price column to use (can be adjusted later)
ADJ_COL = 'Adj Close'

def get_ticker_data(ticker, years_span, adj_col=ADJ_COL):
    """
    Downloads historical data for the given ticker for a period starting from
    today going backwards by years_span (float, in years).

    Returns:
        pd.Series: Price series using the specified column.
    """
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=int(years_span * 365))
    try:
        data = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"),
                           end=end_date.strftime("%Y-%m-%d"), interval='1d')
        if data.empty or adj_col not in data.columns:
            raise ValueError("No data returned or invalid column.")
        return data[adj_col]
    except Exception as e:
        raise ValueError(f"Error downloading data for {ticker}: {e}")

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

# ---------------- Model Functions ----------------
# Each model function fits a model on the returns and returns a one-step-ahead forecast.
# The forecast is then annualized by multiplying by √252.

def run_garch_11(returns, forecast_horizon=1):
    """
    Fits a standard GARCH(1,1) model and returns its one-step-ahead forecast,
    annualized.
    """
    model = arch_model(returns, vol='GARCH', p=1, q=1, dist='normal')
    res = model.fit(disp='off')
    forecast = res.forecast(horizon=forecast_horizon)
    forecast_var = forecast.variance.iloc[-1]
    forecast_vol = np.sqrt(forecast_var) * np.sqrt(252)  # annualize forecast
    return forecast_vol, res

def run_egarch(returns, forecast_horizon=1):
    """
    Fits an EGARCH model and returns its one-step-ahead forecast (annualized).
    """
    model = arch_model(returns, vol='EGARCH', p=1, q=1, dist='normal')
    res = model.fit(disp='off')
    forecast = res.forecast(horizon=forecast_horizon)
    forecast_var = forecast.variance.iloc[-1]
    forecast_vol = np.sqrt(forecast_var) * np.sqrt(252)
    return forecast_vol, res

def run_igarch(returns, forecast_horizon=1):
    """
    Approximates an IGARCH model (forcing α+β=1) and returns its one-step-ahead forecast,
    annualized.
    """
    model = arch_model(returns, vol='GARCH', p=1, q=1, dist='normal')
    res = model.fit(disp='off')
    alpha = res.params['alpha[1]']
    beta = 1 - alpha  # force constraint
    omega = res.params['omega']
    last_sigma2 = res.conditional_volatility.iloc[-1]**2
    last_resid = res.resid.iloc[-1]
    forecast_var = omega + alpha * (last_resid**2) + beta * last_sigma2
    forecast_vol = np.sqrt(forecast_var) * np.sqrt(252)
    return pd.Series([forecast_vol], index=[returns.index[-1] + pd.Timedelta(days=1)]), res

def run_gjr_garch(returns, forecast_horizon=1):
    """
    Fits a GJR-GARCH model (using an extra "o" term) and returns its one-step-ahead forecast,
    annualized.
    """
    model = arch_model(returns, vol='GARCH', p=1, o=1, q=1, dist='normal')
    res = model.fit(disp='off')
    forecast = res.forecast(horizon=forecast_horizon)
    forecast_var = forecast.variance.iloc[-1]
    forecast_vol = np.sqrt(forecast_var) * np.sqrt(252)
    return forecast_vol, res

def run_figarch(returns, forecast_horizon=1):
    raise NotImplementedError("FIGARCH model is not implemented due to lack of available library support.")

def run_aparch(returns, forecast_horizon=1):
    raise NotImplementedError("APARCH model is not implemented.")

def run_cgarch(returns, forecast_horizon=1):
    raise NotImplementedError("CGARCH model is not implemented.")

def run_ngarch(returns, forecast_horizon=1):
    raise NotImplementedError("NGARCH model is not implemented.")

def run_hygarch(returns, forecast_horizon=1):
    raise NotImplementedError("HYGARCH model is not implemented.")

def run_multivariate_garch(returns, forecast_horizon=1):
    raise NotImplementedError("Multivariate GARCH models (BEKK/DCC) are not implemented.")

def run_pgarch(returns, forecast_horizon=1):
    raise NotImplementedError("PGARCH model is not implemented.")

def run_qgarch(returns, forecast_horizon=1):
    raise NotImplementedError("QGARCH model is not implemented.")

# Dictionary mapping model indices to (model name, function)
MODELS = {
    0: ("GARCH(1,1)", run_garch_11),
    1: ("EGARCH", run_egarch),
    2: ("IGARCH", run_igarch),
    3: ("FIGARCH", run_figarch),
    4: ("APARCH", run_aparch),
    5: ("GJR-GARCH", run_gjr_garch),
    6: ("CGARCH", run_cgarch),
    7: ("NGARCH", run_ngarch),
    8: ("HYGARCH", run_hygarch),
    9: ("Multivariate GARCH (BEKK/DCC)", run_multivariate_garch),
    10: ("PGARCH", run_pgarch),
    11: ("QGARCH", run_qgarch)
}

# ---------------- Rolling Forecast Function ----------------

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

# ---------------- Helper Function for File Naming ----------------

def clean_model_name(name):
    """
    Cleans the model name for file naming by removing text in parentheses.
    """
    if "(" in name:
        return name.split('(')[0].strip()
    return name.strip()

# ---------------- Plotting Functions ----------------

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
    
    # Generate timestamp string
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Build title string
    title_str = (f"{model_name} Rolling One-Step-Ahead Forecast vs Historical Volatility - "
                 f"{ticker} - {years_span}years - {forecast_horizon}days Forecast - {timestamp_str}")
    plt.title(title_str)
    plt.xlabel('Date')
    plt.ylabel('Volatility (%)')
    plt.legend()
    
    # Build file name (clean the model name for file naming)
    clean_name = clean_model_name(model_name)
    file_name = f"{clean_name}-{ticker}-{years_span}years-{forecast_horizon}days-{timestamp_str}.png"
    # Ensure the outputs folder exists
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(os.path.join("outputs", file_name))
    plt.show()

def plot_multiple_forecasts(forecast_dict, hist_vol, ticker, years_span, forecast_horizon, same_fig=True):
    """
    Given a dictionary mapping model names to rolling forecast Series,
    plots the historical volatility (common for all) and the rolling forecasted volatility.
    For a same-figure display, the title includes all model names (cleaned) followed by the ticker,
    time span, forecast horizon, and timestamp. The plot is saved accordingly.
    """
    if same_fig:
        plt.figure(figsize=(10, 6))
        # Restrict historical volatility to the period from the earliest forecast date
        earliest = min(fs.index[0] for fs in forecast_dict.values())
        hist_subset = hist_vol.loc[earliest:]
        plt.plot(hist_subset.index, hist_subset, label='Historical Volatility', color='black', linewidth=2)
        for model_name, forecast_series in forecast_dict.items():
            plt.plot(forecast_series.index, forecast_series, linestyle='-', label=f'{model_name} Forecast')
        
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # Build a combined model names string (cleaned)
        model_names_clean = "-".join([clean_model_name(name) for name in forecast_dict.keys()])
        title_str = (f"{model_names_clean} Rolling One-Step-Ahead Forecast Comparison - "
                     f"{ticker} - {years_span}years - {forecast_horizon}days Forecast - {timestamp_str}")
        plt.title(title_str)
        plt.xlabel('Date')
        plt.ylabel('Volatility (%)')
        plt.legend()
        file_name = f"{model_names_clean}-{ticker}-{years_span}years-{forecast_horizon}days-{timestamp_str}.png"
        os.makedirs("outputs", exist_ok=True)
        plt.savefig(os.path.join("outputs", file_name))
        plt.show()
    else:
        for model_name, forecast_series in forecast_dict.items():
            plot_volatility(model_name, hist_vol, forecast_series, ticker, years_span, forecast_horizon)

# ---------------- Main Interactive Function ----------------

def main():
    print("=== Volatility Forecasting with GARCH-Type Models (Rolling Forecast) ===")
    # Prompt for ticker until valid data is downloaded
    while True:
        ticker = input("Enter ticker symbol (e.g., AAPL): ").strip().upper()
        try:
            years_input = float(input("Enter time span in years (e.g., 7.5): "))
            price_series = get_ticker_data(ticker, years_input)
            print(f"Data for {ticker} successfully downloaded!")
            break
        except Exception as e:
            print(e)
            print("Please try again.\n")
    
    # Compute returns and historical volatility
    returns = compute_returns(price_series)
    hist_vol = compute_historical_volatility(returns, window=30)
    
    # For rolling forecast, we use one-step-ahead forecasts with an initial window.
    initial_window = 60  # Adjust as needed
    
    # Prompt for forecast horizon (number of days ahead for static forecast is not used here,
    # but we include it for the title; rolling forecast always computes one-step forecasts)
    try:
        forecast_horizon = int(input("Enter forecast horizon (number of days, e.g., 3): "))
    except:
        forecast_horizon = 3
        print("Invalid input. Using default forecast horizon of 3 days.")
    
    # Display model options
    print("\nSelect display option:")
    print("1. Single model")
    print("2. All models")
    print("3. Compare selected models")
    display_choice = input("Enter 1, 2, or 3: ").strip()
    
    if display_choice == "1":
        print("\nAvailable models:")
        for idx, (name, _) in MODELS.items():
            print(f"  {idx}: {name}")
        try:
            model_choice = int(input("Enter the model index you want to display: "))
        except Exception as e:
            print("Invalid input.")
            return
        if model_choice not in MODELS:
            print("Invalid model index.")
            return
        model_name, model_func = MODELS[model_choice]
        forecast_series = compute_rolling_forecast(model_func, returns, initial_window)
        plot_volatility(model_name, hist_vol, forecast_series, ticker, years_input, forecast_horizon)
    
    elif display_choice == "2":
        print("\nAvailable models:")
        for idx, (name, _) in MODELS.items():
            print(f"  {idx}: {name}")
        all_choice = input("Display all model forecasts on (1) the same graph or (2) individual graphs? Enter 1 or 2: ").strip()
        forecast_dict = {}
        for idx, (name, model_func) in MODELS.items():
            try:
                forecast_series = compute_rolling_forecast(model_func, returns, initial_window)
                forecast_dict[name] = forecast_series
            except NotImplementedError as nie:
                print(f"Skipping {name}: {nie}")
                continue
        if not forecast_dict:
            print("No models available for forecasting.")
            return
        same_fig = True if all_choice == "1" else False
        plot_multiple_forecasts(forecast_dict, hist_vol, ticker, years_input, forecast_horizon, same_fig=same_fig)
    
    elif display_choice == "3":
        print("\nAvailable models:")
        for idx, (name, _) in MODELS.items():
            print(f"  {idx}: {name}")
        sel_str = input("Enter model indices to compare (separated by dashes, e.g., 0-1-2-5): ").strip()
        try:
            selected_indices = [int(x.strip()) for x in sel_str.split('-')]
        except Exception as e:
            print("Invalid format.")
            return
        forecast_dict = {}
        for idx in selected_indices:
            if idx in MODELS:
                name, func = MODELS[idx]
                try:
                    forecast_series = compute_rolling_forecast(func, returns, initial_window)
                    forecast_dict[name] = forecast_series
                except NotImplementedError as nie:
                    print(f"Skipping {name}: {nie}")
            else:
                print(f"Index {idx} is not valid. Skipping.")
        if not forecast_dict:
            print("No valid models selected for comparison.")
            return
        compare_choice = input("Display selected model forecasts on (1) the same graph or (2) individual graphs? Enter 1 or 2: ").strip()
        same_fig = True if compare_choice == "1" else False
        plot_multiple_forecasts(forecast_dict, hist_vol, ticker, years_input, forecast_horizon, same_fig=same_fig)
    
    else:
        print("Invalid selection. Exiting.")

if __name__ == '__main__':
    main()
