#!/usr/bin/env python3


#######6.2 years, 1290 rolling window

"""
Volatility Forecasting using Various GARCH-type Models with Rolling Forecasts

This script downloads price data for the given ticker over a specified
time span, computes returns and historical volatility (via a rolling window),
and fits several GARCH-type models (using the arch library). Instead of computing
a static forecast, the script computes a rolling one-step-ahead forecast for each day
(after an initial estimation window). The resulting rolling forecast series (annualized)
is then plotted along with the historical (annualized) volatility for comparison.

The title of the saved graph includes the ticker, time frame, and a timestamp.
The file is saved in a folder called "outputs" with the file name format:
    "<ModelName>-<Ticker>-<Years>years-<timestamp>.png"

Models included (by index):
    0: GARCH(1,1)
    1: EGARCH
    2: FIGARCH
    3: APARCH
    4: GJR-GARCH
"""

import os
import json
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from arch import arch_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error

# Global variable for the price column to use
ADJ_COL = 'Adj Close'

def get_ticker_data(ticker, years_span, adj_col=ADJ_COL):
    """
    Downloads historical data for the given ticker for a period starting from
    today going backwards by years_span (in years).

    Returns:
        pd.Series: Price series using the specified column.
    """
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=int(years_span * 365))
    try:
        data = yf.download(ticker,
                           start=start_date.strftime("%Y-%m-%d"),
                           end=end_date.strftime("%Y-%m-%d"),
                           interval='1d')
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
    returns = np.log(price_series).diff().dropna()
    return returns

def compute_historical_volatility(returns, window=30):
    """
    Computes the rolling historical volatility (annualized) using a specified window.
    (Calculated as the rolling standard deviation multiplied by √252.)
    """
    hist_vol = returns.rolling(window=window).std() * np.sqrt(252)
    return hist_vol.dropna()

# ---------------- Model Functions ----------------

def run_garch_11(returns, forecast_horizon):
    """
    Fits a standard GARCH(1,1) model and returns its one-step-ahead forecast, annualized.
    """
    model = arch_model(returns, vol='GARCH', p=1, q=1, dist='normal', rescale=False)
    res = model.fit(disp='off')
    forecast = res.forecast(horizon=forecast_horizon)
    forecast_var = forecast.variance.iloc[-1]
    forecast_vol = np.sqrt(forecast_var) * np.sqrt(252)  # annualize forecast
    return forecast_vol, res

def run_egarch(returns, forecast_horizon=1):
    """
    Fits an EGARCH model and returns its one-step-ahead forecast (annualized).
    """
    model = arch_model(returns, vol='EGARCH', p=1, q=1, dist='normal', rescale=False)
    res = model.fit(disp='off')
    forecast = res.forecast(horizon=forecast_horizon)
    forecast_var = forecast.variance.iloc[-1]
    forecast_vol = np.sqrt(forecast_var) * np.sqrt(252)
    return forecast_vol, res

def run_aparch(returns, forecast_horizon=1):
    """
    Fits an APARCH model and returns a one-step-ahead forecast.
    
    This version uses the same extraction logic as your working rolling_forecast code.
    """
    am = arch_model(returns, vol='APARCH', p=1, q=1, dist='normal', rescale=False)
    # Use robust options for fitting as in your working code
    res = am.fit(disp='off', options={'maxiter': 10000, 'ftol': 1e-6})
    forecast = res.forecast(horizon=forecast_horizon)
    variance_forecast = forecast.variance.iloc[-1]
    predicted_vol = np.sqrt(variance_forecast) * np.sqrt(252)
    return pd.Series([predicted_vol]), res

def run_figarch(returns, forecast_horizon=1):
    """
    Fits a FIGARCH model and returns a one-step-ahead forecast.
    
    This version is based on the code that worked for you.
    """
    am = arch_model(returns, vol='FIGARCH', p=1, q=1, dist='normal', rescale=False)
    res = am.fit(disp='off')
    forecast = res.forecast(horizon=forecast_horizon)
    variance_forecast = forecast.variance.iloc[-1]
    predicted_vol = np.sqrt(variance_forecast) * np.sqrt(252)
    return pd.Series([predicted_vol]), res

def run_gjr_garch(returns, forecast_horizon=1):
    """
    Fits a GJR-GARCH model (using an extra "o" term) and returns its one-step-ahead forecast, annualized.
    """
    model = arch_model(returns, vol='GARCH', p=1, o=1, q=1, dist='normal', rescale=False)
    res = model.fit(disp='off')
    forecast = res.forecast(horizon=forecast_horizon)
    forecast_var = forecast.variance.iloc[-1]
    forecast_vol = np.sqrt(forecast_var) * np.sqrt(252)
    return forecast_vol, res

# Dictionary mapping model indices to (model name, function)
MODELS = {
    0: ("GARCH(1,1)", run_garch_11),
    1: ("EGARCH", run_egarch),
    2: ("FIGARCH", run_figarch),
    3: ("APARCH", run_aparch),
    4: ("GJR-GARCH", run_gjr_garch)
}

# ---------------- Rolling Forecast Function ----------------

def compute_rolling_forecast(model_func, returns, initial_window=60):
    """
    Computes a rolling one-step-ahead forecast for each day from initial_window to the end.
    
    For each day t (starting at index=initial_window), the model is fitted to returns[:t]
    and a one-day-ahead forecast is computed. Returns a Series of annualized forecasted volatilities.
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

def plot_volatility(model_name, hist_vol, forecast_series, ticker, years_span):
    """
    Plots the historical volatility and the rolling forecasted volatility.
    The title is augmented with ticker, time span, and timestamp.
    The plot is saved in the "outputs" folder.
    """
    plt.figure(figsize=(10, 6))
    # Limit historical volatility to start at the first forecast date for clarity
    hist_subset = hist_vol.loc[forecast_series.index[0]:]
    plt.plot(hist_subset.index, hist_subset, label='Historical Volatility', color='black', linewidth=1.5)
    plt.plot(forecast_series.index, forecast_series, label=f'{model_name} Rolling Forecast', linestyle='-')
    
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    title_str = (f"{model_name} Rolling One-Step-Ahead Forecast vs Historical Volatility - "
                 f"{ticker} - {years_span}years - {timestamp_str}")
    plt.title(title_str)
    plt.xlabel('Date')
    plt.ylabel('Volatility (%)')
    plt.legend()
    
    clean_name = clean_model_name(model_name)
    file_name = f"{clean_name}-{ticker}-{years_span}years-{timestamp_str}.png"
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(os.path.join("outputs", file_name))
    plt.show()

def plot_multiple_forecasts(forecast_dict, hist_vol, ticker, years_span, same_fig=True):
    """
    Plots the historical volatility (common for all) and the rolling forecasted volatilities
    for multiple models. Can either plot on one figure or on separate figures.
    """
    if same_fig:
        plt.figure(figsize=(10, 6))
        earliest = min(fs.index[0] for fs in forecast_dict.values())
        hist_subset = hist_vol.loc[earliest:]
        plt.plot(hist_subset.index, hist_subset, label='Historical Volatility', color='black', linewidth=2)
        for model_name, forecast_series in forecast_dict.items():
            plt.plot(forecast_series.index, forecast_series, linestyle='-', label=f'{model_name} Forecast')
        
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_names_clean = "-".join([clean_model_name(name) for name in forecast_dict.keys()])
        title_str = (f"{model_names_clean} Rolling One-Step-Ahead Forecast Comparison - "
                     f"{ticker} - {years_span}years - {timestamp_str}")
        plt.title(title_str)
        plt.xlabel('Date')
        plt.ylabel('Volatility (%)')
        plt.legend()
        file_name = f"{model_names_clean}-{ticker}-{years_span}years-{timestamp_str}.png"
        os.makedirs("outputs", exist_ok=True)
        plt.savefig(os.path.join("outputs", file_name))
        plt.show()
    else:
        for model_name, forecast_series in forecast_dict.items():
            plot_volatility(model_name, hist_vol, forecast_series, ticker, years_span)

def compute_metrics(y_true, y_pred):
    """
    Compute evaluation metrics between two Series.
    
    Returns:
        dict: Dictionary containing MAE, MSE, R², and MSLE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    msle = mean_squared_log_error(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "R2": r2, "MSLE": msle}

def evaluate_forecast(forecast_series, hist_vol):
    """
    Evaluates the forecasted volatility against the actual historical volatility.
    Prints evaluation metrics.
    """
    y_true = hist_vol.loc[forecast_series.index]
    y_pred = forecast_series
    metrics = compute_metrics(y_true, y_pred)
    
    print("Evaluation Metrics:")
    print(f"  MAE:  {metrics['MAE']:.4f}")
    print(f"  MSE:  {metrics['MSE']:.4f}")
    print(f"  R²:   {metrics['R2']:.4f}")
    print(f"  MSLE: {metrics['MSLE']:.4f}")
    return metrics

# ---------------- Main Interactive Function ----------------

def main():
    print("=== Volatility Forecasting with GARCH-Type Models (Rolling Forecast) ===")
    print("Select mode:")
    print("1. Testing")
    print("2. Automatic")
    mode = input("Enter 1 for Testing or 2 for Automatic: ").strip()
    
    # Set forecast horizon to 1 for all cases.
    forecast_horizon = 1
    
    if mode == "2":
        # --------------- AUTOMATIC MODE ----------------
        print("\nAutomatic mode selected.")
        try:
            years_input = float(input("Enter time span in years (e.g., 7.5): "))
        except Exception as e:
            print("Invalid input. Using default of 7.5 years.")
            years_input = 7.5
        try:
            initial_window = int(input("Input initial window: "))
        except Exception as e:
            print("Invalid input. Using default initial window of 60.")
            initial_window = 60

        tickers_list = [
            "^GSPC", "^IXIC", "BTC-USD", "GC=F", "EURUSD=X", "EURGBP=X",
            "^FTSE", "^FCHI", "^GDAXI", "FTSEMIB.MI", "^AXJO", "^HSI",
            "^N225", "^NSEI", "^JTOPI", "MERVAL.BA"
        ]
        
        results = {}
        for ticker in tickers_list:
            print(f"\nProcessing ticker: {ticker}")
            try:
                price_series = get_ticker_data(ticker, years_input)
            except Exception as e:
                print(f"  Skipping {ticker} due to error: {e}")
                results[ticker] = {"error": str(e)}
                continue
            
            returns = compute_returns(price_series)
            hist_vol = compute_historical_volatility(returns, window=30)
            results[ticker] = {}
            
            for idx, (model_name, model_func) in MODELS.items():
                print(f"  Running model: {model_name}")
                try:
                    forecast_series = compute_rolling_forecast(model_func, returns, initial_window)
                    metrics = evaluate_forecast(forecast_series, hist_vol)
                    results[ticker][model_name] = metrics
                except Exception as e:
                    print(f"    Error with model {model_name} on {ticker}: {e}")
                    results[ticker][model_name] = {"error": str(e)}
            
            # Print ticker results after processing each ticker.
            print(f"\nFinished ticker: {ticker}. Metrics:")
            print(json.dumps(results[ticker], indent=4))
        
        # Save results to a JSON file
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs("outputs", exist_ok=True)
        file_name = f"automatic_results-{timestamp_str}.json"
        with open(os.path.join("outputs", file_name), "w") as f:
            json.dump(results, f, indent=4)
        print("\nAutomatic mode complete. Final results:")
        print(json.dumps(results, indent=4))
    
    else:
        # --------------- TESTING MODE (Original functionality) ----------------
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
        try:
            initial_window = int(input("Input initial window: "))
        except Exception as e:
            initial_window = 60
            print("Invalid input. Using default initial window of 60.")
        
        # forecast_horizon is fixed to 1
        print("\nDisplay options:")
        while True:
            print("\nSelect display option:")
            print("1. Single model")
            print("2. All models")
            print("3. Compare selected models")
            print("4. Exit")
            display_choice = input("Enter 1, 2, 3 or 4: ").strip()
        
            if display_choice == "4":
                print("Exiting program.")
                break
            elif display_choice == "1":
                print("\nAvailable models:")
                for idx, (name, _) in MODELS.items():
                    print(f"  {idx}: {name}")
                try:
                    model_choice = int(input("Enter the model index you want to display: "))
                except Exception as e:
                    print("Invalid input.")
                    continue
                if model_choice not in MODELS:
                    print("Invalid model index.")
                    continue
                model_name, model_func = MODELS[model_choice]
                forecast_series = compute_rolling_forecast(model_func, returns, initial_window)
                plot_volatility(model_name, hist_vol, forecast_series, ticker, years_input)
                evaluate_forecast(forecast_series, hist_vol)
        
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
                    except Exception as e:
                        print(f"Error with model {name}: {e}")
                if not forecast_dict:
                    print("No models available for forecasting.")
                    continue
                same_fig = True if all_choice == "1" else False
                plot_multiple_forecasts(forecast_dict, hist_vol, ticker, years_input, same_fig=same_fig)
                # Evaluate forecast for each model in option 2
                for model_name, forecast_series in forecast_dict.items():
                    print(f"Evaluation for model {model_name}:")
                    evaluate_forecast(forecast_series, hist_vol)
        
            elif display_choice == "3":
                print("\nAvailable models:")
                for idx, (name, _) in MODELS.items():
                    print(f"  {idx}: {name}")
                sel_str = input("Enter model indices to compare (separated by dashes, e.g., 0-1-3): ").strip()
                try:
                    selected_indices = [int(x.strip()) for x in sel_str.split('-')]
                except Exception as e:
                    print("Invalid format.")
                    continue
                forecast_dict = {}
                for idx in selected_indices:
                    if idx in MODELS:
                        name, func = MODELS[idx]
                        try:
                            forecast_series = compute_rolling_forecast(func, returns, initial_window)
                            forecast_dict[name] = forecast_series
                        except Exception as e:
                            print(f"Error with model {name}: {e}")
                    else:
                        print(f"Index {idx} is not valid. Skipping.")
                if not forecast_dict:
                    print("No valid models selected for comparison.")
                    continue
                compare_choice = input("Display selected model forecasts on (1) the same graph or (2) individual graphs? Enter 1 or 2: ").strip()
                same_fig = True if compare_choice == "1" else False
                plot_multiple_forecasts(forecast_dict, hist_vol, ticker, years_input, same_fig=same_fig)
                # Evaluate forecast for each model in option 3
                for model_name, forecast_series in forecast_dict.items():
                    print(f"Evaluation for model {model_name}:")
                    evaluate_forecast(forecast_series, hist_vol)
        
            else:
                print("Invalid selection.")

if __name__ == '__main__':
    main()
