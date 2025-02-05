import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error

def rolling_forecast(returns, model_type, initial_window):
    """
    Perform a rolling one-step-ahead forecast using the specified ARCH-type model.
    
    Parameters:
      returns (pd.Series): Time series of log returns.
      model_type (str): One of 'GARCH', 'EGARCH', 'GJR-GARCH', 'APARCH', 'FIGARCH'.
      initial_window (int): Number of initial observations to use for the first estimation.
      
    Returns:
      forecast_series (pd.Series): Forecasted volatility (conditional standard deviation).
      actual_series (pd.Series): "Realized" volatility proxy (absolute returns) for the forecast day.
    """
    predictions = []
    actuals = []
    forecast_dates = []
    
    # Loop through the data starting from the end of the initial window.
    for t in range(initial_window, len(returns) - 1):
        # Use data from the beginning up to time t as the training sample.
        train = returns.iloc[:t]
        #train = train * 100
        test = returns.iloc[t + 1]  # the "actual" return on day t+1
        
        try:
            # Set up the appropriate model based on model_type.
            if model_type == 'GARCH':
                am = arch_model(train, vol='Garch', p=1, q=1, dist='normal')
            elif model_type == 'EGARCH':
                am = arch_model(train, vol='EGARCH', p=1, q=1, dist='normal')
            elif model_type == 'GJR-GARCH':
                # In arch_model, GJR-GARCH is specified by setting an 'o' (asymmetry) term.
                am = arch_model(train, vol='Garch', p=1, o=1, q=1, dist='normal')
            elif model_type == 'APARCH':
                am = arch_model(train, vol='APARCH', p=1, o=1, q=1, dist='normal')
                res = am.fit(disp='off', options={'maxiter': 10000, 'ftol': 1e-6})

                #am = arch_model(train, vol='APARCH', p=1, o=1, q=1, dist='normal')
            elif model_type == 'FIGARCH':
                am = arch_model(train, vol='FIGARCH', p=1, q=1, dist='normal')
            else:
                raise ValueError("Unknown model type")
                
            # Fit the model. Setting disp='off' to suppress fitting output.
            res = am.fit(disp='off')
            
            # Forecast one step ahead.
            # The forecast() method returns an object whose 'variance' attribute is a DataFrame.
            forecast = res.forecast(horizon=1)
            # Get the forecasted variance for the next period.
            # (It is common to index using .iloc[-1, 0] which corresponds to the forecast for t+1.)
            variance_forecast = forecast.variance.iloc[-1, 0]
            # Convert variance to standard deviation.
            predicted_vol = np.sqrt(variance_forecast)
        except Exception as e:
            # In case the model fails to converge or another error occurs,
            # record a NaN for this date.
            print(f"Error in {model_type} forecast at index {returns.index[t+1]}: {e}")
            predicted_vol = np.nan
        
        predictions.append(predicted_vol)
        # We use the absolute value of the return as a simple proxy for the realized volatility.
        actuals.append(abs(test))
        forecast_dates.append(returns.index[t+1])
    
    forecast_series = pd.Series(predictions, index=forecast_dates)
    actual_series = pd.Series(actuals, index=forecast_dates)
    return forecast_series, actual_series

def main():
    # Prompt user for ticker symbol and look-back period in years.
    ticker = input("Enter ticker symbol (e.g., AAPL): ").strip().upper()
    try:
        years = float(input("Enter number of years to look back: "))
    except ValueError:
        print("Invalid input for years. Please enter a number.")
        return

    # Define the date range.
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=years)
    print(start_date)
    import time
    
    # Download historical data from Yahoo Finance.
    print(f"Downloading data for {ticker} from {start_date.date()} to {end_date.date()}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        print("No data downloaded. Please check the ticker symbol and try again.")
        return
    
    # Compute log returns from the Adjusted Close prices.
    data['Log_Return'] = np.log(data['Adj Close']).diff()
    returns = data['Log_Return'].dropna()
    
    # Determine an initial training window.
    # Here we use the first 50% of the data for model estimation.
    initial_window = int(len(returns) * 0.5)
    if initial_window < 30:
        initial_window = 30  # Ensure a minimum window length
    
    print(f"Using {initial_window} observations for the initial training window out of {len(returns)} total observations.")
    
    # List of models to run.
    model_names = ['GARCH', 'EGARCH', 'GJR-GARCH', 'APARCH', 'FIGARCH']
    
    # Dictionary to hold results for each model.
    results = {}
    
    # Loop through each model, perform rolling forecast, compute error metrics, and plot results.
    for model in model_names:
        print(f"\nRunning rolling forecast for {model} model...")
        pred_vol, actual_vol = rolling_forecast(returns, model, initial_window)
        
        # Remove any NaN values that might have resulted from estimation errors.
        valid_idx = (~pred_vol.isna()) & (~actual_vol.isna())
        pred_valid = pred_vol[valid_idx]
        actual_valid = actual_vol[valid_idx]
        
        # Calculate error metrics.
        mae = mean_absolute_error(actual_valid, pred_valid)
        mse = mean_squared_error(actual_valid, pred_valid)
        r2 = r2_score(actual_valid, pred_valid)
        # Mean Squared Log Error expects non-negative values; our volatilities are non-negative.
        msle = mean_squared_log_error(actual_valid, pred_valid)
        
        # Save the results.
        results[model] = {
            'forecast': pred_vol,
            'actual': actual_vol,
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'msle': msle
        }
        
        # Plot the historical volatility (absolute returns) and forecasted volatility.
        plt.figure(figsize=(10, 6))
        plt.plot(actual_vol.index, actual_vol, label='Historical Volatility (|Return|)', color='black')
        plt.plot(pred_vol.index, pred_vol, label='Forecasted Volatility', color='orange')
        plt.title(f"{model} Model: Forecasted vs. Historical Volatility")
        plt.xlabel("Date")
        plt.ylabel("Volatility")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # Print out the error metrics for each model.
    print("\nError Metrics for Each Model:")
    for model, metrics in results.items():
        print(f"\n--- {model} ---")
        print(f"Mean Absolute Error (MAE): {metrics['mae']:.6f}")
        print(f"Mean Squared Error (MSE): {metrics['mse']:.6f}")
        print(f"R2 Score: {metrics['r2']:.6f}")
        print(f"Mean Squared Log Error (MSLE): {metrics['msle']:.6f}")

if __name__ == "__main__":
    main()
