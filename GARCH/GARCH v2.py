import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from arch import arch_model  # For comparison

def garch_model(returns, p=1, q=1, forecast_horizon=10):
    """
    Estimate a GARCH(p, q) model and forecast future volatility.

    Parameters:
    - returns: pandas Series of log returns.
    - p: Order of GARCH terms (lagged conditional variances).
    - q: Order of ARCH terms (lagged squared residuals).
    - forecast_horizon: Number of days ahead to forecast volatility.

    Returns:
    - params: Estimated parameters of the GARCH model.
    - sigma2: Estimated conditional variances.
    - forecast_volatility: Forecasted volatility for the specified horizon.
    """
    # Number of observations
    T = len(returns)

    # Mean return
    mu = returns.mean()
    epsilon = returns - mu

    # Initial parameter guesses
    # Parameters: [omega, alpha1, ..., alpha_q, beta1, ..., beta_p]
    initial_omega = 1e-6  # More realistic initial guess
    initial_alphas = [0.05] * q
    initial_betas = [0.90 / p] * p  # Distribute 0.90 equally among betas
    initial_params = [initial_omega] + initial_alphas + initial_betas

    # Bounds: omega > 0, alpha_i >= 0, beta_j >= 0
    bounds = [(1e-6, None)] + [(1e-6, 1) for _ in range(q + p)]

    # Constraints: sum of alphas and betas < 1 (stationarity)
    def constraint(params):
        alpha_params = params[1:1+q]
        beta_params = params[1+q:]
        return 1 - np.sum(alpha_params) - np.sum(beta_params)

    constraints = [{'type': 'ineq', 'fun': constraint}]

    # Define the GARCH(p, q) log-likelihood function (negative for minimization)
    def garch_log_likelihood(params):
        omega = params[0]
        alphas = params[1:1+q]
        betas = params[1+q:]

        sigma2 = np.zeros(T)
        # Initialize with the sample variance
        sigma2[:max(p, q)] = np.var(returns)

        for t in range(max(p, q), T):
            sigma2[t] = omega
            for i in range(1, q+1):
                sigma2[t] += alphas[i-1] * epsilon.iloc[t - i]**2
            for j in range(1, p+1):
                sigma2[t] += betas[j-1] * sigma2[t - j]

            # Ensure positive variance
            if sigma2[t] <= 0:
                sigma2[t] = 1e-6

        # Compute negative log-likelihood only for t >= max(p, q)
        ll_elements = 0.5 * (
            np.log(2 * np.pi) +
            np.log(sigma2[max(p, q):]) +
            (epsilon.iloc[max(p, q):].values**2) / sigma2[max(p, q):]
        )
        negative_log_likelihood = -np.sum(ll_elements)

        return negative_log_likelihood

    # Optimize the parameters using SLSQP method for better constraint handling
    result = minimize(
        garch_log_likelihood,
        initial_params,
        bounds=bounds,
        constraints=constraints,
        method='SLSQP',
        options={'maxiter': 1000, 'disp': False}
    )

    

    # Extract the estimated parameters
    omega = result.x[0]
    alphas = result.x[1:1+q]
    betas = result.x[1+q:]

    params = {'omega': omega, 'alphas': alphas, 'betas': betas}

    # Compute the conditional variances with the estimated parameters
    sigma2 = np.zeros(T)
    sigma2[:max(p, q)] = np.var(returns)

    for t in range(max(p, q), T):
        sigma2[t] = omega
        for i in range(1, q+1):
            sigma2[t] += alphas[i-1] * epsilon.iloc[t - i]**2
        for j in range(1, p+1):
            sigma2[t] += betas[j-1] * sigma2[t - j]

        if sigma2[t] <= 0:
            sigma2[t] = 1e-6

    # Forecast future volatility
    forecast_sigma2 = np.zeros(forecast_horizon)
    # Start from the last computed variance
    forecast_sigma2[0] = omega
    for i in range(1, q+1):
        if T - i >= 0:
            forecast_sigma2[0] += alphas[i-1] * epsilon.iloc[T - i]**2
    for j in range(1, p+1):
        if T - j >= 0:
            forecast_sigma2[0] += betas[j-1] * sigma2[T - j]

    for t in range(1, forecast_horizon):
        forecast_sigma2[t] = omega
        for i in range(q):
            # Future residuals are assumed to be zero
            forecast_sigma2[t] += alphas[i] * 0
        for j in range(p):
            forecast_sigma2[t] += betas[j] * forecast_sigma2[t - j - 1]

    forecast_volatility = np.sqrt(forecast_sigma2)

    return params, sigma2, forecast_volatility

# ---------------------------

# Fetch S&P 500 data using yfinance
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2023, 9, 30)
ticker = '^GSPC'
data = yf.download(ticker, start=start, end=end)

# Calculate daily log returns
data['Returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
returns = data['Returns'].dropna()

# Verify the scale of returns
print("First 5 Returns:")
print(returns.head())
print(f"\nMean Return: {returns.mean()}")
print(f"Variance of Returns: {returns.var()}")

# Calculate historical volatility (21-day rolling window, annualized)
data['Historical_Volatility'] = data['Returns'].rolling(window=21).std() * np.sqrt(252)

# Specify GARCH order
p = 1  # Order of GARCH terms
q = 1  # Order of ARCH terms
forecast_horizon = 10  # Days ahead to forecast

# Estimate the GARCH(p, q) model and forecast volatility
params, sigma2, forecast_volatility = garch_model(returns, p, q, forecast_horizon)

# Print estimated parameters
print("\nEstimated Parameters:")
print(f"omega: {params['omega']}")
for i, alpha in enumerate(params['alphas']):
    print(f"alpha{i+1}: {alpha}")
for j, beta in enumerate(params['betas']):
    print(f"beta{j+1}: {beta}")

# Compare with arch library
print("\nComparison with 'arch' library:")

# Rescale returns for arch library as per the warning
scaled_returns = returns * 100

# Fit GARCH(1,1) model using arch library on scaled returns
am_scaled = arch_model(scaled_returns, vol='Garch', p=1, q=1, mean='Constant', dist='normal')
res_scaled = am_scaled.fit(disp='off')
print(res_scaled.summary())

# Plot the estimated conditional volatility and historical volatility
plt.figure(figsize=(14, 7))

# Custom GARCH Model Estimated Volatility
plt.plot(
    returns.index,
    np.sqrt(sigma2) * np.sqrt(252),
    label='Estimated Volatility (Custom GARCH)',
    color='blue'
)

# arch Library GARCH Model Estimated Volatility
plt.plot(
    scaled_returns.index,
    res_scaled.conditional_volatility / 100 * np.sqrt(252),
    label='Estimated Volatility (arch GARCH)',
    color='orange',
    alpha=0.7
)

# Historical Volatility (21-day Rolling)
plt.plot(
    data.index,
    data['Historical_Volatility'],
    label='Historical Volatility (21-day Rolling)',
    color='red',
   
   
)

# Enhancements
plt.title(f'GARCH({p}, {q}) Estimated Volatility vs. Historical Volatility')
plt.xlabel('Date')
plt.ylabel('Annualized Volatility')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot the forecasted volatility
plt.figure(figsize=(10, 5))
plt.plot(range(1, forecast_horizon+1), forecast_volatility * np.sqrt(252), marker='o', label='Forecasted Volatility (Custom GARCH)')
plt.title('Forecasted Volatility')
plt.xlabel('Days Ahead')
plt.ylabel('Annualized Volatility')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()







