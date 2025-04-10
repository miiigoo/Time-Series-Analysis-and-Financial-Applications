import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import math
from scipy import stats

# ---------------------------
# 1. Load Data (Price Only)
# ---------------------------
# Load price data from CSV file. Assumes no header with price data in the first column.
data_df = pd.read_csv(r"<INSERT FILE PATH>", header=None)
data = data_df.values.astype('float32').flatten()

# ---------------------------
# 2. Split Data: 90% Training, 10% Testing
# ---------------------------
# Split the data into training and testing sets
train_size = int(len(data) * 0.9)
train, test = data[:train_size], data[train_size:]

# ---------------------------
# 3. Define Rolling-Window Forecast Function for ARIMA Model with Fixed Sliding Window
# ---------------------------
def rolling_forecast_ARIMA(train, test, best_order):
    """
    Performs rolling-window forecasting using an ARIMA model with a fixed sliding window.
    - Fits a new ARIMA model for each test point and makes one-step-ahead prediction.
    - Updates the rolling window by replacing the oldest observation with the latest test observation.
    - Calculates RMSE and directional hit rate.
    """
    window = list(train)
    predictions = []
    hits = 0
    n_forecasts = len(test)
    
    for t in range(n_forecasts):
        try:
            # Fit ARIMA model using current window
            model = ARIMA(window, order=best_order)
            model_fit = model.fit()
            yhat = model_fit.forecast(steps=1)  # Forecast one step ahead
        except Exception as e:
            print(f"Error at t={t}: {e}")
            yhat = [np.nan]  # Append NaN if model fitting fails
            
        predictions.append(yhat[0])
        
        # Compute directional hit rate
        previous = window[-1]
        pred_move = yhat[0] - previous
        actual_move = test[t] - previous
        if np.sign(pred_move) == np.sign(actual_move):
            hits += 1
        
        # Update window by removing oldest observation and adding the newest
        window = window[1:] + [test[t]]
    
    # Calculate RMSE and hit rate
    rmse_val = math.sqrt(mean_squared_error(test, predictions))
    hit_rate = hits / n_forecasts
    return predictions, rmse_val, hit_rate

# ---------------------------
# 4. Run Grid Search and Rolling Forecast 10 Times
# ---------------------------
n_runs = 10
rmse_list = []
hitrate_list = []
best_order_list = []
last_run_predictions = None

# ARIMA model grid search over (p, d, q) combinations
for run in range(n_runs):
    aic_values = {}
    for p in range(1, 5):
        for d in range(1, 2):
            for q in range(1, 5):
                try:
                    model = ARIMA(train, order=(p, d, q))
                    model_fit = model.fit()
                    aic_values[(p, d, q)] = model_fit.aic  # Store AIC value for model comparison
                except Exception as e:
                    print(f"Run {run+1}, order ({p},{d},{q}) failed: {e}")
    
    if len(aic_values) == 0:
        print(f"No ARIMA order could be fit for run {run+1}")
        continue

    # Select the best model based on the minimum AIC value
    best_order = min(aic_values, key=aic_values.get)
    best_order_list.append(best_order)
    print(f"Run {run+1}: Best ARIMA order: {best_order} with AIC: {aic_values[best_order]:.4f}")
    
    # Perform rolling-window forecasting with the selected best order
    preds, run_rmse, run_hit = rolling_forecast_ARIMA(train, test, best_order)
    rmse_list.append(run_rmse)
    hitrate_list.append(run_hit)
    last_run_predictions = preds  # Store predictions from the last run for plotting
    print(f"Run {run+1}: RMSE = {run_rmse:.4f}, Hit Rate = {run_hit:.2%}")

# ---------------------------
# 5. Compute Average Metrics and 95% Confidence Intervals
# ---------------------------
# Calculate the average and standard deviation of RMSE and hit rate
rmse_mean = np.mean(rmse_list)
rmse_std = np.std(rmse_list, ddof=1)
hit_mean = np.mean(hitrate_list)
hit_std = np.std(hitrate_list, ddof=1)
n = len(rmse_list)
t_val = stats.t.ppf(0.975, df=n-1)  # For 95% confidence interval calculation

# Compute 95% confidence intervals for RMSE and hit rate
rmse_ci = t_val * (rmse_std / np.sqrt(n))
hit_ci = t_val * (hit_std / np.sqrt(n))

print("\n--- Summary Across 10 Runs ---")
print(f"Average RMSE: {rmse_mean:.4f} ± {rmse_ci:.4f} (95% CI)")
print(f"Average Hit Rate: {hit_mean:.2%} ± {hit_ci:.2%} (95% CI)")
print(f"Best ARIMA orders across runs: {best_order_list}")

# ---------------------------
# 6. Plot Actual vs. Predicted Prices from the Last Run
# ---------------------------
plt.figure(figsize=(12, 6))
plt.plot(test, label='Actual Price')
plt.plot(last_run_predictions, label='Predicted Price', color='red')
plt.title(f'ARIMA Model Rolling Forecast (Best Order from Last Run: {best_order_list[-1]})\n'
          f'Average RMSE: {rmse_mean:.4f} ± {rmse_ci:.4f}, Average Hit Rate: {hit_mean:.2%} ± {hit_ci:.2%}')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.legend()
plt.show()
