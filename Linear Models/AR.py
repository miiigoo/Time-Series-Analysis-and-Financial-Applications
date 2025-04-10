import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import math
from scipy import stats

# ---------------------------
# 1. Load Data (Price Only)
# ---------------------------
# Load EUR/USD price data from a CSV file without headers
data_df = pd.read_csv(r"<INSERT FILE PATH>", header=None)
data = data_df.values.astype('float32').flatten()

# ---------------------------
# 2. Split Data: 90% Training, 10% Testing
# ---------------------------
train_size = int(len(data) * 0.9)
train, test = data[:train_size], data[train_size:]

# ---------------------------
# 3. Rolling-Window Forecast Function for AR Model
# ---------------------------
def rolling_forecast_AR(train, test, best_order):
    """
    Performs rolling-window forecasting using a fixed sliding window approach.
    Returns predictions, RMSE, and directional hit rate.
    """
    window = list(train)  # Initialize sliding window with training data
    predictions = []
    hits = 0
    n_forecasts = len(test)
    
    for t in range(n_forecasts):
        try:
            model = AutoReg(window, lags=best_order, old_names=False)
            model_fit = model.fit()
            yhat = model_fit.predict(start=len(window), end=len(window), dynamic=False)
        except Exception as e:
            print(f"Error at t={t}: {e}")
            yhat = [np.nan]
        
        predictions.append(yhat[0])
        
        # Calculate directional hit rate
        previous = window[-1]
        pred_move = yhat[0] - previous
        actual_move = test[t] - previous
        if np.sign(pred_move) == np.sign(actual_move):
            hits += 1
        
        # Update the window by sliding forward
        window = window[1:] + [test[t]]
    
    rmse_val = math.sqrt(mean_squared_error(test, predictions))
    hit_rate = hits / n_forecasts
    return predictions, rmse_val, hit_rate

# ---------------------------
# 4. AIC Search & Forecasting (Repeated 10 Times)
# ---------------------------
n_runs = 10
max_lag = 5  # Test AR orders from 0 to 5
rmse_list = []
hitrate_list = []
best_order_list = []
last_run_predictions = None

for run in range(n_runs):
    aic_values = {}
    # Search for the best AR model order using AIC
    for lag in range(0, max_lag + 1):
        try:
            model = AutoReg(train, lags=lag, old_names=False)
            model_fit = model.fit()
            aic_values[lag] = model_fit.aic
        except Exception as e:
            print(f"Run {run+1}, order {lag} failed: {e}")
    
    if len(aic_values) == 0:
        print("No AR order could be fit for run", run+1)
        continue
    
    # Select best order with lowest AIC
    current_best_order = min(aic_values, key=aic_values.get)
    best_order_list.append(current_best_order)
    print(f"Run {run+1}: Best AR order: {current_best_order} with AIC: {aic_values[current_best_order]:.4f}")
    
    # Perform forecasting using the selected order
    preds, run_rmse, run_hit = rolling_forecast_AR(train, test, current_best_order)
    rmse_list.append(run_rmse)
    hitrate_list.append(run_hit)
    last_run_predictions = preds  # Store last run predictions for plotting
    print(f"Run {run+1}: RMSE = {run_rmse:.4f}, Hit Rate = {run_hit:.2%}")

# ---------------------------
# 5. Compute Metrics & Confidence Intervals
# ---------------------------
rmse_mean = np.mean(rmse_list)
rmse_std = np.std(rmse_list, ddof=1)
hit_mean = np.mean(hitrate_list)
hit_std = np.std(hitrate_list, ddof=1)
n = n_runs
t_val = stats.t.ppf(0.975, df=n-1)

rmse_ci = t_val * (rmse_std / np.sqrt(n))
hit_ci = t_val * (hit_std / np.sqrt(n))

print("\n--- Summary Across 10 Runs ---")
print(f"Average RMSE: {rmse_mean:.4f} ± {rmse_ci:.4f} (95% CI)")
print(f"Average Hit Rate: {hit_mean:.2%} ± {hit_ci:.2%} (95% CI)")
print(f"Best orders chosen in runs: {best_order_list}")

# ---------------------------
# 6. Plot Predictions vs. Actual Prices
# ---------------------------
plt.figure(figsize=(12, 6))
plt.plot(test, label='Actual Price')
plt.plot(last_run_predictions, label='Predicted Price', color='red')
plt.title(f'AR Model Rolling Forecast (Best Order from Last Run: {best_order_list[-1]})\n'
          f'Average RMSE: {rmse_mean:.4f} ± {rmse_ci:.4f}, Average Hit Rate: {hit_mean:.2%} ± {hit_ci:.2%}')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.legend()
plt.show()
