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
# Load EUR/USD price data from CSV file, assuming it has no headers
data_df = pd.read_csv(r"<INSERT FILE PATH>", header=None)
data = data_df.values.astype('float32').flatten()

# ---------------------------
# 2. Split Data: 90% Training, 10% Testing
# ---------------------------
# Define training and testing size
train_size = int(len(data) * 0.9)
train, test = data[:train_size], data[train_size:]

# ---------------------------
# 3. Define Rolling-Window Forecast Function for ARMA Model Using Fixed Sliding Window
# ---------------------------
def rolling_forecast_ARMA(train, test, best_order):
    """
    Perform rolling-window forecasting using an ARMA model with fixed sliding window.
    - The sliding window is updated by dropping the oldest point and adding the newest.
    - Computes both RMSE and directional hit rate.
    """
    window = list(train)  # Initialize the rolling window with training data
    predictions = []
    hits = 0
    n_forecasts = len(test)
    p, q = best_order  # Extract AR and MA orders from the best_order tuple

    for t in range(n_forecasts):
        try:
            # Fit the ARMA model on the current window
            model = ARIMA(window, order=(p, 0, q))
            model_fit = model.fit()
            yhat = model_fit.forecast(steps=1)  # Forecast next point
        except Exception as e:
            print(f"Error at t={t}: {e}")
            yhat = [np.nan]
        
        predictions.append(yhat[0])
        
        # Compute directional hit: Compare the predicted movement vs. actual movement
        previous = window[-1]
        pred_move = yhat[0] - previous
        actual_move = test[t] - previous
        if np.sign(pred_move) == np.sign(actual_move):
            hits += 1
        
        # Update the rolling window
        window = window[1:] + [test[t]]

    # Calculate RMSE and hit rate
    rmse_val = math.sqrt(mean_squared_error(test, predictions))
    hit_rate = hits / n_forecasts
    return predictions, rmse_val, hit_rate

# ---------------------------
# 4. Run AIC Grid Search and Rolling-Window Forecast 10 Times
# ---------------------------
n_runs = 10  # Number of independent runs
max_p = 5   # Maximum AR order to test
max_q = 5   # Maximum MA order to test
rmse_list = []
hitrate_list = []
best_order_list = []
predictions_last = None

for run in range(n_runs):
    aic_values = {}

    # Perform grid search over (p, q) combinations
    for p in range(0, max_p + 1):
        for q in range(1, max_q + 1):
            try:
                model = ARIMA(train, order=(p, 0, q))
                model_fit = model.fit()
                aic_values[(p, q)] = model_fit.aic  # Store AIC values for comparison
            except Exception as e:
                print(f"Run {run+1}, order ({p},{q}) failed: {e}")
    
    if len(aic_values) == 0:
        print("No valid ARMA model found for run", run+1)
        continue

    # Select the best model based on minimum AIC
    best_order = min(aic_values, key=aic_values.get)
    best_order_list.append(best_order)
    print(f"Run {run+1}: Best ARMA order: {best_order} with AIC: {aic_values[best_order]:.4f}")
    
    # Perform rolling forecast using the selected ARMA order
    preds, run_rmse, run_hit = rolling_forecast_ARMA(train, test, best_order)
    rmse_list.append(run_rmse)
    hitrate_list.append(run_hit)
    predictions_last = preds  # Store predictions from the last run for plotting
    print(f"Run {run+1}: RMSE = {run_rmse:.4f}, Hit Rate = {run_hit:.2%}")

# ---------------------------
# 5. Compute Average Metrics and 95% Confidence Intervals
# ---------------------------
# Calculate the mean and standard deviation of RMSE and hit rate across runs
rmse_mean = np.mean(rmse_list)
rmse_std = np.std(rmse_list, ddof=1)
hit_mean = np.mean(hitrate_list)
hit_std = np.std(hitrate_list, ddof=1)
n = len(rmse_list)
t_val = stats.t.ppf(0.975, df=n-1)  # For 95% confidence interval

# Compute confidence intervals
rmse_ci = t_val * (rmse_std / np.sqrt(n))
hit_ci = t_val * (hit_std / np.sqrt(n))

print("\n--- Summary Across Runs ---")
print(f"Average RMSE: {rmse_mean:.4f} ± {rmse_ci:.4f} (95% CI)")
print(f"Average Hit Rate: {hit_mean:.2%} ± {hit_ci:.2%} (95% CI)")
print(f"Best ARMA orders in runs: {best_order_list}")

# ---------------------------
# 6. Plot Actual vs. Predicted Prices (Last Run)
# ---------------------------
plt.figure(figsize=(12, 6))
plt.plot(test, label='Actual Price')
plt.plot(predictions_last, label='Predicted Price', color='red')
plt.title(f'ARMA Model Rolling Forecast (Best Order from Last Run: {best_order_list[-1]})\n'
          f'Avg RMSE: {rmse_mean:.4f} ± {rmse_ci:.4f}, Avg Hit Rate: {hit_mean:.2%} ± {hit_ci:.2%}')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.legend()
plt.show()
