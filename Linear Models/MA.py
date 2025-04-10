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
# Load the EUR/USD price data from a CSV file (single column of float data, no headers)
data_df = pd.read_csv(r"<INSERT FILE PATH>", header=None)
data = data_df.values.astype("float32").flatten()

# ---------------------------
# 2. Difference the Data for Stationarity
# ---------------------------
# Apply first-order differencing to make the time series stationary
data_diff = np.diff(data)  # Reduces data length by 1

# Store the first original value for later inversion of differencing
first_value = data[0]

# ---------------------------
# 3. Split Data: Train/Test Split
# ---------------------------
# Define the ratio of training to testing data (90% training, 10% testing)
train_ratio = 0.9
train_size = int(len(data_diff) * train_ratio)

# Split the differenced data into training and testing sets
train_diff = data_diff[:train_size]
test_diff = data_diff[train_size:]

# Store the last training point for inverting differencing during prediction
original_train_end = data[train_size]

# ---------------------------
# 4. AIC Search for Best MA(q) Model
# ---------------------------
max_q = 5  # Maximum lag order to test
aic_values = {}

# Test MA models with different orders (q) and store their AIC values
for q in range(1, max_q + 1):
    try:
        model = ARIMA(train_diff, order=(0, 0, q), enforce_invertibility=False)
        model_fit = model.fit()
        aic_values[q] = model_fit.aic  # Store AIC for each order
    except Exception as e:
        print(f"Failed for q={q}: {e}")

# Ensure that at least one model was fitted successfully
if len(aic_values) == 0:
    raise ValueError("No valid MA(q) model found on differenced training data.")

# Select the best order based on the lowest AIC value
best_q = min(aic_values, key=aic_values.get)
print(f"Best MA order on differenced data: q={best_q} (AIC={aic_values[best_q]:.4f})")

# ---------------------------
# 5. Rolling Forecast Function for MA Model
# ---------------------------
def rolling_forecast_MA_diff(train_diff, test_diff, best_q, original_train_end):
    """
    Performs rolling-window forecasting using a fixed sliding window with a selected MA model.
    Converts predictions back to the original scale by inverting differencing.
    """
    window = list(train_diff)  # Initialize rolling window with training data
    predictions_original = []
    last_known_original = original_train_end
    hits = 0

    for t in range(len(test_diff)):
        try:
            # Fit the MA model to the current window and make a prediction
            model = ARIMA(window, order=(0, 0, best_q), enforce_invertibility=False)
            model_fit = model.fit()
            yhat_diff = model_fit.forecast(steps=1)[0]
        except Exception as e:
            print(f"Forecast error at step t={t}: {e}")
            yhat_diff = np.nan
        
        # Convert prediction to original scale by adding the last known value
        forecast_original = last_known_original + yhat_diff
        predictions_original.append(forecast_original)
        
        # Calculate directional hit rate
        actual_original = last_known_original + test_diff[t]
        if np.sign(forecast_original - last_known_original) == np.sign(actual_original - last_known_original):
            hits += 1

        # Update last known original value and the rolling window
        last_known_original = actual_original
        window = window[1:] + [test_diff[t]]
    
    # Compute RMSE between actual and predicted values
    actual_original_array = np.cumsum(np.concatenate(([original_train_end], test_diff)))[1:]
    rmse_val = math.sqrt(mean_squared_error(actual_original_array, predictions_original))
    hit_rate = hits / len(test_diff)
    
    return predictions_original, rmse_val, hit_rate

# ---------------------------
# 6. Perform Multiple Runs & Collect Metrics
# ---------------------------
n_runs = 10  # Number of repeated runs for model evaluation
rmse_list = []
hit_list = []
predictions_last_run = None

# Perform rolling forecast over multiple runs to assess stability
for run in range(n_runs):
    preds_original, rmse_val, hit_rate = rolling_forecast_MA_diff(
        train_diff, test_diff, best_q, data[train_size]
    )
    rmse_list.append(rmse_val)
    hit_list.append(hit_rate)
    predictions_last_run = preds_original  # Store last run predictions for plotting
    print(f"Run {run+1}: RMSE={rmse_val:.4f}, Hit Rate={hit_rate:.2%}")

# ---------------------------
# 7. Calculate Average Metrics & Confidence Intervals
# ---------------------------
rmse_mean = np.mean(rmse_list)
rmse_std = np.std(rmse_list, ddof=1)
hit_mean = np.mean(hit_list)
hit_std = np.std(hit_list, ddof=1)
n = len(rmse_list)
t_val = stats.t.ppf(0.975, df=n-1)

rmse_ci = t_val * (rmse_std / math.sqrt(n))
hit_ci  = t_val * (hit_std / math.sqrt(n))

print("\n--- Summary Across Runs ---")
print(f"Best MA order: q={best_q}")
print(f"Avg RMSE: {rmse_mean:.4f} ± {rmse_ci:.4f} (95% CI)")
print(f"Avg Hit Rate: {hit_mean:.2%} ± {hit_ci:.2%} (95% CI)")

# ---------------------------
# 8. Plot Predictions vs. Actual Prices (Last Run)
# ---------------------------
# Rebuild the actual test array in original scale
test_original = np.cumsum(np.concatenate(([data[train_size]], test_diff)))[1:]

plt.figure(figsize=(12,6))
plt.plot(test_original, label="Actual (Test)", color="blue")
plt.plot(predictions_last_run, label="Forecast (MA, differenced)", color="red", linestyle="--")
plt.title(f"MA(q={best_q}) on Differenced Data\nRMSE={rmse_val:.4f}, HitRate={hit_rate:.2%}")
plt.xlabel("Test Index")
plt.ylabel("Price")
plt.legend()
plt.show()
