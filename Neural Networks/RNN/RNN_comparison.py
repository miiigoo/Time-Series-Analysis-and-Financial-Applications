#Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
from scipy import stats
import math

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------
# 1. Load Data from Specified Path
# ---------------------------
# Load the dataset from the specified path
file_path = "<INSERT FILE PATH>csv"  
data = pd.read_csv(file_path, header=None).values.astype('float32')
# Extract the price data from the first column and reshape to 2D array for scaling
prices = data[:, 0].reshape(-1, 1)

# ---------------------------
# 2. Normalize Data
# ---------------------------
# Scale the price data to the range [0, 1] using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# ---------------------------
# 3. Prepare Sequences
# ---------------------------
def create_sequences(data, sequence_length=5):
    """
    Convert the series data into sequences of the specified length for training.
    Args:
        data (np.array): Scaled price data.
        sequence_length (int): Number of previous steps used to predict the next step.
    Returns:
        X (np.array): Input sequences.
        y (np.array): Corresponding target values.
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

sequence_length = 5  # Number of past steps used to predict the next step
X, y = create_sequences(scaled_prices, sequence_length)

# ---------------------------
# 4. Split into Training, Validation, and Testing
# ---------------------------
total_samples = len(X)
train_size = int(total_samples * 0.8)
val_size = int(total_samples * 0.1)

# Training data (80%)
X_train, y_train = X[:train_size], y[:train_size]

# Validation data (10%)
X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]

# Testing data (10%)
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

# ---------------------------
# 5. Define Hit Rate Calculation Function
# ---------------------------
def calculate_hit_rate(actual, predicted):
    """
    Calculates the directional hit rate by comparing the actual and predicted directions.
    Args:
        actual (np.array): True prices (unscaled).
        predicted (np.array): Predicted prices (unscaled).
    Returns:
        float: Hit rate as a proportion of correctly predicted directions.
    """
    actual_directions = np.diff(actual, axis=0) > 0
    predicted_directions = np.diff(predicted, axis=0) > 0
    correct_predictions = np.sum(actual_directions == predicted_directions)
    return correct_predictions / len(actual_directions)

# ---------------------------
# 6. Define Early Stopping Callback
# ---------------------------
class CustomEarlyStopping(Callback):
    """
    Custom callback to implement early stopping based on validation loss.
    """
    def __init__(self, patience=3):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.wait = 0
        self.prev_loss = None

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        if self.prev_loss is None:
            self.prev_loss = current_loss
        else:
            if current_loss >= self.prev_loss:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    self.model.stop_training = True
            else:
                self.wait = 0
            self.prev_loss = current_loss

# ---------------------------
# 7. Run RNN Model 10 Times and Collect Metrics
# ---------------------------
n_runs = 10
mse_list = []
hitrate_list = []
last_run_predictions = None

for run in range(n_runs):
    # Build the RNN model
    model = Sequential([
        SimpleRNN(50, activation='relu', input_shape=(sequence_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model with early stopping
    early_stop_callback = CustomEarlyStopping(patience=3)
    model.fit(
        X_train, y_train,
        epochs=500,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop_callback],
        verbose=1
    )

    # Generate predictions for the test data
    y_pred = model.predict(X_test)

    # Reverse normalization for evaluation
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)

    # Calculate RMSE
    rmse = math.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mse_list.append(rmse)

    # Calculate Hit Rate
    hit_rate = calculate_hit_rate(y_test_inv, y_pred_inv)
    hitrate_list.append(hit_rate)

    last_run_predictions = y_pred_inv  # Save the last run's predictions

    print(f"Run {run+1}: RMSE = {rmse:.4f}, Hit Rate = {hit_rate*100:.2f}%")

# ---------------------------
# 8. Compute Average Metrics and 95% Confidence Intervals
# ---------------------------
rmse_mean = np.mean(mse_list)
rmse_std = np.std(mse_list, ddof=1)
hit_mean = np.mean(hitrate_list)
hit_std = np.std(hitrate_list, ddof=1)
n = len(mse_list)
t_val = stats.t.ppf(0.975, df=n-1)

rmse_ci = t_val * (rmse_std / np.sqrt(n))
hit_ci = t_val * (hit_std / np.sqrt(n))

print("\n--- Summary Across 10 Runs ---")
print(f"Average RMSE: {rmse_mean:.4f} ± {rmse_ci:.4f} (95% CI)")
print(f"Average Hit Rate: {hit_mean*100:.2f}% ± {hit_ci*100:.2f}% (95% CI)")

# ---------------------------
# 9. Plot Forecasted vs Actual Prices from the Last Run
# ---------------------------
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Actual Price', color='blue')
plt.plot(last_run_predictions, label='Forecasted Price', color='red', linestyle='--')
plt.title(f'RNN Forecast vs Actual Price (Last Run)\n'
          f'Average RMSE: {rmse_mean:.4f} ± {rmse_ci:.4f}, Average Hit Rate: {hit_mean*100:.2f}% ± {hit_ci*100:.2f}%')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
