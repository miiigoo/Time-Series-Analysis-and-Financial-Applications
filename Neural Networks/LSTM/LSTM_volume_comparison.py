import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import Callback
from scipy import stats
import math

# ---------------------------
# 1. Set Random Seeds for Reproducibility
# ---------------------------
np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------
# 2. Load and Preprocess Data
# ---------------------------
# Load CSV file containing price and volume data (two columns: price, volume)
data_df = pd.read_csv('<INSERT FILE PATH>', header=None)

# Convert data to float format; shape: (num_samples, 2)
data = data_df.values.astype('float32')

# --- Separate Scaling for Price and Volume ---
price_scaler = MinMaxScaler(feature_range=(0, 1))  # Scale prices to [0, 1]
prices = data[:, 0].reshape(-1, 1)
prices_scaled = price_scaler.fit_transform(prices)

volume_scaler = MinMaxScaler(feature_range=(0, 0.2))  # Scale volume to [0, 0.2]
volumes = data[:, 1].reshape(-1, 1)
volumes_scaled = volume_scaler.fit_transform(volumes)

# Combine scaled prices and volumes back together
data_scaled = np.concatenate([prices_scaled, volumes_scaled], axis=1)

# ---------------------------
# 3. Create Dataset
# ---------------------------
look_back = 20  # Number of previous time steps used for prediction

def create_dataset(dataset, look_back=20):
    """
    Create sequences from the dataset for LSTM training.
    Inputs: All features over a rolling window of length 'look_back'.
    Target: The price value at the next time step.
    """
    X, y = [], []
    for i in range(len(dataset) - look_back):
        seq = dataset[i:(i + look_back), :]
        target = dataset[i + look_back, 0]  # Target is the price at the next time step
        X.append(seq)
        y.append(target)
    return np.array(X), np.array(y)

# Split data into training (80%), validation (10%), and testing (10%)
n = len(data_scaled)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

train_data = data_scaled[:train_end]
val_data = data_scaled[train_end:val_end]
test_data = data_scaled[val_end:]

# Create sequences for training, validation, and testing
X_train, y_train = create_dataset(train_data, look_back)
X_val, y_val = create_dataset(val_data, look_back)
test_input = np.concatenate((val_data[-look_back:], test_data), axis=0)
X_test, y_test = create_dataset(test_input, look_back)

# Reshape inputs to fit LSTM model (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 2))
X_val   = X_val.reshape((X_val.shape[0],   X_val.shape[1],   2))
X_test  = X_test.reshape((X_test.shape[0],  X_test.shape[1],  2))

# ---------------------------
# 4. Define Custom Early Stopping Callback
# ---------------------------
class CustomEarlyStopping(Callback):
    """
    Custom Early Stopping callback to prevent overfitting.
    Stops training when validation loss does not improve for 'patience' epochs.
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
# 5. Build LSTM Model
# ---------------------------
def build_lstm_model():
    """
    Build the LSTM model with 100 units and a dropout layer.
    """
    model = Sequential()
    model.add(LSTM(100, input_shape=(look_back, 2)))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer for price prediction
    return model

# ---------------------------
# 6. Train Model 10 Times and Collect Metrics
# ---------------------------
n_runs = 10
rmse_list = []
hitrate_list = []
last_run_predictions = None

epochs = 1000
batch_size = 16

for run in range(n_runs):
    print(f"\n--- Run {run+1} ---")
    model = build_lstm_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    early_stop_callback = CustomEarlyStopping(patience=3)

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stop_callback],
                        verbose=1)

    predictions = model.predict(X_test)
    predictions_inv = price_scaler.inverse_transform(predictions)

    y_test_inv = price_scaler.inverse_transform(y_test.reshape(-1, 1))

    test_rmse = np.sqrt(mean_squared_error(y_test_inv, predictions_inv))
    rmse_list.append(test_rmse)
    last_run_predictions = predictions_inv  # Store predictions from the last run

# ---------------------------
# 7. Compute Confidence Intervals
# ---------------------------
rmse_mean = np.mean(rmse_list)
rmse_std = np.std(rmse_list, ddof=1)
n = len(rmse_list)
t_val = stats.t.ppf(0.975, df=n-1)
rmse_ci = t_val * (rmse_std / np.sqrt(n))

print("\n--- Summary Across 10 Runs ---")
print(f"Average RMSE: {rmse_mean:.4f} ± {rmse_ci:.4f} (95% CI)")

# ---------------------------
# 8. Plot Actual vs Predicted Prices
# ---------------------------
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Actual Prices', color='blue')
plt.plot(last_run_predictions, label='Predicted Prices', color='red', linestyle='--')
plt.title(f"Actual vs Predicted Prices (Last Run)\nAverage RMSE: {rmse_mean:.4f} ± {rmse_ci:.4f}")
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.legend()
plt.show()
