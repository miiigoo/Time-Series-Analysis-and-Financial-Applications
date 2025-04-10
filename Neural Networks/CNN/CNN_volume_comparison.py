import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy import stats
import math

# Ensure plots display in the notebook
%matplotlib inline

# ---------------------------
# 1. Load Data from Specified Path
# ---------------------------
# Load dataset containing Price and Volume from CSV file
file_path = "<INSERT FILE PATH>"
data = pd.read_csv(file_path, header=None)
data.columns = ['Price', 'Volume']

# Display first few rows to inspect data
print(data.head())
print("Total observations:", len(data))

# Plot original price data for visualization
plt.figure(figsize=(12, 6))
plt.plot(data['Price'], label='Original Price')
plt.title('Original Price Data')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.legend()
plt.show()

# ---------------------------
# 2. Preprocess Data and Split into Train/Val/Test
# ---------------------------
# Set sliding window size for sequences
window_size = 20

# Convert data to NumPy array, containing both features (Price and Volume)
features = data[['Price', 'Volume']].values
N = len(features)
train_end = int(0.8 * N)
val_end = train_end + int(0.1 * N)

# Scale the 'Price' data to [0, 1] using MinMaxScaler
price_scaler = MinMaxScaler(feature_range=(0, 1))
price_scaler.fit(features[:train_end, 0].reshape(-1, 1))
prices_scaled = price_scaler.transform(features[:, 0].reshape(-1, 1))

# Scale the 'Volume' data to a smaller range [0, 0.2]
volume_scaler = MinMaxScaler(feature_range=(0, 0.2))
volume_scaler.fit(features[:train_end, 1].reshape(-1, 1))
volumes_scaled = volume_scaler.transform(features[:, 1].reshape(-1, 1))

# Combine scaled 'Price' and 'Volume' into a single dataset
features_scaled = np.concatenate([prices_scaled, volumes_scaled], axis=1)

# Prepare train, validation, and test datasets
train_scaled = features_scaled[:train_end]
val_scaled = features_scaled[train_end - window_size: val_end]
test_scaled = features_scaled[val_end - window_size:]

# Plot scaled 'Price' data for inspection
plt.figure(figsize=(12, 6))
plt.plot(prices_scaled, label='Scaled Price')
plt.title('Scaled Price Data')
plt.xlabel('Time Step')
plt.ylabel('Scaled Price')
plt.legend()
plt.show()

# ---------------------------
# 3. Create Sliding Window Sequences
# ---------------------------
# Generates training sequences from the scaled dataset
def create_sequences(series, window_size):
    X, y = [], []
    for i in range(window_size, len(series)):
        X.append(series[i - window_size:i])
        y.append(series[i, 0])  # Target is only the Price (not Volume)
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, window_size)
X_val, y_val = create_sequences(val_scaled, window_size)
X_test, y_test = create_sequences(test_scaled, window_size)

# Display number of samples in each dataset
print("Training samples:", X_train.shape[0])
print("Validation samples:", X_val.shape[0])
print("Test samples:", X_test.shape[0])

# ---------------------------
# 4. Define the CNN Model
# ---------------------------
def build_cnn_model():
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=(window_size, 2)),
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1)  # Predicting only the price
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_cnn_model()
model.summary()

# ---------------------------
# 5. Early Stopping Callback
# ---------------------------
class CustomEarlyStopping(tf.keras.callbacks.Callback):
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
# 6. Run the CNN Model 10 Times and Collect Metrics
# ---------------------------
n_runs = 10
rmse_list = []

def safe_inverse_transform(scaled_data):
    dummy_volume = np.zeros((scaled_data.shape[0], 1))
    combined = np.concatenate([scaled_data, dummy_volume], axis=1)
    return price_scaler.inverse_transform(combined)[:, 0].reshape(-1, 1)

for run in range(n_runs):
    print(f"\n--- Run {run+1} ---")
    model = build_cnn_model()
    early_stop_callback = CustomEarlyStopping(patience=3)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=500,
        batch_size=16,
        callbacks=[early_stop_callback],
        verbose=1
    )

    predictions = model.predict(X_test)
    predictions_inv = safe_inverse_transform(predictions)
    y_test_inv = safe_inverse_transform(y_test.reshape(-1, 1))

    test_rmse = math.sqrt(mean_squared_error(y_test_inv, predictions_inv))
    rmse_list.append(test_rmse)

# ---------------------------
# 7. Compute Average Metrics and 95% Confidence Intervals
# ---------------------------
rmse_mean = np.mean(rmse_list)
rmse_std = np.std(rmse_list, ddof=1)
n = len(rmse_list)
t_val = stats.t.ppf(0.975, df=n-1)
rmse_ci = t_val * (rmse_std / np.sqrt(n))

print("\n--- Summary Across 10 Runs ---")
print(f"Average RMSE: {rmse_mean:.4f} ± {rmse_ci:.4f} (95% CI)")

# ---------------------------
# 8. Plot Forecasted vs Actual Prices from the Last Run
# ---------------------------
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Actual Prices', color='blue')
plt.plot(predictions_inv, label='Predicted Prices', color='red', linestyle='--')
plt.title(f'Forecasted vs Actual Prices (Last Run)\nAverage RMSE: {rmse_mean:.4f} ± {rmse_ci:.4f}')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.legend()
plt.show()
