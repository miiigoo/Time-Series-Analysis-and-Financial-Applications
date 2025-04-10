import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy import stats
import math

# Ensure plots display in the notebook
%matplotlib inline

# ---------------------------
# 1. Load Data from Specified Path
# ---------------------------
# Load price data from a CSV file
file_path = "<INSERT FILE PATH>"
data = pd.read_csv(file_path, header=None)
data.columns = ['Price']

# Display the first few rows and total number of observations
print(data.head())
print("Total observations:", len(data))

# Plot original price data to visualize trends
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
window_size = 15  # Look-back period for sequences

# Convert price data to NumPy array and reshape for scaling
prices = data['Price'].values.reshape(-1, 1)
N = len(prices)
train_end = int(0.8 * N)
val_end = train_end + int(0.1 * N)

# Scale the prices using MinMaxScaler for normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(prices[:train_end])
prices_scaled = scaler.transform(prices)

# Clip scaled values to ensure they remain in [0,1] range
prices_scaled = np.clip(prices_scaled, 0, 1)

# Plot scaled price data to confirm preprocessing
plt.figure(figsize=(12, 6))
plt.plot(prices_scaled, label='Scaled Price')
plt.title('Scaled Price Data')
plt.xlabel('Time Step')
plt.ylabel('Scaled Price')
plt.legend()
plt.show()

# Create train, validation, and test datasets
train_scaled = prices_scaled[:train_end]
val_scaled = prices_scaled[train_end - window_size: val_end]
test_scaled = prices_scaled[val_end - window_size:]

# Function to create sequences using sliding windows
def create_sequences(series, window_size):
    X, y = [], []
    for i in range(window_size, len(series)):
        X.append(series[i - window_size:i])
        y.append(series[i])
    return np.array(X), np.array(y)

# Generate input-output pairs for training, validation, and testing
X_train, y_train = create_sequences(train_scaled, window_size)
X_val, y_val = create_sequences(val_scaled, window_size)
X_test, y_test = create_sequences(test_scaled, window_size)

# Display the number of samples in each set
print("Training samples:", X_train.shape[0])
print("Validation samples:", X_val.shape[0])
print("Test samples:", X_test.shape[0])

# ---------------------------
# 3. Define Custom Loss Function and Build the CNN Model
# ---------------------------
def combined_loss(alpha=1.0):
    """
    Custom loss function combining RMSE with a directional penalty.
    RMSE captures prediction accuracy, while the directional penalty
    helps ensure correct trend prediction.
    """
    def loss_fn(y_true, y_pred):
        rmse_loss = K.sqrt(K.mean(K.square(y_true - y_pred)))
        direction_loss = K.mean(K.abs(K.sign(y_true) - K.sign(y_pred))) / 2.0
        return rmse_loss + alpha * direction_loss
    return loss_fn

def build_cnn_model():
    """
    Builds a simple CNN model with two Conv1D layers followed by dense layers.
    """
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', padding='same', input_shape=(window_size, 1)),
        Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=combined_loss(alpha=1.0))
    return model

model = build_cnn_model()
model.summary()

# ---------------------------
# 4. Early Stopping Callback (Base Provided)
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
# 5. Run the CNN Model 10 Times and Collect Metrics
# ---------------------------
n_runs = 10
rmse_list = []
hitrate_list = []
last_run_predictions = None

for run in range(n_runs):
    print(f"\n--- Run {run+1} ---")
    model = build_cnn_model()
    early_stop_callback = CustomEarlyStopping(patience=3)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=500,
        batch_size=16,
        callbacks=[early_stop_callback],
        verbose=1
    )

    predictions = model.predict(X_test)
    predictions_inv = scaler.inverse_transform(predictions)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    test_rmse = math.sqrt(mean_squared_error(y_test_inv, predictions_inv))
    rmse_list.append(test_rmse)
    last_run_predictions = predictions_inv

# ---------------------------
# 6. Compute Average Metrics and 95% Confidence Intervals
# ---------------------------
rmse_mean = np.mean(rmse_list)
rmse_std = np.std(rmse_list, ddof=1)
n = len(rmse_list)
t_val = stats.t.ppf(0.975, df=n-1)
rmse_ci = t_val * (rmse_std / np.sqrt(n))

print("\n--- Summary Across 10 Runs ---")
print(f"Average RMSE: {rmse_mean:.4f} ± {rmse_ci:.4f} (95% CI)")

# ---------------------------
# 7. Plot Forecasted vs Actual Prices from the Last Run
# ---------------------------
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Actual Prices', color='blue')
plt.plot(last_run_predictions, label='Predicted Prices', color='red', linestyle='--')
plt.title(f'Forecasted vs Actual Prices (Last Run)\nAverage RMSE: {rmse_mean:.4f} ± {rmse_ci:.4f}')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.legend()
plt.show()
