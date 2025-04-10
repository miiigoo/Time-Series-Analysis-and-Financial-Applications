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
# 1. Set Seeds for Reproducibility
# ---------------------------
np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------
# 2. Load and Preprocess Data
# ---------------------------
file_path = "<INSERT FILE PATH>"  # File path to the dataset
data_df = pd.read_csv(file_path, header=None)
data = data_df.values.astype('float32').flatten()  # Convert data to a 1D numpy array

# Reshape data to 2D for scaling
data = data.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# ---------------------------
# 3. Prepare Sequences
# ---------------------------
def create_dataset(dataset, look_back=5):
    """
    Generates input-output sequences for the LSTM model.
    Each sequence consists of 'look_back' time steps, and the target is the subsequent value.
    """
    X, y = [], []
    for i in range(len(dataset) - look_back):
        seq = dataset[i:(i + look_back), 0]
        target = dataset[i + look_back, 0]
        X.append(seq)
        y.append(target)
    return np.array(X), np.array(y)

# Define the look-back period for sequence generation
look_back = 5
X, y = create_dataset(data_scaled, look_back=look_back)

# ---------------------------
# 4. Split Data into Train, Validation, and Test Sets
# ---------------------------
n = len(data_scaled)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

# Split data into training, validation, and testing sets
train_data = data_scaled[:train_end]
val_data = data_scaled[train_end:val_end]
test_data = data_scaled[val_end:]

# Generate sequences for each dataset
X_train, y_train = create_dataset(train_data, look_back=look_back)
X_val, y_val = create_dataset(val_data, look_back=look_back)

# Prepare test data by appending the last 'look_back' points from the validation set
test_input = np.concatenate((val_data[-look_back:], test_data), axis=0)
X_test, y_test = create_dataset(test_input, look_back=look_back)

# Reshape inputs to match LSTM requirements: [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_val   = np.reshape(X_val,   (X_val.shape[0],   X_val.shape[1],   1))
X_test  = np.reshape(X_test,  (X_test.shape[0],  X_test.shape[1],  1))

# ---------------------------
# 5. Define Custom Early Stopping Callback
# ---------------------------
class CustomEarlyStopping(Callback):
    """
    Early stopping mechanism to prevent overfitting by monitoring validation loss.
    """
    def __init__(self, patience=3):  # Patience set to 3 epochs
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

# Define custom RMSE loss function
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# ---------------------------
# 6. Train and Evaluate the LSTM Model
# ---------------------------
n_runs = 10
rmse_list = []
hitrate_list = []
last_run_predictions = None

for run in range(n_runs):
    print(f"\n--- Run {run+1} ---")
    
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(100, input_shape=(look_back, 1)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=rmse)
    
    model.summary()
    
    # Train the model with early stopping
    early_stop_callback = CustomEarlyStopping(patience=3)
    history = model.fit(X_train, y_train,
                        epochs=1000,
                        batch_size=16,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stop_callback],
                        verbose=0)

    # Generate predictions on the test set
    predictions = model.predict(X_test)

    # Inverse-transform predictions and actual test values
    predictions_inv = scaler.inverse_transform(predictions)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate RMSE
    test_rmse = math.sqrt(mean_squared_error(y_test_inv, predictions_inv))
    
    # Calculate Directional Hit Rate
    last_values_scaled = X_test[:, -1, 0]
    last_values_unscaled = scaler.inverse_transform(last_values_scaled.reshape(-1, 1)).flatten()
    preds_flat = predictions_inv.flatten()
    y_test_flat = y_test_inv.flatten()
    pred_diff = preds_flat - last_values_unscaled
    actual_diff = y_test_flat - last_values_unscaled
    pred_direction = np.sign(pred_diff)
    actual_direction = np.sign(actual_diff)
    hit_count = np.sum(pred_direction == actual_direction)
    hit_rate = hit_count / len(pred_direction)
    
    rmse_list.append(test_rmse)
    hitrate_list.append(hit_rate)
    last_run_predictions = predictions_inv

# ---------------------------
# 7. Compute Average Metrics and 95% Confidence Intervals
# ---------------------------
rmse_mean = np.mean(rmse_list)
rmse_std = np.std(rmse_list, ddof=1)
hit_mean = np.mean(hitrate_list)
hit_std = np.std(hitrate_list, ddof=1)
n = len(rmse_list)
t_val = stats.t.ppf(0.975, df=n-1)
rmse_ci = t_val * (rmse_std / np.sqrt(n))
hit_ci = t_val * (hit_std / np.sqrt(n))

print(f"\nAverage RMSE: {rmse_mean:.4f} ± {rmse_ci:.4f}")
print(f"Average Hit Rate: {hit_mean*100:.2f}% ± {hit_ci*100:.2f}%")

# ---------------------------
# 8. Plot Results
# ---------------------------
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Actual Prices', color='blue')
plt.plot(last_run_predictions, label='Predicted Prices', color='red', linestyle='--')
plt.title('LSTM Model: Actual vs Predicted Prices')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.legend()
plt.show()
