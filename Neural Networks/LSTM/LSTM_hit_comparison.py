import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K  # for our custom loss
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
data_df = pd.read_csv('<INSERT FILE PATH>', header=None)
data = data_df.values.astype('float32').flatten()
data = data.reshape(-1, 1)

# Normalize data to the range [0,1]
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

look_back = 10  # Fixed look-back period for sequence generation

# ---------------------------
# 3. Prepare Sequences
# ---------------------------
def create_dataset(dataset, look_back=10):
    """
    Generates input-output sequences for training.
    Inputs are sequences of length 'look_back', targets are the subsequent value.
    """
    X, y = [], []
    for i in range(len(dataset) - look_back):
        seq = dataset[i:(i + look_back), 0]
        target = dataset[i + look_back, 0]
        X.append(seq)
        y.append(target)
    return np.array(X), np.array(y)

# Splitting data into training, validation, and test sets
n = len(data_scaled)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

train_data = data_scaled[:train_end]
val_data = data_scaled[train_end:val_end]
test_data = data_scaled[val_end:]

X_train, y_train = create_dataset(train_data, look_back=look_back)
X_val, y_val = create_dataset(val_data, look_back=look_back)

# Prepare test data by appending the last 'look_back' points from validation
test_input = np.concatenate((val_data[-look_back:], test_data), axis=0)
X_test, y_test = create_dataset(test_input, look_back=look_back)

# Reshape inputs for LSTM: [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_val   = np.reshape(X_val,   (X_val.shape[0],   X_val.shape[1],   1))
X_test  = np.reshape(X_test,  (X_test.shape[0],  X_test.shape[1],  1))

# ---------------------------
# 4. Define Custom Callback and Loss Function
# ---------------------------
class CustomEarlyStopping(Callback):
    """
    Custom early stopping to terminate training when validation loss plateaus.
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

def combined_loss(alpha=1.0):
    """
    Custom loss combining RMSE and a directional penalty for incorrect predictions.
    """
    def loss_fn(y_true, y_pred):
        rmse_loss = K.sqrt(K.mean(K.square(y_true - y_pred)))
        direction_loss = K.mean(K.abs(K.sign(y_true) - K.sign(y_pred))) / 2.0
        return rmse_loss + alpha * direction_loss
    return loss_fn

# ---------------------------
# 5. Training and Evaluation Over 10 Runs
# ---------------------------
n_runs = 10
rmse_list = []
hitrate_list = []
val_loss_list = []
last_run_predictions = None

epochs = 1000
batch_size = 16

for run in range(n_runs):
    print(f"\n--- Run {run+1} ---")
    
    # Define LSTM model architecture
    model = Sequential()
    model.add(LSTM(100, input_shape=(look_back, 1)))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    # Compile with the custom loss function
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=combined_loss(alpha=1.0))

    # Define early stopping callback
    early_stop_callback = CustomEarlyStopping(patience=3)
    
    # Train the model
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stop_callback],
                        verbose=0)

    # Generate predictions on the test set
    predictions = model.predict(X_test)
    predictions_inv = scaler.inverse_transform(predictions)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate RMSE on the original scale
    test_rmse = np.sqrt(mean_squared_error(y_test_inv, predictions_inv))

    # Calculate directional hit rate
    last_values_scaled = X_test[:, -1, 0]
    last_values_unscaled = scaler.inverse_transform(last_values_scaled.reshape(-1, 1)).flatten()
    preds_flat = predictions_inv.flatten()
    y_test_flat = y_test_inv.flatten()
    pred_diff = preds_flat - last_values_unscaled
    actual_diff = y_test_flat - last_values_unscaled
    hitrate = np.sum(np.sign(pred_diff) == np.sign(actual_diff)) / len(pred_diff)

    rmse_list.append(test_rmse)
    hitrate_list.append(hitrate)
    val_loss_list.append(min(history.history['val_loss']))
    last_run_predictions = predictions_inv

# ---------------------------
# 6. Compute Confidence Intervals
# ---------------------------
rmse_mean = np.mean(rmse_list)
rmse_std = np.std(rmse_list, ddof=1)
hit_mean = np.mean(hitrate_list)
hit_std = np.std(hitrate_list, ddof=1)
n = len(rmse_list)
t_val = stats.t.ppf(0.975, df=n-1)

rmse_ci = t_val * (rmse_std / np.sqrt(n))
hit_ci = t_val * (hit_std / np.sqrt(n))

print("\n--- Summary Across 10 Runs ---")
print(f"Average RMSE: {rmse_mean:.4f} ± {rmse_ci:.4f} (95% CI)")
print(f"Average Hit Rate: {hit_mean*100:.2f}% ± {hit_ci*100:.2f}% (95% CI)")

# ---------------------------
# 7. Plot Forecasted vs Actual Prices from the Last Run
# ---------------------------
plt.figure(figsize=(14, 7))
x_axis = np.arange(len(y_test_inv))
plt.plot(x_axis, y_test_inv, label='Actual Test Data', color='black', linewidth=2)
plt.plot(x_axis, last_run_predictions, label='Predicted Prices (Last Run)', color='red', linestyle='--')
plt.title('Test Data vs Forecast (Last Run)')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.legend()
plt.show()
