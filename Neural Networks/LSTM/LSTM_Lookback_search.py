import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import Callback

# ---------------------------
# 1. Set Seeds for Reproducibility
# ---------------------------
np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------
# 2. Load and Preprocess Data
# ---------------------------
# Load dataset; expects CSV without headers, with prices in the first column
data_df = pd.read_csv('<INSERT FILE PATH>', header=None)
data = data_df.values.astype('float32').flatten()  # Flatten to 1D array

# Reshape data for scaling (expected by MinMaxScaler)
data = data.reshape(-1, 1)

# Normalize data to the range [0,1]
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# ---------------------------
# 3. Create Dataset Function
# ---------------------------
def create_dataset(dataset, look_back=20):
    """
    Converts a time series dataset into sequences for supervised learning.
    Each input sequence consists of 'look_back' previous time steps,
    with the next time step as the target output.
    """
    X, y = [], []
    for i in range(len(dataset) - look_back):
        seq = dataset[i:(i + look_back), 0]  # Extract sequence of length 'look_back'
        target = dataset[i + look_back, 0]   # Target is the next time step
        X.append(seq)
        y.append(target)
    return np.array(X), np.array(y)

# ---------------------------
# 4. Custom EarlyStopping Callback
# ---------------------------
class CustomEarlyStopping(Callback):
    """
    Custom EarlyStopping callback to halt training if validation loss doesn't improve after a specified patience period.
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
# 5. Define Custom RMSE Loss Function
# ---------------------------
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# ---------------------------
# 6. Define Look-back Periods for Testing
# ---------------------------
lookback_list = [5, 10, 15, 20]  # Different look-back periods for testing

# Dictionary to store results for each look-back period
results = {}

# ---------------------------
# 7. Split Data into Training, Validation, and Test Sets
# ---------------------------
n = len(data_scaled)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

train_data = data_scaled[:train_end]
val_data = data_scaled[train_end:val_end]
test_data = data_scaled[val_end:]

# ---------------------------
# 8. Train Model for Each Look-back Period
# ---------------------------
for look_back in lookback_list:
    print("\n" + "="*40)
    print(f"Running experiment with look_back = {look_back}")

    # Create datasets for training, validation, and testing
    X_train, y_train = create_dataset(train_data, look_back=look_back)
    X_val, y_val = create_dataset(val_data, look_back=look_back)
    test_input = np.concatenate((val_data[-look_back:], test_data), axis=0)
    X_test, y_test = create_dataset(test_input, look_back=look_back)

    # Reshape inputs to match LSTM input format [samples, time steps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # ---------------------------
    # 9. Build and Compile the LSTM Model
    # ---------------------------
    model = Sequential()
    model.add(LSTM(100, input_shape=(look_back, 1)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=rmse)

    # Display model summary for the first look-back period only
    if look_back == lookback_list[0]:
        model.summary()

    # ---------------------------
    # 10. Train the Model
    # ---------------------------
    epochs = 200
    batch_size = 16
    early_stop_callback = CustomEarlyStopping(patience=3)

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stop_callback],
                        verbose=0)

    # ---------------------------
    # 11. Make Predictions and Inverse Transform
    # ---------------------------
    predictions = model.predict(X_test)
    predictions_inv = scaler.inverse_transform(predictions)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # ---------------------------
    # 12. Calculate RMSE and Hit Rate
    # ---------------------------
    test_rmse = np.sqrt(mean_squared_error(y_test_inv, predictions_inv))

    last_values_scaled = X_test[:, -1, 0]
    last_values_unscaled = scaler.inverse_transform(last_values_scaled.reshape(-1, 1)).flatten()
    preds_flat = predictions_inv.flatten()
    y_test_flat = y_test_inv.flatten()

    pred_diff = preds_flat - last_values_unscaled
    actual_diff = y_test_flat - last_values_unscaled
    hit_count = np.sum(np.sign(pred_diff) == np.sign(actual_diff))
    hitrate = hit_count / len(pred_diff)

    results[look_back] = {'predictions': preds_flat, 'rmse': test_rmse, 'hitrate': hitrate}

    print(f"Look-back {look_back} -> RMSE: {test_rmse:.4f}, Hit Rate: {hitrate:.2%}")

# ---------------------------
# 13. Plot Results
# ---------------------------
plt.figure(figsize=(14, 7))
x_axis = np.arange(len(y_test_flat))
plt.plot(x_axis, y_test_flat, label='Actual Test Data', color='black', linewidth=2)

colors = ['red', 'blue', 'green', 'orange']
for i, look_back in enumerate(lookback_list):
    pred = results[look_back]['predictions']
    plt.plot(x_axis, pred, label=f'Look-back {look_back}', color=colors[i], linestyle='--')

plt.title('LSTM Forecasts for Different Look-back Periods')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.legend()
plt.show()
