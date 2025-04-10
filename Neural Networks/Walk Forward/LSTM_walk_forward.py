import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Scikit‑learn utilities
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import Callback

# ---------------------------
# 1. Custom Early Stopping Callback
# ---------------------------
class CustomEarlyStopping(Callback):
    """Simple patience‑based early stopping without restoring best weights."""
    def __init__(self, patience=4):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience          # epochs to wait for improvement
        self.wait = 0                     # epochs since last improvement
        self.prev_loss = None             # last seen validation loss

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        if self.prev_loss is None:
            self.prev_loss = current_loss  # first epoch baseline
        else:
            if current_loss >= self.prev_loss:  # no improvement
                self.wait += 1
                if self.wait >= self.patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    self.model.stop_training = True
            else:
                self.wait = 0  # reset counter on improvement
            self.prev_loss = current_loss

# Define a custom RMSE loss for Keras
# (Keras uses reduction=mean by default, so this is true RMSE)

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# Directional hit rate function

def compute_directional_hit_rate(predictions, actuals, last_values=None):
    """Return the fraction of times the model got the direction right."""
    # If last_values provided, compare against that reference point
    if last_values is not None:
        pred_diff = predictions - last_values
        actual_diff = actuals - last_values
    else:  # otherwise use consecutive differences
        pred_diff = predictions[1:] - predictions[:-1]
        actual_diff = actuals[1:] - actuals[:-1]

    pred_sign = np.sign(pred_diff)
    actual_sign = np.sign(actual_diff)
    total = len(pred_diff) if last_values is not None else len(predictions) - 1

    hit_count = np.sum(pred_sign == actual_sign)
    hit_rate = hit_count / total if total > 0 else 0.0
    return hit_rate

# ---------------------------
# 2. Utility to Create Dataset
# ---------------------------

def create_dataset(dataset, look_back=10):
    """Convert a scaled 1‑D array into (X, y) pairs for supervised learning."""
    X, y = [], []
    for i in range(len(dataset) - look_back):
        seq = dataset[i:(i + look_back), 0]  # past look_back timesteps
        target = dataset[i + look_back, 0]   # next value to predict
        X.append(seq)
        y.append(target)
    return np.array(X), np.array(y)

# ---------------------------
# 3. Build Model with Fixed Hyperparams
# ---------------------------

def build_lstm_model(input_shape,
                     num_layers=1,
                     hidden_units=50,
                     dropout_rate=0.2,
                     learning_rate=0.001):
    """Create a simple stacked LSTM followed by a Dense output layer."""
    model = Sequential()

    # First LSTM layer (return sequences if more layers follow)
    model.add(LSTM(hidden_units, input_shape=input_shape, return_sequences=(num_layers > 1)))
    model.add(Dropout(dropout_rate))

    # Optional additional LSTM layers
    for layer_i in range(2, num_layers + 1):
        if layer_i < num_layers:
            model.add(LSTM(hidden_units, return_sequences=True))
            model.add(Dropout(dropout_rate))
        else:  # last LSTM layer should not return sequences
            model.add(LSTM(hidden_units, return_sequences=False))
            model.add(Dropout(dropout_rate))

    model.add(Dense(1))  # single output for regression
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=rmse)
    return model

# =============================================================================
#                            FIXED BEST CONFIG
# =============================================================================
# These hyperparameters were determined outside this script
best_look_back = 10
best_num_layers = 1
best_hidden_units = 50
best_lr = 0.001

# ---------------------------
# 4. Load and Scale Data
# ---------------------------
file_path = "<INSERT FILE PATH>"  # path to CSV with close prices

data_df = pd.read_csv(file_path, header=None)
# Flatten to 1‑D float32 array then scale to [0,1]

data = data_df.values.astype('float32').flatten()

data = data.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

n_total = len(data_scaled)
n_90pct = int(0.9 * n_total)  # last 10 % reserved for walk‑forward

# ---------------------------
# 5. Split Data
# ---------------------------
# First 90 % is split 80/10/10 into train/val/test
train_end = int(0.8 * n_90pct)
val_end = int(0.9 * n_90pct)

data_train = data_scaled[:train_end]
data_val = data_scaled[train_end:val_end]
data_test = data_scaled[val_end:n_90pct]
data_walk = data_scaled[n_90pct:]

# Prepare supervised datasets
X_train, y_train = create_dataset(data_train, best_look_back)
X_val, y_val = create_dataset(data_val, best_look_back)
X_test_, y_test_ = create_dataset(data_test, best_look_back)

# Reshape to (samples, timesteps, features) for Keras LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test_ = X_test_.reshape((X_test_.shape[0], X_test_.shape[1], 1))

X_val_size = X_val.shape[0]  # used later for rolling validation

# ---------------------------
# 6. Initial Training (Patience = 4)
# ---------------------------
model_init = build_lstm_model(
    input_shape=(best_look_back, 1),
    num_layers=best_num_layers,
    hidden_units=best_hidden_units,
    learning_rate=best_lr
)

initial_patience = 4
early_stop_init = CustomEarlyStopping(patience=initial_patience)

history_init = model_init.fit(
    X_train, y_train,
    epochs=1000,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[early_stop_init],
    verbose=1
)

# --- Separate plots for train vs validation loss ---
plt.figure()
plt.plot(history_init.history['loss'], label='Train Loss', color='blue')
plt.title(f"Initial Training - Train Loss (Patience={initial_patience})")
plt.xlabel("Epoch")
plt.ylabel("Loss (RMSE)")
plt.legend()
plt.show()

plt.figure()
plt.plot(history_init.history['val_loss'], label='Validation Loss', color='orange')
plt.title(f"Initial Training - Validation Loss (Patience={initial_patience})")
plt.xlabel("Epoch")
plt.ylabel("Loss (RMSE)")
plt.legend()
plt.show()

# Evaluate on the internal test set

test_preds_ = model_init.predict(X_test_)
test_preds_inv = scaler.inverse_transform(test_preds_)
y_test_inv = scaler.inverse_transform(y_test_.reshape(-1, 1))

test_rmse_ = math.sqrt(mean_squared_error(y_test_inv, test_preds_inv))
test_hit_rate_ = compute_directional_hit_rate(
    predictions=test_preds_inv.flatten(),
    actuals=y_test_inv.flatten(),
    last_values=None
)

print(f"Internal Test RMSE: {test_rmse_:.4f}, Hit Rate: {test_hit_rate_*100:.2f}%")

# --- Plot predictions on Train, Val, and Test subsets ---
X_train_plot, y_train_plot = create_dataset(data_train, best_look_back)
X_val_plot, y_val_plot = create_dataset(data_val, best_look_back)
X_test_plot, y_test_plot = create_dataset(data_test, best_look_back)

X_train_plot = X_train_plot.reshape((X_train_plot.shape[0], best_look_back, 1))
X_val_plot = X_val_plot.reshape((X_val_plot.shape[0], best_look_back, 1))
X_test_plot = X_test_plot.reshape((X_test_plot.shape[0], best_look_back, 1))

train_preds_ = model_init.predict(X_train_plot)
val_preds_ = model_init.predict(X_val_plot)
test_preds_ = model_init.predict(X_test_plot)

train_preds_inv_ = scaler.inverse_transform(train_preds_)
val_preds_inv_ = scaler.inverse_transform(val_preds_)
test_preds_inv_ = scaler.inverse_transform(test_preds_)

y_train_plot_inv = scaler.inverse_transform(y_train_plot.reshape(-1, 1))
y_val_plot_inv = scaler.inverse_transform(y_val_plot.reshape(-1, 1))
y_test_plot_inv = scaler.inverse_transform(y_test_plot.reshape(-1, 1))

# --- Plot training subset
plt.figure()
plt.plot(y_train_plot_inv, label='Actual (Train)')
plt.plot(train_preds_inv_, label='Predicted (Train)', linestyle='--')
plt.title("Train Subset Predictions")
plt.xlabel("Time Step")
plt.ylabel("Price")
plt.legend()
plt.show()

# --- Plot validation subset
plt.figure()
plt.plot(y_val_plot_inv, label='Actual (Val)')
plt.plot(val_preds_inv_, label='Predicted (Val)', linestyle='--')
plt.title("Validation Subset Predictions")
plt.xlabel("Time Step")
plt.ylabel("Price")
plt.legend()
plt.show()

# --- Plot test subset
plt.figure()
plt.plot(y_test_plot_inv, label='Actual (Test)')
plt.plot(test_preds_inv_, label='Predicted (Test)', linestyle='--')
plt.title(f"Test Subset Predictions\nRMSE={test_rmse_:.4f}, HitRate={test_hit_rate_*100:.2f}%")
plt.xlabel("Time Step")
plt.ylabel("Price")
plt.legend()
plt.show()

# =============================================================================
# 8. WALK-FORWARD on the Last 10% with the SAME MODEL (no re-initialization)
# =============================================================================

rolling_train_data = np.concatenate([data_train, data_val, data_test], axis=0)

n_walk = len(data_walk)
rolling_val_window = X_val_size  # keep validation window size constant
walk_forward_patience = 1

# Reuse the trained model to continue learning day by day
model_wf = model_init

wf_predictions = []
wf_actuals = []

temp_preds = []
temp_actuals = []
plot_counter = 0

for i in range(n_walk):
    day_index = n_90pct + i  # absolute index in full dataset

    # 1) Build input sequence for today
    if len(rolling_train_data) < best_look_back:
        print("Not enough data to form a sequence. Exiting.")
        break

    recent_seq = rolling_train_data[-best_look_back:]
    X_pred = recent_seq.reshape((1, best_look_back, 1))

    # 2) Prepare training data including all history up to yesterday
    X_full, y_full = create_dataset(rolling_train_data, best_look_back)
    X_full = X_full.reshape((X_full.shape[0], X_full.shape[1], 1))

    # Rolling validation slice (last X_val_size sequences)
    if X_full.shape[0] > rolling_val_window:
        val_start = X_full.shape[0] - rolling_val_window
        X_train_rolling = X_full[:val_start]
        y_train_rolling = y_full[:val_start]
        X_val_rolling = X_full[val_start:]
        y_val_rolling = y_full[val_start:]
    else:  # not enough data yet for separate val
        X_train_rolling = X_full
        y_train_rolling = y_full
        X_val_rolling = None
        y_val_rolling = None

    # 3) Continue training with small patience each day
    wf_stop = CustomEarlyStopping(patience=walk_forward_patience)
    if X_val_rolling is not None:
        model_wf.fit(
            X_train_rolling, y_train_rolling,
            epochs=1000,
            batch_size=16,
            validation_data=(X_val_rolling, y_val_rolling),
            callbacks=[wf_stop],
            verbose=0
        )
    else:
        model_wf.fit(
            X_train_rolling, y_train_rolling,
            epochs=100,
            batch_size=16,
            verbose=0
        )

    # 4) Predict next value
    pred_scaled = model_wf.predict(X_pred)
    pred_unscaled = scaler.inverse_transform(pred_scaled)[0, 0]

    actual_scaled = data_scaled[day_index, 0]
    actual_unscaled = scaler.inverse_transform([[actual_scaled]])[0, 0]

    wf_predictions.append(pred_unscaled)
    wf_actuals.append(actual_unscaled)

    # 5) Append today to rolling history before next iteration
    rolling_train_data = np.concatenate([rolling_train_data, data_scaled[day_index].reshape(1, 1)], axis=0)

    # Accumulate for partial plots every 10 steps
    temp_preds.append(pred_unscaled)
    temp_actuals.append(actual_unscaled)

    if (i + 1) % 10 == 0 or (i + 1) == n_walk:
        plot_counter += 1
        part_preds = np.array(temp_preds)
        part_actuals = np.array(temp_actuals)
        part_rmse = math.sqrt(mean_squared_error(part_actuals, part_preds))
        part_hit = compute_directional_hit_rate(
            predictions=part_preds,
            actuals=part_actuals,
            last_values=None
        )

        plt.figure()
        plt.plot(part_actuals, label='Actual (last 10 steps)')
        plt.plot(part_preds, label='Predicted (last 10 steps)', linestyle='--')
        plt.title(f"Walk-Forward Partial Plot #{plot_counter}\n"
                  f"RMSE={part_rmse:.4f}, HitRate={part_hit*100:.2f}%")
        plt.xlabel("Step in Last 10 Predictions")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

        temp_preds = []
        temp_actuals = []

# ---------------------------
# 9. Final Out-of-Sample Results & Full Plot
# ---------------------------
wf_predictions = np.array(wf_predictions)
wf_actuals = np.array(wf_actuals)

final_rmse = math.sqrt(mean_squared_error(wf_actuals, wf_predictions))
final_hit = compute_directional_hit_rate(
    predictions=wf_predictions,
    actuals=wf_actuals,
    last_values=None
)

print("\n=== Final Walk-Forward Results on the Last 10% ===")
print(f"Out-of-Sample RMSE: {final_rmse:.4f}")
print(f"Out-of-Sample Directional Hit Rate: {final_hit*100:.2f}%")

# Plot the full out-of-sample predictions vs. actual
plt.figure(figsize=(10, 6))
plt.plot(wf_actuals, label='Actual (Out-of-Sample)')
plt.plot(wf_predictions, label='Predicted (Out-of-Sample)', linestyle='--')
plt.title("Full Out-of-Sample Predictions vs. Actual")
plt.xlabel("Index in Walk-Forward Period")
plt.ylabel("Price")
plt.legend()
plt.show()
