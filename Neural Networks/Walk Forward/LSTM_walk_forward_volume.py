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
    """Simple patience‑based early stopping."""
    def __init__(self, patience=4):
        super().__init__()
        self.patience = patience
        self.wait = 0
        self.prev_loss = None

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        if self.prev_loss is None:
            self.prev_loss = current_loss  # first epoch baseline
        else:
            if current_loss >= self.prev_loss:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    self.model.stop_training = True
            else:
                self.wait = 0  # reset when we improve
            self.prev_loss = current_loss

# RMSE loss (used throughout)

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# ---------------------------
# Directional hit‑rate helper (evaluation only)
# ---------------------------

def compute_directional_hit_rate(predictions, actuals, last_values=None):
    """Fraction of correct up/down moves."""
    if last_values is not None:
        pred_diff = predictions - last_values
        actual_diff = actuals - last_values
    else:
        pred_diff = predictions[1:] - predictions[:-1]
        actual_diff = actuals[1:] - actuals[:-1]

    pred_sign = np.sign(pred_diff)
    actual_sign = np.sign(actual_diff)
    total = len(pred_diff) if last_values is not None else len(predictions) - 1
    return np.sum(pred_sign == actual_sign) / total if total > 0 else 0.0

# ---------------------------
# 2. Create Dataset with Price & Volume features
# ---------------------------

def create_dataset(dataset, look_back=10):
    """Return X (look_back × 2) and y (next price) pairs."""
    X, y = [], []
    for i in range(len(dataset) - look_back):
        seq = dataset[i:i + look_back, :]   # shape (look_back, 2)
        target = dataset[i + look_back, 0]  # price column only
        X.append(seq); y.append(target)
    return np.array(X), np.array(y)

# ---------------------------
# 3. Build LSTM Model (regresses next price)
# ---------------------------

def build_lstm_model(input_shape,
                     num_layers=1,
                     hidden_units=50,
                     dropout_rate=0.2,
                     learning_rate=0.001):
    model = Sequential()
    model.add(LSTM(hidden_units, input_shape=input_shape, return_sequences=(num_layers > 1)))
    model.add(Dropout(dropout_rate))

    # Optional stacked layers
    for layer_i in range(2, num_layers + 1):
        return_seq = layer_i < num_layers
        model.add(LSTM(hidden_units, return_sequences=return_seq))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1))  # scalar price output
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=rmse)
    return model

# =============================================================================
# 4. Fixed Hyperparameters (chosen elsewhere)
# =============================================================================
best_look_back = 10
best_num_layers = 1
best_hidden_units = 50
best_lr = 0.001

# ---------------------------
# 5. Load & Scale Data (Price + Volume)
# ---------------------------
file_path = "<INSERT FILE PATH>"  # CSV with two columns

data_df = pd.read_csv(file_path, header=None)

data = data_df.values.astype('float32')  # shape (N, 2)

price_data = data[:, 0:1]   # column 0
volume_data = data[:, 1:2]  # column 1

scaler_price = MinMaxScaler((0, 1))
scaler_volume = MinMaxScaler((0, 0.5))  # keep volume in smaller range

price_scaled = scaler_price.fit_transform(price_data)
volume_scaled = scaler_volume.fit_transform(volume_data)

data_scaled = np.hstack((price_scaled, volume_scaled))  # shape (N, 2)

n_total = len(data_scaled)
n_90pct = int(0.9 * n_total)  # hold‑out last 10 %

# ---------------------------
# 6. Split 80/10/10 + 10
# ---------------------------
train_end = int(0.8 * n_90pct)
val_end = int(0.9 * n_90pct)

data_train = data_scaled[:train_end]
data_val = data_scaled[train_end:val_end]
data_test = data_scaled[val_end:n_90pct]
data_walk = data_scaled[n_90pct:]

# Build supervised sets
X_train, y_train = create_dataset(data_train, best_look_back)
X_val, y_val = create_dataset(data_val, best_look_back)
X_test_, y_test_ = create_dataset(data_test, best_look_back)

# Reshape to (samples, timesteps, features)
X_train = X_train.reshape((-1, best_look_back, 2))
X_val = X_val.reshape((-1, best_look_back, 2))
X_test_ = X_test_.reshape((-1, best_look_back, 2))

X_val_size = X_val.shape[0]

# ---------------------------
# 7. Initial Training
# ---------------------------
model_init = build_lstm_model((best_look_back, 2), best_num_layers, best_hidden_units, learning_rate=best_lr)

history_init = model_init.fit(
    X_train, y_train,
    epochs=1000,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[CustomEarlyStopping(patience=4)],
    verbose=1
)

# --- Loss curves ---
for key, title in [('loss', 'Train'), ('val_loss', 'Validation')]:
    plt.figure(); plt.plot(history_init.history[key]);
    plt.title(f"Initial Training - {title} Loss"); plt.xlabel("Epoch"); plt.ylabel("RMSE"); plt.show()

# Internal test metrics (inverse‑transform price only)

test_preds_inv = scaler_price.inverse_transform(model_init.predict(X_test_))
y_test_inv = scaler_price.inverse_transform(y_test_.reshape(-1, 1))

test_rmse_ = math.sqrt(mean_squared_error(y_test_inv, test_preds_inv))

test_hit_rate_ = compute_directional_hit_rate(test_preds_inv.flatten(), y_test_inv.flatten())
print(f"Internal Test RMSE: {test_rmse_:.4f}, Hit Rate: {test_hit_rate_*100:.2f}%")

# ---------------------------
# 8. Quick subset plots (Train / Val / Test)
# ---------------------------

def prepare_plot_data(X, y):
    preds = scaler_price.inverse_transform(model_init.predict(X))
    actual = scaler_price.inverse_transform(y.reshape(-1, 1))
    return preds.flatten(), actual.flatten()

for name, Xs, ys in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test_, y_test_)]:
    preds, actual = prepare_plot_data(Xs, ys)
    plt.figure(); plt.plot(actual, label='Actual'); plt.plot(preds, '--', label='Predicted')
    title = name if name != 'Test' else f"Test\nRMSE={test_rmse_:.4f}, HitRate={test_hit_rate_*100:.2f}%"
    plt.title(f"{title} Subset Predictions"); plt.xlabel("Time Step"); plt.ylabel("Price"); plt.legend(); plt.show()

# =============================================================================
# 9. Walk‑Forward Training & Prediction
# =============================================================================
rolling_train_data = np.concatenate([data_train, data_val, data_test])

n_walk = len(data_walk)
rolling_val_window = X_val_size

model_wf = model_init  # continue training

wf_predictions, wf_actuals = [], []

temp_preds, temp_actuals = [], []
plot_counter = 0

for i in range(n_walk):
    day_idx = n_90pct + i

    # Build today sequence
    if len(rolling_train_data) < best_look_back:
        print("Not enough data to form a sequence. Exiting."); break
    recent_seq = rolling_train_data[-best_look_back:].reshape((1, best_look_back, 2))

    # Prepare full history sets
    X_full, y_full = create_dataset(rolling_train_data, best_look_back)
    X_full = X_full.reshape((-1, best_look_back, 2))
    if X_full.shape[0] > rolling_val_window:
        val_start = X_full.shape[0] - rolling_val_window
        X_train_roll, y_train_roll = X_full[:val_start], y_full[:val_start]
        X_val_roll, y_val_roll = X_full[val_start:], y_full[val_start:]
    else:
        X_train_roll, y_train_roll = X_full, y_full
        X_val_roll = y_val_roll = None

    # One‑epoch early‑stop training each day
    model_wf.fit(X_train_roll, y_train_roll,
                 epochs=1000, batch_size=16,
                 validation_data=(X_val_roll, y_val_roll) if X_val_roll is not None else None,
                 callbacks=[CustomEarlyStopping(patience=1)], verbose=0)

    # Predict today (inverse‑transform price)
    pred_price = scaler_price.inverse_transform(model_wf.predict(recent_seq))[0, 0]
    actual_price = scaler_price.inverse_transform([[data_scaled[day_idx, 0]]])[0, 0]

    wf_predictions.append(pred_price); wf_actuals.append(actual_price)

    # Append today (price & volume) to history
    rolling_train_data = np.concatenate([rolling_train_data, data_scaled[day_idx].reshape(1, 2)])

    # Partial plot data
    temp_preds.append(pred_price); temp_actuals.append(actual_price)
    if (i + 1) % 10 == 0 or (i + 1) == n_walk:
        plot_counter += 1
        part_rmse = math.sqrt(mean_squared_error(temp_actuals, temp_preds))
        part_hit = compute_directional_hit_rate(np.array(temp_preds), np.array(temp_actuals))
        plt.figure(); plt.plot(temp_actuals, label='Actual'); plt.plot(temp_preds, '--', label='Predicted')
        plt.title(f"Walk‑Forward Partial #{plot_counter}\nRMSE={part_rmse:.4f}, HitRate={part_hit*100:.2f}%")
        plt.xlabel("Step in Last 10 Predictions"); plt.ylabel("Price"); plt.legend(); plt.show()
        temp_preds, temp_actuals = [], []

# ---------------------------
# 10. Final Out‑of‑Sample Metrics
# ---------------------------
wf_predictions = np.array(wf_predictions)
wf_actuals = np.array(wf_actuals)

final_rmse = math.sqrt(mean_squared_error(wf_actuals, wf_predictions))
final_hit = compute_directional_hit_rate(wf_predictions, wf_actuals)
print("\n=== Final Walk‑Forward Results ===")
print(f"Out‑of‑Sample RMSE: {final_rmse:.4f}")
print(f"Out‑of‑Sample Directional Hit Rate: {final_hit*100:.2f}%")

plt.figure(figsize=(10, 6))
plt.plot(wf_actuals, label='Actual'); plt.plot(wf_predictions, '--', label='Predicted')
plt.title("Full Out‑of‑Sample Predictions vs. Actual (Price + Volume Input)")
plt.xlabel("Index in Walk‑Forward Period"); plt.ylabel("Price"); plt.legend(); plt.show()
