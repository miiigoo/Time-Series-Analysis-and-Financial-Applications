import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Scikit‑learn helpers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K  # needed for sign ops in custom loss

# ---------------------------
# 1. Custom Early Stopping Callback
# ---------------------------
class CustomEarlyStopping(Callback):
    """Stop training if validation loss hasn't improved for `patience` epochs."""
    def __init__(self, patience=4):
        super().__init__()
        self.patience = patience
        self.wait = 0
        self.prev_loss = None

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        if self.prev_loss is None:
            self.prev_loss = current_loss  # baseline on first epoch
        else:
            if current_loss >= self.prev_loss:  # no improvement
                self.wait += 1
                if self.wait >= self.patience:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    self.model.stop_training = True
            else:
                self.wait = 0  # reset counter when we improve
            self.prev_loss = current_loss

# Simple RMSE (kept for reference / metrics)

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# ---------------------------
# 2A. Combined Loss: RMSE + Direction Penalty
# ---------------------------

def combined_loss(alpha=1.0):
    """Return a loss_fn that adds a directional mismatch term to RMSE."""
    def loss_fn(y_true, y_pred):
        rmse_loss = K.sqrt(K.mean(K.square(y_true - y_pred)))
        # direction_loss = 0 when signs match, 1 when they don't
        direction_loss = K.mean(K.abs(K.sign(y_true) - K.sign(y_pred))) / 2.0
        return rmse_loss + alpha * direction_loss
    return loss_fn

# ---------------------------
# 2B. Directional Hit‑Rate Metric (for evaluation only)
# ---------------------------

def compute_directional_hit_rate(predictions, actuals, last_values=None):
    """Fraction of times model predicts the correct price direction."""
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
# 3. Helper: sliding window dataset
# ---------------------------

def create_dataset(dataset, look_back=10):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i + look_back, 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

# ---------------------------
# 4. Build LSTM Model (uses combined loss)
# ---------------------------

def build_lstm_model(input_shape,
                     num_layers=1,
                     hidden_units=50,
                     dropout_rate=0.2,
                     learning_rate=0.001):
    model = Sequential()
    # First LSTM layer (return sequences if stacking)
    model.add(LSTM(hidden_units, input_shape=input_shape, return_sequences=(num_layers > 1)))
    model.add(Dropout(dropout_rate))

    # Optional stacked layers
    for layer_idx in range(2, num_layers + 1):
        return_seq = layer_idx < num_layers
        model.add(LSTM(hidden_units, return_sequences=return_seq))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1))  # scalar output
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=combined_loss(alpha=1.0))
    return model

# =============================================================================
#                            FIXED BEST CONFIG
# =============================================================================
best_look_back = 10
best_num_layers = 1
best_hidden_units = 50
best_lr = 0.001

# ---------------------------
# 5. Load & Scale Data
# ---------------------------
file_path = "<INSERT FILE PATH>"  # CSV with close prices

data_df = pd.read_csv(file_path, header=None)

data = data_df.values.astype('float32').flatten().reshape(-1, 1)

scaler = MinMaxScaler((0, 1))
data_scaled = scaler.fit_transform(data)

n_total = len(data_scaled)
n_90pct = int(0.9 * n_total)  # last 10 % held for walk‑forward

# ---------------------------
# 6. Split Data 80/10/10 + 10
# ---------------------------
train_end = int(0.8 * n_90pct)
val_end = int(0.9 * n_90pct)

data_train = data_scaled[:train_end]
data_val = data_scaled[train_end:val_end]
data_test = data_scaled[val_end:n_90pct]
data_walk = data_scaled[n_90pct:]

# Build supervised datasets
X_train, y_train = create_dataset(data_train, best_look_back)
X_val, y_val = create_dataset(data_val, best_look_back)
X_test_, y_test_ = create_dataset(data_test, best_look_back)

# Reshape to (samples, timesteps, features)
X_train = X_train.reshape((-1, best_look_back, 1))
X_val = X_val.reshape((-1, best_look_back, 1))
X_test_ = X_test_.reshape((-1, best_look_back, 1))

X_val_size = X_val.shape[0]  # keep for rolling val window later

# ---------------------------
# 7. Initial Training
# ---------------------------
model_init = build_lstm_model((best_look_back, 1), best_num_layers, best_hidden_units, learning_rate=best_lr)

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
    plt.figure()
    plt.plot(history_init.history[key])
    plt.title(f"Initial Training - {title} Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss (RMSE + Dir)"); plt.show()

# Internal test metrics

test_preds_inv = scaler.inverse_transform(model_init.predict(X_test_))
y_test_inv = scaler.inverse_transform(y_test_.reshape(-1, 1))

test_rmse_ = math.sqrt(mean_squared_error(y_test_inv, test_preds_inv))

test_hit_rate_ = compute_directional_hit_rate(test_preds_inv.flatten(), y_test_inv.flatten())
print(f"Internal Test RMSE: {test_rmse_:.4f}, Hit Rate: {test_hit_rate_*100:.2f}%")

# Quick subset plots
for subset_name, X_plot, y_plot in [
    ("Train", *create_dataset(data_train, best_look_back)),
    ("Val", *create_dataset(data_val, best_look_back)),
    ("Test", *create_dataset(data_test, best_look_back))
]:
    X_plot = X_plot.reshape((-1, best_look_back, 1))
    preds_inv = scaler.inverse_transform(model_init.predict(X_plot))
    actual_inv = scaler.inverse_transform(y_plot.reshape(-1, 1))
    plt.figure(); plt.plot(actual_inv, label='Actual'); plt.plot(preds_inv, '--', label='Predicted')
    if subset_name == 'Test':
        plt.title(f"{subset_name} Subset\nRMSE={test_rmse_:.4f}, HitRate={test_hit_rate_*100:.2f}%")
    else:
        plt.title(f"{subset_name} Subset Predictions")
    plt.xlabel("Time Step"); plt.ylabel("Price"); plt.legend(); plt.show()

# =============================================================================
# 8. Walk‑Forward (continual training)
# =============================================================================
rolling_train_data = np.concatenate([data_train, data_val, data_test])

n_walk = len(data_walk)
rolling_val_window = X_val_size

model_wf = model_init  # keep weights & optimizer

wf_predictions, wf_actuals = [], []

temp_preds, temp_actuals = [], []
plot_counter = 0

for i in range(n_walk):
    day_idx = n_90pct + i

    # Build today sequence
    if len(rolling_train_data) < best_look_back:
        print("Not enough data to form a sequence. Exiting."); break
    recent_seq = rolling_train_data[-best_look_back:].reshape((1, best_look_back, 1))

    # Prepare full train/val up to yesterday
    X_full, y_full = create_dataset(rolling_train_data, best_look_back)
    X_full = X_full.reshape((-1, best_look_back, 1))
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
                 callbacks=[CustomEarlyStopping(patience=1)],
                 verbose=0)

    # Predict today
    pred_unscaled = scaler.inverse_transform(model_wf.predict(recent_seq))[0, 0]
    actual_unscaled = scaler.inverse_transform([[data_scaled[day_idx, 0]]])[0, 0]

    wf_predictions.append(pred_unscaled); wf_actuals.append(actual_unscaled)

    # Append today to history
    rolling_train_data = np.concatenate([rolling_train_data, data_scaled[day_idx].reshape(1, 1)])

    # Partial plots every 10 steps
    temp_preds.append(pred_unscaled); temp_actuals.append(actual_unscaled)
    if (i + 1) % 10 == 0 or (i + 1) == n_walk:
        plot_counter += 1
        part_rmse = math.sqrt(mean_squared_error(temp_actuals, temp_preds))
        part_hit = compute_directional_hit_rate(np.array(temp_preds), np.array(temp_actuals))
        plt.figure(); plt.plot(temp_actuals, label='Actual'); plt.plot(temp_preds, '--', label='Predicted')
        plt.title(f"Walk‑Forward Partial #{plot_counter}\nRMSE={part_rmse:.4f}, HitRate={part_hit*100:.2f}%")
        plt.xlabel("Step in Last 10 Predictions"); plt.ylabel("Price"); plt.legend(); plt.show()
        temp_preds, temp_actuals = [], []

# ---------------------------
# 9. Final Out‑of‑Sample Metrics
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
plt.title("Full Out‑of‑Sample Predictions vs. Actual")
plt.xlabel("Index in Walk‑Forward Period"); plt.ylabel("Price"); plt.legend(); plt.show()
