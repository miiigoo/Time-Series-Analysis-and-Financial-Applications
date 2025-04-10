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

# ---------------------------
# 1. Custom Early Stopping Callback
# ---------------------------
class CustomEarlyStopping(Callback):
    """Stop when validation loss hasn't improved for `patience` epochs."""
    def __init__(self, patience=4):
        super().__init__()
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
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    self.model.stop_training = True
            else:
                self.wait = 0
            self.prev_loss = current_loss

# RMSE loss

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# ---------------------------
# Directional hit‑rate metric (evaluation only)
# ---------------------------

def compute_directional_hit_rate(predictions, actuals, last_values=None):
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
# 2. Sliding‑window dataset builder
# ---------------------------

def create_dataset(dataset, look_back=10):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i + look_back, 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

# ---------------------------
# 3. LSTM model factory
# ---------------------------

def build_lstm_model(input_shape,
                     num_layers=1,
                     hidden_units=50,
                     dropout_rate=0.2,
                     learning_rate=0.001):
    model = Sequential()
    model.add(LSTM(hidden_units, input_shape=input_shape, return_sequences=(num_layers > 1)))
    model.add(Dropout(dropout_rate))
    for layer_i in range(2, num_layers + 1):
        return_seq = layer_i < num_layers
        model.add(LSTM(hidden_units, return_sequences=return_seq))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=rmse)
    return model

# =============================================================================
# 4. Fixed hyperparameters
# =============================================================================
best_look_back = 10
best_num_layers = 1
best_hidden_units = 50
best_lr = 0.001

# ---------------------------
# 5. Load & scale price data
# ---------------------------
file_path = "<INSERT FILE PATH>"  # close‑price CSV

data_df = pd.read_csv(file_path, header=None)

data = data_df.values.astype('float32').flatten().reshape(-1, 1)

scaler = MinMaxScaler((0, 1))
data_scaled = scaler.fit_transform(data)

n_total = len(data_scaled)
n_90pct = int(0.9 * n_total)  # reserve last 10 %

# ---------------------------
# 6. Train/Val/Test split 80/10/10 + 10
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
X_train = X_train.reshape((-1, best_look_back, 1))
X_val = X_val.reshape((-1, best_look_back, 1))
X_test_ = X_test_.reshape((-1, best_look_back, 1))

X_val_size = X_val.shape[0]

# ---------------------------
# 7. Initial model training
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

# --- Plot loss curves ---
for key, title in [('loss', 'Train'), ('val_loss', 'Validation')]:
    plt.figure(); plt.plot(history_init.history[key]);
    plt.title(f"Initial Training - {title} Loss"); plt.xlabel("Epoch"); plt.ylabel("RMSE"); plt.show()

# Internal test metrics

test_preds_inv = scaler.inverse_transform(model_init.predict(X_test_))
y_test_inv = scaler.inverse_transform(y_test_.reshape(-1, 1))

test_rmse_ = math.sqrt(mean_squared_error(y_test_inv, test_preds_inv))

test_hit_rate_ = compute_directional_hit_rate(test_preds_inv.flatten(), y_test_inv.flatten())
print(f"Internal Test RMSE: {test_rmse_:.4f}, Hit Rate: {test_hit_rate_*100:.2f}%")

# ---------------------------
# 8. Plot Train / Val / Test predictions
# ---------------------------

def plot_subset(name, Xs, ys):
    preds = scaler.inverse_transform(model_init.predict(Xs))
    actual = scaler.inverse_transform(ys.reshape(-1, 1))
    plt.figure(); plt.plot(actual, label='Actual'); plt.plot(preds, '--', label='Predicted')
    if name == 'Test':
        plt.title(f"{name} Subset\nRMSE={test_rmse_:.4f}, HitRate={test_hit_rate_*100:.2f}%")
    else:
        plt.title(f"{name} Subset Predictions")
    plt.xlabel("Time Step"); plt.ylabel("Price"); plt.legend(); plt.show()

plot_subset("Train", X_train, y_train)
plot_subset("Val", X_val, y_val)
plot_subset("Test", X_test_, y_test_)

# =============================================================================
# 9. Walk‑forward forecasting + simple trading strategy
# =============================================================================
rolling_train_data = np.concatenate([data_train, data_val, data_test])

n_walk = len(data_walk)
rolling_val_window = X_val_size

model_wf = model_init  # keep weights & optimizer

wf_predictions, wf_actuals = [], []

# Trading state vars
account_balance = 100000.0
position_open = False
position_direction = None  # 'long' / 'short'
entry_price = 0.0
forecast_directions = [None] * n_walk  # store model direction per day
position_array = np.zeros(n_walk, dtype=int)  # +1 long, -1 short, 0 flat

temp_preds, temp_actuals = [], []
plot_counter = 0

# Helper: daily & weekly bias from out‑of‑sample prices
out_of_sample_prices = scaler.inverse_transform(data_walk)[:, 0]

def get_overall_bias_outsample(day_i):
    if day_i - 6 < 0 or day_i - 2 < 0:
        return None
    price_yesterday = out_of_sample_prices[day_i - 1]
    weekly_bias = 'up' if price_yesterday > out_of_sample_prices[day_i - 6] else 'down'
    daily_bias = 'up' if price_yesterday > out_of_sample_prices[day_i - 2] else 'down'
    return weekly_bias if weekly_bias == daily_bias else None

# --- Walk‑forward loop ---
for i in range(n_walk):
    day_idx = n_90pct + i

    # Build today sequence
    recent_seq = rolling_train_data[-best_look_back:].reshape((1, best_look_back, 1))

    # Re‑train on all history (quick early‑stop)
    X_full, y_full = create_dataset(rolling_train_data, best_look_back)
    X_full = X_full.reshape((-1, best_look_back, 1))
    if X_full.shape[0] > rolling_val_window:
        val_start = X_full.shape[0] - rolling_val_window
        X_train_roll, y_train_roll = X_full[:val_start], y_full[:val_start]
        X_val_roll, y_val_roll = X_full[val_start:], y_full[val_start:]
    else:
        X_train_roll, y_train_roll = X_full, y_full
        X_val_roll = y_val_roll = None
    model_wf.fit(X_train_roll, y_train_roll,
                 epochs=1000, batch_size=16,
                 validation_data=(X_val_roll, y_val_roll) if X_val_roll is not None else None,
                 callbacks=[CustomEarlyStopping(patience=1)], verbose=0)

    # Predict next price
    pred_price = scaler.inverse_transform(model_wf.predict(recent_seq))[0, 0]
    actual_price = scaler.inverse_transform([[data_scaled[day_idx, 0]]])[0, 0]

    wf_predictions.append(pred_price); wf_actuals.append(actual_price)

    # Forecast direction vs yesterday actual
    if i == 0:
        forecast_directions[i] = None
    else:
        forecast_directions[i] = 'up' if pred_price > wf_actuals[i - 1] else 'down'

    # Append today to training history
    rolling_train_data = np.concatenate([rolling_train_data, data_scaled[day_idx].reshape(1, 1)])

    # Partial plots every 10 steps
    temp_preds.append(pred_price); temp_actuals.append(actual_price)
    if (i + 1) % 10 == 0 or (i + 1) == n_walk:
        plot_counter += 1
        part_rmse = math.sqrt(mean_squared_error(temp_actuals, temp_preds))
        part_hit = compute_directional_hit_rate(np.array(temp_preds), np.array(temp_actuals))
        plt.figure(); plt.plot(temp_actuals, label='Actual'); plt.plot(temp_preds, '--', label='Predicted')
        plt.title(f"Walk‑Forward Partial #{plot_counter}\nRMSE={part_rmse:.4f}, HitRate={part_hit*100:.2f}%")
        plt.xlabel("Step in Last 10 Predictions"); plt.ylabel("Price"); plt.legend(); plt.show()
        temp_preds, temp_actuals = [], []

    # --- Record current position in position_array ---
    if position_open:
        position_array[i] = +1 if position_direction == 'long' else -1

# ---------------------------
# 10. Trading logic (open/close positions)
# ---------------------------
position_open = False; position_direction = None; entry_price = 0.0; account_balance = 100000.0

for i in range(n_walk):
    # Close logic if a position exists and direction flips
    if position_open and i > 0 and forecast_directions[i] and forecast_directions[i - 1]:
        if forecast_directions[i] != forecast_directions[i - 1]:
            close_price = wf_actuals[i - 1]
            pnl = close_price - entry_price if position_direction == 'long' else entry_price - close_price
            account_balance += pnl
            print(f"Closing {position_direction} on day {i - 1}, price={close_price:.2f}, PNL={pnl:.2f}, bal={account_balance:.2f}")
            position_open = False

    # Open logic if flat
    if not position_open:
        bias = get_overall_bias_outsample(i)
        if forecast_directions[i] and bias and forecast_directions[i] == bias and i > 0:
            open_price = wf_actuals[i - 1]
            position_open = True
            position_direction = 'long' if bias == 'up' else 'short'
            entry_price = open_price
            print(f"Opening {position_direction.upper()} on day {i - 1}, price={open_price:.2f}, bal={account_balance:.2f}")

    position_array[i] = +1 if (position_open and position_direction == 'long') else -1 if (position_open and position_direction == 'short') else 0

# Close any residual position at end
if position_open:
    close_price = wf_actuals[-1]
    pnl = close_price - entry_price if position_direction == 'long' else entry_price - close_price
    account_balance += pnl
    print(f"Closing final {position_direction} on last day, price={close_price:.2f}, PNL={pnl:.2f}, bal={account_balance:.2f}")

print(f"\n=== Final Account Balance: {account_balance:.2f} ===")

# ---------------------------
# 11. Final out‑of‑sample metrics & plot
# ---------------------------
final_rmse = math.sqrt(mean_squared_error(wf_actuals, wf_predictions))
final_hit = compute_directional_hit_rate(np.array(wf_predictions), np.array(wf_actuals))
print(f"Out‑of‑Sample RMSE: {final_rmse:.4f}, HitRate: {final_hit*100:.2f}%")

plt.figure(figsize=(10, 6))
plt.plot(wf_actuals, label='Actual'); plt.plot(wf_predictions, '--', label='Predicted')
plt.title("Full Out‑of‑Sample Predictions vs. Actual"); plt.xlabel("Index in Walk‑Forward"); plt.ylabel("Price"); plt.legend(); plt.show()

# =========================================
# 12. Buy‑and‑hold vs strategy equity curves
# =========================================
initial_capital = 100000.0

# Buy & Hold curve
buy_hold_balances = [initial_capital]
for i in range(1, n_walk):
    pct_change = (wf_actuals[i] - wf_actuals[i - 1]) / wf_actuals[i - 1]
    buy_hold_balances.append(buy_hold_balances[-1] * (1 + pct_change))

# Strategy curve using position_array
strategy_balances = [initial_capital]
for i in range(1, n_walk):
    pos = position_array[i - 1]
    if pos == 0:
        strategy_balances.append(strategy_balances[-1])
    else:
        underlying_ret = (wf_actuals[i] - wf_actuals[i - 1]) / wf_actuals[i - 1]
        strategy_balances.append(strategy_balances[-1] * (1 + (underlying_ret if pos > 0 else -underlying_ret)))

# Print total returns
bh_return_pct = (buy_hold_balances[-1] - initial_capital) / initial_capital * 100
strat_return_pct = (strategy_balances[-1] - initial_capital) / initial_capital * 100
print(f"SPX Buy‑and‑Hold Return: {bh_return_pct:.2f}%")
print(f"Strategy Return:         {strat_return_pct:.2f}%")

# Plot curves
plt.figure(figsize=(10, 6))
plt.plot(buy_hold_balances, label='Buy & Hold')
plt.plot(strategy_balances, label='Strategy')
plt.title("Account Balances ($100k start)")
plt.xlabel("Day in Out‑of‑Sample"); plt.ylabel("Balance ($)"); plt.legend(); plt.show()
