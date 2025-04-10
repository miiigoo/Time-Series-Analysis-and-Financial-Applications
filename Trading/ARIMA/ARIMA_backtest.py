import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from scipy import stats

# -------------------------------------------------------------------
# 1. Load two‑column CSV  [Close, Open]
# -------------------------------------------------------------------
# Column 0 – daily close (used for modelling / evaluation)
# Column 1 – daily open  (used for trade entries & exits)
# -------------------------------------------------------------------

data_2col = pd.read_csv(
    r"<INSERT FILE PATH>",
    header=None
).values.astype('float32')

close_prices = data_2col[:, 0]  # modelling target
open_prices = data_2col[:, 1]   # execution price for strategy

train_size = int(len(close_prices) * 0.9)  # 90 % in‑sample
instrument_price_before_oot = close_prices[train_size - 1]

# -------------------------------------------------------------------
# 2. Split 90 % train / 10 % test
# -------------------------------------------------------------------
train_close, test_close = close_prices[:train_size], close_prices[train_size:]
train_open, test_open = open_prices[:train_size], open_prices[train_size:]

# -------------------------------------------------------------------
# 3. Rolling‑window ARIMA forecast helper
# -------------------------------------------------------------------

def rolling_forecast_ARIMA(train_series, test_series, best_order):
    """Walk‑forward ARIMA forecasting with hit‑rate metric."""
    window = list(train_series)
    predictions, hits = [], 0
    for t, actual in enumerate(test_series):
        try:
            model = ARIMA(window, order=best_order)
            yhat = model.fit().forecast()[0]
        except Exception as e:
            print(f"Error at t={t}: {e}"); yhat = np.nan
        predictions.append(yhat)
        # direction hit?
        if np.sign(yhat - window[-1]) == np.sign(actual - window[-1]):
            hits += 1
        window = window[1:] + [actual]  # slide window
    rmse_val = math.sqrt(mean_squared_error(test_series, predictions))
    hit_rate = hits / len(test_series)
    return predictions, rmse_val, hit_rate

# -------------------------------------------------------------------
# 4. Simple grid search for (p,d,q) on AIC then rolling forecast
# -------------------------------------------------------------------

aic_scores, rmse_list, hitrate_list, best_order_list = [], [], [], []
last_run_predictions = None

for run in range(1):  # set >1 for multiple random splits
    best_aic = {}
    for p in range(1, 6):
        for d in range(1, 2):
            for q in range(1, 6):
                try:
                    aic = ARIMA(train_close, order=(p, d, q)).fit().aic
                    best_aic[(p, d, q)] = aic
                except Exception as e:
                    print(f"Order ({p},{d},{q}) failed: {e}")
    if not best_aic:
        print("No ARIMA order fit"); continue
    best_order = min(best_aic, key=best_aic.get)
    best_order_list.append(best_order)
    print(f"Best ARIMA order {best_order} (AIC={best_aic[best_order]:.2f})")

    preds, rmse_val, hit_val = rolling_forecast_ARIMA(train_close, test_close, best_order)
    rmse_list.append(rmse_val); hitrate_list.append(hit_val)
    last_run_predictions = preds
    print(f"RMSE={rmse_val:.4f}, HitRate={hit_val:.2%}")

# Aggregate stats across runs (trivial when n_runs=1)
rmse_mean, hit_mean = np.mean(rmse_list), np.mean(hitrate_list)
rmse_ci = hit_ci = 0.0
if len(rmse_list) > 1:
    t_val = stats.t.ppf(0.975, df=len(rmse_list) - 1)
    rmse_ci = t_val * np.std(rmse_list, ddof=1) / math.sqrt(len(rmse_list))
    hit_ci = t_val * np.std(hitrate_list, ddof=1) / math.sqrt(len(rmse_list))

print(f"\nAvg RMSE {rmse_mean:.4f} ± {rmse_ci:.4f}, Avg Hit {hit_mean:.2%} ± {hit_ci:.2%}")

# -------------------------------------------------------------------
# 5. Plot actual vs predicted (last run)
# -------------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(test_close, label='Actual Close')
if last_run_predictions is not None:
    plt.plot(last_run_predictions, label='Predicted', color='red')
    plt.title(f"ARIMA Rolling Forecast order={best_order_list[-1]}")
else:
    plt.title("No ARIMA predictions")
plt.xlabel("Test index"); plt.ylabel("Price"); plt.legend(); plt.show()

# -------------------------------------------------------------------
# 6. Simple trading strategy using forecast direction + daily/weekly bias
# -------------------------------------------------------------------
if last_run_predictions is None:
    print("No predictions – skip strategy.")
else:
    print(f"\nClose before OOS: {instrument_price_before_oot:.2f}")

    preds = np.array(last_run_predictions)
    closes = np.array(test_close)
    opens = np.array(test_open)

    # Direction of forecast each day (vs prev close)
    forecast_dirs = [None] + ['up' if preds[i] > closes[i - 1] else 'down' for i in range(1, len(preds))]

    # Helper for weekly + daily bias confirmation
    def overall_bias(idx):
        return None if idx < 6 else (
            'up' if (closes[idx - 1] > closes[idx - 6] and closes[idx - 1] > closes[idx - 2]) else
            'down' if (closes[idx - 1] < closes[idx - 6] and closes[idx - 1] < closes[idx - 2]) else None)

    account_balance = 100000.0
    position_open = False; position_dir = None; entry_price = 0.0
    trades, daily_balance = [], []

    for i in range(len(preds)):
        # mark balance at start of day i
        daily_balance.append(account_balance if i == 0 else daily_balance[-1])

        # 1) close logic
        if position_open and i > 0 and forecast_dirs[i] and forecast_dirs[i - 1] and forecast_dirs[i] != forecast_dirs[i - 1]:
            close_px = opens[i]
            pnl = close_px - entry_price if position_dir == 'long' else entry_price - close_px
            account_balance += pnl; daily_balance[-1] = account_balance
            position_open = False; position_dir = None
            trades[-1]['close'] = (i, close_px, pnl)

        # 2) open logic
        if not position_open and i > 0 and forecast_dirs[i] == overall_bias(i):
            open_px = opens[i]
            position_open = True; position_dir = 'long' if forecast_dirs[i] == 'up' else 'short'; entry_price = open_px
            trades.append({'dir': position_dir, 'open': (i, open_px)})

    # final close if still open
    if position_open:
        close_px = opens[-1]
        pnl = close_px - entry_price if position_dir == 'long' else entry_price - close_px
        account_balance += pnl; daily_balance[-1] = account_balance
        trades[-1]['close'] = (len(preds) - 1, close_px, pnl)

    print(f"\nFinal balance: {account_balance:.2f}")

    # Buy‑and‑hold comparison (1:1 exposure on closes)
    bh_balance = 100000.0 + (closes - closes[0])

    # --- Plot balances ---
    plt.figure(figsize=(10, 6))
    plt.plot(daily_balance, label='Strategy')
    plt.plot(bh_balance, label='Buy & Hold', linestyle='--')
    plt.title("Account Value – Strategy vs Buy & Hold"); plt.xlabel("Day"); plt.ylabel("Balance ($)"); plt.legend(); plt.show()

    # --- Plot close prices with trade markers ---
    plt.figure(figsize=(10, 6))
    plt.plot(closes, label='Close')
    for t in trades:
        ox, oy = t['open']
        plt.scatter(ox, oy, marker='^' if t['dir'] == 'long' else 'v', color='green' if t['dir'] == 'long' else 'red', s=100)
        if 'close' in t:
            cx, cy, _ = t['close']
            plt.scatter(cx, cy, marker='v' if t['dir'] == 'long' else '^', color='red' if t['dir'] == 'long' else 'green', s=100)
    plt.title("Trades on Test Data"); plt.xlabel("Day"); plt.ylabel("Price"); plt.legend(); plt.show()

# -------------------------------------------------------------------
# 7. Daily compounded returns (open‑to‑open) – separate evaluation
# -------------------------------------------------------------------
if last_run_predictions is not None:
    n_days = len(opens)
    position_array = np.zeros(n_days, dtype=int)
    for t in trades:
        od = t['open'][0]; cd = t['close'][0] if 'close' in t else n_days
        position_array[od:cd] = +1 if t['dir'] == 'long' else -1

    strat_bal = [100000.0]
    for i in range(1, n_days):
        ret = (opens[i] - opens[i - 1]) / opens[i - 1]
        if position_array[i - 1] < 0:
            ret = -ret
        strat_bal.append(strat_bal[-1] * (1 + ret) if position_array[i - 1] != 0 else strat_bal[-1])

    bh_bal = [100000.0]
    for i in range(1, n_days):
        ret = (closes[i] - closes[i - 1]) / closes[i - 1]
        bh_bal.append(bh_bal[-1] * (1 + ret))

    print(f"Strategy return {(strat_bal[-1] - 100000) / 1000:.2f}% vs Buy&Hold {(bh_bal[-1] - 100000) / 1000:.2f}%")

    plt.figure(figsize=(10, 6))
    plt.plot(strat_bal, label='Strategy'); plt.plot(bh_bal, label='Buy & Hold');
    plt.title("Daily Compounded Balances"); plt.xlabel("Day"); plt.ylabel("Balance ($)"); plt.legend(); plt.show()
