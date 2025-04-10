import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def run_equity_curve():
    """Compute and plot strategy vs. buy‑and‑hold equity curves from a CSV."""

    # -------------------------------------------------------------
    # 1) Read CSV – expected columns: Day, Close, Open, Action(s)
    # -------------------------------------------------------------
    file_path = r"<INSERT FILE PATH>"

    df = pd.read_csv(file_path, header=None).iloc[:, :4]
    df.columns = ['DayExcel', 'Close', 'Open', 'Action']

    # ensure numeric price columns & drop bad rows
    df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(subset=['Open', 'Close'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    n_days = len(df)
    if n_days == 0:
        print("No valid data after cleaning."); return

    # -------------------------------------------------------------
    # 2) Initialise equity arrays & position state
    # -------------------------------------------------------------
    initial_capital = 100_000.0
    strategy_balances = np.zeros(n_days)
    spx_balances = np.zeros(n_days)

    account_balance = initial_capital
    position_open = False
    position_direction = None  # 'long' or 'short'
    entry_price = 0.0

    strategy_balances[0] = account_balance
    spx_balances[0] = initial_capital

    # -------------------------------------------------------------
    # 3) Iterate day by day, process actions, update balances
    # -------------------------------------------------------------
    for i in range(n_days):
        open_price = df.loc[i, 'Open']
        actions = [a.strip() for a in str(df.loc[i, 'Action']).split(',') if a.strip()]

        # --- handle each action in order ---
        for action in actions:
            if action == 'Close Long' and position_open and position_direction == 'long':
                pct_ret = (open_price - entry_price) / entry_price
                account_balance *= (1 + pct_ret)
                position_open = False; position_direction = None; entry_price = 0.0

            elif action == 'Close Short' and position_open and position_direction == 'short':
                pct_ret = (entry_price - open_price) / entry_price
                account_balance *= (1 + pct_ret)
                position_open = False; position_direction = None; entry_price = 0.0

            elif action == 'Open Long' and not position_open:
                position_open = True; position_direction = 'long'; entry_price = open_price

            elif action == 'Open Short' and not position_open:
                position_open = True; position_direction = 'short'; entry_price = open_price

        # store strategy balance for today
        strategy_balances[i] = account_balance

        # buy‑and‑hold balance (open‑to‑open compounding)
        if i > 0:
            bh_ret = (open_price - df.loc[i - 1, 'Open']) / df.loc[i - 1, 'Open']
            spx_balances[i] = spx_balances[i - 1] * (1 + bh_ret)

    # -------------------------------------------------------------
    # 4) Final stats & plot
    # -------------------------------------------------------------
    strat_ret_pct = (strategy_balances[-1] - initial_capital) / initial_capital * 100
    bh_ret_pct = (spx_balances[-1] - initial_capital) / initial_capital * 100

    print("\n=== Final Results ===")
    print(f"Strategy final balance = ${strategy_balances[-1]:,.2f} => {strat_ret_pct:.2f}%")
    print(f"SPX final balance      = ${spx_balances[-1]:,.2f} => {bh_ret_pct:.2f}%")

    plt.figure(figsize=(10, 6))
    plt.plot(spx_balances, label='Buy & Hold')
    plt.plot(strategy_balances, label='Strategy')
    plt.title("Daily Account Balances – Strategy vs Buy & Hold\n($100k start)")
    plt.xlabel("Day in Out‑of‑Sample")
    plt.ylabel("Account Balance ($)")
    plt.legend(); plt.show()


# Execute when script is run directly
run_equity_curve()
