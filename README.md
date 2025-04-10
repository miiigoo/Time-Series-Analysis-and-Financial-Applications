# Financial‑Time‑Series Modelling & Trading Strategies

A public collection of my experiments in forecasting and trading the S&P 500, FX, equities and more.  
It spans traditional linear models, deep‑learning architectures, walk‑forward evaluation, and a small utility toolbox — all fully reproducible thanks to the included datasets.

## Folder structure

```text
├── Utils
│   ├── 95_CI.py
│   ├── clean_data.py
│   ├── flip_data.py
│   └── plot.py
│
├── Trading
│   ├── ARIMA
│   │   ├── ARIMA_backtest.py
│   │   └── ARIMA_Live.py
│   └── LSTM
│       └── LSTM_backtest.py
│
├── Neural Networks
│   ├── CNN
│   │   ├── CNN_comparison.py
│   │   ├── CNN_hit_comparison.py
│   │   └── CNN_volume_comparison.py
│   ├── LSTM
│   │   ├── LSTM_comparison.py
│   │   ├── LSTM_hit_comparison.py
│   │   ├── LSTM_Lookback_search.py
│   │   └── LSTM_volume_comparison.py
│   ├── RNN
│   │   ├── RNN_comparison.py
│   │   └── RNN_sequence_search.py
│   ├── SLP & MLP
│   │   ├── Multi_Layer_Perceptron*.py
│   │   └── Single*_Perceptron*.py
│   └── Walk Forward
│       ├── LSTM_walk_forward*.py
│       └── …
│
├── Linear Models
│   ├── AR.py
│   ├── MA.py
│   ├── ARMA.py
│   └── ARIMA.py
│
└── Datasets
    ├── Close+open_prices
    ├── Closing_price_only
    ├── Closing_price+volume
    └── MNIST
