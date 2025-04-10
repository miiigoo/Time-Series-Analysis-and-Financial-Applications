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
│   └── Walk Forward
│       ├── LSTM_walk_forward.py
│       ├── LSTM_walk_forward_hit_rate.py
│       ├── LSTM_walk_forward_periodic.py
│       └── LSTM_walk_forward_volume.py
│
├── Linear Models
│   ├── AR.py
│   ├── MA.py
│   ├── ARMA.py
│   └── ARIMA.py
│
├── Extra Exercises
│   ├── Multi_Layer_Perceptron.py
│   ├── Multi_Layer_Perceptron_alpha_test.py
│   ├── Multi_Layer_Perceptron_cross_validation.py
│   ├── Multi_Layer_Perceptron_epoch_no_test.py
│   ├── Multi_Layer_Perceptron_hidden_layers_test.py
│   ├── Multi_Layer_Perceptron_validation+early_stop.py
│   ├── Single_Layer_Perceptron.py
│   ├── Single_Layer_Perceptron_L.R_test.py
│   ├── Single_Perceptron.py
│   └── Single_Perceptron_random.py
│
└── Datasets
    ├── Close+open_prices
    │   ├── S&P_01_2019_01_2024.csv
    │   └── S&P_03_2020_03_2025.csv
    ├── Closing_price_only
    │   ├── EURUSD_01_2019_01_2024.csv
    │   ├── Melb_12_1985_12_1990.csv
    │   ├── S&P_01_2019_01_2024.csv
    │   └── Tesla_01_2019_01_2024.csv
    ├── Closing_price+volume
    │   ├── S&P_01_2019_01_2024_price_volume.csv
    │   └── Tesla_01_2019_01_2024_price_volume.csv
    └── MNIST
        ├── t10k-images.idx3‑ubyte
        ├── t10k-labels.idx1‑ubyte
        ├── train-images.idx3‑ubyte
        └── train-labels.idx1‑ubyte
```
## How these folders map to the report

1. **Neural Networks** and **Linear Models** underpin the testing results in **Section 4** (datasets: **Closing_price_only**, **Closing_price+volume**).  
2. **Trading** scripts drive the testing results in **Section 5**:  
   * `ARIMA_backtest.py` and `ARIMA_Live.py` use datasets in **Close+open_prices**.  
   * `LSTM_backtest.py` uses `S&P_01_2019_01_2024.csv` from **Closing_price_only**.  
3. **Extra Exercises** showcases basic ML concepts (`*Perceptron*` scripts) trained on **MNIST**.  
4. **Utils** provides helper scripts for data cleaning, confidence intervals, and plotting.
