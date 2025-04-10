import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# ---------------------------
# 1. Define Utility Functions
# ---------------------------

# Function to prepare sequences for RNN model training
def create_sequences(data, sequence_length=1):
    """
    Splits the input data into sequences of a given length for training.
    Args:
        data (np.array): Scaled price data.
        sequence_length (int): Number of past observations used to predict the next value.
    Returns:
        np.array: Features (X) and targets (y) prepared for RNN training.
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# Function to calculate directional hit rate
def calculate_hit_rate(actual, predicted):
    """
    Calculates the percentage of correctly predicted directions (up/down).
    Args:
        actual (np.array): True prices (unscaled).
        predicted (np.array): Predicted prices (unscaled).
    Returns:
        float: Hit rate as a proportion of correctly predicted directions.
    """
    actual_directions = np.diff(actual, axis=0) > 0
    predicted_directions = np.diff(predicted, axis=0) > 0
    correct_predictions = np.sum(actual_directions == predicted_directions)
    return correct_predictions / len(actual_directions)

# ---------------------------
# 2. Load and Preprocess Data
# ---------------------------

# Load the CSV file containing price data (assumed to have no header)
file_path = "<INSERT FILE PATH>"
data = pd.read_csv(file_path, header=None)
prices = data.iloc[:, 0].values.reshape(-1, 1)

# Normalize the price data to the [0, 1] range for training
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# ---------------------------
# 3. Define Experiment Parameters
# ---------------------------

# Different sequence lengths to experiment with
sequence_lengths = [5, 10, 15, 20]
batch_size = 32  # Batch size for model training

# Store results for each sequence length
results = {"sequence_length": [], "hit_rate": [], "rmse": []}
predictions = {}  # To store predictions for plotting

# ---------------------------
# 4. Train and Evaluate Model for Each Sequence Length
# ---------------------------

for sequence_length in sequence_lengths:
    # Prepare the dataset for the given sequence length
    X, y = create_sequences(scaled_prices, sequence_length)
    
    # Split the data into training and testing sets (80% train, 20% test)
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # Inverse transform test labels for evaluation
    y_test_unscaled = scaler.inverse_transform(y_test)

    # ---------------------------
    # 5. Build and Compile the Model
    # ---------------------------
    model = Sequential([
        SimpleRNN(50, activation='relu', input_shape=(sequence_length, 1)),
        Dense(1)  # Output layer for predicting the next price
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # ---------------------------
    # 6. Train the Model
    # ---------------------------
    model.fit(X_train, y_train, epochs=50, batch_size=batch_size, verbose=0)

    # ---------------------------
    # 7. Generate Predictions
    # ---------------------------
    y_pred = model.predict(X_test)
    y_pred_unscaled = scaler.inverse_transform(y_pred)  # Convert predictions to original scale

    # ---------------------------
    # 8. Calculate Metrics
    # ---------------------------
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_unscaled))
    hit_rate = calculate_hit_rate(y_test_unscaled, y_pred_unscaled)

    # Store results for current sequence length
    results["sequence_length"].append(sequence_length)
    results["hit_rate"].append(hit_rate * 100)  # Convert to percentage
    results["rmse"].append(rmse)
    predictions[sequence_length] = y_pred_unscaled  # Store predictions for plotting

# ---------------------------
# 9. Plot Predictions vs Actual Prices
# ---------------------------

plt.figure(figsize=(12, 6))
plt.plot(y_test_unscaled, label='Actual Price', color='blue', linewidth=2)
for sequence_length in sequence_lengths:
    plt.plot(predictions[sequence_length], label=f'Predicted (Seq Length={sequence_length})', linestyle='--')
plt.title('Forecasted Price vs Actual Price for Different Sequence Lengths')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.legend()
plt.show()

# ---------------------------
# 10. Plot Hit Rate and RMSE vs Sequence Length
# ---------------------------

fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot RMSE on left y-axis
ax1.set_xlabel('Sequence Length')
ax1.set_ylabel('RMSE', color='tab:red')
ax1.plot(results["sequence_length"], results["rmse"], color='tab:red', marker='o')
ax1.tick_params(axis='y', labelcolor='tab:red')

# Plot Hit Rate on right y-axis
ax2 = ax1.twinx()
ax2.set_ylabel('Hit Rate (%)', color='tab:blue')
ax2.plot(results["sequence_length"], results["hit_rate"], color='tab:blue', marker='o')
ax2.tick_params(axis='y', labelcolor='tab:blue')

fig.suptitle('Sequence Length vs Hit Rate and RMSE')
fig.tight_layout()
plt.show()
