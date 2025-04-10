import numpy as np
import struct
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Function to load MNIST .idx files
def load_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
        return images

def load_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# Sigmoid activation function and its derivative
def sigmoid(v):
    return 1 / (1 + np.exp(-v))

def sigmoid_derivative(v):
    return v * (1 - v)

# Initialize weights and biases
def initialize_weights(input_size, hidden_size, output_size):
    w_hidden = np.random.rand(hidden_size, input_size) * 0.01
    b_hidden = np.random.rand(hidden_size) * 0.01

    w_output = np.random.rand(output_size, hidden_size) * 0.01
    b_output = np.random.rand(output_size) * 0.01

    return w_hidden, b_hidden, w_output, b_output

# Forward pass
def forward_propagation(x, w_hidden, b_hidden, w_output, b_output):
    v_hidden = np.dot(w_hidden, x) + b_hidden
    y_hidden = sigmoid(v_hidden)

    v_output = np.dot(w_output, y_hidden) + b_output
    y_output = sigmoid(v_output)

    return v_hidden, y_hidden, v_output, y_output

# Backward pass
def backward_propagation(x, d, v_hidden, y_hidden, y_output, w_hidden, b_hidden, w_output, b_output, alpha):
    error_output = d - y_output
    delta_output = error_output * sigmoid_derivative(y_output)

    error_hidden = np.dot(w_output.T, delta_output)
    delta_hidden = error_hidden * sigmoid_derivative(y_hidden)

    w_output += alpha * np.outer(delta_output, y_hidden)
    b_output += alpha * delta_output

    w_hidden += alpha * np.outer(delta_hidden, x)
    b_hidden += alpha * delta_hidden

    return w_hidden, b_hidden, w_output, b_output, np.sum(error_output**2)

# Train the MLP
def train_mlp(trainX, trainY, valX, valY, hidden_nodes, alpha, epochs, patience):
    input_size = trainX.shape[1]
    output_size = trainY.shape[1]

    w_hidden, b_hidden, w_output, b_output = initialize_weights(input_size, hidden_nodes, output_size)

    train_errors = []
    val_errors = []

    best_val_error = float("inf")
    no_improvement = 0

    for epoch in range(epochs):
        total_train_error = 0
        for x, d in zip(trainX, trainY):
            v_hidden, y_hidden, v_output, y_output = forward_propagation(x, w_hidden, b_hidden, w_output, b_output)
            w_hidden, b_hidden, w_output, b_output, error = backward_propagation(
                x, d, v_hidden, y_hidden, y_output, w_hidden, b_hidden, w_output, b_output, alpha
            )
            total_train_error += error

        # Compute validation error
        total_val_error = 0
        for x, d in zip(valX, valY):
            _, _, _, y_output = forward_propagation(x, w_hidden, b_hidden, w_output, b_output)
            val_error = np.sum((d - y_output) ** 2)
            total_val_error += val_error

        train_errors.append(total_train_error)
        val_errors.append(total_val_error)

        # Early stopping logic
        if total_val_error < best_val_error:
            best_val_error = total_val_error
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= patience:
            break

    return w_hidden, b_hidden, w_output, b_output, val_errors[-1]

# Cross-validate to find the best learning rate
def cross_validate_learning_rate(X, y, learning_rates, hidden_nodes, epochs, patience, k=5):
    best_learning_rate = None
    best_val_error = float("inf")
    results = {}

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for lr in learning_rates:
        print(f"Testing learning rate: {lr}")
        val_errors = []

        for train_idx, val_idx in kf.split(X):
            trainX, valX = X[train_idx], X[val_idx]
            trainY, valY = y[train_idx], y[val_idx]

            # Train the MLP
            _, _, _, _, val_error = train_mlp(trainX, trainY, valX, valY, hidden_nodes, lr, epochs, patience)
            val_errors.append(val_error)

        # Average validation error across folds
        avg_val_error = np.mean(val_errors)
        results[lr] = avg_val_error
        print(f"Learning rate {lr}, Average Validation Error: {avg_val_error:.4f}")

        # Track the best learning rate
        if avg_val_error < best_val_error:
            best_val_error = avg_val_error
            best_learning_rate = lr

    print(f"Best Learning Rate: {best_learning_rate} with Validation Error: {best_val_error:.4f}")
    return best_learning_rate, results

# Main function
def main():
    # Specify paths to MNIST files
    train_images_path = r"<INSERT FILE PATH>"
    train_labels_path = r"<INSERT FILE PATH>"

    # Load MNIST data
    print("Loading MNIST dataset...")
    trainX = load_mnist_images(train_images_path)
    trainY = load_mnist_labels(train_labels_path)

    # Normalize images and one-hot encode labels
    trainX = trainX / 255.0
    trainY = np.eye(10)[trainY]

    # Parameters
    hidden_nodes = 15
    epochs = 10
    patience = 5
    learning_rates = [0.0001, 0.001, 0.01, 0.1]

    # Perform cross-validation to find the best learning rate
    print("Performing Cross-Validation...")
    best_lr, results = cross_validate_learning_rate(trainX, trainY, learning_rates, hidden_nodes, epochs, patience)

    # Plot validation errors for different learning rates
    plt.plot(list(results.keys()), list(results.values()), marker='o')
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Average Validation Error")
    plt.title("Cross-Validation for Learning Rate")
    plt.grid()
    plt.show()

    print(f"The optimal learning rate is: {best_lr}")

if __name__ == "__main__":
    main()
