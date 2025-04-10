import numpy as np
import struct
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

        print(f"Epoch {epoch + 1}/{epochs}, Training Error: {total_train_error:.4f}, Validation Error: {total_val_error:.4f}")

        # Early stopping logic
        if total_val_error < best_val_error:
            best_val_error = total_val_error
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return w_hidden, b_hidden, w_output, b_output, train_errors, val_errors

# Test the MLP
def test_mlp(testX, testY, w_hidden, b_hidden, w_output, b_output):
    correct_predictions = 0

    for x, d in zip(testX, testY):
        _, _, _, y_output = forward_propagation(x, w_hidden, b_hidden, w_output, b_output)
        predicted_class = np.argmax(y_output)
        true_class = np.argmax(d)

        if predicted_class == true_class:
            correct_predictions += 1

    accuracy = correct_predictions / len(testX)
    return accuracy

# Main function
def main():
    # Specify paths to MNIST files
    train_images_path = r"<INSERT FILE PATH>"
    train_labels_path = r"<INSERT FILE PATH>"
    test_images_path = r"<INSERT FILE PATH>"
    test_labels_path = r"<INSERT FILE PATH>"

    # Load MNIST data
    print("Loading MNIST dataset...")
    trainX = load_mnist_images(train_images_path)
    trainY = load_mnist_labels(train_labels_path)
    testX = load_mnist_images(test_images_path)
    testY = load_mnist_labels(test_labels_path)

    # Normalize images and one-hot encode labels
    trainX = trainX / 255.0
    testX = testX / 255.0
    trainY = np.eye(10)[trainY]
    testY = np.eye(10)[testY]

    # Split training data into training and validation sets
    trainX_train, trainX_val, trainY_train, trainY_val = train_test_split(
        trainX, trainY, test_size=0.15, random_state=42
    )

    # Parameters
    hidden_nodes = 15
    alpha = 0.1
    epochs = 50
    patience = 3

    # Train the MLP
    print("Training MLP...")
    w_hidden, b_hidden, w_output, b_output, train_errors, val_errors = train_mlp(
        trainX_train, trainY_train, trainX_val, trainY_val, hidden_nodes, alpha, epochs, patience
    )

    # Test the MLP
    print("Testing MLP...")
    test_accuracy = test_mlp(testX, testY, w_hidden, b_hidden, w_output, b_output)

    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Plot training and validation error with dual axes
    fig, ax1 = plt.subplots()

    ax1.plot(range(1, len(train_errors) + 1), train_errors, 'b-', label="Training Error")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Error", color="b")
    ax1.tick_params(axis='y', labelcolor="b")

    ax2 = ax1.twinx()
    ax2.plot(range(1, len(val_errors) + 1), val_errors, 'r-', label="Validation Error")
    ax2.set_ylabel("Validation Error", color="r")
    ax2.tick_params(axis='y', labelcolor="r")

    fig.suptitle("Training and Validation Error Over Epochs")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
