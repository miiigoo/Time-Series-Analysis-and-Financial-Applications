import numpy as np
import struct
import matplotlib.pyplot as plt

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
def train_mlp(trainX, trainY, hidden_nodes, alpha, epochs):
    input_size = trainX.shape[1]
    output_size = trainY.shape[1]

    w_hidden, b_hidden, w_output, b_output = initialize_weights(input_size, hidden_nodes, output_size)

    for epoch in range(epochs):
        total_error = 0
        for x, d in zip(trainX, trainY):
            v_hidden, y_hidden, v_output, y_output = forward_propagation(x, w_hidden, b_hidden, w_output, b_output)
            w_hidden, b_hidden, w_output, b_output, error = backward_propagation(
                x, d, v_hidden, y_hidden, y_output, w_hidden, b_hidden, w_output, b_output, alpha
            )
            total_error += error

    return w_hidden, b_hidden, w_output, b_output

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
    # Paths to MNIST files
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

    # Parameters
    hidden_nodes = 15  # Fixed number of hidden nodes
    alpha = 0.1  # Learning rate
    epoch_list = [10, 15, 20, 25]  # Number of epochs to test

    train_accuracies = []
    test_accuracies = []

    # Iterate over different numbers of epochs
    for epochs in epoch_list:
        print(f"Training with {epochs} epochs...")
        w_hidden, b_hidden, w_output, b_output = train_mlp(trainX, trainY, hidden_nodes, alpha, epochs)

        # Calculate accuracies
        train_accuracy = test_mlp(trainX, trainY, w_hidden, b_hidden, w_output, b_output)
        test_accuracy = test_mlp(testX, testY, w_hidden, b_hidden, w_output, b_output)

        train_accuracies.append(train_accuracy * 100)  # Convert to percentage
        test_accuracies.append(test_accuracy * 100)    # Convert to percentage

        print(f"Epochs: {epochs}, Training Accuracy: {train_accuracy * 100:.2f}%, Test Accuracy: {test_accuracy * 100:.2f}%")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_list, train_accuracies, marker='o', label='Training Accuracy')
    plt.plot(epoch_list, test_accuracies, marker='x', label='Test Accuracy')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Number of Epochs")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
