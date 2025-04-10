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

# Normalize images to [0, 1]
def normalize_images(images):
    return images / 255.0

# Convert labels to one-hot encoding
def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

# Sigmoid activation function
def sigmoid(v):
    return 1 / (1 + np.exp(-v))

# Initialize weights and biases
def initialize_weights(input_dim, num_classes):
    weights = np.random.rand(num_classes, input_dim) * 0.01  # Shape: (10, 784)
    biases = np.random.rand(num_classes) * 0.01             # Shape: (10,)
    return weights, biases

# Train the single-layer perceptron
def train_single_layer_perceptron(trainX, trainY, num_classes, alpha=0.1, epochs=10):
    num_samples, input_dim = trainX.shape
    weights, biases = initialize_weights(input_dim, num_classes)

    for epoch in range(epochs):
        total_error = 0

        # Shuffle data
        indices = np.random.permutation(num_samples)
        trainX, trainY = trainX[indices], trainY[indices]

        for x, d in zip(trainX, trainY):  # x: input vector, d: target output vector
            # Compute outputs for all perceptrons
            v = np.dot(weights, x) + biases  # Shape: (10,)
            y = sigmoid(v)  # Shape: (10,)

            # Update weights and biases for each perceptron
            for i in range(num_classes):
                error = d[i] - y[i]
                delta = alpha * error * y[i] * (1 - y[i])
                weights[i] += delta * x
                biases[i] += delta

            # Accumulate error for monitoring
            total_error += np.sum((d - y) ** 2)

        print(f"Epoch {epoch + 1}/{epochs}, Total Error: {total_error:.4f}")

    return weights, biases

# Test the single-layer perceptron
def test_single_layer_perceptron(testX, testY, weights, biases):
    num_samples = testX.shape[0]
    correct_predictions = 0

    for x, d in zip(testX, testY):
        # Compute outputs for all perceptrons
        v = np.dot(weights, x) + biases  # Shape: (10,)
        y = sigmoid(v)  # Shape: (10,)

        # Predicted class is the index with the maximum output
        predicted_class = np.argmax(y)
        true_class = np.argmax(d)

        if predicted_class == true_class:
            correct_predictions += 1

    accuracy = correct_predictions / num_samples
    return accuracy

# Main function to train and test the perceptron
def main(epochs=10, alpha=0.1):
    # Paths to the MNIST files
    train_images_path = r"<INSERT FILE PATH>"
    train_labels_path = r"<INSERT FILE PATH>"
    test_images_path = r"<INSERT FILE PATH>"
    test_labels_path = r"<INSERT FILE PATH>"

    # Load the MNIST dataset
    print("Loading MNIST dataset...")
    trainX = load_mnist_images(train_images_path)
    trainY = load_mnist_labels(train_labels_path)
    testX = load_mnist_images(test_images_path)
    testY = load_mnist_labels(test_labels_path)

    # Normalize images and convert labels to one-hot encoding
    trainX = normalize_images(trainX)
    trainY_one_hot = one_hot_encode(trainY)
    testX = normalize_images(testX)
    testY_one_hot = one_hot_encode(testY)

    # Use 10% of the training set and test set
    trainX_subset = trainX[:6000]
    trainY_subset = trainY_one_hot[:6000]
    testX_subset = testX[:1000]
    testY_subset = testY_one_hot[:1000]

    # Train the perceptron
    print("Training the single-layer perceptron...")
    weights, biases = train_single_layer_perceptron(trainX_subset, trainY_subset, num_classes=10, alpha=alpha, epochs=epochs)

    # Test the perceptron
    print("Testing the single-layer perceptron...")
    train_accuracy = test_single_layer_perceptron(trainX_subset, trainY_subset, weights, biases)
    test_accuracy = test_single_layer_perceptron(testX_subset, testY_subset, weights, biases)

    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    return weights, biases, train_accuracy, test_accuracy

# Run the program with adjustable parameters
if __name__ == "__main__":
    # Example: Change epochs and learning rate here
    weights, biases, train_accuracy, test_accuracy = main(epochs=10, alpha=0.05)
