import numpy as np
import random
import matplotlib.pyplot as plt

# Step 1: Define the data (binary images of "1" and "0")
digit_1 = [
    [0, 0, 1, 0, 0,  # Pattern 1
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0],
    #Pattern 2
    [0, 1, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 1, 1, 1, 0],
    #Pattern 3
    [0, 0, 1, 0, 0,
     0, 1, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0],
    #Pattern 4
    [0, 0, 0, 1, 0,
     0, 0, 1, 1, 0,
     0, 0, 1, 0, 0,
     0, 1, 1, 0, 0,
     0, 1, 0, 0, 0],
    #Pattern 5
    [0, 0, 0, 0, 1,
     0, 0, 0, 1, 0,
     0, 0, 1, 0, 0,
     0, 1, 0, 0, 0,
     1, 0, 0, 0, 0],
    #Pattern 6
    [0, 0, 0, 0, 1,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0],
]

digit_0 = [
    [0, 1, 1, 1, 0,  #Pattern 1
     0, 1, 0, 1, 0,
     0, 1, 0, 1, 0,
     0, 1, 0, 1, 0,
     0, 1, 1, 1, 0],
    #Pattern 2
    [0, 0, 1, 0, 0,
     0, 1, 0, 1, 0,
     0, 1, 0, 1, 0,
     0, 1, 0, 1, 0,
     0, 0, 1, 0, 0],
    #Pattern 3
    [0, 0, 1, 0, 0,
     0, 1, 0, 1, 0,
     0, 1, 0, 1, 0,
     0, 1, 0, 1, 0,
     0, 1, 1, 1, 0],
    #Pattern 4
    [0, 0, 1, 1, 0,
     0, 1, 0, 0, 1,
     0, 1, 0, 0, 1,
     0, 1, 0, 1, 0,
     0, 1, 1, 0, 0],
    #Pattern 5
    [1, 1, 1, 1, 1,
     1, 0, 0, 0, 1,
     1, 0, 1, 0, 1,
     1, 0, 0, 0, 1,
     1, 1, 1, 1, 1],
    #Pattern 6
    [0, 1, 1, 1, 0,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     0, 1, 1, 1, 0],
]

# Function to prepare data (randomly split into training and test sets)
def prepare_data():
    all_data_1 = np.array(digit_1)  # All patterns for "1"
    all_data_0 = np.array(digit_0)  # All patterns for "0"

    # Randomly shuffle and split
    random_indices_1 = np.random.permutation(len(all_data_1))
    random_indices_0 = np.random.permutation(len(all_data_0))

    # Select training and test data
    train_1 = all_data_1[random_indices_1[:4]]  # 4 random patterns for "1"
    test_1 = all_data_1[random_indices_1[4:]]  # Remaining 2 patterns for "1"
    train_0 = all_data_0[random_indices_0[:4]]  # 4 random patterns for "0"
    test_0 = all_data_0[random_indices_0[4:]]  # Remaining 2 patterns for "0"

    # Combine and label data
    train_data = np.vstack((train_1, train_0))  # Combine training patterns
    train_labels = np.array([1] * 4 + [-1] * 4)  # Labels for training data

    test_data = np.vstack((test_1, test_0))  # Combine test patterns
    test_labels = np.array([1] * 2 + [-1] * 2)  # Labels for test data

    return train_data, train_labels, test_data, test_labels

# Function to initialize perceptron
def initialize_perceptron(input_dim):
    weights = np.random.rand(input_dim) * 0.01
    bias = np.random.rand() * 0.01
    return weights, bias

# Function to train perceptron
def train_perceptron(train_data, train_labels, alpha=0.1, epochs=10):
    weights, bias = initialize_perceptron(train_data.shape[1])
    for epoch in range(epochs):
        total_error = 0
        for x, d in zip(train_data, train_labels):
            y = np.sign(np.dot(weights, x) + bias)
            error = d - y
            total_error += abs(error)
            weights += alpha * error * x
            bias += alpha * error
        if total_error == 0:
            break
    return weights, bias

# Function to test perceptron
def test_perceptron(test_data, test_labels, weights, bias):
    predictions = [np.sign(np.dot(weights, x) + bias) for x in test_data]
    accuracy = np.mean(predictions == test_labels)
    return accuracy

# Main function to run trials
def run_trials(num_trials=50):
    accuracies = []

    for trial in range(num_trials):
        # Prepare random training and test data
        train_data, train_labels, test_data, test_labels = prepare_data()

        # Train the perceptron
        weights, bias = train_perceptron(train_data, train_labels)

        # Test the perceptron
        accuracy = test_perceptron(test_data, test_labels, weights, bias)*100
        accuracies.append(accuracy)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_trials + 1), accuracies, marker='o', label="Trial Accuracy")
    plt.axhline(np.mean(accuracies), color='r', linestyle='--', label=f"Mean Accuracy = {np.mean(accuracies):.2f}")
    plt.title("Perceptron Accuracy Over 50 Trials")
    plt.xlabel("Trial Number")
    plt.ylabel("Accuracy /%")
    plt.legend()
    plt.grid()
    plt.show()

    print(f"Mean Accuracy over {num_trials} trials: {np.mean(accuracies):.2f}")

# Run the trials
run_trials()